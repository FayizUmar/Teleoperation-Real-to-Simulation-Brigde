"""Runtime scene control for so101-lite (Gazebo-style move / hide / show).

Commands reach the **running** sim over a local Unix domain socket — no HTTP,
no network, just a socket file (default ``/tmp/so101_scene.sock``). A second
terminal uses ``so101-lite scene ...`` to talk to it.

MuJoCo compiles its model once and freezes it, so unlike Gazebo we cannot
truly add or delete bodies at runtime. What we *can* do live:

  * **move**  a body  (free-joint bodies via ``data.qpos``; static bodies via
    ``model.body_pos``)
  * **hide**  ("delete") a body by parking it far off-world and disabling its
    geoms' collision + rendering
  * **show**  ("spawn") it back at its saved pose with collision restored

Changing geometry (board size, new object types) still needs a fresh env.

Thread-safety: socket requests land on a background thread and are pushed onto
a queue. The sim loop calls :meth:`SceneSocketServer.drain` every tick and
applies the mutations on the physics thread, so model/data are never touched
concurrently.

Wire protocol: one newline-terminated JSON object per request, one JSON object
per response. Requests:

    {"action": "list"}
    {"action": "info", "name": "chess_piece"}
    {"action": "move", "name": "chess_piece", "x": .., "y": .., "z": ..}
    {"action": "hide", "name": "chess_piece"}
    {"action": "show", "name": "chess_piece"}
"""

from __future__ import annotations

import json
import logging
import os
import queue
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# Default socket path the sim binds and the `scene` CLI connects to.
DEFAULT_SOCKET_PATH = "/tmp/so101_scene.sock"

# Where hidden bodies are parked (far from the scene, below the floor).
_PARK_POS = np.array([100.0, 100.0, -10.0])


@dataclass
class _BodyInfo:
    """Cached per-body handles + saved render/collision state for hide/show."""

    name: str
    body_id: int
    is_free: bool
    qpos_addr: int | None  # free joint qpos start (3 pos + 4 quat) or None
    dof_addr: int | None  # free joint qvel start (6) or None
    geom_ids: list[int]
    saved_contype: list[int] = field(default_factory=list)
    saved_conaffinity: list[int] = field(default_factory=list)
    saved_alpha: list[float] = field(default_factory=list)
    saved_pos: np.ndarray | None = None
    hidden: bool = False


class SceneController:
    """Moves / hides / shows bodies in a live MuJoCo model+data.

    All methods MUST be called from the physics thread (the sim loop), which is
    guaranteed by routing them through :class:`SceneSocketServer`'s queue.
    """

    def __init__(self, model: Any, data: Any, env: Any = None) -> None:
        self.model = model
        self.data = data
        self._env = env  # unwrapped gym env, for source/target square marks
        self._bodies: dict[str, _BodyInfo] = {}
        self._scan_bodies()

    def _scan_bodies(self) -> None:
        model = self.model
        for bid in range(1, model.nbody):  # skip world body (0)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if name is None:
                continue

            is_free = False
            qpos_addr: int | None = None
            dof_addr: int | None = None
            jnt_num = int(model.body_jntnum[bid])
            jnt_adr = int(model.body_jntadr[bid])
            for j in range(jnt_adr, jnt_adr + jnt_num):
                if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                    is_free = True
                    qpos_addr = int(model.jnt_qposadr[j])
                    dof_addr = int(model.jnt_dofadr[j])
                    break

            geom_adr = int(model.body_geomadr[bid])
            geom_num = int(model.body_geomnum[bid])
            geom_ids = list(range(geom_adr, geom_adr + geom_num))

            self._bodies[name] = _BodyInfo(
                name=name,
                body_id=bid,
                is_free=is_free,
                qpos_addr=qpos_addr,
                dof_addr=dof_addr,
                geom_ids=geom_ids,
            )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def _require(self, name: str) -> _BodyInfo:
        info = self._bodies.get(name)
        if info is None:
            raise KeyError(f"unknown body '{name}'. Known: {sorted(self._bodies)}")
        return info

    def _world_pos(self, info: _BodyInfo) -> list[float]:
        return [round(float(v), 5) for v in self.data.xpos[info.body_id]]

    # ------------------------------------------------------------------
    # Mutations (physics-thread only)
    # ------------------------------------------------------------------

    def _set_pos(self, info: _BodyInfo, pos: np.ndarray) -> None:
        if info.is_free and info.qpos_addr is not None:
            a = info.qpos_addr
            self.data.qpos[a : a + 3] = pos
            if info.dof_addr is not None:
                self.data.qvel[info.dof_addr : info.dof_addr + 6] = 0.0
        else:
            self.model.body_pos[info.body_id] = pos
        mujoco.mj_forward(self.model, self.data)

    def move(self, name: str, x: float, y: float, z: float) -> dict:
        info = self._require(name)
        self._set_pos(info, np.array([x, y, z], dtype=float))
        return {"name": name, "position": self._world_pos(info), "hidden": info.hidden}

    def hide(self, name: str) -> dict:
        info = self._require(name)
        if info.hidden:
            return {"name": name, "hidden": True, "note": "already hidden"}

        # Remember where it is now so show() can restore it.
        if info.is_free and info.qpos_addr is not None:
            a = info.qpos_addr
            info.saved_pos = np.array(self.data.qpos[a : a + 3], dtype=float)
        else:
            info.saved_pos = np.array(self.model.body_pos[info.body_id], dtype=float)

        # Disable collision + rendering for every geom on the body.
        info.saved_contype = [int(self.model.geom_contype[g]) for g in info.geom_ids]
        info.saved_conaffinity = [
            int(self.model.geom_conaffinity[g]) for g in info.geom_ids
        ]
        info.saved_alpha = [float(self.model.geom_rgba[g, 3]) for g in info.geom_ids]
        for g in info.geom_ids:
            self.model.geom_contype[g] = 0
            self.model.geom_conaffinity[g] = 0
            self.model.geom_rgba[g, 3] = 0.0

        self._set_pos(info, _PARK_POS)
        info.hidden = True
        return {"name": name, "hidden": True}

    def show(self, name: str) -> dict:
        info = self._require(name)
        if not info.hidden:
            return {"name": name, "hidden": False, "note": "already visible"}

        for idx, g in enumerate(info.geom_ids):
            self.model.geom_contype[g] = info.saved_contype[idx]
            self.model.geom_conaffinity[g] = info.saved_conaffinity[idx]
            self.model.geom_rgba[g, 3] = info.saved_alpha[idx]

        if info.saved_pos is not None:
            self._set_pos(info, info.saved_pos)
        info.hidden = False
        return {"name": name, "hidden": False, "position": self._world_pos(info)}

    def info(self, name: str) -> dict:
        info = self._require(name)
        return {
            "name": name,
            "position": self._world_pos(info),
            "hidden": info.hidden,
            "free_joint": info.is_free,
        }

    # -- source/target square masks (overhead-only visualization) -------

    def mark(self, kind: str, square: str) -> dict:
        if self._env is None or not hasattr(self._env, "set_mark"):
            raise RuntimeError("this env does not support source/target marks")
        return self._env.set_mark(kind, square)

    def mark_clear(self) -> dict:
        if self._env is None or not hasattr(self._env, "clear_marks"):
            raise RuntimeError("this env does not support source/target marks")
        return self._env.clear_marks()

    def list_bodies(self) -> dict:
        return {
            "bodies": [
                {
                    "name": info.name,
                    "position": self._world_pos(info),
                    "hidden": info.hidden,
                    "free_joint": info.is_free,
                }
                for info in self._bodies.values()
            ]
        }


@dataclass
class _Command:
    fn: Callable[[SceneController], Any]
    event: threading.Event
    result: Any = None
    error: str | None = None


def _recv_line(conn: socket.socket) -> str:
    """Read bytes until a newline; return the decoded line (without the \\n)."""
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf.split(b"\n", 1)[0].decode()


def _send_line(conn: socket.socket, payload: dict) -> None:
    conn.sendall((json.dumps(payload) + "\n").encode())


class SceneSocketServer:
    """Unix-socket front-end that queues commands for the sim loop to apply."""

    def __init__(self, controller: SceneController, socket_path: str = DEFAULT_SOCKET_PATH) -> None:
        self._controller = controller
        self._queue: "queue.Queue[_Command]" = queue.Queue()
        self.socket_path = socket_path
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.socket_path)
        self._sock.listen(8)
        self._sock.settimeout(0.5)  # so the accept loop can notice _stop
        self._thread = threading.Thread(
            target=self._serve, name="so101-scene-sock", daemon=True
        )
        self._thread.start()
        print(
            f"  scene API listening on unix:{self.socket_path}\n"
            "      try:  so101-lite scene list  |  scene move <name> x y z  |  "
            "scene hide/show <name>"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

    def drain(self) -> None:
        """Apply all queued commands. Call once per sim-loop iteration."""
        while True:
            try:
                cmd = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                cmd.result = cmd.fn(self._controller)
            except Exception as exc:  # noqa: BLE001 — report back to caller
                cmd.error = str(exc)
            finally:
                cmd.event.set()

    # -- serving --------------------------------------------------------

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            try:
                line = _recv_line(conn)
            except OSError:
                return
            if not line:
                return
            try:
                req = json.loads(line)
            except ValueError as exc:
                _send_line(conn, {"ok": False, "error": f"bad JSON: {exc}"})
                return
            payload = self._dispatch(req)
            try:
                _send_line(conn, payload)
            except OSError:
                pass

    def _dispatch(self, req: dict) -> dict:
        action = req.get("action")
        if action == "list":
            return self._submit(lambda c: c.list_bodies())
        if action == "info":
            name = req.get("name", "")
            return self._submit(lambda c: c.info(name))
        if action == "move":
            name = req.get("name", "")
            try:
                x, y, z = float(req["x"]), float(req["y"]), float(req["z"])
            except (KeyError, TypeError, ValueError):
                return {"ok": False, "error": "move requires numeric x, y, z"}
            return self._submit(lambda c: c.move(name, x, y, z))
        if action == "hide":
            name = req.get("name", "")
            return self._submit(lambda c: c.hide(name))
        if action == "show":
            name = req.get("name", "")
            return self._submit(lambda c: c.show(name))
        if action == "mark":
            kind = req.get("kind", "")
            square = req.get("square", "")
            return self._submit(lambda c: c.mark(kind, square))
        if action == "mark_clear":
            return self._submit(lambda c: c.mark_clear())
        return {"ok": False, "error": f"unknown action {action!r}"}

    def _submit(self, fn: Callable[[SceneController], Any], timeout: float = 2.0) -> dict:
        cmd = _Command(fn=fn, event=threading.Event())
        self._queue.put(cmd)
        if not cmd.event.wait(timeout):
            return {"ok": False, "error": "sim loop did not process command (paused/quit?)"}
        if cmd.error is not None:
            return {"ok": False, "error": cmd.error}
        return {"ok": True, "result": cmd.result}


# ----------------------------------------------------------------------
# Client (used by the `so101-lite scene ...` CLI subcommand)
# ----------------------------------------------------------------------


def send_command(request: dict, socket_path: str = DEFAULT_SOCKET_PATH, timeout: float = 5.0) -> dict:
    """Connect to a running sim's socket, send one request, return its reply."""
    if not os.path.exists(socket_path):
        raise ConnectionError(
            f"no sim socket at {socket_path}. Is the sim running with --scene-socket?"
        )
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(socket_path)
        s.sendall((json.dumps(request) + "\n").encode())
        line = _recv_line(s)
    finally:
        s.close()
    if not line:
        raise ConnectionError("sim closed the connection without replying")
    return json.loads(line)
