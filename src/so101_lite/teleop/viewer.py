"""Interactive teleop + recording loop for so101-lite.

Live camera feeds (wrist + global/overhead) render to OpenCV windows so this
works from a plain ``python`` interpreter on macOS (the MuJoCo passive 3D
viewer needs ``mjpython`` and is opened opportunistically as a bonus).

Keys (focus a camera window):
    SPACE  start / stop+save the current episode
    n      stop+save current episode, then reset a fresh scene
    r      discard current recording and reset the scene
    q/ESC  quit (finalizes the dataset)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
import so101_lite.envs  # registers all 6 gym env IDs
import gymnasium as gym
import numpy as np

from so101_lite.teleop.follower import SimFollower
from so101_lite.teleop.leader import (
    DummyLeader,
    apply_wrist_roll_offset_deg,
    format_leader_connection_error,
    get_leader,
)
from so101_lite.teleop.recorder import DatasetWriter, RecordingState

logger = logging.getLogger(__name__)

# ASCII keycodes returned by cv2.waitKey.
_KEY_SPACE = 32
_KEY_ESC = 27


@dataclass
class TeleopConfig:
    """User-facing knobs for an interactive teleop recording session."""

    env_id: str = "MuJoCoChessBoard-v1"
    port: str = ""
    robot_type: str = "so101"
    leader_id: str = "so101_leader"
    repo_id: str = ""
    fps: int = 30
    wrist_wh: tuple[int, int] = (320, 240)
    overhead_wh: tuple[int, int] = (640, 360)
    wrist_roll_offset_deg: float = -90.0
    dataset_root: str | None = None
    num_episodes: int = 0  # 0 == unlimited; stop with 'q'
    success_hold_seconds: float = 0.5
    show_windows: bool = True
    sim_only: bool = False
    world_config_path: str | None = None  # Python file defining WORLD = WorldConfig(...)
    scene_socket: str | None = None  # path enables the runtime scene-control socket
    second_camera: str = "global"  # which 2nd camera to RECORD: "global" or "overhead"


def _to_bgr(rgb: np.ndarray):
    import cv2

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _try_launch_passive(follower: SimFollower):
    """Open the MuJoCo passive 3D viewer if the platform supports it."""
    try:
        import mujoco.viewer

        unwrapped = follower.env.unwrapped
        return mujoco.viewer.launch_passive(unwrapped.model, unwrapped.data)
    except Exception as exc:  # noqa: BLE001 — optional convenience only
        logger.info("MuJoCo passive viewer unavailable (%s); using camera windows.", exc)
        return None


def _say(msg: str) -> None:
    """Print an immediately-flushed progress line during the slow startup."""
    print(msg, flush=True)


class _TerminalKeys:
    """Read single keypresses from the terminal without blocking.

    macOS OpenCV builds often fail to capture keys via ``cv2.waitKey`` unless a
    camera window is focused (and sometimes not even then). Reading stdin in
    cbreak mode lets SPACE/n/r/q work straight from the terminal regardless of
    which window has focus. No-ops when stdin is not a TTY.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._old = None
        self._fd = None

    def __enter__(self) -> "_TerminalKeys":
        import sys

        try:
            import termios
            import tty

            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
                self._old = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                self._enabled = True
        except Exception:  # noqa: BLE001 — fall back to cv2-only key capture
            self._enabled = False
        return self

    def read_key(self) -> int:
        """Return an ASCII keycode if one is waiting, else -1 (non-blocking)."""
        if not self._enabled:
            return -1
        import select
        import sys

        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            return ord(ch) if ch else -1
        return -1

    def __exit__(self, *exc) -> None:
        if self._enabled and self._old is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)


def run_teleop(cfg: TeleopConfig) -> Path | None:
    """Run the interactive teleop loop; return the dataset root if anything saved."""
    import cv2

    if cfg.sim_only:
        _say("[1/3] Sim-only mode — no leader arm required.")
        leader = DummyLeader()
        leader.connect()
        _say("      dummy leader ready.")
    else:
        _say(f"[1/3] Connecting to leader arm on {cfg.port or '<no port>'} ...")
        leader = get_leader(cfg.robot_type, cfg.port, cfg.leader_id)
        try:
            leader.connect()
        except Exception as exc:  # noqa: BLE001 — surface a friendly hint
            raise RuntimeError(format_leader_connection_error(cfg.port, exc)) from exc
        _say("      leader connected.")

    resolved_config = None
    if cfg.world_config_path:
        from so101_lite.world import build_chess_config, load_world_config

        world = load_world_config(cfg.world_config_path)
        resolved_config = build_chess_config(world)
        _say(f"      loaded world config from {cfg.world_config_path}")

    follower = SimFollower(
        cfg.env_id,
        wrist_wh=cfg.wrist_wh,
        overhead_wh=cfg.overhead_wh,
        config=resolved_config,
    )
    # Seed the sim at the operator's current physical pose so episode 1 doesn't
    # start with the arm snapping from the rest pose.
    try:
        first = apply_wrist_roll_offset_deg(leader.get_action(), cfg.wrist_roll_offset_deg)
        follower.set_initial_leader_action(first)
    except Exception:  # noqa: BLE001
        pass
    _say(f"[2/3] Building simulator '{cfg.env_id}' (loading model, first render) ...")
    follower.connect()
    _say("      simulator ready.")

    repo_id = cfg.repo_id or _default_repo_id(cfg.env_id)
    resolved_root = _resolve_dataset_root(cfg.dataset_root)
    if resolved_root != cfg.dataset_root:
        _say(
            f"      note: '{cfg.dataset_root}' already exists; "
            f"recording to '{resolved_root}' instead."
        )
    cfg.dataset_root = resolved_root
    writer: DatasetWriter | None = None
    saved_any = False
    task = getattr(follower.env.unwrapped, "task_description", "") or cfg.env_id

    state = RecordingState(task_description=task)
    passive = _try_launch_passive(follower)

    scene_server = None
    if cfg.scene_socket:
        from so101_lite.teleop.scene_api import SceneController, SceneSocketServer

        unwrapped = follower.env.unwrapped
        controller = SceneController(unwrapped.model, unwrapped.data, env=unwrapped)
        scene_server = SceneSocketServer(controller, socket_path=cfg.scene_socket)
        scene_server.start()

    use_cv = False
    if cfg.show_windows:
        try:
            cv2.namedWindow("global", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("wrist", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("overhead", cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow("global", 60, 60)
            cv2.moveWindow("wrist", 60 + cfg.overhead_wh[0] + 30, 60)
            cv2.moveWindow("overhead", 60 + cfg.overhead_wh[0] + 30 + cfg.wrist_wh[0] + 30, 60)
            cv2.waitKey(1)
            use_cv = True
        except cv2.error:
            print(
                "  OpenCV windows unavailable (mjpython/macOS conflict) — "
                "MuJoCo 3D viewer active. Close its window to quit."
            )

    quit_requested = False
    episodes_saved = 0
    frame_dt = 1.0 / cfg.fps

    def stop_and_save() -> bool:
        """Save the in-progress episode. Return True if the quota is now met."""
        nonlocal writer, saved_any, episodes_saved
        if not state.is_recording:
            return False
        state.is_recording = False
        if state.num_frames == 0:
            print("  (no frames recorded; nothing saved)")
            return False
        if writer is None:
            writer = DatasetWriter(
                repo_id,
                fps=cfg.fps,
                root=cfg.dataset_root,
                wrist_wh=cfg.wrist_wh,
                overhead_wh=cfg.overhead_wh,
            )
        n = writer.add_episode(state, task)
        saved_any = True
        episodes_saved += 1
        print(f"  saved episode {episodes_saved} ({n} frames)")
        return bool(cfg.num_episodes) and episodes_saved >= cfg.num_episodes

    _say(
        f"[3/3] Teleop ready on {cfg.env_id} via {cfg.port or '<no port>'}.\n"
        "      (camera windows may open BEHIND this terminal -- Cmd+Tab to find them)\n"
        "      SPACE start/stop episode | n save+next | r reset | q quit"
    )

    _say("      TIP: keys also work from THIS terminal (no need to focus a window).")

    term_keys = _TerminalKeys()
    term_keys.__enter__()
    try:
        while not quit_requested:
            loop_start = time.monotonic()

            if scene_server is not None:
                scene_server.drain()  # apply any queued move/hide/show commands

            leader_action = apply_wrist_roll_offset_deg(
                leader.get_action(), cfg.wrist_roll_offset_deg
            )
            sent = follower.send_action(leader_action)
            obs = follower.get_observation()
            wrist = obs.get("wrist")
            global_img = obs.get("global")
            overhead = obs.get("overhead")

            if state.is_recording:
                if cfg.second_camera == "overhead":
                    second = overhead if overhead is not None else global_img
                else:
                    second = global_img if global_img is not None else overhead
                state.append(
                    action=sent,
                    state=obs,
                    wrist=wrist,
                    overhead=second,
                    qpos=follower.current_qpos(),
                )
                info = follower.last_step_info()
                if _should_autostop(state, info, cfg):
                    print("  task success held; auto-stopping episode")
                    if stop_and_save():
                        break
                    _reset_scene(follower, leader, state, cfg)

            key = -1
            if use_cv:
                _draw(cv2, wrist, global_img, overhead, state)
                key = cv2.waitKey(1) & 0xFF
            # Terminal keypress takes precedence (robust on macOS).
            term_key = term_keys.read_key()
            if term_key != -1:
                key = term_key
            if key not in (-1, 255):
                quit_requested = _handle_key(
                    key, state, follower, leader, cfg, stop_and_save
                )
            if passive is not None:
                passive.sync()
                if not passive.is_running():
                    quit_requested = True

            elapsed = time.monotonic() - loop_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    finally:
        try:
            term_keys.__exit__()
        except Exception:  # noqa: BLE001
            pass
        if state.is_recording:
            stop_and_save()
        if writer is not None:
            writer.finalize()
        if scene_server is not None:
            scene_server.stop()
        if passive is not None:
            passive.close()
        if use_cv:
            cv2.destroyAllWindows()  # closes wrist, global, overhead
        follower.disconnect()
        leader.disconnect()

    if saved_any and writer is not None:
        root = Path(writer.dataset.root)
        print(f"Dataset written to {root}")
        return root
    print("No episodes saved.")
    return None


def _resolve_dataset_root(root: str | None) -> str | None:
    """If *root* already exists, return a fresh sibling (``demo1`` -> ``demo1_2``).

    LeRobot's ``create()`` refuses to write into an existing dataset dir, so we
    pick a unique path up front to avoid crashing (and losing a take) at save.
    """
    if not root:
        return root
    p = Path(root)
    if not p.exists():
        return root
    i = 2
    while (cand := p.parent / f"{p.name}_{i}").exists():
        i += 1
    return str(cand)


def _default_repo_id(env_id: str) -> str:
    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = env_id.replace("/", "-").replace(" ", "_")
    return f"local/teleop-{safe}-{ts}"


def _should_autostop(state: RecordingState, info, cfg: TeleopConfig) -> bool:
    """Stop after a successful placement is held for ``success_hold_seconds``."""
    if info is None or not info.terminated:
        return False
    if state.terminated_at_frame is None:
        state.terminated_at_frame = state.num_frames
    hold_frames = max(0, round(cfg.success_hold_seconds * cfg.fps))
    return (state.num_frames - state.terminated_at_frame) >= hold_frames


def _reset_scene(follower: SimFollower, leader, state: RecordingState, cfg: TeleopConfig) -> None:
    """Randomize a fresh scene, seeding the arm at the operator's current pose."""
    state.clear_episode()
    options = None
    try:
        first = apply_wrist_roll_offset_deg(leader.get_action(), cfg.wrist_roll_offset_deg)
        options = {"init_qpos": follower._leader_to_qpos(first)}
    except Exception:  # noqa: BLE001 — fall back to the env's default rest pose
        options = None
    follower.env.reset(options=options)


def _handle_key(key, state, follower, leader, cfg, stop_and_save) -> bool:
    """Apply a keypress. Return True to request quit."""
    if key in (ord("q"), _KEY_ESC):
        return True
    if key == _KEY_SPACE:
        if state.is_recording:
            _say("  ■ STOPPED — saving episode ...")
            stop_and_save()
        else:
            state.clear_episode()
            state.replay_meta = follower.scene_metadata()
            state.replay_meta["wrist_cam"] = follower.wrist_camera_state()
            state.replay_meta["fps"] = cfg.fps
            state.replay_meta["second_camera"] = cfg.second_camera
            state.replay_meta["marks"] = follower.current_marks()
            state.is_recording = True
            _say("  ● RECORDING STARTED — move the arm, press SPACE again to save")
    elif key == ord("n"):
        stop_and_save()
        _reset_scene(follower, leader, state, cfg)
        _say("  → next episode")
    elif key == ord("r"):
        state.is_recording = False
        _reset_scene(follower, leader, state, cfg)
        _say("  ↺ reset (discarded)")
    return False


def _status_banner(cv2, img, state: RecordingState) -> None:
    """Draw a prominent REC / IDLE banner across the top of a camera window."""
    w = img.shape[1]
    if state.is_recording:
        cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 210), -1)  # red bar (BGR)
        cv2.circle(img, (16, 15), 7, (255, 255, 255), -1)
        cv2.putText(
            img, f"REC  {state.num_frames}", (32, 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
    else:
        cv2.rectangle(img, (0, 0), (w, 30), (55, 55, 55), -1)  # grey bar
        cv2.putText(
            img, "IDLE - press SPACE here to record", (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1,
        )


def _draw(cv2, wrist, global_img, overhead, state: RecordingState) -> None:
    """Render live camera windows with a prominent recording indicator."""
    if wrist is not None:
        img = _to_bgr(wrist)
        _status_banner(cv2, img, state)
        cv2.imshow("wrist", img)
    if global_img is not None:
        img = _to_bgr(global_img)
        _status_banner(cv2, img, state)
        cv2.imshow("global", img)
    if overhead is not None:
        img = _to_bgr(overhead)
        _status_banner(cv2, img, state)
        cv2.imshow("overhead", img)
