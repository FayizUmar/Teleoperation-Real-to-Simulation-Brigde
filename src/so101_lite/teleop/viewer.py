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

import numpy as np

from so101_lite.teleop.follower import SimFollower
from so101_lite.teleop.leader import (
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


def run_teleop(cfg: TeleopConfig) -> Path | None:
    """Run the interactive teleop loop; return the dataset root if anything saved."""
    import cv2

    _say(f"[1/3] Connecting to leader arm on {cfg.port or '<no port>'} ...")
    leader = get_leader(cfg.robot_type, cfg.port, cfg.leader_id)
    try:
        leader.connect()
    except Exception as exc:  # noqa: BLE001 — surface a friendly hint
        raise RuntimeError(format_leader_connection_error(cfg.port, exc)) from exc
    _say("      leader connected.")

    follower = SimFollower(
        cfg.env_id,
        wrist_wh=cfg.wrist_wh,
        overhead_wh=cfg.overhead_wh,
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
    writer: DatasetWriter | None = None
    saved_any = False
    task = getattr(follower.env.unwrapped, "task_description", "") or cfg.env_id

    state = RecordingState(task_description=task)
    passive = _try_launch_passive(follower) if cfg.show_windows else None

    if cfg.show_windows:
        # Create + position the windows up front so they appear immediately and
        # don't get lost behind the terminal on macOS.
        cv2.namedWindow("global", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("wrist", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("global", 60, 60)
        cv2.moveWindow("wrist", 60 + cfg.overhead_wh[0] + 30, 60)
        cv2.waitKey(1)

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

    try:
        while not quit_requested:
            loop_start = time.monotonic()

            leader_action = apply_wrist_roll_offset_deg(
                leader.get_action(), cfg.wrist_roll_offset_deg
            )
            sent = follower.send_action(leader_action)
            obs = follower.get_observation()
            wrist = obs.get("wrist")
            overhead = obs.get("overhead")

            if state.is_recording:
                state.append(action=sent, state=obs, wrist=wrist, overhead=overhead)
                info = follower.last_step_info()
                if _should_autostop(state, info, cfg):
                    print("  task success held; auto-stopping episode")
                    if stop_and_save():
                        break
                    _reset_scene(follower, leader, state, cfg)

            if cfg.show_windows:
                _draw(cv2, wrist, overhead, state)
                key = cv2.waitKey(1) & 0xFF
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
        if state.is_recording:
            stop_and_save()
        if writer is not None:
            writer.finalize()
        if passive is not None:
            passive.close()
        if cfg.show_windows:
            cv2.destroyAllWindows()
        follower.disconnect()
        leader.disconnect()

    if saved_any and writer is not None:
        root = Path(writer.dataset.root)
        print(f"Dataset written to {root}")
        return root
    print("No episodes saved.")
    return None


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
            stop_and_save()
        else:
            state.clear_episode()
            state.is_recording = True
            print("  recording...")
    elif key == ord("n"):
        stop_and_save()
        _reset_scene(follower, leader, state, cfg)
        print("  next episode")
    elif key == ord("r"):
        state.is_recording = False
        _reset_scene(follower, leader, state, cfg)
        print("  reset (discarded)")
    return False


def _draw(cv2, wrist, overhead, state: RecordingState) -> None:
    """Render live camera windows with a recording indicator."""
    if wrist is not None:
        img = _to_bgr(wrist)
        if state.is_recording:
            cv2.circle(img, (12, 12), 6, (0, 0, 255), -1)
        cv2.imshow("wrist", img)
    if overhead is not None:
        img = _to_bgr(overhead)
        label = f"REC {state.num_frames}" if state.is_recording else "idle"
        cv2.putText(
            img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        cv2.imshow("global", img)
