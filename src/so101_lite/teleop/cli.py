"""Command-line entry point for so101-lite teleop recording."""

from __future__ import annotations

import argparse
import os
import sys


def _setup_backend() -> None:
    """Select a MuJoCo GL backend before any rendering happens."""
    if "MUJOCO_GL" not in os.environ and sys.platform == "darwin":
        os.environ["MUJOCO_GL"] = "cgl"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="so101-lite",
        description="Lean SO101 MuJoCo teleop recorder (live camera windows + keypress control).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("teleop", help="Run an interactive teleop recording session.")
    t.add_argument("--env-id", default="MuJoCoChessBoard-v1", help="Gym env id to record.")
    t.add_argument("--port", default="", help="Leader-arm serial port, e.g. /dev/tty.usbmodemXXXX.")
    t.add_argument("--robot-type", default="so101", choices=["so100", "so101"])
    t.add_argument("--leader-id", default="so101_leader", help="LeRobot leader calibration id.")
    t.add_argument("--repo-id", default="", help="Dataset repo id (default: local/teleop-<env>-<ts>).")
    t.add_argument("--fps", type=int, default=30)
    t.add_argument("--wrist-res", default="320x240", help="Wrist camera WxH.")
    t.add_argument("--overhead-res", default="640x360", help="Global/overhead camera WxH.")
    t.add_argument("--wrist-roll-offset-deg", type=float, default=-90.0)
    t.add_argument("--dataset-root", default=None, help="Where to write the dataset on disk.")
    t.add_argument("--num-episodes", type=int, default=0, help="0 = unlimited (stop with 'q').")
    t.add_argument("--success-hold-seconds", type=float, default=0.5)
    t.add_argument("--no-windows", action="store_true", help="Headless: no live camera windows.")
    return p


def _parse_res(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def main(argv: list[str] | None = None) -> int:
    _setup_backend()
    args = build_parser().parse_args(argv)

    if args.command == "teleop":
        from so101_lite.teleop.viewer import TeleopConfig, run_teleop

        cfg = TeleopConfig(
            env_id=args.env_id,
            port=args.port,
            robot_type=args.robot_type,
            leader_id=args.leader_id,
            repo_id=args.repo_id,
            fps=args.fps,
            wrist_wh=_parse_res(args.wrist_res),
            overhead_wh=_parse_res(args.overhead_res),
            wrist_roll_offset_deg=args.wrist_roll_offset_deg,
            dataset_root=args.dataset_root,
            num_episodes=args.num_episodes,
            success_hold_seconds=args.success_hold_seconds,
            show_windows=not args.no_windows,
        )
        run_teleop(cfg)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
