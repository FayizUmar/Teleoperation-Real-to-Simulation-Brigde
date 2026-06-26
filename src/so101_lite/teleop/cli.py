"""Command-line entry point for so101-lite teleop recording."""

from __future__ import annotations

import argparse
import json
import os
import sys

# Kept in sync with scene_api.DEFAULT_SOCKET_PATH; duplicated here so building the
# parser / `scene` client does not import mujoco.
DEFAULT_SOCKET_PATH = "/tmp/so101_scene.sock"


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
    t.add_argument("--sim", action="store_true", help="Run without a physical leader arm (DummyLeader).")
    t.add_argument(
        "--second-camera",
        choices=["global", "overhead"],
        default="global",
        help="Which 2nd camera to record alongside wrist (default: global).",
    )
    t.add_argument(
        "--world-config",
        default=None,
        help="Python file defining WORLD = WorldConfig(...) for board + camera layout.",
    )
    t.add_argument(
        "--scene-socket",
        nargs="?",
        const=DEFAULT_SOCKET_PATH,
        default=None,
        help=(
            "Enable runtime scene control over a local socket. Bare flag uses "
            f"{DEFAULT_SOCKET_PATH}; pass a path to override."
        ),
    )

    # ---- scene: companion command to drive a running sim over the socket ----
    s = sub.add_parser("scene", help="Send a command to a running teleop sim.")
    s.add_argument(
        "--socket",
        default=DEFAULT_SOCKET_PATH,
        help=f"Socket path the sim is listening on (default {DEFAULT_SOCKET_PATH}).",
    )
    s_sub = s.add_subparsers(dest="scene_action", required=True)
    s_sub.add_parser("list", help="List all bodies, positions, and hidden state.")
    s_info = s_sub.add_parser("info", help="Show one body's position + state.")
    s_info.add_argument("name")
    s_move = s_sub.add_parser("move", help="Move a body to an absolute x y z (metres).")
    s_move.add_argument("name")
    s_move.add_argument("x", type=float)
    s_move.add_argument("y", type=float)
    s_move.add_argument("z", type=float)
    s_hide = s_sub.add_parser("hide", help='"Delete" a body (park off-world, no collision).')
    s_hide.add_argument("name")
    s_show = s_sub.add_parser("show", help="Restore a previously hidden body.")
    s_show.add_argument("name")
    s_msrc = s_sub.add_parser("mark-source", help="Mark a source square red (e.g. e1).")
    s_msrc.add_argument("square")
    s_mtgt = s_sub.add_parser("mark-target", help="Mark a target square green (e.g. e3).")
    s_mtgt.add_argument("square")
    s_sub.add_parser("mark-clear", help="Clear both source/target square marks.")

    # ---- replay: domain-randomised re-render of recorded demos ----
    rp = sub.add_parser("replay", help="Replay recorded demos under DR variations.")
    rp.add_argument("--in", dest="in_root", required=True, help="Recorded dataset root.")
    rp.add_argument("--out-repo", default="local/chess-dr", help="Output dataset repo id.")
    rp.add_argument("--out-root", default=None, help="Where to write the DR dataset.")
    rp.add_argument("--dr-config", default=None, help="Python file defining DR = DRConfig(...).")
    rp.add_argument(
        "--limit-episodes", type=int, default=0, help="Cap recorded episodes replayed (0 = all)."
    )
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
            sim_only=args.sim,
            world_config_path=args.world_config,
            scene_socket=args.scene_socket,
            second_camera=args.second_camera,
        )
        run_teleop(cfg)
        return 0

    if args.command == "scene":
        return _run_scene_command(args)

    if args.command == "replay":
        return _run_replay_command(args)

    return 1


def _run_replay_command(args) -> int:
    """Generate a domain-randomised dataset from recorded demos."""
    from so101_lite.dr.replay import replay_dataset
    from so101_lite.dr.variations import discover_dr, load_dr_config

    dr = load_dr_config(args.dr_config) if args.dr_config else discover_dr()
    print(
        f"DR grid: {len(dr.pieces)} piece sets x {len(dr.boards)} boards "
        f"= {dr.n_combos} combos per demo"
    )
    try:
        n = replay_dataset(
            args.in_root,
            args.out_repo,
            dr,
            out_root=args.out_root,
            limit_episodes=args.limit_episodes or None,
        )
    except FileNotFoundError as exc:
        print(f"replay: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {n} domain-randomised episodes to dataset '{args.out_repo}'.")
    return 0


def _run_scene_command(args) -> int:
    """Send one scene command to a running sim over its socket and print the reply."""
    from so101_lite.teleop.scene_api import send_command

    action = args.scene_action
    request: dict = {"action": action}
    if action in ("info", "hide", "show"):
        request["name"] = args.name
    elif action == "move":
        request.update(name=args.name, x=args.x, y=args.y, z=args.z)
    elif action == "mark-source":
        request = {"action": "mark", "kind": "source", "square": args.square}
    elif action == "mark-target":
        request = {"action": "mark", "kind": "target", "square": args.square}
    elif action == "mark-clear":
        request = {"action": "mark_clear"}

    try:
        reply = send_command(request, socket_path=args.socket)
    except (ConnectionError, FileNotFoundError, OSError) as exc:
        print(f"scene: {exc}", file=sys.stderr)
        return 1

    if not reply.get("ok", False):
        print(f"scene: {reply.get('error', 'command failed')}", file=sys.stderr)
        return 1

    result = reply.get("result", {})
    if action == "list":
        for b in result.get("bodies", []):
            flag = " [hidden]" if b["hidden"] else ""
            free = "free" if b["free_joint"] else "static"
            x, y, z = b["position"]
            print(f"  {b['name']:<18} ({x:+.3f}, {y:+.3f}, {z:+.3f})  {free}{flag}")
    else:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
