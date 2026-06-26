"""Domain-randomised replay: re-render recorded demos under texture variations.

State playback — for each recorded frame we set the full MuJoCo ``qpos`` and
re-render, so the generated frames match the recorded actions exactly (zero
physics drift). Each (piece-material, board) combo produces one new episode with
the *same* action labels but different pixels, all written into one mixed
LeRobot dataset.

Pipeline:
    recorded dataset/replay_meta/episode_XXXX.{npz,json}
        -> for each DR combo: build env, set qpos[t], render, write episode
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from so101_lite.dr.variations import BoardVariant, DRConfig, PieceMaterials
from so101_lite.teleop.recorder import DatasetWriter, RecordingState


def _build_replay_config(meta: dict, piece: PieceMaterials, board: BoardVariant):
    """A ChessBoardConfig that reproduces the recorded geometry with DR materials."""
    from so101_lite.config import ChessBoardConfig
    from so101_lite.observations import (
        EndEffectorPose,
        GlobalCamera,
        GraspState,
        JointPositions,
        OverheadCamera,
        WristCamera,
    )

    ww, wh = meta["wrist_wh"]
    ow, oh = meta["overhead_wh"]
    observations = [
        JointPositions(),
        EndEffectorPose(),
        GraspState(),
        WristCamera(width=ww, height=wh),
        GlobalCamera(width=ow, height=oh, elevation_deg=-38.0, azimuth_deg=120.0),
        OverheadCamera(width=ow, height=oh),
    ]
    return ChessBoardConfig(
        board_center=tuple(meta["board_center"]),
        square_size=meta["square_size"],
        board_thickness=meta["board_thickness"],
        mesh_dir=meta.get("mesh_dir"),
        board_texture=board.texture,
        light_square_color=board.light_color,
        dark_square_color=board.dark_color,
        white_piece_rgba=piece.white_rgba,
        black_piece_rgba=piece.black_rgba,
        white_texture=piece.white_texture,
        black_texture=piece.black_texture,
        observations=observations,
    )


def _apply_wrist_cam(env, wrist_cam: tuple | None) -> None:
    """Restore the recorded wrist-camera pose so the replayed feed matches."""
    if wrist_cam is None:
        return
    cid = getattr(env, "_wrist_cam_id", None)
    if cid is None:
        return
    pos, quat, fovy = wrist_cam
    env.model.cam_pos[cid] = pos
    env.model.cam_quat[cid] = quat
    env.model.cam_fovy[cid] = float(fovy)


def _apply_marks(env, marks: dict | None) -> None:
    """Re-place the recorded source/target square masks for DR re-rendering."""
    if not marks:
        return
    for kind in ("source", "target"):
        square = marks.get(kind)
        if square:
            env.set_mark(kind, square)


def _load_episode(npz_path: Path) -> tuple[dict, dict]:
    data = dict(np.load(npz_path))
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    return data, meta


def replay_dataset(
    in_root: str | Path,
    out_repo_id: str,
    dr: DRConfig,
    out_root: str | Path | None = None,
    limit_episodes: int | None = None,
    progress: bool = True,
) -> int:
    """Replay every recorded episode under every DR combo. Return episodes written."""
    from so101_lite.envs.chess_env import ChessBoardEnv

    in_root = Path(in_root)
    meta_dir = in_root / "replay_meta"
    if not meta_dir.is_dir():
        raise FileNotFoundError(
            f"no replay_meta/ under {in_root}. Record with qpos capture first."
        )
    episodes = sorted(meta_dir.glob("episode_*.npz"))
    if limit_episodes:
        episodes = episodes[:limit_episodes]
    if not episodes:
        raise FileNotFoundError(f"no episode_*.npz files in {meta_dir}")

    writer: DatasetWriter | None = None
    written = 0
    n_combos = dr.n_combos

    for ep_i, npz_path in enumerate(episodes):
        data, meta = _load_episode(npz_path)
        qpos = data["qpos"]
        actions = data["actions"]
        states = data["states"]
        task = meta.get("task", "")
        fps = int(meta.get("fps", 30))
        # Re-render the same 2nd camera that was recorded into the "overhead" field.
        second_key = (
            "overhead_camera"
            if meta.get("second_camera", "global") == "overhead"
            else "global_camera"
        )
        wrist_wh = tuple(meta["wrist_wh"])
        overhead_wh = tuple(meta["overhead_wh"])
        wrist_cam = None
        if "wrist_cam_pos" in data:
            wrist_cam = (
                data["wrist_cam_pos"],
                data["wrist_cam_quat"],
                float(np.asarray(data["wrist_cam_fovy"]).ravel()[0]),
            )

        for c_i, (piece, board) in enumerate(dr.combos()):
            cfg = _build_replay_config(meta, piece, board)
            env = ChessBoardEnv(config=cfg, render_mode="rgb_array")
            env.reset()
            _apply_wrist_cam(env, wrist_cam)
            _apply_marks(env, meta.get("marks"))

            rs = RecordingState(task_description=task)
            rs.episode_actions = [a.astype(np.float32) for a in actions]
            rs.episode_states = [s.astype(np.float32) for s in states]
            for t in range(len(qpos)):
                env.data.qpos[:] = qpos[t]
                mujoco.mj_forward(env.model, env.data)
                obs = env._get_obs()
                rs.episode_wrist_images.append(np.asarray(obs["wrist_camera"]))
                overhead = obs.get(second_key)
                if overhead is None:
                    overhead = obs.get("global_camera")
                if overhead is None:
                    overhead = obs.get("overhead_camera")
                rs.episode_overhead_images.append(np.asarray(overhead))

            if writer is None:
                writer = DatasetWriter(
                    out_repo_id,
                    fps=fps,
                    root=str(out_root) if out_root else None,
                    wrist_wh=wrist_wh,
                    overhead_wh=overhead_wh,
                )
            writer.add_episode(rs, task)
            env.close()
            written += 1
            if progress:
                print(
                    f"  episode {ep_i + 1}/{len(episodes)} "
                    f"combo {c_i + 1}/{n_combos} "
                    f"[{piece.name} | {board.name}] -> {len(qpos)} frames",
                    flush=True,
                )

    if writer is not None:
        writer.finalize()
    return written
