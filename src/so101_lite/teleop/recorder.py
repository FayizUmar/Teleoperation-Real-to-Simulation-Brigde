"""Recording state buffers and LeRobot dataset writer for so101-lite teleop.

The hot loop lives in :mod:`so101_lite.teleop.viewer`; this module just holds
per-episode buffers and turns them into a LeRobot dataset on disk so the output
stays drop-in compatible with policy training pipelines.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from so101_lite.config import SO101_JOINT_NAMES
from so101_lite.teleop.dataset import (
    FieldSelection,
    build_features,
    build_frame,
)


@dataclass
class RecordingState:
    """Per-episode buffers filled by the teleop loop."""

    is_recording: bool = False
    task_description: str = ""
    terminated_at_frame: int | None = None

    episode_actions: list[np.ndarray] = field(default_factory=list)
    episode_states: list[np.ndarray] = field(default_factory=list)
    episode_wrist_images: list[np.ndarray] = field(default_factory=list)
    episode_overhead_images: list[np.ndarray] = field(default_factory=list)
    # Full MuJoCo qpos per frame (robot + all pieces) for deterministic replay.
    episode_qpos: list[np.ndarray] = field(default_factory=list)
    # Scene + wrist-camera snapshot captured when recording starts.
    replay_meta: dict = field(default_factory=dict)

    def clear_episode(self) -> None:
        """Drop everything recorded for the current episode."""
        self.episode_actions.clear()
        self.episode_states.clear()
        self.episode_wrist_images.clear()
        self.episode_overhead_images.clear()
        self.episode_qpos.clear()
        self.replay_meta = {}
        self.terminated_at_frame = None

    @property
    def num_frames(self) -> int:
        return len(self.episode_actions)

    def append(
        self,
        *,
        action: dict[str, float],
        state: dict[str, float],
        wrist: np.ndarray | None,
        overhead: np.ndarray | None,
        qpos: np.ndarray | None = None,
    ) -> None:
        """Append one frame's action/state vectors, camera images, and qpos."""
        self.episode_actions.append(dict_to_vector(action))
        self.episode_states.append(dict_to_vector(state))
        if wrist is not None:
            self.episode_wrist_images.append(wrist)
        if overhead is not None:
            self.episode_overhead_images.append(overhead)
        if qpos is not None:
            self.episode_qpos.append(np.asarray(qpos, dtype=np.float32))


def dict_to_vector(
    motor_dict: Mapping[str, object],
    joint_names: tuple[str, ...] = SO101_JOINT_NAMES,
) -> np.ndarray:
    """Extract ``<joint>.pos`` values in canonical joint order as float32."""
    return np.array(
        [float(cast("float", motor_dict[f"{name}.pos"])) for name in joint_names],
        dtype=np.float32,
    )


class DatasetWriter:
    """Thin wrapper over ``LeRobotDataset`` for teleop episodes."""

    def __init__(
        self,
        repo_id: str,
        *,
        fps: int,
        root: str | Path | None = None,
        selection: FieldSelection | None = None,
        wrist_wh: tuple[int, int] = (320, 240),
        overhead_wh: tuple[int, int] = (640, 360),
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.selection = selection or FieldSelection()
        action_features = {f"{name}.pos": float for name in SO101_JOINT_NAMES}
        ww, wh = wrist_wh
        ow, oh = overhead_wh
        follower_features = {
            **action_features,
            "wrist": (wh, ww, 3),
            "overhead": (oh, ow, 3),
        }
        features = build_features(self.selection, follower_features, action_features)
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            robot_type="sim_so_follower",
            root=root,
            use_videos=True,
        )
        # Sidecar dir holding per-episode qpos trajectories + scene metadata that
        # the DR replay engine reads back (kept out of the LeRobot dataset proper).
        self._replay_dir = Path(self.dataset.root) / "replay_meta"
        self._replay_dir.mkdir(parents=True, exist_ok=True)
        self._episode_idx = 0

    def add_episode(self, state: RecordingState, task: str) -> int:
        """Write one buffered episode and return the number of frames saved."""
        n = state.num_frames
        for i in range(n):
            frame = build_frame(
                self.selection,
                state=state.episode_states[i],
                action=state.episode_actions[i],
                task=task,
                wrist_image=state.episode_wrist_images[i]
                if i < len(state.episode_wrist_images)
                else None,
                overhead_image=state.episode_overhead_images[i]
                if i < len(state.episode_overhead_images)
                else None,
            )
            self.dataset.add_frame(frame)
        self.dataset.save_episode()
        self._write_replay_sidecar(state, task)
        self._episode_idx += 1
        return n

    def _write_replay_sidecar(self, state: RecordingState, task: str) -> None:
        """Persist qpos trajectory + scene/wrist-cam metadata for DR replay."""
        if not state.episode_qpos:
            return  # nothing to replay (qpos wasn't captured)
        ep = self._episode_idx
        meta = dict(state.replay_meta)
        wrist_cam = meta.pop("wrist_cam", None) or {}

        arrays = {
            "qpos": np.asarray(state.episode_qpos, dtype=np.float32),
            "actions": np.asarray(state.episode_actions, dtype=np.float32),
            "states": np.asarray(state.episode_states, dtype=np.float32),
        }
        if wrist_cam:
            arrays["wrist_cam_pos"] = np.asarray(wrist_cam["cam_pos"], dtype=np.float32)
            arrays["wrist_cam_quat"] = np.asarray(wrist_cam["cam_quat"], dtype=np.float32)
            arrays["wrist_cam_fovy"] = np.asarray([wrist_cam["cam_fovy"]], dtype=np.float32)
        np.savez(self._replay_dir / f"episode_{ep:04d}.npz", **arrays)

        meta_out = {**meta, "task": task, "num_frames": int(state.num_frames)}
        (self._replay_dir / f"episode_{ep:04d}.json").write_text(json.dumps(meta_out, indent=2))

    def finalize(self) -> None:
        self.dataset.finalize()

    def push_to_hub(self, **kwargs: Any) -> None:
        self.dataset.push_to_hub(**kwargs)
