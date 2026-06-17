"""Recording state buffers and LeRobot dataset writer for so101-lite teleop.

The hot loop lives in :mod:`so101_lite.teleop.viewer`; this module just holds
per-episode buffers and turns them into a LeRobot dataset on disk so the output
stays drop-in compatible with policy training pipelines.
"""

from __future__ import annotations

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

    def clear_episode(self) -> None:
        """Drop everything recorded for the current episode."""
        self.episode_actions.clear()
        self.episode_states.clear()
        self.episode_wrist_images.clear()
        self.episode_overhead_images.clear()
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
    ) -> None:
        """Append one frame's action/state vectors and camera images."""
        self.episode_actions.append(dict_to_vector(action))
        self.episode_states.append(dict_to_vector(state))
        if wrist is not None:
            self.episode_wrist_images.append(wrist)
        if overhead is not None:
            self.episode_overhead_images.append(overhead)


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
        return n

    def finalize(self) -> None:
        self.dataset.finalize()

    def push_to_hub(self, **kwargs: Any) -> None:
        self.dataset.push_to_hub(**kwargs)
