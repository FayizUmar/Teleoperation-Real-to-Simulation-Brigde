"""Headless smoke test: follower -> recorder -> LeRobot dataset round-trip."""

import os
import tempfile

import so101_lite.envs  # noqa: F401 — register envs
from so101_lite.config import SO101_JOINT_NAMES
from so101_lite.teleop.follower import SimFollower
from so101_lite.teleop.leader import apply_wrist_roll_offset_deg
from so101_lite.teleop.recorder import DatasetWriter, RecordingState


class FakeLeader:
    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_action(self):
        return {f"{n}.pos": 0.0 for n in SO101_JOINT_NAMES[:-1]} | {"gripper.pos": 50.0}


def main() -> None:
    wrist_wh, overhead_wh = (64, 48), (96, 54)
    f = SimFollower("MuJoCoChessBoard-v1", wrist_wh=wrist_wh, overhead_wh=overhead_wh)
    leader = FakeLeader()
    leader.connect()
    f.set_initial_leader_action(apply_wrist_roll_offset_deg(leader.get_action(), -90.0))
    f.connect()

    state = RecordingState(task_description="test")
    state.is_recording = True
    for _ in range(12):
        a = apply_wrist_roll_offset_deg(leader.get_action(), -90.0)
        sent = f.send_action(a)
        obs = f.get_observation()
        state.append(action=sent, state=obs, wrist=obs.get("wrist"), overhead=obs.get("overhead"))

    assert state.num_frames == 12
    assert state.episode_wrist_images[0].shape == (48, 64, 3)
    assert state.episode_overhead_images[0].shape == (54, 96, 3)

    root = os.path.join(tempfile.mkdtemp(prefix="so101lite_"), "ds")
    w = DatasetWriter(
        "local/so101lite-smoke", fps=30, root=root, wrist_wh=wrist_wh, overhead_wh=overhead_wh
    )
    n = w.add_episode(state, "test task")
    w.finalize()

    f.disconnect()

    # Validate the on-disk dataset via metadata only. We deliberately avoid
    # constructing LeRobotDataset (which eagerly probes torchcodec/ffmpeg to
    # decode video) so this smoke test does not depend on the host ffmpeg ABI.
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("local/so101lite-smoke", root=root)
    assert meta.total_frames == n == 12, (meta.total_frames, n)
    assert meta.total_episodes == 1, meta.total_episodes
    assert meta.fps == 30, meta.fps
    assert meta.features["action"]["shape"] == (len(SO101_JOINT_NAMES),)
    assert "observation.images.wrist" in meta.features
    assert "observation.images.overhead" in meta.features
    print(
        f"on-disk dataset: {meta.total_frames} frames / {meta.total_episodes} ep, "
        f"fps={meta.fps} -- DATASET WRITE OK"
    )


if __name__ == "__main__":
    main()
