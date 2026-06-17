# so101-lite

A lean, low-latency successor to **so101-nexus** for the SO-101 arm: MuJoCo
simulation, direct teleoperation from a physical leader arm, recording, and
LeRobot-compatible dataset creation — with the heavy abstraction layers removed.

## What changed vs Nexus

The Nexus teleop hot loop routed every frame through a LeRobot `Robot` adapter
(`SimSOFollower`), which rebuilt a `FeetechMotorsBus` per call and round-tripped
joint values through tick/normalize/unnormalize math against a synthetic
calibration file, then drove a Gradio polling UI and a separate `SimCamera`
render path.

so101-lite replaces that with a **direct simulator follower**:

| Concern              | Nexus                                            | so101-lite                                   |
|----------------------|--------------------------------------------------|----------------------------------------------|
| Leader → sim mapping | degrees→ticks→normalize→unnormalize→ticks→rad    | closed-form `radians()` + gripper lerp       |
| Motors bus           | `FeetechMotorsBus` constructed per frame         | none                                         |
| Calibration          | synthetic JSON written/read on disk              | none                                         |
| Cameras              | `SimCamera` abstraction, separate render         | read straight from env `step` obs            |
| UI                   | Gradio walkthrough (polling)                     | OpenCV camera windows + keypresses           |

The mapping is provably equivalent to Nexus (verified to within sub-tick
rounding, ~0.09°), so recorded datasets stay drop-in compatible.

The cancellation: Nexus's synthetic calibration uses `drive_mode=0`,
`homing_offset=0`, and symmetric body ranges sharing one midpoint, so the whole
tick round-trip collapses to:

```
body joints:  qpos_rad = radians(leader_degrees)
gripper:      qpos_rad = gripper_low + (percent / 100) * (gripper_high - gripper_low)
```

## Layout

```
src/so101_lite/
  __init__.py          asset paths + public re-exports
  config.py            EnvironmentConfig + all task configs
  constants.py         colors / YCB tables
  observations.py      observation components (cameras, poses, ...)
  rewards.py camera_utils.py objects.py ycb_*.py visualization.py
  assets/SO101_menagerie/   vendored MuJoCo model
  envs/
    __init__.py        gym.register all MuJoCo* env ids
    base_env.py        shared MuJoCo base env
    reach_/pick_/pick_and_place/look_at_/move_/chess_env.py
  teleop/
    leader.py          physical leader-arm factory + port diagnostics
    follower.py        SimFollower: direct leader<->sim mapping
    recorder.py        RecordingState buffers + LeRobot DatasetWriter
    dataset.py         feature schema + frame builders
    viewer.py          interactive loop (camera windows + keypresses)
    cli.py             `so101-lite teleop` entry point
```

## Environments

`MuJoCoReach-v1`, `MuJoCoPickLift-v1`, `MuJoCoPickAndPlace-v1`,
`MuJoCoLookAt-v1`, `MuJoCoMove-v1`, `MuJoCoChessBoard-v1`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[teleop]"     # teleop extra pulls in lerobot
```

> While package indexes are unreachable, you can run against the existing Nexus
> virtualenv instead:
> `MUJOCO_GL=cgl PYTHONPATH=src /path/to/so101-nexus/.venv/bin/python ...`

## Record a session

```bash
so101-lite teleop \
  --env-id MuJoCoChessBoard-v1 \
  --port /dev/tty.usbmodemXXXX \
  --repo-id local/chess-demos \
  --fps 30
```

Live wrist + global camera windows open. With a camera window focused:

| Key    | Action                              |
|--------|-------------------------------------|
| SPACE  | start / stop+save the episode       |
| n      | save current episode, reset a fresh scene |
| r      | discard current recording, reset    |
| q/ESC  | quit (finalizes the dataset)        |

Recording runs until you stop it — there is no fixed step cap. For the chess
task, an episode also auto-stops once the piece is placed and the arm has
returned home (held for `--success-hold-seconds`).

On macOS the MuJoCo passive 3D viewer needs `mjpython`; the OpenCV camera
windows work from a plain `python`, and the 3D viewer is opened automatically
when available.
