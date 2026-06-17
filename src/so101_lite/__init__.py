"""Public API for so101-lite: lean SO101 MuJoCo sim + teleop stack.

This is a slimmed-down, low-latency successor to so101-nexus. The teleop loop
drives the simulator directly (no LeRobot ``Robot`` adapter, no per-frame
Feetech bus construction, no synthetic calibration files).
"""

from pathlib import Path

from so101_lite.config import (
    DIRECTION_VECTORS as DIRECTION_VECTORS,
)
from so101_lite.config import (
    EXTENDED_POSE as EXTENDED_POSE,
)
from so101_lite.config import (
    POSES as POSES,
)
from so101_lite.config import (
    REST_POSE as REST_POSE,
)
from so101_lite.config import (
    ROBOT_CAMERA_PRESETS as ROBOT_CAMERA_PRESETS,
)
from so101_lite.config import (
    SO101_JOINT_NAMES as SO101_JOINT_NAMES,
)
from so101_lite.config import (
    ChessBoardConfig as ChessBoardConfig,
)
from so101_lite.config import (
    ControlMode as ControlMode,
)
from so101_lite.config import (
    EnvironmentConfig as EnvironmentConfig,
)
from so101_lite.config import (
    LookAtConfig as LookAtConfig,
)
from so101_lite.config import (
    MoveConfig as MoveConfig,
)
from so101_lite.config import (
    MoveDirection as MoveDirection,
)
from so101_lite.config import (
    ObsMode as ObsMode,
)
from so101_lite.config import (
    PickAndPlaceConfig as PickAndPlaceConfig,
)
from so101_lite.config import (
    PickConfig as PickConfig,
)
from so101_lite.config import (
    Pose as Pose,
)
from so101_lite.config import (
    ReachConfig as ReachConfig,
)
from so101_lite.config import (
    RenderConfig as RenderConfig,
)
from so101_lite.config import (
    RewardConfig as RewardConfig,
)
from so101_lite.config import (
    RobotCameraPreset as RobotCameraPreset,
)
from so101_lite.config import (
    RobotConfig as RobotConfig,
)
from so101_lite.config import (
    YcbModelId as YcbModelId,
)
from so101_lite.config import (
    describe_pick_target as describe_pick_target,
)
from so101_lite.constants import (
    COLOR_MAP as COLOR_MAP,
)
from so101_lite.constants import (
    YCB_OBJECTS as YCB_OBJECTS,
)
from so101_lite.constants import (
    ColorConfig as ColorConfig,
)
from so101_lite.constants import (
    ColorName as ColorName,
)
from so101_lite.constants import (
    sample_color as sample_color,
)
from so101_lite.objects import (  # noqa: F401
    CubeObject,
    MeshObject,
    SceneObject,
    YCBObject,
)
from so101_lite.observations import (  # noqa: F401
    EndEffectorPose,
    GazeDirection,
    GlobalCamera,
    GraspState,
    JointPositions,
    ObjectOffset,
    ObjectPose,
    Observation,
    OverheadCamera,
    TargetOffset,
    TargetPosition,
    WristCamera,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def get_so101_mujoco_model_dir() -> Path:
    """Return the directory holding the vendored MuJoCo Menagerie SO101 model."""
    return ASSETS_DIR / "SO101_menagerie"


def get_so101_mujoco_model_path() -> Path:
    """Return the path to the MJCF model used by the MuJoCo backend (menagerie)."""
    return get_so101_mujoco_model_dir() / "so101.xml"


from so101_lite.rewards import (  # noqa: F401, E402
    orientation_progress,
    reach_progress,
    simple_reward,
)
from so101_lite.ycb_assets import (  # noqa: E402
    ensure_ycb_assets as ensure_ycb_assets,
)
from so101_lite.ycb_assets import (  # noqa: E402
    get_ycb_collision_mesh as get_ycb_collision_mesh,
)
from so101_lite.ycb_assets import (  # noqa: E402
    get_ycb_mesh_dir as get_ycb_mesh_dir,
)
from so101_lite.ycb_assets import (  # noqa: E402
    get_ycb_texture_file as get_ycb_texture_file,
)
from so101_lite.ycb_assets import (  # noqa: E402
    get_ycb_visual_mesh as get_ycb_visual_mesh,
)
from so101_lite.ycb_geometry import (  # noqa: E402
    get_maniskill_ycb_spawn_z as get_maniskill_ycb_spawn_z,
)
from so101_lite.ycb_geometry import (  # noqa: E402
    get_mujoco_ycb_rest_pose as get_mujoco_ycb_rest_pose,
)
