"""Direct simulator follower for so101-lite teleop.

This replaces so101-nexus's ``SimSOFollower`` + LeRobot ``Robot`` adapter +
``FeetechMotorsBus`` normalization + synthetic-calibration round-trips with a
single closed-form affine mapping that is provably identical to the Nexus path
(modulo sub-tick rounding):

    body joints:  qpos_rad = radians(leader_degrees)
    gripper:      qpos_rad = lerp(gripper_low, gripper_high, percent / 100)

The cancellation falls out of Nexus's synthetic calibration (drive_mode=0,
homing_offset=0, symmetric body ranges sharing one ``mid``). No motors bus is
constructed per frame, so the hot loop is pure NumPy/scalar arithmetic.

Camera frames are read straight from the env's observation dict
(``wrist_camera``/``global_camera``/``overhead_camera``) — there is no separate
camera abstraction; the env already renders them every step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from so101_lite.config import SO101_JOINT_NAMES
from so101_lite.observations import GlobalCamera, OverheadCamera, WristCamera
from so101_lite.teleop.leader import import_backend_for_env_id

_GRIPPER_IDX = SO101_JOINT_NAMES.index("gripper")
_BODY_NAMES = SO101_JOINT_NAMES[:-1]
_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


@dataclass(frozen=True)
class StepInfo:
    """Termination metadata captured from the most recent ``env.step``."""

    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


def _coerce_flag(value: object) -> bool:
    arr = np.asarray(value)
    return bool(arr.item()) if arr.shape == () else bool(arr.any())


def _resize_camera(comp, width: int, height: int):
    """Return *comp* resized to (width, height), preserving its other fields."""
    if isinstance(comp, WristCamera):
        return WristCamera(
            width=width,
            height=height,
            fov_deg_range=comp.fov_deg_range,
            pitch_deg_range=comp.pitch_deg_range,
            pos_x_noise=comp.pos_x_noise,
            pos_y_center=comp.pos_y_center,
            pos_y_noise=comp.pos_y_noise,
            pos_z_center=comp.pos_z_center,
            pos_z_noise=comp.pos_z_noise,
        )
    if isinstance(comp, GlobalCamera):
        return GlobalCamera(
            width=width,
            height=height,
            fov_deg=comp.fov_deg,
            elevation_deg=comp.elevation_deg,
            azimuth_deg=comp.azimuth_deg,
            lookat_x_offset=getattr(comp, "lookat_x_offset", 0.0),
            lookat_y_offset=getattr(comp, "lookat_y_offset", 0.0),
            distance_scale=getattr(comp, "distance_scale", 1.0),
        )
    if isinstance(comp, OverheadCamera):
        return OverheadCamera(
            width=width,
            height=height,
            fov_deg=comp.fov_deg,
            x_offset=getattr(comp, "x_offset", 0.0),
            y_offset=getattr(comp, "y_offset", 0.0),
            height_offset=getattr(comp, "height_offset", 0.0),
        )
    return comp


def _wire_cameras(observations: list, wrist_wh, overhead_wh) -> list:
    """Ensure a wrist + global + overhead camera are present at the given sizes."""
    ww, wh = wrist_wh
    ow, oh = overhead_wh
    out: list = []
    found_wrist = found_global = found_overhead = False
    for comp in observations:
        if isinstance(comp, WristCamera):
            out.append(_resize_camera(comp, ww, wh))
            found_wrist = True
        elif isinstance(comp, GlobalCamera):
            out.append(_resize_camera(comp, ow, oh))
            found_global = True
        elif isinstance(comp, OverheadCamera):
            out.append(_resize_camera(comp, ow, oh))
            found_overhead = True
        else:
            out.append(comp)
    if not found_wrist:
        out.append(WristCamera(width=ww, height=wh))
    if not found_global and not found_overhead:
        out.append(OverheadCamera(width=ow, height=oh))
    return out


class SimFollower:
    """Drives a so101-lite MuJoCo env directly from leader-arm joint readings."""

    def __init__(
        self,
        env_id: str,
        *,
        wrist_wh: tuple[int, int] = (320, 240),
        overhead_wh: tuple[int, int] = (640, 360),
        config: object | None = None,
        control_mode: str = "pd_joint_pos",
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.env_id = env_id
        self.wrist_wh = wrist_wh
        self.overhead_wh = overhead_wh
        self.control_mode = control_mode
        self._config = config
        self._extra_env_kwargs = dict(env_kwargs or {})

        self._env: gym.Env | None = None
        self._gripper_low = 0.0
        self._gripper_high = 1.0
        self._target_low: np.ndarray | None = None
        self._target_high: np.ndarray | None = None
        self._last_step_info: StepInfo | None = None
        self._pending_init_action: dict[str, float] | None = None
        self._last_obs: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._env is not None

    @property
    def env(self) -> gym.Env:
        if self._env is None:
            raise RuntimeError("SimFollower is not connected.")
        return self._env

    def _build_config(self):
        """Resolve and camera-wire the env config for recording."""
        config = self._config
        if config is None:
            import gymnasium

            spec = gymnasium.spec(self.env_id)
            ctor = spec.entry_point
            cfg_cls = None
            if isinstance(ctor, str):
                import importlib

                mod, attr = ctor.split(":")
                cfg_cls = getattr(getattr(importlib.import_module(mod), attr), "default_config_cls", None)
            if cfg_cls is None:
                return None
            config = cfg_cls()

        observations = getattr(config, "observations", None)
        if observations is None:
            return config
        wired = _wire_cameras(observations, self.wrist_wh, self.overhead_wh)
        attrs = vars(config).copy()
        attrs["observations"] = wired
        return config.__class__(**attrs)

    def connect(self) -> None:
        """Create the env, settle it, and cache control/gripper bounds."""
        import_backend_for_env_id(self.env_id)

        make_kwargs: dict[str, Any] = {
            "render_mode": "rgb_array",
            "control_mode": self.control_mode,
        }
        config = self._build_config()
        if config is not None:
            make_kwargs["config"] = config
        make_kwargs.update(self._extra_env_kwargs)

        self._env = gym.make(self.env_id, **make_kwargs)
        self._env.reset()

        unwrapped = self._env.unwrapped
        self._target_low = np.asarray(unwrapped._target_low, dtype=np.float64)
        self._target_high = np.asarray(unwrapped._target_high, dtype=np.float64)
        self._gripper_low = float(self._target_low[_GRIPPER_IDX])
        self._gripper_high = float(self._target_high[_GRIPPER_IDX])

        if self._pending_init_action is not None:
            init_qpos = self._leader_to_qpos(self._pending_init_action)
            self._env.reset(options={"init_qpos": init_qpos})
            self._pending_init_action = None
        self._last_step_info = None

    def disconnect(self) -> None:
        env = self._env
        self._env = None
        self._last_step_info = None
        self._pending_init_action = None
        if env is not None:
            env.close()

    # ------------------------------------------------------------------
    # Mapping (the lean replacement for the Feetech normalization chain)
    # ------------------------------------------------------------------

    def _leader_to_qpos(self, action: dict[str, float]) -> np.ndarray:
        """Map a leader action dict (degrees + gripper %) to sim joint radians."""
        qpos = np.zeros(len(SO101_JOINT_NAMES), dtype=np.float64)
        for i, name in enumerate(_BODY_NAMES):
            qpos[i] = float(action[f"{name}.pos"]) * _DEG2RAD
        pct = float(action["gripper.pos"])
        qpos[_GRIPPER_IDX] = self._gripper_low + (pct / 100.0) * (
            self._gripper_high - self._gripper_low
        )
        return np.clip(qpos, self._target_low, self._target_high)

    def _qpos_to_leader(self, qpos: np.ndarray) -> dict[str, float]:
        """Map sim joint radians back to leader units (degrees + gripper %)."""
        out: dict[str, float] = {}
        for i, name in enumerate(_BODY_NAMES):
            out[f"{name}.pos"] = float(qpos[i]) * _RAD2DEG
        span = self._gripper_high - self._gripper_low
        pct = 0.0 if span == 0 else (float(qpos[_GRIPPER_IDX]) - self._gripper_low) / span * 100.0
        out["gripper.pos"] = float(np.clip(pct, 0.0, 100.0))
        return out

    # ------------------------------------------------------------------
    # Hot loop
    # ------------------------------------------------------------------

    def set_initial_leader_action(self, action: dict[str, float] | None) -> None:
        """Seed ``env.reset(options={'init_qpos': ...})`` from a leader pose."""
        self._pending_init_action = None if action is None else dict(action)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Step the env toward *action*; return the applied pose in leader units."""
        qpos = self._leader_to_qpos(action)
        _obs, _reward, terminated, truncated, info = self.env.step(qpos)
        self._last_step_info = StepInfo(
            terminated=_coerce_flag(terminated),
            truncated=_coerce_flag(truncated),
            info=dict(info) if isinstance(info, dict) else {},
        )
        self._last_obs = _obs
        return self._qpos_to_leader(qpos)

    def get_observation(self) -> dict[str, Any]:
        """Return current joint state (leader units) plus camera frames."""
        unwrapped = self.env.unwrapped
        qpos = np.asarray(unwrapped._get_current_qpos(), dtype=np.float64)
        obs: dict[str, Any] = self._qpos_to_leader(qpos)

        last = getattr(self, "_last_obs", None)
        if isinstance(last, dict):
            wrist = last.get("wrist_camera")
            global_img = last.get("global_camera")
            overhead_img = last.get("overhead_camera")
            if wrist is not None:
                obs["wrist"] = wrist
            if global_img is not None:
                obs["global"] = global_img
            if overhead_img is not None:
                obs["overhead"] = overhead_img
        return obs

    def last_step_info(self) -> StepInfo | None:
        return self._last_step_info

    # ------------------------------------------------------------------
    # Replay capture helpers (for domain-randomised state playback)
    # ------------------------------------------------------------------

    def current_qpos(self) -> np.ndarray:
        """Return a copy of the full MuJoCo qpos (robot + every piece)."""
        return np.asarray(self.env.unwrapped.data.qpos, dtype=np.float64).copy()

    def wrist_camera_state(self) -> dict[str, Any] | None:
        """Snapshot the (randomized) wrist camera pose so replay can reproduce it."""
        u = self.env.unwrapped
        cid = getattr(u, "_wrist_cam_id", None)
        if cid is None:
            return None
        return {
            "cam_pos": np.asarray(u.model.cam_pos[cid], dtype=np.float64).copy(),
            "cam_quat": np.asarray(u.model.cam_quat[cid], dtype=np.float64).copy(),
            "cam_fovy": float(u.model.cam_fovy[cid]),
        }

    def current_marks(self) -> dict[str, Any]:
        """Return the currently selected source/target squares, if any."""
        u = self.env.unwrapped
        return u.get_marks() if hasattr(u, "get_marks") else {}

    def scene_metadata(self) -> dict[str, Any]:
        """Return everything replay needs to rebuild an identical env."""
        u = self.env.unwrapped
        c = u.config
        return {
            "env_id": self.env_id,
            "board_center": list(getattr(c, "board_center", (0.235, 0.0))),
            "square_size": getattr(c, "square_size", 0.03375),
            "board_thickness": getattr(c, "board_thickness", 0.006),
            "mesh_dir": getattr(c, "mesh_dir", None),
            "wrist_wh": list(self.wrist_wh),
            "overhead_wh": list(self.overhead_wh),
        }
