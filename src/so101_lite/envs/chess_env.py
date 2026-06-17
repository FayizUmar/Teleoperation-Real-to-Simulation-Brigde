"""MuJoCo chess board pick-and-place environment.

Provides ChessBoardEnv where the robot picks a chess piece from one board
square and places it on a highlighted target square.

Two cameras:
  - GlobalCamera: angled fixed view above the board (macro / board state).
  - WristCamera:  ego-centric view from the gripper (millimetre alignment).
"""

from __future__ import annotations

import tempfile
from typing import ClassVar

import mujoco
import numpy as np

from so101_lite import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_lite.config import (
    ChessBoardConfig,
    ControlMode,
)
from so101_lite.constants import sample_color
from so101_lite.observations import ObjectOffset, ObjectPose, TargetOffset, TargetPosition
from so101_lite.rewards import reach_progress
from so101_lite.envs.base_env import SCENE_OPTION_XML, SO101NexusMuJoCoBaseEnv
from so101_lite.envs.spawn_utils import random_yaw_quat

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()


def _board_geoms_xml(square_size: float, board_thickness: float) -> str:
    """Return MJCF for the fixed chessboard body: one collision base + 64 visual squares."""
    board_half = 4.0 * square_size
    half_thick = board_thickness / 2.0
    surface_z = board_thickness + 0.0001
    sq_half = square_size / 2.0

    lines: list[str] = []
    # Single physical box the pieces rest on
    lines.append(
        f'<geom name="board_base" type="box" '
        f'size="{board_half:.5f} {board_half:.5f} {half_thick:.5f}" '
        f'pos="0 0 {half_thick:.5f}" '
        f'rgba="0.50 0.30 0.15 1" '
        f"contype=\"1\" conaffinity=\"1\"/>"
    )
    # 64 visual-only coloured squares
    for row in range(8):
        for col in range(8):
            color = "0.93 0.87 0.74 1" if (row + col) % 2 == 0 else "0.18 0.11 0.06 1"
            x_rel = (col - 3.5) * square_size
            y_rel = (row - 3.5) * square_size
            lines.append(
                f'<geom name="sq_{row}_{col}" type="box" '
                f'size="{sq_half:.5f} {sq_half:.5f} 0.00010" '
                f'pos="{x_rel:.5f} {y_rel:.5f} {surface_z:.5f}" '
                f'rgba="{color}" '
                f'contype="0" conaffinity="0"/>'
            )
    return "\n      ".join(lines)


def _build_chess_scene_xml(
    board_cx: float,
    board_cy: float,
    square_size: float,
    board_thickness: float,
    piece_color: list[float],
    target_color: list[float],
    source_color: list[float],
    piece_radius: float,
    piece_half_height: float,
    piece_mass: float,
    ground_color: list[float],
) -> str:
    robot_path = str(_SO101_XML)
    gr, gg, gb, ga = ground_color
    pr, pg, pb, _ = piece_color
    tr, tg, tb, ta = target_color
    sr, sg, sb, sa = source_color
    board_geoms = _board_geoms_xml(square_size, board_thickness)
    piece_z = board_thickness + piece_half_height
    tgt_radius = square_size * 0.42  # visual disc fits inside a single square

    return f"""\
<mujoco model="chess_scene">
  <compiler angle="radian"/>

  <include file="{robot_path}"/>
  {SCENE_OPTION_XML}

  <visual>
    <headlight diffuse="0.0 0.0 0.0" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <worldbody>
    <light pos="1 1 3.5" dir="-0.27 -0.27 -0.92" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="0 0 0.01" rgba="{gr} {gg} {gb} {ga}"
          pos="0 0 0" contype="1" conaffinity="1"/>

    <body name="chessboard" pos="{board_cx:.5f} {board_cy:.5f} 0">
      {board_geoms}
    </body>

    <body name="chess_piece" pos="{board_cx:.5f} {board_cy:.5f} {piece_z:.5f}">
      <freejoint name="chess_piece_joint"/>
      <geom name="chess_piece_geom" type="cylinder"
            size="{piece_radius:.5f} {piece_half_height:.5f}"
            rgba="{pr} {pg} {pb} 1" mass="{piece_mass}"
            contype="1" conaffinity="1" condim="4" friction="1 0.05 0.001"
            solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>

    <body name="target" pos="{board_cx:.5f} {board_cy:.5f} {board_thickness + 0.001:.5f}">
      <geom name="target_disc" type="cylinder" size="{tgt_radius:.5f} 0.001"
            rgba="{tr} {tg} {tb} {ta}" contype="0" conaffinity="0"/>
    </body>

    <body name="source" pos="{board_cx:.5f} {board_cy:.5f} {board_thickness + 0.001:.5f}">
      <geom name="source_disc" type="cylinder" size="{tgt_radius:.5f} 0.001"
            rgba="{sr} {sg} {sb} {sa}" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


class ChessBoardEnv(SO101NexusMuJoCoBaseEnv):
    """Pick-and-place on an 8×8 chessboard.

    The robot must pick a cylindrical chess piece from its source square and
    place it on the highlighted target square. Episode resets randomise both
    the source and target squares (guaranteed distinct).

    Observations (default):
      - ``state``: joint positions, end-effector pose, grasp state
      - ``wrist_camera``: ego-centric RGB from the gripper camera
      - ``global_camera``: angled RGB overview of the full board
    """

    config: ChessBoardConfig
    default_config_cls: ClassVar[type[ChessBoardConfig]] = ChessBoardConfig

    def __init__(
        self,
        config: ChessBoardConfig | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ):
        if config is None:
            config = ChessBoardConfig()
        self._init_common(
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        self.task_description = config.task_description
        cx, cy = config.board_center

        xml_string = _build_chess_scene_xml(
            board_cx=cx,
            board_cy=cy,
            square_size=config.square_size,
            board_thickness=config.board_thickness,
            piece_color=sample_color(config.piece_colors),
            target_color=sample_color(config.target_colors),
            source_color=sample_color(config.source_colors),
            piece_radius=config.piece_radius,
            piece_half_height=config.piece_half_height,
            piece_mass=config.piece_mass,
            ground_color=sample_color(config.ground_colors),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        self._piece_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "chess_piece"
        )
        self._obj_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "chess_piece_geom"
        )
        piece_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "chess_piece_joint"
        )
        self._piece_qpos_addr = self.model.jnt_qposadr[piece_joint_id]
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self._source_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "source"
        )

        self._finish_model_setup()

    # ------------------------------------------------------------------
    # Square geometry helpers
    # ------------------------------------------------------------------

    def _square_world_pos(self, row: int, col: int) -> np.ndarray:
        """Return the world XY centre of a board square."""
        cfg = self.config
        cx, cy = cfg.board_center
        x = cx + (col - 3.5) * cfg.square_size
        y = cy + (row - 3.5) * cfg.square_size
        z = cfg.board_thickness + cfg.piece_half_height
        return np.array([x, y, z])

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def _get_piece_pose(self) -> np.ndarray:
        addr = self._piece_qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _get_target_pos(self) -> np.ndarray:
        return self.data.xpos[self._target_body_id].copy()

    # ------------------------------------------------------------------
    # Observation component dispatch
    # ------------------------------------------------------------------

    def _get_component_data(self, component: object) -> np.ndarray:
        if isinstance(component, ObjectPose):
            return self._get_piece_pose()
        if isinstance(component, ObjectOffset):
            tcp_pos = self._get_tcp_pose()[:3]
            obj_pos = self._get_piece_pose()[:3]
            return obj_pos - tcp_pos
        if isinstance(component, TargetPosition):
            return self._get_target_pos()
        if isinstance(component, TargetOffset):
            obj_pos = self._get_piece_pose()[:3]
            target_pos = self._get_target_pos()
            return target_pos - obj_pos
        return super()._get_component_data(component)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _task_reset(self) -> None:
        rng = self.np_random

        # Choose two distinct squares
        src_row, src_col = int(rng.integers(0, 8)), int(rng.integers(0, 8))
        tgt_row, tgt_col = src_row, src_col
        while tgt_row == src_row and tgt_col == src_col:
            tgt_row, tgt_col = int(rng.integers(0, 8)), int(rng.integers(0, 8))

        # Place the target marker
        tgt_pos = self._square_world_pos(tgt_row, tgt_col)
        self.model.body_pos[self._target_body_id] = [
            tgt_pos[0],
            tgt_pos[1],
            self.config.board_thickness + 0.001,
        ]

        # Place the source marker
        src_pos = self._square_world_pos(src_row, src_col)
        self.model.body_pos[self._source_body_id] = [
            src_pos[0],
            src_pos[1],
            self.config.board_thickness + 0.001,
        ]

        # Place the piece at the source square with a random yaw
        piece_quat = random_yaw_quat(rng)
        addr = self._piece_qpos_addr
        self.data.qpos[addr : addr + 3] = src_pos
        self.data.qpos[addr + 3 : addr + 7] = piece_quat
        self.data.qvel[addr : addr + 6] = 0.0

        self._initial_piece_z = float(src_pos[2])

    def _refresh_reset_reference_state(self) -> None:
        self._initial_piece_z = float(self._get_piece_pose()[2])
        self._home_tcp_pos = self._get_tcp_pose()[:3].copy()

    # ------------------------------------------------------------------
    # Info & reward
    # ------------------------------------------------------------------

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        piece_pose = self._get_piece_pose()
        piece_pos = piece_pose[:3]
        target_pos = self._get_target_pos()
        is_grasped = self._is_grasping()

        xy_dist = float(np.linalg.norm(piece_pos[:2] - target_pos[:2]))
        on_board_z = piece_pos[2] < self.config.board_thickness + self.config.piece_half_height * 2 + 0.01
        is_placed = xy_dist <= self.config.goal_thresh and on_board_z
        is_static = self._is_robot_static()
        lift_height = float(piece_pos[2] - self._initial_piece_z)
        home_dist = float(np.linalg.norm(tcp_pos - self._home_tcp_pos))
        is_home = home_dist <= self.config.home_thresh

        return {
            "piece_to_target_dist": xy_dist,
            "is_placed": is_placed,
            "is_grasped": is_grasped,
            "is_robot_static": is_static,
            "lift_height": lift_height,
            "tcp_to_home_dist": home_dist,
            "is_home": is_home,
            "success": is_placed and not is_grasped and is_home and is_static,
            "tcp_to_piece_dist": float(np.linalg.norm(piece_pos - tcp_pos)),
        }

    def _compute_reward(self, info: dict) -> float:
        scale = self.config.reward.tanh_shaping_scale
        rp = reach_progress(info["tcp_to_piece_dist"], scale=scale)
        is_grasped = info["is_grasped"] > 0.5
        placement_progress = (
            reach_progress(info["piece_to_target_dist"], scale=scale) if is_grasped else 0.0
        )
        return self.config.reward.compute(
            reach_progress=rp,
            is_grasped=is_grasped,
            task_progress=placement_progress,
            is_complete=info["success"],
            action_delta_norm=info.get("action_delta_norm", 0.0),
            energy_norm=info.get("energy_norm", 0.0),
        )
