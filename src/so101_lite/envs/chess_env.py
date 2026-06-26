"""MuJoCo chess environment with a full 32-piece mesh set.

The board carries a complete chess starting position built from user-supplied
OBJ meshes (one per piece type, shared between colours — colour is a material,
not geometry). White occupies the two ranks nearest the arm; black the two far
ranks. Every piece has a free joint so any can be picked up or knocked over.

This env is built for teleop recording + domain-randomised replay, so colour /
texture of the pieces and board are configurable (the DR axes) while geometry
stays fixed. The RL reward is a light reach shaping toward a designated piece;
the real signal comes from recorded demonstrations.

Cameras: WristCamera (ego), GlobalCamera (angled overview), OverheadCamera (top).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import ClassVar

import mujoco
import numpy as np

from so101_lite import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_lite.camera_utils import build_scene_cameras_xml
from so101_lite.config import ChessBoardConfig, ControlMode
from so101_lite.constants import sample_color
from so101_lite.observations import (
    GlobalCamera,
    ObjectOffset,
    ObjectPose,
    OverheadCamera,
    TargetOffset,
    TargetPosition,
)
from so101_lite.rewards import reach_progress
from so101_lite.envs.base_env import SCENE_OPTION_XML, SO101NexusMuJoCoBaseEnv

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()

_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"
_DEFAULT_MESH_DIR = _ASSETS_DIR / "meshes" / "chess"

_PIECE_TYPES = ("pawn", "rook", "knight", "bishop", "queen", "king")
# Back-rank order by file a..h.
_BACK_RANK = ("rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook")

# Geom group used for the source/target square masks. Group 5 is unused by the
# SO-101 model (group 4 holds the red collision_gripper_mesh geoms!), so it is
# invisible everywhere unless a renderer explicitly enables it (we enable it
# only for the overhead/global cameras, never the wrist).
MASK_GEOM_GROUP = 5
_MASK_PARK = (0.0, 0.0, -1.0)  # where an unset mask body hides


def square_to_rowcol(square: str) -> tuple[int, int]:
    """Map algebraic notation ('e1') to (row=file, col=rank-1).

    Files a-h -> row 0-7 (sideways); ranks 1-8 -> col 0-7 (col 0 nearest arm,
    the white back rank). So 'e1' is the e-file, rank-1 square (white's king).
    """
    s = square.strip().lower()
    if len(s) < 2 or not s[0].isalpha() or not s[1:].isdigit():
        raise ValueError(f"bad square {square!r}; expected like 'e1'")
    row = ord(s[0]) - ord("a")
    col = int(s[1:]) - 1
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError(f"square {square!r} out of board range a1..h8")
    return row, col

# Designated "primary" piece for grasp/reward bookkeeping: central white pawn
# (d-file, rank 2). row == file index, col == rank index.
_PRIMARY_ROW, _PRIMARY_COL = 3, 1


def _obj_xy_extent(path: Path) -> float:
    """Return the larger of the OBJ's X/Y bounding-box extents (metres)."""
    min_xy = [1e9, 1e9]
    max_xy = [-1e9, -1e9]
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                x, y = float(parts[1]), float(parts[2])
                min_xy[0], max_xy[0] = min(min_xy[0], x), max(max_xy[0], x)
                min_xy[1], max_xy[1] = min(min_xy[1], y), max(max_xy[1], y)
    return max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])


def _compute_piece_scale(mesh_dir: Path, square_size: float, fill_frac: float) -> float:
    """Uniform scale so the widest piece fills ``fill_frac`` of a square."""
    widest = 0.0
    for ptype in _PIECE_TYPES:
        p = mesh_dir / f"{ptype}_white.obj"
        if p.is_file():
            widest = max(widest, _obj_xy_extent(p))
    if widest <= 0.0:
        return 1.0
    return (fill_frac * square_size) / widest


def _yaw_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    h = math.radians(yaw_deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def starting_layout() -> list[dict]:
    """Return the 32-piece starting position.

    Each entry: name, ptype, color ('white'/'black'), row (file 0-7),
    col (rank 0-7; 0 nearest the arm), yaw_deg.
    White on cols 0-1 (near arm), black on cols 6-7 (far).
    """
    layout: list[dict] = []
    for file_idx in range(8):
        # White back rank (col 0) + pawns (col 1)
        layout.append(_entry(_BACK_RANK[file_idx], "white", file_idx, 0, 0.0))
        layout.append(_entry("pawn", "white", file_idx, 1, 0.0))
        # Black pawns (col 6) + back rank (col 7); knights yaw 180 to face white
        layout.append(_entry("pawn", "black", file_idx, 6, 180.0))
        back = _BACK_RANK[file_idx]
        layout.append(_entry(back, "black", file_idx, 7, 180.0 if back == "knight" else 0.0))
    return layout


def _entry(ptype: str, color: str, row: int, col: int, yaw_deg: float) -> dict:
    name = f"piece_{color[0]}_{ptype}_{row}_{col}"
    return {"name": name, "ptype": ptype, "color": color, "row": row, "col": col, "yaw_deg": yaw_deg}


def _board_squares_xml(square_size: float, board_thickness: float, light, dark) -> str:
    """Procedural 64-square visual board (fallback when no board texture)."""
    surface_z = board_thickness + 0.0001
    sq_half = square_size / 2.0
    lr, lg, lb, la = light
    dr, dg, db, da = dark
    lines: list[str] = []
    for row in range(8):
        for col in range(8):
            r, g, b, a = (lr, lg, lb, la) if (row + col) % 2 == 0 else (dr, dg, db, da)
            x_rel = (col - 3.5) * square_size
            y_rel = (row - 3.5) * square_size
            lines.append(
                f'<geom name="sq_{row}_{col}" type="box" '
                f'size="{sq_half:.5f} {sq_half:.5f} 0.00010" '
                f'pos="{x_rel:.5f} {y_rel:.5f} {surface_z:.5f}" '
                f'rgba="{r} {g} {b} {a}" contype="0" conaffinity="0"/>'
            )
    return "\n      ".join(lines)


def _build_chess_scene_xml(config: ChessBoardConfig, scale: float, cameras_xml: str) -> str:
    cx, cy = config.board_center
    square_size = config.square_size
    board_thickness = config.board_thickness
    board_half = 4.0 * square_size
    half_thick = board_thickness / 2.0
    mesh_dir = Path(config.mesh_dir).resolve() if config.mesh_dir else _DEFAULT_MESH_DIR
    gr, gg, gb, ga = sample_color(config.ground_colors)
    piece_z = board_thickness + 0.0005  # mesh base sits at local z=0

    # --- assets: 6 piece meshes + piece/board materials ---
    mesh_assets = "".join(
        f'    <mesh name="cm_{t}" file="{(mesh_dir / f"{t}_white.obj").as_posix()}" '
        f'scale="{scale:.6f} {scale:.6f} {scale:.6f}"/>\n'
        for t in _PIECE_TYPES
    )

    def _piece_material(name: str, rgba, texture: str | None) -> str:
        if texture:
            return (
                f'    <texture name="{name}_tex" type="2d" file="{Path(texture).resolve().as_posix()}"/>\n'
                f'    <material name="{name}_mat" texture="{name}_tex" texuniform="true"/>\n'
            )
        r, g, b, a = rgba
        return f'    <material name="{name}_mat" rgba="{r} {g} {b} {a}" specular="0.3" shininess="0.4"/>\n'

    mat_assets = _piece_material("white", config.white_piece_rgba, config.white_texture)
    mat_assets += _piece_material("black", config.black_piece_rgba, config.black_texture)

    board_surface_xml: str
    if config.board_texture:
        tex = Path(config.board_texture).resolve().as_posix()
        mat_assets += (
            f'    <texture name="board_tex" type="2d" file="{tex}"/>\n'
            f'    <material name="board_mat" texture="board_tex" texuniform="false"/>\n'
        )
        board_surface_xml = (
            f'<geom name="board_surface" type="box" '
            f'size="{board_half:.5f} {board_half:.5f} 0.00040" '
            f'pos="0 0 {board_thickness + 0.0002:.5f}" material="board_mat" '
            f'contype="0" conaffinity="0"/>'
        )
    else:
        board_surface_xml = _board_squares_xml(
            square_size, board_thickness, config.light_square_color, config.dark_square_color
        )

    # --- piece bodies in starting position (world coords) ---
    piece_bodies: list[str] = []
    for e in starting_layout():
        x = cx + (e["col"] - 3.5) * square_size
        y = cy + (e["row"] - 3.5) * square_size
        qw, qx, qy, qz = _yaw_quat(e["yaw_deg"])
        mat = f'{e["color"]}_mat'
        name = e["name"]
        piece_bodies.append(
            f'    <body name="{name}" pos="{x:.5f} {y:.5f} {piece_z:.5f}" '
            f'quat="{qw:.5f} {qx:.5f} {qy:.5f} {qz:.5f}">\n'
            f'      <freejoint name="{name}_joint"/>\n'
            f'      <geom name="{name}_col" type="mesh" mesh="cm_{e["ptype"]}" group="3" '
            f'mass="{config.piece_mass}" contype="1" conaffinity="1" condim="4" '
            f'friction="1 0.05 0.001" solref="0.01 1" solimp="0.95 0.99 0.001"/>\n'
            f'      <geom name="{name}_vis" type="mesh" mesh="cm_{e["ptype"]}" '
            f'material="{mat}" group="2" contype="0" conaffinity="0" mass="0"/>\n'
            f"    </body>\n"
        )

    # Source (red) / target (green) square masks: flat quads in group 4, parked
    # off-board until a square is selected. Rendered only by overhead/global.
    sq_half = square_size / 2.0
    px, py, pz = _MASK_PARK
    masks_xml = (
        f'    <body name="mask_source" pos="{px} {py} {pz}">\n'
        f'      <geom name="mask_source_geom" type="box" '
        f'size="{sq_half:.5f} {sq_half:.5f} 0.00030" rgba="0.90 0.10 0.10 0.55" '
        f'group="{MASK_GEOM_GROUP}" contype="0" conaffinity="0"/>\n'
        f"    </body>\n"
        f'    <body name="mask_target" pos="{px} {py} {pz}">\n'
        f'      <geom name="mask_target_geom" type="box" '
        f'size="{sq_half:.5f} {sq_half:.5f} 0.00030" rgba="0.10 0.80 0.10 0.55" '
        f'group="{MASK_GEOM_GROUP}" contype="0" conaffinity="0"/>\n'
        f"    </body>\n"
    )

    # Rotate the board surface in place (pieces are separate top-level bodies and
    # are unaffected). Square centres map onto themselves for 90deg multiples.
    bqw, bqx, bqy, bqz = _yaw_quat(config.board_yaw_deg)

    robot_path = str(_SO101_XML)
    return f"""\
<mujoco model="chess_scene">
  <compiler angle="radian"/>

  <include file="{robot_path}"/>
  {SCENE_OPTION_XML}

  <asset>
{mesh_assets}{mat_assets}  </asset>

  <visual>
    <headlight diffuse="0.0 0.0 0.0" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <worldbody>
    <light pos="1 1 3.5" dir="-0.27 -0.27 -0.92" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="0 0 0.01" rgba="{gr} {gg} {gb} {ga}"
          pos="0 0 0" contype="1" conaffinity="1"/>
{cameras_xml}
    <body name="chessboard" pos="{cx:.5f} {cy:.5f} 0" quat="{bqw:.5f} {bqx:.5f} {bqy:.5f} {bqz:.5f}">
      <geom name="board_base" type="box"
            size="{board_half:.5f} {board_half:.5f} {half_thick:.5f}"
            pos="0 0 {half_thick:.5f}" rgba="0.30 0.20 0.12 1"
            contype="1" conaffinity="1"/>
      {board_surface_xml}
    </body>

{masks_xml}
{"".join(piece_bodies)}  </worldbody>
</mujoco>
"""


class ChessBoardEnv(SO101NexusMuJoCoBaseEnv):
    """Full 32-piece chess scene for teleop recording + DR replay.

    Pieces start in the standard position (white nearest the arm). All pieces
    are free-jointed and graspable. A central white pawn is designated the
    "primary" piece for grasp detection and reach-reward bookkeeping.
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

        mesh_dir = Path(config.mesh_dir) if config.mesh_dir else _DEFAULT_MESH_DIR
        self._piece_scale = _compute_piece_scale(
            mesh_dir, config.square_size, config.piece_fill_frac
        )

        obs_list = config.observations or []
        overhead_comp = next((c for c in obs_list if isinstance(c, OverheadCamera)), None)
        global_comp = next((c for c in obs_list if isinstance(c, GlobalCamera)), None)
        cameras_xml = build_scene_cameras_xml(
            spawn_center=config.spawn_center,
            spawn_max_radius=config.spawn_max_radius,
            global_elevation_deg=getattr(global_comp, "elevation_deg", -38.0),
            global_azimuth_deg=getattr(global_comp, "azimuth_deg", 120.0),
            global_fov_deg=getattr(global_comp, "fov_deg", 50.0),
            overhead_x_offset=getattr(overhead_comp, "x_offset", 0.0),
            overhead_y_offset=getattr(overhead_comp, "y_offset", 0.0),
            overhead_height_offset=getattr(overhead_comp, "height_offset", 0.0),
            global_lookat_x_offset=getattr(global_comp, "lookat_x_offset", 0.0),
            global_lookat_y_offset=getattr(global_comp, "lookat_y_offset", 0.0),
            global_distance_scale=getattr(global_comp, "distance_scale", 1.0),
        )

        xml_string = _build_chess_scene_xml(config, self._piece_scale, cameras_xml)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        # Index every piece body + its free-joint qpos address + collision geom.
        self._layout = starting_layout()
        self._pieces: dict[str, dict] = {}
        for e in self._layout:
            name = e["name"]
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_col")
            self._pieces[name] = {
                "body_id": bid,
                "qpos_addr": int(self.model.jnt_qposadr[jid]),
                "geom_id": gid,
                "entry": e,
            }

        primary = f"piece_w_pawn_{_PRIMARY_ROW}_{_PRIMARY_COL}"
        p = self._pieces[primary]
        self._piece_body_id = p["body_id"]
        self._piece_qpos_addr = p["qpos_addr"]
        self._obj_geom_id = p["geom_id"]

        # Source/target square mask bodies (overhead-only visualization).
        self._mask_bodies = {
            "source": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mask_source"),
            "target": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mask_target"),
        }
        self._marks: dict[str, str | None] = {"source": None, "target": None}

        self._finish_model_setup()

    # ------------------------------------------------------------------
    # Source/target square masks (rendered only by overhead/global cameras)
    # ------------------------------------------------------------------

    def set_mark(self, kind: str, square: str) -> dict:
        """Place the source/target mask on a board square (algebraic, e.g. 'e1')."""
        if kind not in self._mask_bodies:
            raise ValueError(f"mark kind must be 'source' or 'target', got {kind!r}")
        row, col = square_to_rowcol(square)
        pos = self._square_world_pos(row, col)
        z = self.config.board_thickness + 0.0006
        self.model.body_pos[self._mask_bodies[kind]] = [pos[0], pos[1], z]
        self._marks[kind] = square.strip().lower()
        mujoco.mj_forward(self.model, self.data)
        return {"kind": kind, "square": self._marks[kind]}

    def clear_marks(self) -> dict:
        """Park both masks off-board (hidden)."""
        for kind, bid in self._mask_bodies.items():
            self.model.body_pos[bid] = list(_MASK_PARK)
            self._marks[kind] = None
        mujoco.mj_forward(self.model, self.data)
        return {"source": None, "target": None}

    def get_marks(self) -> dict[str, str | None]:
        return dict(self._marks)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _square_world_pos(self, row: int, col: int) -> np.ndarray:
        cfg = self.config
        cx, cy = cfg.board_center
        x = cx + (col - 3.5) * cfg.square_size
        y = cy + (row - 3.5) * cfg.square_size
        z = cfg.board_thickness + 0.0005
        return np.array([x, y, z])

    def _get_piece_pose(self) -> np.ndarray:
        addr = self._piece_qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    # ------------------------------------------------------------------
    # Observation component dispatch (default obs has no Target/Object comps)
    # ------------------------------------------------------------------

    def _get_component_data(self, component: object) -> np.ndarray:
        if isinstance(component, ObjectPose):
            return self._get_piece_pose()
        if isinstance(component, ObjectOffset):
            return self._get_piece_pose()[:3] - self._get_tcp_pose()[:3]
        if isinstance(component, (TargetPosition, TargetOffset)):
            # No discrete goal square in the full-board scene; report the
            # primary piece's own position (offset 0) to keep shapes valid.
            if isinstance(component, TargetPosition):
                return self._get_piece_pose()[:3]
            return np.zeros(3)
        return super()._get_component_data(component)

    # ------------------------------------------------------------------
    # Reset: place every piece in the fixed starting position
    # ------------------------------------------------------------------

    def _task_reset(self) -> None:
        for e in self._layout:
            p = self._pieces[e["name"]]
            addr = p["qpos_addr"]
            pos = self._square_world_pos(e["row"], e["col"])
            quat = _yaw_quat(e["yaw_deg"])
            self.data.qpos[addr : addr + 3] = pos
            self.data.qpos[addr + 3 : addr + 7] = quat
            self.data.qvel[addr : addr + 6] = 0.0
        self._initial_piece_z = float(self.config.board_thickness + 0.0005)

    def _refresh_reset_reference_state(self) -> None:
        self._initial_piece_z = float(self._get_piece_pose()[2])
        self._home_tcp_pos = self._get_tcp_pose()[:3].copy()

    # ------------------------------------------------------------------
    # Info & reward (light reach shaping toward the primary piece)
    # ------------------------------------------------------------------

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        piece_pos = self._get_piece_pose()[:3]
        is_grasped = self._is_grasping()
        return {
            "tcp_to_piece_dist": float(np.linalg.norm(piece_pos - tcp_pos)),
            "piece_to_target_dist": 0.0,
            "is_grasped": is_grasped,
            "is_robot_static": self._is_robot_static(),
            "lift_height": float(piece_pos[2] - self._initial_piece_z),
            "is_placed": False,
            "success": False,
        }

    def _compute_reward(self, info: dict) -> float:
        scale = self.config.reward.tanh_shaping_scale
        rp = reach_progress(info["tcp_to_piece_dist"], scale=scale)
        is_grasped = info["is_grasped"] > 0.5
        return self.config.reward.compute(
            reach_progress=rp,
            is_grasped=is_grasped,
            task_progress=0.0,
            is_complete=False,
            action_delta_norm=info.get("action_delta_norm", 0.0),
            energy_norm=info.get("energy_norm", 0.0),
        )
