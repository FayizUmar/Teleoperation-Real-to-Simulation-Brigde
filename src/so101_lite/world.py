"""WorldConfig: a single, human-editable source of truth for the chess scene.

A user edits a plain Python file (``world_config.py``) that assigns a module
level ``WORLD = WorldConfig(...)``. The teleop CLI loads it with
``--world-config world_config.py`` and turns it into a ``ChessBoardConfig``
(plus correctly-offset observation cameras) via :func:`build_chess_config`.

Why a Python file instead of YAML/TOML: zero new dependencies, IDE
auto-completion, and the dataclasses validate themselves on construction.

Note on scope: board geometry (size, square count, thickness) is baked into
the MuJoCo model at build time, so changing it requires a fresh env. Camera
offsets are applied at env-build time too. For *runtime* object moving /
hiding while the sim is live, see :mod:`so101_lite.teleop.scene_api`.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BoardSpec:
    """Chessboard geometry, placement, and surface appearance."""

    center: tuple[float, float] = (0.235, 0.0)  # XY world position of board centre (m)
    square_size: float = 0.03375  # 27 cm board / 8
    thickness: float = 0.006  # board thickness above the floor (m)
    yaw_deg: float = 0.0  # rotate board surface in place (pieces unaffected)
    texture: str | None = None  # image-file board surface; None => procedural squares
    light_color: tuple[float, float, float, float] = (0.93, 0.87, 0.74, 1.0)
    dark_color: tuple[float, float, float, float] = (0.18, 0.11, 0.06, 1.0)


@dataclass
class PieceSpec:
    """The 32-piece mesh set: geometry fixed, colour/texture are DR axes."""

    mesh_dir: str | None = None  # dir with {pawn,rook,...}_white.obj; None => bundled assets
    fill_frac: float = 0.75  # widest piece fills this fraction of a square
    mass: float = 0.02  # kg per piece
    white_rgba: tuple[float, float, float, float] = (0.90, 0.89, 0.85, 1.0)
    black_rgba: tuple[float, float, float, float] = (0.12, 0.12, 0.14, 1.0)
    white_texture: str | None = None  # DR: image-file material for white pieces
    black_texture: str | None = None  # DR: image-file material for black pieces


@dataclass
class OverheadCam:
    """Top-down camera. Offsets nudge it relative to the auto-fit pose (metres)."""

    fov_deg: float = 45.0
    x_offset: float = 0.0  # + forward (robot +X) / - backward
    y_offset: float = 0.0  # + left (world +Y) / - right
    height_offset: float = 0.0  # + up / - down
    width: int = 640
    height: int = 360


@dataclass
class GlobalCam:
    """Angled overview camera. Elevation/azimuth rotate it; offsets re-aim it."""

    fov_deg: float = 50.0
    elevation_deg: float = -38.0  # negative = looking down from above
    azimuth_deg: float = 120.0
    lookat_x_offset: float = 0.0  # shift aim point forward/back (m)
    lookat_y_offset: float = 0.0  # shift aim point sideways (m)
    distance_scale: float = 1.0  # >1 pulls camera back, <1 pushes it closer
    width: int = 640
    height: int = 360


@dataclass
class WristCam:
    """Ego-centric gripper camera."""

    enabled: bool = True
    pos_y_center: float = 0.04
    pos_z_center: float = -0.04
    width: int = 320
    height: int = 240


@dataclass
class WorldConfig:
    """Full scene description: board + piece + the three cameras."""

    board: BoardSpec = field(default_factory=BoardSpec)
    piece: PieceSpec = field(default_factory=PieceSpec)
    overhead: OverheadCam = field(default_factory=OverheadCam)
    global_cam: GlobalCam = field(default_factory=GlobalCam)
    wrist: WristCam = field(default_factory=WristCam)
    robot_color: str = "yellow"  # arm body colour (e.g. "black", "gray", "blue")


DEFAULT_WORLD = WorldConfig()


def load_world_config(path: str | Path) -> WorldConfig:
    """Import a Python file and return its top-level ``WORLD`` object."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"world config file not found: {p}")
    spec = importlib.util.spec_from_file_location("so101_world_config", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load world config from {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    world = getattr(module, "WORLD", None)
    if world is None:
        raise ValueError(f"{p} must define a top-level `WORLD = WorldConfig(...)`")
    if not isinstance(world, WorldConfig):
        raise TypeError(
            f"WORLD in {p} must be a WorldConfig, got {type(world).__name__}"
        )
    return world


def build_chess_config(world: WorldConfig):
    """Construct a ``ChessBoardConfig`` (with offset cameras) from a WorldConfig."""
    from so101_lite.config import ChessBoardConfig
    from so101_lite.observations import (
        EndEffectorPose,
        GlobalCamera,
        GraspState,
        JointPositions,
        OverheadCamera,
        WristCamera,
    )

    observations: list = [JointPositions(), EndEffectorPose(), GraspState()]
    if world.wrist.enabled:
        observations.append(
            WristCamera(
                width=world.wrist.width,
                height=world.wrist.height,
                pos_y_center=world.wrist.pos_y_center,
                pos_z_center=world.wrist.pos_z_center,
            )
        )
    observations.append(
        GlobalCamera(
            width=world.global_cam.width,
            height=world.global_cam.height,
            fov_deg=world.global_cam.fov_deg,
            elevation_deg=world.global_cam.elevation_deg,
            azimuth_deg=world.global_cam.azimuth_deg,
            lookat_x_offset=world.global_cam.lookat_x_offset,
            lookat_y_offset=world.global_cam.lookat_y_offset,
            distance_scale=world.global_cam.distance_scale,
        )
    )
    observations.append(
        OverheadCamera(
            width=world.overhead.width,
            height=world.overhead.height,
            fov_deg=world.overhead.fov_deg,
            x_offset=world.overhead.x_offset,
            y_offset=world.overhead.y_offset,
            height_offset=world.overhead.height_offset,
        )
    )

    return ChessBoardConfig(
        board_center=world.board.center,
        square_size=world.board.square_size,
        board_thickness=world.board.thickness,
        board_yaw_deg=world.board.yaw_deg,
        board_texture=world.board.texture,
        light_square_color=world.board.light_color,
        dark_square_color=world.board.dark_color,
        mesh_dir=world.piece.mesh_dir,
        piece_fill_frac=world.piece.fill_frac,
        piece_mass=world.piece.mass,
        white_piece_rgba=world.piece.white_rgba,
        black_piece_rgba=world.piece.black_rgba,
        white_texture=world.piece.white_texture,
        black_texture=world.piece.black_texture,
        robot_colors=world.robot_color,
        observations=observations,
    )
