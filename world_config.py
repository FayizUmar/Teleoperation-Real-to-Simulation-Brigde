"""Editable scene layout for so101-lite teleop.

Tune the board, piece, and cameras here, then launch:

    PYTHONPATH=src python -m so101_lite.teleop.cli teleop --sim \
        --world-config world_config.py

For the 3D passive viewer (macOS) use `mjpython` instead of `python`.

All distances are in metres. Camera offsets nudge the auto-fitted cameras:
  overhead  x_offset > 0 = forward, < 0 = backward
            y_offset > 0 = left,    < 0 = right
            height_offset > 0 = higher, < 0 = lower
  global    elevation_deg / azimuth_deg rotate it; lookat_*_offset re-aim it;
            distance_scale > 1 pulls it back, < 1 pushes it closer.
"""

from so101_lite.world import (
    BoardSpec,
    GlobalCam,
    OverheadCam,
    PieceSpec,
    WorldConfig,
    WristCam,
)

WORLD = WorldConfig(
    board=BoardSpec(
        center=(0.335, 0.0),     # near edge ~20 cm from the arm base (+10 cm further)
        square_size=0.03375,     # 27 cm board / 8
        thickness=0.006,
        yaw_deg=90.0,            # rotate board surface 90 deg in place (pieces unaffected)
        texture="assets/textures/board/chess_board_standard.png",  # None => procedural
    ),
    piece=PieceSpec(
        mesh_dir=None,           # None => bundled assets/meshes/chess
        fill_frac=0.75,
        mass=0.02,
        # white near the arm, black opposite; colour is a material (DR axis)
        white_rgba=(0.90, 0.89, 0.85, 1.0),
        black_rgba=(0.12, 0.12, 0.14, 1.0),
    ),
    overhead=OverheadCam(
        fov_deg=45.0,
        x_offset=0.0,        # e.g. 0.04 to move 4 cm forward
        y_offset=0.0,
        height_offset=0.0,   # e.g. -0.08 to drop the camera 8 cm
    ),
    global_cam=GlobalCam(
        fov_deg=50.0,
        elevation_deg=-38.0,
        azimuth_deg=120.0,
        lookat_x_offset=0.0,
        lookat_y_offset=0.0,
        distance_scale=1.0,
    ),
    wrist=WristCam(),
    robot_color="black",   # arm body colour ("yellow", "gray", "blue", ...)
)
