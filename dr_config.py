"""Editable domain-randomisation grid for chess replay.

By default this auto-discovers materials from ``assets/textures/`` (piece PNGs
under ``pieces/white`` + ``pieces/black``, board PNGs under ``board/``) and adds
procedural board recolours. Drop PNGs in to expand the grid with no edits here.

Run replay against a recorded dataset with:

    PYTHONPATH=src python -m so101_lite.teleop.cli replay \
        --in <recorded_dataset_root> --out-repo local/chess-dr \
        --dr-config dr_config.py

To customise, replace ``DR`` with an explicit DRConfig, e.g.:

    from so101_lite.dr.variations import DRConfig, PieceMaterials, BoardVariant
    DR = DRConfig(
        pieces=[
            PieceMaterials("marble_set", white_texture="assets/textures/pieces/white/marble.png",
                           black_texture="assets/textures/pieces/black/ebony.png"),
            PieceMaterials("ivory_ebony", (0.90, 0.89, 0.85, 1.0), (0.12, 0.12, 0.14, 1.0)),
        ],
        boards=[
            BoardVariant("wood", texture="assets/textures/board/chess_board_standard.png"),
            BoardVariant("green_white", None, (0.93, 0.93, 0.88, 1.0), (0.30, 0.45, 0.30, 1.0)),
        ],
    )
"""

from so101_lite.dr.variations import discover_dr

DR = discover_dr()
