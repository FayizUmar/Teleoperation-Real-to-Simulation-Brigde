"""Domain-randomisation variation registry for the chess scene.

A ``DRConfig`` is the Cartesian product of:
  * **piece materials** — a (white, black) appearance pair, either flat RGBA or
    image-file textures applied to the bare meshes.
  * **board variants** — a board surface, either an image texture or a
    procedural light/dark square recolour.

Each (piece, board) combo becomes one re-rendered copy of every recorded demo.
Materials are auto-discovered from ``assets/textures/`` so dropping PNGs in
expands the variation set with no code changes.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"

RGBA = tuple[float, float, float, float]


@dataclass
class PieceMaterials:
    """Appearance for the two piece colours (geometry stays fixed)."""

    name: str
    white_rgba: RGBA = (0.90, 0.89, 0.85, 1.0)
    black_rgba: RGBA = (0.12, 0.12, 0.14, 1.0)
    white_texture: str | None = None
    black_texture: str | None = None


@dataclass
class BoardVariant:
    """Board surface appearance: image texture, or procedural square colours."""

    name: str
    texture: str | None = None
    light_color: RGBA = (0.93, 0.87, 0.74, 1.0)
    dark_color: RGBA = (0.18, 0.11, 0.06, 1.0)


@dataclass
class DRConfig:
    """The full variation grid."""

    pieces: list[PieceMaterials] = field(default_factory=list)
    boards: list[BoardVariant] = field(default_factory=list)

    @property
    def n_combos(self) -> int:
        return len(self.pieces) * len(self.boards)

    def combos(self):
        """Yield every (PieceMaterials, BoardVariant) pair."""
        for p in self.pieces:
            for b in self.boards:
                yield p, b


# Flat-colour fallbacks used when no piece-texture PNGs are present yet.
_FALLBACK_PIECES = [
    PieceMaterials("ivory_ebony", (0.90, 0.89, 0.85, 1.0), (0.12, 0.12, 0.14, 1.0)),
    PieceMaterials("cream_charcoal", (0.96, 0.93, 0.84, 1.0), (0.22, 0.22, 0.25, 1.0)),
    PieceMaterials("sand_walnut", (0.85, 0.78, 0.62, 1.0), (0.30, 0.19, 0.11, 1.0)),
]

# Procedural board recolours that complement any wood image variants.
_PROCEDURAL_BOARDS = [
    BoardVariant("green_white", None, (0.93, 0.93, 0.88, 1.0), (0.30, 0.45, 0.30, 1.0)),
    BoardVariant("gray_white", None, (0.92, 0.92, 0.92, 1.0), (0.40, 0.40, 0.42, 1.0)),
    BoardVariant("blue_cream", None, (0.95, 0.92, 0.82, 1.0), (0.25, 0.35, 0.55, 1.0)),
]


def discover_dr(assets_dir: Path = _ASSETS_DIR) -> DRConfig:
    """Build a DRConfig from the staged assets + sensible procedural defaults."""
    # Piece materials: pair white/black PNGs by sorted order; else flat fallbacks.
    pieces: list[PieceMaterials] = []
    wdir = assets_dir / "textures" / "pieces" / "white"
    bdir = assets_dir / "textures" / "pieces" / "black"
    wpngs = sorted(wdir.glob("*.png")) if wdir.is_dir() else []
    bpngs = sorted(bdir.glob("*.png")) if bdir.is_dir() else []
    for i, wp in enumerate(wpngs):
        bp = bpngs[i] if i < len(bpngs) else (bpngs[-1] if bpngs else None)
        pieces.append(
            PieceMaterials(
                name=wp.stem,
                white_texture=str(wp),
                black_texture=str(bp) if bp else None,
            )
        )
    if not pieces:
        pieces = list(_FALLBACK_PIECES)

    # Board variants: every board PNG + the procedural recolours.
    boards: list[BoardVariant] = []
    board_dir = assets_dir / "textures" / "board"
    if board_dir.is_dir():
        for png in sorted(board_dir.glob("*.png")):
            boards.append(BoardVariant(name=png.stem, texture=str(png)))
    boards += list(_PROCEDURAL_BOARDS)

    return DRConfig(pieces=pieces, boards=boards)


def load_dr_config(path: str | Path) -> DRConfig:
    """Import a Python file and return its top-level ``DR`` (a DRConfig)."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"DR config file not found: {p}")
    spec = importlib.util.spec_from_file_location("so101_dr_config", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load DR config from {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dr = getattr(module, "DR", None)
    if dr is None:
        raise ValueError(f"{p} must define a top-level `DR = DRConfig(...)`")
    if not isinstance(dr, DRConfig):
        raise TypeError(f"DR in {p} must be a DRConfig, got {type(dr).__name__}")
    return dr
