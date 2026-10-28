"""Consistent, publication-quality matplotlib styling."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#132238"
MUTED = "#64748b"
GRID = "#dbe3ee"
BLUE = "#2563eb"
TEAL = "#0d9488"
AMBER = "#d97706"
ROSE = "#e11d48"
VIOLET = "#7c3aed"
PALETTE = [BLUE, TEAL, AMBER, ROSE, VIOLET, "#0891b2", "#65a30d", "#db2777"]


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 170,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, path: Path, tight: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
