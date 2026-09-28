"""Figure style of the paper: method colours, fonts and panel export."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

METHODS = ["Segger", "10x Cell", "10x Nucleus", "Proseg", "Baysor", "Bering"]

METHOD_COLORS = {
    "Segger": "#0072B2",
    "10x Cell": "#163A5F",
    "10x Nucleus": "#B24C4C",
    "Proseg": "#009E73",
    "Baysor": "#E69F00",
    "Bering": "#CC79A7",
    "Cellpose": "#3A3A3A",
}

INK = "#1A1A1A"
MUTED = "#8A8F98"
TILE_BACKGROUND = "#FEFCF9"

MM = 1 / 25.4  # inches per millimetre


def set_style(font_size: float = 6.5) -> None:
    """Matplotlib defaults for journal-size panels (Nature column width is 89 mm)."""
    logging.getLogger("fontTools").setLevel(logging.WARNING)  # PDF font subsetting is verbose
    installed = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in installed] + ["DejaVu Sans"]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": fonts,
            "font.size": font_size,
            "axes.titlesize": font_size + 0.5,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
            "legend.fontsize": font_size - 0.5,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_panel(fig: plt.Figure, path: str | Path) -> None:
    """Save a panel as PDF (vector) and PNG next to each other."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix))
