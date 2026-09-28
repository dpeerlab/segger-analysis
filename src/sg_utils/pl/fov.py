"""Field-of-view tiles: morphology images, cell outlines and transcripts."""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from shapely.strtree import STRtree

from .style import INK, MUTED, TILE_BACKGROUND

# Morphology channel colours used in the paper.
DAPI_RED = "#FF4533"
MEMBRANE_BLUE = "#2F6BFF"
MEMBRANE_CYAN = "#3CC8FA"
INTERIOR_CYAN = "#3CC8E6"
# Categorical fill of reference cells; neighbouring cells get different colours.
CELL_FILL = ["#C8C84A", "#4DA84F", "#4E9FD1", "#56308F", "#D23264", "#E89A1F", "#E8C630", "#C9742B"]
CONTAMINANT_ORANGE = "#F28E2B"
HOST_BLUE = "#6BAED6"


def morphology_rgb(
    image: np.ndarray,
    colors: list[str],
    percentile: float | list[float] = 99.5,
    gamma: float | list[float] = 1.25,
    background: float | list[float] = 0.0,
    gain: float = 1.0,
) -> np.ndarray:
    """Additive colour composite of morphology channels.

    Parameters
    ----------
    image
        ``(channels, rows, cols)`` intensities, e.g. from :func:`sg_utils.io.read_morphology`.
    colors
        One colour per channel.
    percentile
        Intensity percentile mapped to full brightness, per channel or for all.
    gamma
        Gamma applied after normalisation.
    background
        Intensity percentile subtracted as background.
    gain
        Overall brightness.

    Returns
    -------
    ``(rows, cols, 3)`` RGB image in [0, 1].
    """
    n = len(colors)
    per_channel = [np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in (percentile, gamma, background)]
    rgb = np.zeros((*image.shape[1:], 3))
    for channel, color, top, g, bottom in zip(image.astype(np.float64), colors, *per_channel, strict=True):
        lo, hi = np.percentile(channel, bottom), np.percentile(channel, top)
        scaled = np.clip((channel - lo) / (hi - lo + 1e-9), 0, 1) ** g
        rgb += scaled[..., None] * np.asarray(mcolors.to_rgb(color))[None, None, :]
    return np.clip(rgb * gain, 0, 1)


def tile(ax: Axes, bbox: tuple[float, float, float, float], background: str = TILE_BACKGROUND) -> None:
    """Field of view in µm without axes; y increases downwards as in the image."""
    x0, y0, x1, y1 = bbox
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_facecolor(background)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def image(ax: Axes, rgb: np.ndarray, bbox: tuple[float, float, float, float]) -> None:
    """Show an RGB morphology crop covering ``bbox``."""
    x0, y0, x1, y1 = bbox
    tile(ax, bbox, background="black")
    ax.imshow(rgb, extent=(x0, x1, y1, y0), interpolation="nearest")


def outlines(ax: Axes, geoms, color: str = INK, lw: float = 0.8, **kwargs) -> None:
    """Draw polygon outlines."""
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        for poly in getattr(geom, "geoms", [geom]):
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color, lw=lw, solid_joinstyle="round", **kwargs)


def categorical_fill(masks, palette: list[str] = CELL_FILL, gap: float = 0.5) -> pd.Series:
    """Colour per cell such that cells closer than ``gap`` µm get different colours.

    Cells are coloured left to right, each with the least-used colour that none of
    its neighbours has, so all colours appear about equally often.
    """
    geoms = list(masks.values)
    tree = STRtree(geoms)
    order = np.lexsort((masks.centroid.y.to_numpy(), masks.centroid.x.to_numpy()))
    colors = np.full(len(geoms), -1)
    used = np.zeros(len(palette), dtype=int)
    for i in order:
        taken = {colors[j] for j in tree.query(geoms[i].buffer(gap), predicate="intersects") if j != i}
        free = [c for c in range(len(palette)) if c not in taken] or list(range(len(palette)))
        colors[i] = min(free, key=lambda c: used[c])
        used[colors[i]] += 1
    return pd.Series([palette[c] for c in colors], index=masks.index)


def filled(ax: Axes, masks, colors: pd.Series, alpha: float = 0.95) -> None:
    """Fill polygons with one colour each."""
    polys, faces = [], []
    for cell, geom in masks.items():
        for poly in getattr(geom, "geoms", [geom]):
            polys.append(np.asarray(poly.exterior.coords))
            faces.append(colors[cell])
    ax.add_collection(PolyCollection(polys, facecolors=faces, edgecolors="none", alpha=alpha, zorder=1))


def box(ax: Axes, bbox: tuple[float, float, float, float], color: str = "#BDBDBD", lw: float = 0.8) -> None:
    """Outline a sub-region, e.g. the zoom of the next row."""
    x0, y0, x1, y1 = bbox
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=lw, zorder=10))


def scale_bar(
    ax: Axes,
    length: float,
    label: str | None = None,
    color: str = "white",
    fontsize: float | None = None,
) -> None:
    """Scale bar in the lower-right corner (length in µm), outlined so it reads on any background."""
    from matplotlib import patheffects

    x0, x1 = sorted(ax.get_xlim())
    y_top, y_bottom = sorted(ax.get_ylim())
    width, height = x1 - x0, y_bottom - y_top
    xe = x1 - 0.06 * width
    y = y_bottom - 0.08 * height
    halo = [patheffects.withStroke(linewidth=1.6, foreground="black" if color == "white" else "white")]
    ax.plot([xe - length, xe], [y, y], color=color, lw=2.0, solid_capstyle="butt", path_effects=halo)
    ax.text(
        xe - length / 2, y - 0.035 * height, label or f"{length:g} µm",
        color=color, fontsize=fontsize, ha="center", va="bottom", fontweight="bold", path_effects=halo,
    )


def contamination(
    ax: Axes,
    transcripts: pl.DataFrame,
    method: str,
    masks,
    contaminants: list[str],
    bbox: tuple[float, float, float, float],
    size: float = 1.2,
    contaminant_size: float = 9.0,
) -> int:
    """Host-cell transcripts and contaminating transcripts of one segmentation.

    Parameters
    ----------
    transcripts
        Transcripts of the field of view with the cell-id column ``method``.
    method
        Segmentation to draw.
    masks
        ``GeoSeries`` of the host cells (e.g. epithelial and tumour cells), indexed
        by cell id; only transcripts assigned to these cells are drawn.
    contaminants
        Genes drawn in orange, e.g. stromal genes absent from epithelial cells.
    bbox
        Field of view ``(xmin, ymin, xmax, ymax)`` in µm.

    Returns
    -------
    Number of contaminating transcripts assigned to host cells.
    """
    tile(ax, bbox)
    inside = transcripts.filter(pl.col(method).is_in(list(masks.index)))
    is_contaminant = inside.get_column("feature_name").is_in(contaminants).to_numpy()
    xy = inside.select("x", "y").to_numpy()
    ax.scatter(*xy[~is_contaminant].T, s=size, c=HOST_BLUE, lw=0, alpha=0.8, rasterized=True)
    outlines(ax, masks.values, color=INK, lw=0.7, zorder=3)
    ax.scatter(*xy[is_contaminant].T, s=contaminant_size, c=CONTAMINANT_ORANGE, lw=0, zorder=4)
    return int(is_contaminant.sum())


def marker_recall(
    ax: Axes,
    transcripts: pl.DataFrame,
    method: str,
    focal: str,
    center: tuple[float, float],
    markers: dict[str, str],
    masks,
    bbox: tuple[float, float, float, float],
    radius: float = 10.0,
    size: float = 14.0,
) -> tuple[int, int]:
    """Marker transcripts around one focal cell.

    Filled points are marker transcripts assigned to the focal cell, crosses are
    marker transcripts within ``radius`` of its centroid assigned elsewhere or not
    at all, and red rings mark genes present within ``radius`` without any
    transcript in the focal cell.

    Parameters
    ----------
    transcripts
        Transcripts of the field of view.
    method
        Segmentation to draw.
    focal
        Cell id of the focal cell in ``method``.
    center
        Centroid of the focal cell.
    markers
        Colour of each marker gene.
    masks
        ``GeoSeries`` of outlines in the field of view, indexed by cell id.

    Returns
    -------
    Number of marker genes recovered by the focal cell and number available.
    """
    tile(ax, bbox)
    others = masks.drop(focal, errors="ignore")
    outlines(ax, others.values, color="#C9CCD1", lw=0.5, zorder=1)
    if focal in masks.index:
        outlines(ax, [masks[focal]], color=INK, lw=1.0, zorder=3)
    cx, cy = center
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta), ls=(0, (3, 2)), color="#555555", lw=0.7, zorder=2)

    tx = transcripts.filter(pl.col("feature_name").is_in(list(markers))).with_columns(
        (((pl.col("x") - cx) ** 2 + (pl.col("y") - cy) ** 2).sqrt() <= radius).alias("near"),
        (pl.col(method) == focal).fill_null(False).alias("own"),
    )
    recovered, available = 0, 0
    for gene, color in markers.items():
        g = tx.filter(pl.col("feature_name") == gene)
        own = g.filter(pl.col("own")).select("x", "y").to_numpy()
        missed = g.filter(pl.col("near") & ~pl.col("own")).select("x", "y").to_numpy()
        ax.scatter(*own.T, s=size, c=color, lw=0, zorder=5)
        ax.scatter(*missed.T, s=size * 0.55, c=color, marker="x", lw=0.9, zorder=4)
        if g.filter(pl.col("near")).height:
            available += 1
            if len(own):
                recovered += 1
            else:
                ax.scatter(*missed.T, s=size * 2.6, facecolors="none", edgecolors="#D62728", lw=0.8, zorder=6)
    return recovered, available


def marker_recall_legend(ax: Axes, markers: dict[str, str], radius: float = 10.0) -> None:
    """Legend for :func:`marker_recall`."""
    handles = [
        Line2D([], [], ls="", marker="o", ms=5, mfc="none", mec="#D62728", mew=1.0, label="omitted marker gene"),
        Line2D([], [], ls="", marker="o", ms=3.5, color=MUTED, label="captured transcript"),
        Line2D([], [], ls="", marker="x", ms=3.5, color=MUTED, mew=1.0, label="omitted transcript"),
        Line2D([], [], color=INK, lw=1.0, label="segmentation mask"),
        Line2D([], [], color="#555555", lw=0.8, ls=(0, (3, 2)), label=f"{radius:g} µm radius"),
    ] + [Line2D([], [], ls="", marker="o", ms=3.5, color=c, label=g) for g, c in markers.items()]
    ax.legend(handles=handles, loc="center left", ncol=5, handletextpad=0.3, columnspacing=1.2)
    ax.axis("off")


def contamination_legend(ax: Axes, channels: dict[str, str] | None = None) -> None:
    """Legend for :func:`contamination`, optionally with the morphology channels."""
    handles = []
    if channels:
        handles += [Line2D([], [], ls="", marker="s", ms=5, color=c, label=name) for name, c in channels.items()]
    handles += [
        Line2D([], [], ls="", marker="o", ms=3.5, color=HOST_BLUE, label="host-cell transcript"),
        Line2D([], [], ls="", marker="o", ms=4.5, color=CONTAMINANT_ORANGE, label="contaminating transcript"),
        Line2D([], [], color=INK, lw=1.0, label="host-cell mask"),
    ]
    ax.legend(handles=handles, loc="center left", ncol=len(handles), handletextpad=0.3, columnspacing=1.2)
    ax.axis("off")
