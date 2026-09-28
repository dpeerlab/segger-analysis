"""Neighbourhood graph plots."""

from __future__ import annotations

import numpy as np
from adjustText import adjust_text
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

from .fov import outlines, tile
from .style import INK, METHOD_COLORS, MUTED

EGO = "#CC79A7"  # neighbourhood edges and centroids
CONTEXT = "#D5D8DD"


def ego_graph(
    ax: Axes,
    focal: str,
    neighbors: list[str],
    masks,
    centroids: dict[str, tuple[float, float]],
    bbox: tuple[float, float, float, float],
    radius: float | None = None,
    dilation: float | None = None,
) -> None:
    """Neighbours of one focal cell in a field of view.

    Parameters
    ----------
    focal
        Cell id of the central cell.
    neighbors
        Cell ids joined to ``focal`` in the graph.
    masks
        ``GeoSeries`` of outlines in the field of view, indexed by cell id.
    centroids
        Centroid of the focal cell and of every neighbour.
    radius
        Draw the kNN search radius (µm) around the focal centroid.
    dilation
        Draw the neighbours' outlines dilated by this many µm, dashed (mask-contact graph).
    """
    tile(ax, bbox, background="white")
    involved = [focal, *neighbors]
    context = masks.drop([c for c in involved if c in masks.index])
    outlines(ax, context.values, color=CONTEXT, lw=0.5, zorder=1)
    shown = masks.reindex([c for c in neighbors if c in masks.index])
    if dilation:
        grown = [g.buffer(dilation) for g in shown.values]
        outlines(ax, grown, color=INK, lw=0.8, ls=(0, (3, 1.5)), zorder=2)
        if focal in masks.index:
            outlines(ax, [masks[focal].buffer(dilation)], color=INK, lw=0.8, ls=(0, (3, 1.5)), zorder=2)
    else:
        outlines(ax, shown.values, color="#3B3F45", lw=0.8, zorder=2)
        if focal in masks.index:
            outlines(ax, [masks[focal]], color=INK, lw=1.6, zorder=3)
    cx, cy = centroids[focal]
    if radius:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta), color=EGO, lw=0.8, ls=(0, (4, 2)), zorder=2)
    for cell in neighbors:
        nx, ny = centroids[cell]
        ax.plot([cx, nx], [cy, ny], color=EGO, lw=0.9, zorder=4)
    xy = np.array([centroids[c] for c in involved])
    ax.scatter(xy[:, 0], xy[:, 1], s=[14] + [7] * len(neighbors), color=EGO, lw=0, zorder=5)
    ax.text(0.04, 0.96, f"n = {len(neighbors)}", transform=ax.transAxes, ha="left", va="top")


def density_clouds(
    ax: Axes,
    degrees: dict[str, tuple[np.ndarray, np.ndarray]],
    reference: str,
    fragmentation: str,
    overexpansion: str,
    xlim: tuple[float, float] = (0, 19),
    ylim: tuple[float, float] = (0, 10),
    max_cells: int = 6000,
    seed: int = 0,
    colors: dict[str, str] = METHOD_COLORS,
) -> dict[str, tuple[float, float]]:
    """Per-cell kNN neighbours against mask contacts as one density cloud per method.

    Up to ``max_cells`` cells per method are drawn, and integer degrees are
    jittered by ±0.35 before the Gaussian kernel density estimate. Dashed lines
    mark the ``reference`` means; the shaded half-planes start midway between the
    reference and the method with the most kNN neighbours (``fragmentation``, x)
    or the most contacts (``overexpansion``, y).

    Returns
    -------
    Mean (kNN, mask) degree per method.
    """
    gx, gy = np.meshgrid(np.linspace(*xlim, 170), np.linspace(*ylim, 170))
    grid = np.vstack([gx.ravel(), gy.ravel()])
    means, clouds = {}, {}
    for k, (method, (knn, mask)) in enumerate(degrees.items()):
        rng = np.random.default_rng([seed, k])
        knn, mask = np.asarray(knn, float), np.asarray(mask, float)
        means[method] = (knn.mean(), mask.mean())
        if len(knn) > max_cells:
            pick = rng.choice(len(knn), max_cells, replace=False)
            knn, mask = knn[pick], mask[pick]
        jitter = rng.uniform(-0.35, 0.35, size=(2, len(knn)))
        clouds[method] = gaussian_kde(np.vstack([knn + jitter[0], mask + jitter[1]]))(grid).reshape(gx.shape)

    rx, ry = means[reference]
    ax.axvspan(0.5 * (rx + means[fragmentation][0]), xlim[1], color=colors[fragmentation], alpha=0.10, lw=0, zorder=0)
    ax.axhspan(0.5 * (ry + means[overexpansion][1]), ylim[1], color=colors[overexpansion], alpha=0.10, lw=0, zorder=0)
    ax.axvline(rx, color=MUTED, lw=0.6, ls=(0, (3, 2)), alpha=0.6, zorder=1)
    ax.axhline(ry, color=MUTED, lw=0.6, ls=(0, (3, 2)), alpha=0.6, zorder=1)
    far_first = sorted(clouds, key=lambda m: -np.hypot(means[m][0] - rx, means[m][1] - ry))
    for method in far_first:
        z, color = clouds[method], colors[method]
        levels = np.linspace(0.55 * z.max(), z.max(), 5)
        cmap = LinearSegmentedColormap.from_list(method, ["#FFFFFF", color])
        ax.contourf(gx, gy, z, levels=levels, cmap=cmap, alpha=0.42, zorder=2)
        ax.contour(gx, gy, z, levels=[levels[0]], colors=[color], linewidths=0.7, zorder=3)
    texts = []
    for method, (mx, my) in means.items():
        ax.scatter(mx, my, s=34, color=colors[method], edgecolor="white", lw=0.7, zorder=6)
        texts.append(ax.text(mx, my, method, zorder=7))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    xy = np.array(list(means.values()))
    adjust_text(texts, x=xy[:, 0], y=xy[:, 1], ax=ax, expand=(1.6, 2.2), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    return means
