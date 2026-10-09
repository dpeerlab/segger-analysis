"""Tiling maps for ED Fig. 2."""

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from shapely.plotting import patch_from_polygon


def plot_tiling(
    tiling,
    values,
    ax,
    vmin: float = 1e2,
    vmax: float = 1e5,
    cmap: str = 'rocket',
    linewidth: float = 0.125,
    edgecolor: str = 'w',
):
    """Draw a tiling as filled polygons colored by a per-tile value.

    Args:
        tiling: A segger ``Tiling`` instance.
        values: One value per tile, in ``tiling.tiles`` order (array or tensor).
        ax: Axes to draw on.
        vmin, vmax: Limits of the log color scale.
        cmap: Colormap name.
        linewidth: Tile outline width.
        edgecolor: Tile outline color.

    Returns:
        The ``PatchCollection`` added to ``ax``.
    """
    if hasattr(values, 'cpu'):
        values = values.cpu().numpy()
    patches = [patch_from_polygon(geom) for geom in tiling.tiles.geometry]
    collection = PatchCollection(
        patches,
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        linewidth=linewidth,
        edgecolor=edgecolor,
    )
    collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection(collection)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')
    return collection
