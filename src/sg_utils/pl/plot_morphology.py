"""Segmentation footprints over a reference segmentation (Supp. Fig. 11).

Reference (Cellpose) cells are drawn filled; each method's cells are outlined on
top and colored by the reference cell they map to.
"""

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from shapely import concave_hull as _concave_hull
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.plotting import patch_from_polygon


def crop_and_colorize(image, xmin, ymin, xmax, ymax, mpp=0.2125 * 2):
    """Crop the morphology image to a ROI and false-color it.

    Args:
        image: (H, W, 2) uint8 image; channel 0 nuclei, channel 1 membrane.
        xmin, ymin, xmax, ymax: ROI bounds in microns.
        mpp: Microns per pixel.

    Returns:
        An (h, w, 3) uint8 image with membrane in cyan and nuclei in red.
    """
    crop = image[
        int(ymin / mpp): int(ymax / mpp),
        int(xmin / mpp): int(xmax / mpp),
    ]
    mem = np.tile(crop[:, :, 1, np.newaxis], (1, 1, 3)) * np.array([0, 1, 1])
    nuc = np.tile(crop[:, :, 0, np.newaxis], (1, 1, 3)) * np.array([1, 0, 0])
    return (mem + nuc).astype('uint8')


def plot_morphology_image(
    image, boundaries, xmin, ymin, xmax, ymax, ax, mpp=0.2125 * 2,
):
    """Overlay Cellpose boundaries on the false-colored morphology image.

    Args:
        image: (H, W, 2) membrane/DAPI image.
        boundaries: GeoDataFrame of Cellpose polygons in micron coordinates.
        xmin, ymin, xmax, ymax: ROI bounds in microns.
        ax: Axes to draw on.
        mpp: Microns per pixel of ``image``.
    """
    roi = crop_and_colorize(image, xmin, ymin, xmax, ymax, mpp=mpp)
    ax.imshow(roi, zorder=0, extent=(xmin, xmax, ymin, ymax), origin='lower')
    for geom in boundaries.cx[xmin:xmax, ymin:ymax].geometry:
        ax.add_patch(patch_from_polygon(
            geom, fill=False, lw=0.25, edgecolor='w', zorder=1))


def plot_cell_outlines(
    bins_ref: pl.DataFrame,
    bins_col: pl.DataFrame,
    reference: str,
    column: str,
    xmin: float, xmax: float, ymin: float, ymax: float,
    bin_width: float = 1,
    mapping: pl.DataFrame | None = None,
    ax: plt.Axes | None = None,
    linewidth: float = 1.0,
    alpha: float = 0.5,
    concave_hull: bool = False,
    ref_colors: dict | None = None,
):
    """Draw reference cells filled, with method-cell outlines colored by match.

    Args:
        bins_ref: Binned footprints of the reference segmentation.
        bins_col: Binned footprints of the method (``column``) segmentation.
        reference: Reference cell-id column.
        column: Method cell-id column.
        xmin, xmax, ymin, ymax: ROI bounds in microns.
        bin_width: Grid bin size used to build the footprints.
        mapping: ``(column, reference)`` matches; method cells inherit their
            reference cell's color.
        ax: Axes to draw on (created if ``None``).
        linewidth: Outline width for method cells.
        alpha: Fill darkening applied to reference cells.
        concave_hull: If True, smooth footprints with a concave hull.
        ref_colors: ``{reference id: color}``, e.g. from
            :func:`reference_colors`, to keep colors consistent across
            panels; by default colors cycle in drawing order.

    Returns:
        The Axes drawn on.
    """
    bxmin = int(np.floor(xmin / bin_width))
    bxmax = int(np.ceil(xmax / bin_width))
    bymin = int(np.floor(ymin / bin_width))
    bymax = int(np.ceil(ymax / bin_width))

    def filter_region(df):
        return df.filter(
            pl.col('x_bin').is_between(bxmin, bxmax),
            pl.col('y_bin').is_between(bymin, bymax),
        )

    def blend_with_black(color, a):
        r, g, b = to_rgb(color)
        return (r * a, g * a, b * a)

    def bins_to_shape(xs, ys):
        shape = unary_union([
            box(x * bin_width, y * bin_width,
                (x + 1) * bin_width, (y + 1) * bin_width)
            for x, y in zip(xs, ys)
        ])
        if concave_hull:
            shape = _concave_hull(shape, ratio=0.2).buffer(-bin_width / 2)
        return shape

    ref_region = filter_region(bins_ref)
    col_region = filter_region(bins_col)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.set_facecolor('black')

    cycler = plt.rcParams['axes.prop_cycle']()
    ref_colors = dict(ref_colors or {})
    for (cell_id,), group in ref_region.group_by([reference], maintain_order=True):
        color = ref_colors.setdefault(cell_id, next(cycler)['color'])
        shape = bins_to_shape(group['x_bin'].to_numpy(), group['y_bin'].to_numpy())
        for geom in getattr(shape, 'geoms', [shape]):
            ax.fill(*geom.exterior.xy, color=blend_with_black(color, alpha))

    col_colors = {}
    if mapping is not None:
        for row in mapping.iter_rows(named=True):
            if row[reference] in ref_colors:
                col_colors[row[column]] = ref_colors[row[reference]]

    for (cell_id,), group in col_region.group_by([column]):
        color = col_colors.get(cell_id, 'white')
        shape = bins_to_shape(group['x_bin'].to_numpy(), group['y_bin'].to_numpy())
        for geom in getattr(shape, 'geoms', [shape]):
            ax.plot(*geom.exterior.xy, color=color, linewidth=linewidth)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    return ax
