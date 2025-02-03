from matplotlib.collections import PolyCollection
from matplotlib import pyplot as plt, ticker, axes
from numpy.typing import ArrayLike
import geopandas as gpd
import numpy as np
import shapely


def plot_segmentation_comparison(
    bd_true: gpd.GeoSeries,
    bd_pred: gpd.GeoSeries,
    ax: axes.Axes,
    region: shapely.Polygon,
    bd_true_kwargs: dict = {},
    bd_pred_kwargs: dict = {},
    img: ArrayLike = None,
    show_outlines: bool = False,
):
    # Filled reference cell boundaries
    colors = np.random.random((len(bd_true), 3))
    defaults = dict(facecolors=colors, edgecolors=colors, linewidths=0.25, alpha=0.5, zorder=0)
    defaults.update(bd_true_kwargs)
    plot_segmentation_boundaries(bd_true, ax, **defaults)

    # Predicted edges of cell boundaries
    defaults = dict(facecolors=[0]*4, edgecolors='w', linewidths=0.25, zorder=1)
    defaults.update(bd_pred_kwargs)
    plot_segmentation_boundaries(bd_pred, ax, **defaults)

    # Outlines around lesions
    if show_outlines:
        defaults = dict(facecolors=[0]*4, edgecolors='w', linewidths=0.75, zorder=2)
        plot_segmentation_outlines(bd_true, ax, **defaults)

    # Background IF image
    if img is not None:
        plot_image(img, region, ax, zorder=-1)

    # Explicitly crop to region
    ax.set_xlim(region.bounds[0], region.bounds[2])
    ax.set_ylim(region.bounds[1], region.bounds[3])
    
    format_ax(ax, region)


def plot_image(
    image: np.ndarray,
    region: shapely.Polygon,
    ax: axes.Axes,
    **kwargs,
):
    xmin, ymin, xmax, ymax = region.bounds
    xcoords = np.linspace(xmin, xmax, image.shape[1])
    ycoords = np.linspace(ymin, ymax, image.shape[0])
    ax.pcolormesh(xcoords, ycoords, image, **kwargs)
    

def plot_segmentation_outlines(
    boundaries: gpd.GeoSeries,
    ax: axes.Axes,
    **kwargs,
):
    # Join shapes and smooth
    r = np.sqrt(boundaries.area / np.pi).mean() / 4
    outlines = shapely.unary_union(boundaries.geometry.buffer(r))
    outlines = outlines.buffer(-2*r).buffer(r)

    # Collect inner and outer outlines
    ext_vertices = []
    int_vertices = []
    for contour in list(outlines.geoms):
        v = list(zip(*contour.exterior.coords.xy))
        ext_vertices.append(v)
        for interior in contour.interiors:
            # Heuristic cutoff for large interior outlines
            if interior.crosses(boundaries).sum() > 6:
                v = list(zip(*interior.coords.xy))
                int_vertices.append(v)

    # Exterior outlines
    if 'linewidths' not in kwargs:
        kwargs['linewidths'] = 1.5
    ext_collection = PolyCollection(ext_vertices, **kwargs)
    ax.add_collection(ext_collection)

    # Interior outlines
    kwargs['linewidths'] *= 0.75
    int_collection = PolyCollection(int_vertices, **kwargs)
    ax.add_collection(int_collection)


def plot_segmentation_boundaries(
    boundaries: gpd.GeoSeries,
    ax: axes.Axes,
    **kwargs,
):
    vertices = []
    for poly in boundaries.geometry:
        if poly is None or poly.is_empty:
            continue
        elif isinstance(poly, shapely.MultiPolygon):
            for geom in poly.geoms:
                v = list(zip(*geom.exterior.coords.xy))
                if len(v) > 4: vertices.append(v)
        else:
            v = list(zip(*poly.exterior.coords.xy))
            if len(v) > 4: vertices.append(v)
    collection = PolyCollection(vertices, **kwargs)
    ax.add_collection(collection)


def format_ax(
    ax: axes.Axes,
    region: shapely.Polygon,
):
    xmin, ymin, xmax, ymax = region.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([xmin, xmax])
    ax.set_yticks([ymin, ymax])
    fmt = ticker.FormatStrFormatter('%d µm')
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(which='major', length=2, width=0.5, labelsize=4, pad=2)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.spines[list(ax.spines)].set_visible(False)