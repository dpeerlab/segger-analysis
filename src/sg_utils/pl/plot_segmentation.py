from matplotlib.collections import PolyCollection
from matplotlib import pyplot as plt, ticker, axis
from numpy.typing import ArrayLike
from typing import List, Tuple
import geopandas as gpd
import seaborn as sns
import pandas as pd
import numpy as np
import shapely
import random


def plot_segmentation(
    transcripts: pd.DataFrame,
    x: str,
    y: str,
    seg_col: str,
    pos_col: str,
    ax: axis.Axis,
    image: ArrayLike = None,
    boundaries: gpd.GeoDataFrame = None,
    cell_id_colors: List = None,
    xlim: Tuple[float] = None,
    ylim: Tuple[float] = None,
    transcript_styles: dict = {},
    negative_styles: dict = {},
    boundary_styles: dict = {},
    outlines: bool = False,
):
    if xlim is None:
        xlim = transcripts[x].min(), transcripts[x].max()
    if ylim is None:
        ylim = transcripts[y].min(), transcripts[y].max()
    
    palette = sns.color_palette(
        cell_id_colors,
        transcripts[seg_col].nunique()
    )
    random.shuffle(palette)

    if image is not None:
        ax.pcolormesh(
            np.linspace(*xlim, image.shape[1]),
            np.linspace(*ylim, image.shape[0]),
            image,
            rasterized=True,
        )
    else:
        ax.set_facecolor([0]*4)

    # Cell segmentation
    defaults = dict(s=0.15, alpha=0.75, lw=0, legend=False, rasterized=True)
    defaults.update(transcript_styles)
    sns.scatterplot(
        data=transcripts,
        x=x, y=y,
        hue=seg_col,
        palette=palette,
        zorder=1,
        **defaults,
    )

    # False positives
    defaults = dict(s=0.15, lw=0., legend=False, rasterized=True, c='tab:red')
    defaults.update(negative_styles)
    sns.scatterplot(
        data=transcripts[~transcripts[pos_col]],
        x=x, y=y,
        zorder=2,
        **defaults
    )

    # Boundaries
    if boundaries is not None:
        outline_styles = boundary_styles.copy()
        if 'linewidths' in outline_styles:
            outline_styles['linewidths'] *= 2
        plot_segmentation_boundaries(
            boundaries,
            ax,
            cell_styles=boundary_styles,
            outline_styles=outline_styles,
            show_outlines=outlines,
        )

    # Formatting
    format_ax(ax, xlim, ylim)


def plot_segmentation_boundaries(
    boundaries: gpd.GeoDataFrame,
    ax: axis.Axis,
    cell_styles: dict = {},
    outline_styles: dict = {},
    show_outlines: bool = False,
):
    # Plot cell outlines
    
    vertices = boundaries.geometry.apply(
        lambda x: list(zip(*x.exterior.coords.xy))
    )
    defaults = dict(facecolors=[0]*4, edgecolors='w', linewidths=0.25)
    defaults.update(cell_styles)
    collection = PolyCollection(vertices, **defaults)
    ax.add_collection(collection)

    # All cell borders
    if show_outlines:
        outlines = shapely.unary_union(boundaries.geometry.buffer(2))
        outlines = outlines.buffer(-4).buffer(2)
        ext_vertices = []
        int_vertices = []
        for contour in list(outlines.geoms):
            v = list(zip(*contour.exterior.coords.xy))
            ext_vertices.append(v)
            for interior in contour.interiors:
                # Heuristic cutoff for large interior outlines
                if interior.crosses(boundaries).sum() > 3:
                    v = list(zip(*interior.coords.xy))
                    int_vertices.append(v)

        # Exterior outlines
        defaults = dict(linewidths=0.75, facecolors=[0]*4, edgecolors='w')
        defaults.update(outline_styles)
        ext_collection = PolyCollection(ext_vertices, **defaults)
        ax.add_collection(ext_collection)

        # Interior outlines
        defaults['linewidths'] *= 0.75
        int_collection = PolyCollection(int_vertices, **defaults)
        ax.add_collection(int_collection)


def format_ax(
    ax: axis.Axis,
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
