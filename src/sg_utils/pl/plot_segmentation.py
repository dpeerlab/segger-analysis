from matplotlib.collections import PolyCollection
from matplotlib import pyplot as plt, ticker, axis
from numpy.typing import ArrayLike
from typing import List, Tuple, Dict, Union
import geopandas as gpd
import seaborn as sns
import pandas as pd
import numpy as np
import shapely
import random
from matplotlib.axes import Axes
from tifffile import imread
from sg_utils.tl.generate_boundaries import generate_boundaries


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





def plot_segmentation_advanced(
    transcripts: pd.DataFrame,
    x: str,
    y: str,
    ax: Axes,
    seg_col: str = None,
    marker_col: str = None,
    cell_type_filter: str = None,
    ome_tiff_path: str = None,
    negative_markers: List[str] = None,
    boundaries: bool = True,
    marker_colors: Dict[str, str] = None,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    transcript_styles: Dict = None,
    type_color: str = 'n',
    s: int = 50,
    cmap: str = 'inferno',
    boundary_styles: Dict = {},
) -> Tuple[int, int, Union[gpd.GeoDataFrame, None]]:
    """
    Plots segmentation results, optionally overlaying an OME-TIFF image and plotting cell boundaries.

    Parameters:
    ----------
    transcripts : pd.DataFrame
        DataFrame containing transcript information.
    x, y : str
        Column names for x and y coordinates.
    ax : Axes
        Matplotlib axis to plot on.
    seg_col : str, optional
        Column indicating segmentation labels.
    marker_col : str, optional
        Column indicating marker types.
    cell_type_filter : str, optional
        If specified, filters transcripts by this cell type.
    ome_tiff_path : str, optional
        Path to an OME-TIFF file to overlay as background.
    negative_markers : List[str], optional
        List of negative marker names for coloring.
    boundaries : bool, default=True
        Whether to generate and plot cell boundaries.
    marker_colors : Dict[str, str], optional
        Custom color mapping for marker types.
    xlim, ylim : Tuple[float, float], optional
        Limits for the x and y axes.
    transcript_styles : Dict, optional
        Styling parameters for transcripts.
    type_color : str, default='n'
        Determines color scheme:
        - 'm': Colors transcripts by marker type.
        - 'n': Colors negative markers distinctly.
        - 'c': Colors by segmentation label.
    s : int, default=50
        Marker size for scatter plots.
    cmap : str, default='inferno'
        Colormap for the OME-TIFF background.
    boundary_styles : Dict, optional
        Styling parameters for cell boundaries.

    Returns:
    -------
    Tuple[int, int, Union[gpd.GeoDataFrame, None]]
        Number of unique cells, number of transcripts, and the boundary GeoDataFrame (if `boundaries=True`).
    """

    if transcript_styles is None:
        transcript_styles = {}

    # Filter transcripts by cell type
    if type_color != 'c' and cell_type_filter:
        transcripts = transcripts[transcripts['cell_type'] == cell_type_filter]

    # Define axis limits
    xlim = xlim or (transcripts[x].min(), transcripts[x].max())
    ylim = ylim or (transcripts[y].min(), transcripts[y].max())

    # Filter transcripts within axis limits
    transcripts = transcripts[
        (transcripts[x].between(xlim[0], xlim[1])) &
        (transcripts[y].between(ylim[0], ylim[1]))
    ]

    # Load OME-TIFF image if provided
    if ome_tiff_path:
        image = np.array(imread(ome_tiff_path))
        mpp = 0.2125  # Microns per pixel
        image = image[
            int(ylim[0] / mpp): int(ylim[1] / mpp),
            int(xlim[0] / mpp): int(xlim[1] / mpp),
        ]
        ax.pcolormesh(
            np.linspace(*xlim, image.shape[1]),
            np.linspace(*ylim, image.shape[0]),
            image,
            rasterized=True,
            cmap=cmap,
            vmin=300,
            vmax=1200
        )
    else:
        ax.set_facecolor([0] * 4)

    # Default scatter plot styles
    scatter_defaults = dict(s=s, alpha=0.5, lw=0, legend=False, rasterized=True)
    scatter_defaults.update(transcript_styles)

    # Plot transcripts based on type_color mode
    if type_color == 'm' and marker_col:
        sns.scatterplot(
            data=transcripts,
            x=x, y=y,
            hue=marker_col,
            palette=marker_colors,
            zorder=2,
            size=5,
            ax=ax,
            **scatter_defaults
        )

    elif type_color == 'n' and negative_markers:
        transcripts['negative'] = transcripts['feature_name'].isin(negative_markers)
        transcripts['size'] = transcripts['negative'].astype(int) * 5 + (~transcripts['negative']).astype(int) * 2
        sns.scatterplot(
            data=transcripts,
            x=x, y=y,
            hue='negative',
            palette={True: '#C72228', False: '#8bb7f4'},
            zorder=2,
            size='size',
            ax=ax,
            **scatter_defaults
        )

    elif type_color == 'c' and seg_col:
        palette = sns.color_palette("hls", 5)
        random.shuffle(palette)
        sns.scatterplot(
            data=transcripts,
            x=x, y=y,
            hue=seg_col,
            palette=palette,
            zorder=2,
            ax=ax,
            **scatter_defaults
        )

    # Generate and plot boundaries if requested
    bd = None
    if boundaries and seg_col:
        bs = generate_boundaries(transcripts, cell_id=seg_col).dropna().set_index('cell_id')
        bs = bs[~bs.is_empty]
        bs.geometry = bs.geometry.apply(
            lambda x: x if not isinstance(x, shapely.MultiPolygon) else list(x.geoms)[0]
        )

        # Process boundaries
        bd = bs.copy().buffer(5).buffer(-5).buffer(0.5)
        bd = bd.apply(lambda x: x if not isinstance(x, shapely.MultiPolygon) else list(x.geoms)[0])
        bd = bd.dropna()
        bd = bd[bd.apply(lambda x: len(x.exterior.coords)) > 3]

        plot_segmentation_boundaries(bd, ax, cell_styles=boundary_styles)

    # Formatting
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.yaxis.set_inverted(True)
    ax.set_title(f"Segmentation Results for {cell_type_filter}")

    # Remove legend
    ax.legend().remove()

    # Compute cell and transcript counts
    cells = transcripts[seg_col].nunique() if seg_col else 0
    tx = len(transcripts)

    return (cells, tx, bd) if boundaries else (cells, tx)
