"""Readers for the inputs of the paper notebooks.

All tables are per transcript and share ``row_index``, the row position in the
platform's ``transcripts.parquet``. This is the key written by ``segger segment``
to ``segger_segmentation.parquet``, so the assignments of every method can be
joined on it without spatial matching.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Control probes and codewords removed by ``segger segment`` before training
# (``segger/io/fields.py``). Genomic controls of Xenium Prime panels
# (``Intergenic_Region_*``) are kept by segger and therefore here too.
CONTROL_PATTERN = (
    "NegControlProbe|antisense|NegControlCodeword|BLANK|DeprecatedCodeword|UnassignedCodeword"
)

# Column names of the comparison-method assignments in ``segmentations.parquet``.
METHOD_COLUMNS = {
    "Cellpose": "cellpose_cell_id",
    "Proseg": "proseg_cell_id",
    "Baysor": "baysor_cell_id",
    "Bering": "bering_cell_id",
}

# Cell ids that mean "not assigned" in vendor outputs (Xenium v2+ and v1).
_UNASSIGNED = ["UNASSIGNED", "-1", ""]


def read_transcripts(
    path: str | Path,
    columns: list[str] | None = None,
    min_qv: float = 20.0,
    bbox: tuple[float, float, float, float] | None = None,
    controls: bool = False,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Read a Xenium ``transcripts.parquet`` with the filters of ``segger segment``.

    Transcripts with a quality value below ``min_qv`` and control probes are removed.

    Parameters
    ----------
    path
        Path to ``transcripts.parquet``.
    columns
        Columns to keep in addition to ``row_index``, ``x``, ``y`` and ``feature_name``,
        e.g. ``["cell_id", "overlaps_nucleus", "z"]``.
    min_qv
        Minimum Phred-scaled quality value.
    bbox
        Optional ``(xmin, ymin, xmax, ymax)`` window in µm.
    controls
        Keep control probes and codewords.
    lazy
        Return a polars ``LazyFrame``, e.g. to join large sections without
        materialising intermediate tables.

    Returns
    -------
    One row per transcript, coordinates in µm; gene names and vendor cell ids as
    categorical columns.
    """
    rename = {"x_location": "x", "y_location": "y", "z_location": "z"}
    lf = (
        pl.scan_parquet(path)
        .with_row_index("row_index")
        .rename(rename, strict=False)
        .filter(pl.col("qv") >= min_qv)
    )
    if not controls:
        lf = lf.filter(~pl.col("feature_name").cast(pl.Utf8).str.contains(CONTROL_PATTERN))
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        lf = lf.filter(pl.col("x").is_between(x0, x1) & pl.col("y").is_between(y0, y1))
    base = ["row_index", "x", "y", "feature_name"]
    keep = base + [c for c in dict.fromkeys(columns or []) if c not in base]
    categorical = [pl.col(c).cast(pl.Utf8).cast(pl.Categorical) for c in ("feature_name", "cell_id") if c in keep]
    lf = lf.select(keep).with_columns(*categorical, pl.col("row_index").cast(pl.Int64))
    return lf if lazy else lf.collect()


def read_segger(path: str | Path, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """Read ``segger_segmentation.parquet``.

    Outputs written before segger v0.3.0 have no ``filtered`` column. It is added
    here with the rule of the current writer: the transcript has an assigned cell,
    its gene threshold converged and its similarity is at or above the threshold.
    The oldest outputs have no ``converged`` column either; their thresholds are
    taken as converged, as in ``segger export``. Cell ids are returned as a
    categorical column; ``lazy`` returns a ``LazyFrame``.
    """
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    if "filtered" not in schema:
        converged = pl.col("converged") if "converged" in schema else pl.lit(True)
        lf = lf.with_columns(
            (
                pl.col("segger_cell_id").is_not_null()
                & converged
                & (pl.col("segger_similarity") >= pl.col("similarity_threshold"))
            ).alias("filtered")
        )
    lf = lf.with_columns(
        pl.col("row_index").cast(pl.Int64),
        pl.col("segger_cell_id").cast(pl.Utf8).cast(pl.Categorical),
    )
    return lf if lazy else lf.collect()


def read_segmentations(path: str | Path, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """Read the per-transcript cell ids of the comparison methods.

    Columns are renamed to the method names used in the paper (``Cellpose``,
    ``Proseg``, ``Baysor``, ``Bering``) and returned as categorical columns;
    ``lazy`` returns a ``LazyFrame``.
    """
    lf = pl.scan_parquet(path)
    names = lf.collect_schema().names()
    methods = {m: c for m, c in METHOD_COLUMNS.items() if c in names}
    lf = lf.select(
        pl.col("row_index").cast(pl.Int64),
        *[_null_unassigned(pl.col(c).cast(pl.Utf8)).cast(pl.Categorical).alias(m) for m, c in methods.items()],
    )
    return lf if lazy else lf.collect()


def _null_unassigned(col: pl.Expr) -> pl.Expr:
    return pl.when(col.is_in(_UNASSIGNED)).then(None).otherwise(col)


def join_assignments(
    transcripts: pl.DataFrame | pl.LazyFrame,
    segger: pl.DataFrame | pl.LazyFrame | None = None,
    segmentations: pl.DataFrame | pl.LazyFrame | None = None,
    how: str = "inner",
) -> pl.DataFrame:
    """Add one cell-id column per segmentation to a transcript table.

    ``10x Cell`` and ``10x Nucleus`` come from the vendor ``cell_id`` (the latter
    only for transcripts overlapping the nucleus). ``Segger`` holds the kept
    assignments (``filtered``). By default, transcripts that Segger did not score
    are dropped, so all methods are scored on the same molecules; ``how="left"``
    keeps them with a null ``Segger`` column.

    Parameters
    ----------
    transcripts
        Output of :func:`read_transcripts`, optionally with ``cell_id`` and
        ``overlaps_nucleus``.
    segger
        Output of :func:`read_segger`.
    segmentations
        Output of :func:`read_segmentations`.

    Returns
    -------
    The joined table in the row order of ``transcripts``. Lazy inputs are joined
    in one query and collected once.
    """
    tx = transcripts.lazy()
    names = tx.collect_schema().names()
    if "cell_id" in names:
        vendor = _null_unassigned(pl.col("cell_id").cast(pl.Utf8))
        tx = tx.with_columns(vendor.cast(pl.Categorical).alias("10x Cell"))
        if "overlaps_nucleus" in names:
            nucleus = pl.when(pl.col("overlaps_nucleus") == 1).then(pl.col("10x Cell")).otherwise(None)
            tx = tx.with_columns(nucleus.alias("10x Nucleus"))
    if segger is not None:
        kept = pl.when(pl.col("filtered")).then(pl.col("segger_cell_id")).otherwise(None)
        tx = tx.join(segger.lazy().select("row_index", kept.alias("Segger")), on="row_index", how=how, maintain_order="left")
    if segmentations is not None:
        tx = tx.join(segmentations.lazy(), on="row_index", how="left", maintain_order="left")
    return tx.collect()


def read_xenium_boundaries(
    path: str | Path,
    cells: list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
):
    """Vendor outlines from ``cell_boundaries.parquet`` or ``nucleus_boundaries.parquet``.

    Parameters
    ----------
    path
        Xenium boundary table (one row per polygon vertex).
    cells
        Cell ids to keep.
    bbox
        Keep cells with at least one vertex in ``(xmin, ymin, xmax, ymax)`` (µm).

    Returns
    -------
    ``geopandas.GeoSeries`` indexed by cell id: a polygon per cell, or a
    multipolygon for cells with several nuclei (one ring per ``label_id``).
    """
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon

    lf = pl.scan_parquet(path).with_columns(pl.col("cell_id").cast(pl.Utf8))
    ring = ["cell_id", "label_id"] if "label_id" in lf.collect_schema() else ["cell_id"]
    if cells is not None:
        lf = lf.filter(pl.col("cell_id").is_in(list(cells)))
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        inside = lf.filter(pl.col("vertex_x").is_between(x0, x1) & pl.col("vertex_y").is_between(y0, y1))
        lf = lf.join(inside.select("cell_id").unique(), on="cell_id", how="semi")
    df = lf.select(*ring, "vertex_x", "vertex_y").collect()
    rings = df.group_by(ring, maintain_order=True).agg("vertex_x", "vertex_y")
    parts: dict[str, list] = {}
    for cell, xs, ys in zip(rings["cell_id"], rings["vertex_x"], rings["vertex_y"]):
        parts.setdefault(cell, []).append(Polygon(zip(xs, ys)))
    polygons = [p[0] if len(p) == 1 else MultiPolygon(p) for p in parts.values()]
    return gpd.GeoSeries(polygons, index=pd.Index(list(parts), name="cell_id"))


def read_polygons(
    path: str | Path,
    bbox: tuple[float, float, float, float] | None = None,
    id_column: str | None = None,
):
    """Cell polygons from a GeoParquet file, e.g. a Cellpose reference segmentation.

    Parameters
    ----------
    path
        GeoParquet with one polygon per cell.
    bbox
        Keep polygons intersecting ``(xmin, ymin, xmax, ymax)`` (µm).
    id_column
        Column holding the cell id; the index by default.

    Returns
    -------
    ``geopandas.GeoSeries`` indexed by cell id (as strings). Invalid polygons are
    repaired with ``make_valid``, keeping their polygonal parts; cells without any
    are dropped.
    """
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.ops import unary_union

    def polygonal(geom):
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            return geom
        parts = [g for g in getattr(geom, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
        return unary_union(parts) if parts else None

    gdf = gpd.read_parquet(path)
    if bbox is not None:
        gdf = gdf.iloc[np.sort(gdf.sindex.query(box(*bbox), predicate="intersects"))]
    ids = pd.Index(pd.Series(gdf[id_column] if id_column is not None else gdf.index).astype(str).values, name="cell_id")
    geoms = gpd.GeoSeries(gdf.geometry.values, index=ids)
    invalid = ~geoms.is_valid
    geoms[invalid] = [polygonal(g) for g in geoms[invalid].make_valid()]
    return geoms[geoms.notna() & ~geoms.is_empty]


def read_morphology(
    path: str | Path,
    bbox: tuple[float, float, float, float],
    channels: list[int] | None = None,
    pixel_size: float = 0.2125,
) -> np.ndarray:
    """Crop a Xenium morphology image to a window given in µm.

    Parameters
    ----------
    path
        ``morphology_focus/morphology_focus_0000.ome.tif`` (Xenium v2+, one file per
        channel, read together as one series), ``morphology_focus.ome.tif`` (v1), or a
        ``(rows, cols, channels)`` NumPy array saved as ``.npy``.
    bbox
        ``(xmin, ymin, xmax, ymax)`` in µm.
    channels
        Channel indices to read, all by default.
    pixel_size
        µm per pixel at full resolution.

    Returns
    -------
    Array of shape ``(channels, rows, cols)``; row 0 is ``ymin``. Parts of the
    window outside the image are zero.
    """
    import tifffile
    import zarr

    x0, y0, x1, y1 = bbox
    c0, c1 = round(x0 / pixel_size), round(x1 / pixel_size)
    r0, r1 = round(y0 / pixel_size), round(y1 / pixel_size)
    if str(path).endswith(".npy"):
        img = np.load(path, mmap_mode="r")
        crop = np.moveaxis(_crop(img, r0, r1, c0, c1, axes=(0, 1)), -1, 0)
        return crop if channels is None else crop[list(channels)]
    # Xenium v2 stores one channel per file; tifffile warns that it cannot read
    # the per-file pyramids together, but reads the full-resolution series.
    tiff_log = logging.getLogger("tifffile")
    level = tiff_log.level
    tiff_log.setLevel(logging.ERROR)
    try:
        with tifffile.TiffFile(path) as tif:
            img = zarr.open(tif.series[0].aszarr(), mode="r")
            if isinstance(img, zarr.hierarchy.Group):  # pyramid: level 0 is full resolution
                img = img["0"]
            if img.ndim == 2:
                return _crop(img, r0, r1, c0, c1, axes=(0, 1))[None]
            channels = range(img.shape[0]) if channels is None else channels
            return np.stack([_crop(img, r0, r1, c0, c1, axes=(1, 2), channel=c) for c in channels])
    finally:
        tiff_log.setLevel(level)


def _crop(img, r0: int, r1: int, c0: int, c1: int, axes: tuple[int, int], channel: int | None = None) -> np.ndarray:
    """``img[r0:r1, c0:c1]`` on the two spatial ``axes``, zero-padded where the window leaves the image."""
    rows, cols = img.shape[axes[0]], img.shape[axes[1]]
    ra, rb, ca, cb = max(r0, 0), min(r1, rows), max(c0, 0), min(c1, cols)
    index = [slice(None)] * img.ndim
    if channel is not None:
        index[0] = channel
    index[axes[0]], index[axes[1]] = slice(ra, max(ra, rb)), slice(ca, max(ca, cb))
    part = np.asarray(img[tuple(index)])
    if (ra, rb, ca, cb) == (r0, r1, c0, c1):
        return part
    shape = list(part.shape)
    spatial = [a - (channel is not None) for a in axes]
    shape[spatial[0]], shape[spatial[1]] = r1 - r0, c1 - c0
    out = np.zeros(shape, dtype=part.dtype)
    index = [slice(None)] * part.ndim
    index[spatial[0]] = slice(ra - r0, ra - r0 + part.shape[spatial[0]])
    index[spatial[1]] = slice(ca - c0, ca - c0 + part.shape[spatial[1]])
    out[tuple(index)] = part
    return out


def write_spatialdata(
    path: str | Path,
    transcripts: pl.DataFrame,
    tables: dict,
    boundaries: dict | None = None,
    overwrite: bool = False,
):
    """Write transcripts, cell tables and cell outlines as a SpatialData Zarr store.

    Element names follow ``segger export spatialdata``: the points element
    ``transcripts`` holds one cell-id column per segmentation; a segmentation with
    outlines gets a shapes element ``cell_boundaries_<name>`` and a table
    ``table_<name>`` annotating it through ``region`` and ``cell_id``, as read by
    SOPA and napari-spatialdata. Tables without outlines are written unannotated.

    Parameters
    ----------
    path
        Output ``.zarr`` directory.
    transcripts
        One row per transcript with ``x``, ``y``, ``feature_name`` and the cell-id
        columns, e.g. from :func:`join_assignments`.
    tables
        Cell x gene ``AnnData`` of each segmentation, e.g. from
        :func:`sg_utils.tl.metrics.cell_by_gene`.
    boundaries
        Outlines of each segmentation as a ``GeoSeries`` indexed by cell id; cells
        without an outline are dropped from the matching table.
    overwrite
        Replace an existing store at ``path``.

    Returns
    -------
    The written store, opened with ``spatialdata.read_zarr``.
    """
    import warnings

    import geopandas as gpd

    with warnings.catch_warnings():
        # spatialdata < 0.5 runs on the legacy dask DataFrame, which warns on import.
        warnings.filterwarnings("ignore", category=FutureWarning, module="dask")
        import dask.dataframe as dd
        import spatialdata as sd
        from spatialdata.models import PointsModel, ShapesModel, TableModel

    path = Path(path)
    boundaries = boundaries or {}
    name = {m: m.lower().replace(" ", "_") for m in tables}

    # Points go through a temporary Parquet file so that large sections are
    # streamed into the store instead of copied into pandas.
    staging = path.with_name(path.name + ".transcripts.parquet")
    points = transcripts.with_columns(pl.col(pl.Categorical).cast(pl.Utf8))
    points.write_parquet(staging)
    coordinates = {"x": "x", "y": "y"} | ({"z": "z"} if "z" in points.columns else {})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not serialize pd.DataFrame.attrs")
        points = PointsModel.parse(dd.read_parquet(staging), coordinates=coordinates, feature_key="feature_name")

    shapes, annotated = {}, {}
    for method, geoms in boundaries.items():
        key = f"cell_boundaries_{name[method]}"
        shapes[key] = ShapesModel.parse(gpd.GeoDataFrame(geometry=geoms.values, index=pd.Index(geoms.index.astype(str), name="cell_id")))
    for method, adata in tables.items():
        table = adata.copy()
        table.obs.index.name = None
        if method in boundaries:
            region = f"cell_boundaries_{name[method]}"
            table = table[table.obs_names.isin(boundaries[method].index.astype(str))].copy()
            table.obs["region"] = pd.Categorical([region] * table.n_obs, categories=[region])
            table.obs["cell_id"] = table.obs_names.to_numpy()
            table = TableModel.parse(table, region=region, region_key="region", instance_key="cell_id")
        else:
            table = TableModel.parse(table)
        annotated[f"table_{name[method]}"] = table

    sd_log = logging.getLogger("spatialdata._logging")  # its INFO notes on store paths
    level = sd_log.level
    sd_log.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not serialize pd.DataFrame.attrs")
            sd.SpatialData(points={"transcripts": points}, shapes=shapes, tables=annotated).write(path, overwrite=overwrite)
    finally:
        staging.unlink(missing_ok=True)
        sd_log.setLevel(level)
    return sd.read_zarr(path)
