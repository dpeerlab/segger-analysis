"""Cell geometry and neighbourhood graphs.

Two graphs are built identically for every segmentation and for the reference:
a nearest-neighbour graph of cell centroids and a mask-contact graph joining
cells whose transcript territories touch after a small dilation. Agreement with
the reference is measured on transcript sets.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree
from shapely import make_valid
from shapely.errors import GEOSException
from shapely.ops import unary_union
from shapely.strtree import STRtree
from skimage.segmentation import expand_labels


def knn_graph(xy: np.ndarray, n_neighbors: int = 15, radius: float = 20.0) -> np.ndarray:
    """Undirected centroid graph joining each cell to its ``n_neighbors`` nearest cells within ``radius`` µm.

    The union of the directed neighbour lists, so a cell can have more than
    ``n_neighbors`` neighbours.

    Returns
    -------
    ``(n_edges, 2)`` array of cell indices with ``i < j``.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    k = min(n_neighbors + 1, len(xy))
    dist, idx = cKDTree(xy).query(xy, k=k)
    src = np.repeat(np.arange(len(xy)), k - 1)
    dst = idx[:, 1:].ravel()
    keep = (dist[:, 1:].ravel() <= radius) & (src != dst)
    edges = np.sort(np.stack([src[keep], dst[keep]], axis=1), axis=1)
    return np.unique(edges, axis=0)


def contact_graph(
    transcripts: pl.DataFrame,
    method: str,
    cells: list[str] | None = None,
    dilation: float = 1.0,
    pixel_size: float = 1.0,
    tile_size: float = 1500.0,
    overlap: float = 30.0,
) -> pl.DataFrame:
    """Mask-contact graph: cells whose transcript territories touch after ``dilation`` µm.

    Each cell's transcripts are painted on a ``pixel_size`` grid anchored at the
    lowest transcript coordinate, and the labels are grown by ``dilation`` (rounded
    to whole pixels) into the background without crossing each other
    (``skimage.segmentation.expand_labels``). Cells sharing a pixel edge are in
    contact. The section is processed in tiles of ``tile_size`` µm that overlap by
    ``overlap`` µm.

    Parameters
    ----------
    transcripts
        Transcripts with ``x``, ``y`` and the cell-id column ``method``.
    cells
        Cells to include, all assigned cells by default.

    Returns
    -------
    One row per contact with columns ``cell_a`` and ``cell_b``.
    """
    tx = transcripts.lazy().filter(pl.col(method).is_not_null())
    if cells is not None:
        tx = tx.filter(pl.col(method).is_in(list(cells)))
    tx = tx.select(method, "x", "y").collect()
    if tx.height == 0:
        return pl.DataFrame({"cell_a": [], "cell_b": []}, schema={"cell_a": pl.Utf8, "cell_b": pl.Utf8})
    ids = tx.get_column(method)
    if ids.dtype != pl.Categorical:
        ids = ids.cast(pl.Utf8).cast(pl.Categorical)
    _, first, codes = np.unique(ids.to_physical().to_numpy(), return_index=True, return_inverse=True)
    names = ids.gather(first).cast(pl.Utf8).to_numpy()
    labels = (codes + 1).astype(np.uint32)
    x, y = tx.get_column("x").to_numpy(), tx.get_column("y").to_numpy()
    grow = int(round(dilation / pixel_size))

    pairs = []
    n_x = max(1, math.ceil((x.max() - x.min()) / tile_size))
    n_y = max(1, math.ceil((y.max() - y.min()) / tile_size))
    for x0 in x.min() + tile_size * np.arange(n_x):
        for y0 in y.min() + tile_size * np.arange(n_y):
            xa, xb, ya, yb = x0 - overlap, x0 + tile_size + overlap, y0 - overlap, y0 + tile_size + overlap
            inside = (x >= xa) & (x < xb) & (y >= ya) & (y < yb)
            if not inside.any():
                continue
            width = int(math.ceil((xb - xa) / pixel_size)) + 1
            height = int(math.ceil((yb - ya) / pixel_size)) + 1
            cols = np.clip(((x[inside] - xa) / pixel_size).astype(np.int64), 0, width - 1)
            rows = np.clip(((y[inside] - ya) / pixel_size).astype(np.int64), 0, height - 1)
            image = np.zeros((height, width), dtype=np.uint32)
            image[rows, cols] = labels[inside]
            image = expand_labels(image, distance=grow)
            for a, b in ((image[:, :-1], image[:, 1:]), (image[:-1, :], image[1:, :])):
                touch = (a != b) & (a > 0) & (b > 0)
                pairs.append(np.sort(np.stack([a[touch], b[touch]], axis=1), axis=1))
    pairs = np.unique(np.vstack(pairs), axis=0) if pairs else np.zeros((0, 2), np.uint32)
    return pl.DataFrame({"cell_a": names[pairs[:, 0] - 1], "cell_b": names[pairs[:, 1] - 1]})


def degree(edges: np.ndarray, n: int) -> np.ndarray:
    """Number of neighbours of each of ``n`` nodes in an undirected edge list."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return np.bincount(edges.ravel(), minlength=n)


def edge_index(edges: pl.DataFrame, cells: pd.Index) -> np.ndarray:
    """Convert an edge table of cell ids to indices into ``cells``; edges to other cells are dropped."""
    lookup = pd.Series(np.arange(len(cells)), index=cells)
    a = lookup.reindex(edges.get_column("cell_a").to_numpy()).to_numpy()
    b = lookup.reindex(edges.get_column("cell_b").to_numpy()).to_numpy()
    keep = ~(np.isnan(a) | np.isnan(b))
    return np.stack([a[keep], b[keep]], axis=1).astype(np.int64)


def match_reference(
    transcripts: pl.DataFrame,
    method: str,
    reference: str,
    min_transcripts: int = 10,
) -> pd.DataFrame:
    """Match every cell of a segmentation to the reference cell sharing most of its transcripts.

    Only transcripts assigned in both segmentations are counted, so the
    intersection over union (IoU) and Dice coefficient measure how the two
    segmentations partition the same molecules.

    Returns
    -------
    One row per cell with at least ``min_transcripts`` such transcripts: the matched
    reference cell, the shared, own and reference transcript counts, IoU and Dice.
    """
    tx = transcripts.lazy().filter(pl.col(method).is_not_null() & pl.col(reference).is_not_null()).select(method, reference)
    as_str = [pl.col(method).cast(pl.Utf8), pl.col(reference).cast(pl.Utf8)]
    shared = tx.group_by(method, reference).len("shared").with_columns(as_str).collect()
    sizes = tx.group_by(method).len("n").with_columns(as_str[0]).collect()
    ref_sizes = tx.group_by(reference).len("n_reference").with_columns(as_str[1]).collect()
    # Ties in the shared count go to the reference cell with the smallest id.
    best = (
        shared.sort(["shared", reference], descending=[True, False], maintain_order=True)
        .group_by(method, maintain_order=True)
        .first()
        .join(sizes, on=method)
        .join(ref_sizes, on=reference)
        .filter(pl.col("n") >= min_transcripts)
        .with_columns(
            (pl.col("shared") / (pl.col("n") + pl.col("n_reference") - pl.col("shared"))).alias("iou"),
            (2 * pl.col("shared") / (pl.col("n") + pl.col("n_reference"))).alias("dice"),
        )
    )
    return best.rename({method: "cell", reference: "reference_cell"}).to_pandas().set_index("cell")


def overlap_fraction(polygons) -> np.ndarray:
    """Fraction of each polygon's area covered by the other polygons of the same set."""
    polygons = [make_valid(p) for p in polygons]
    tree = STRtree(polygons)
    out = np.zeros(len(polygons))
    for i, poly in enumerate(polygons):
        others = [polygons[j] for j in tree.query(poly, predicate="intersects") if j != i]
        if others and poly.area > 0:
            out[i] = min(_shared_area(poly, unary_union(others)) / poly.area, 1.0)
    return out


def _shared_area(a, b) -> float:
    try:
        return a.intersection(b).area
    except GEOSException:  # robustness failure on nearly coincident edges
        return a.buffer(0).intersection(b.buffer(0)).area


def masks_per_cell(reference, polygons, min_area: float = 0.05) -> np.ndarray:
    """Number of polygons overlapping each reference polygon by more than ``min_area`` µm²."""
    polygons = [make_valid(p) for p in polygons]
    tree = STRtree(polygons)
    out = np.zeros(len(reference))
    for i, ref in enumerate(reference):
        ref = make_valid(ref)
        hits = tree.query(ref, predicate="intersects")
        out[i] = sum(_shared_area(ref, polygons[j]) > min_area for j in hits)
    return out
