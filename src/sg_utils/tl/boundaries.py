"""Cell outlines from assigned transcripts.

The ``"delaunay"`` outline is the algorithm of ``segger export boundaries``
(``segger/export/boundary.py`` in segger v0.3.0): the Delaunay triangulation of a
cell's transcripts is pruned of long and obtuse boundary edges and the remaining
outline is optionally smoothed by Chaikin corner cutting. The ``"concave"``
outline is the earlier version of the same pruning (:mod:`sg_utils.tl.generate_boundaries`)
used for the cell geometry of the NSCLC section. Both are reproduced here so that outlines
can be drawn for every method on a CPU-only machine. Outlines are used for
display and for the geometric statistics of cells (area, overlap between masks);
assignment metrics are computed on transcript sets.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import polygonize


def _as_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda p: p.area)
    return geom if geom.geom_type == "Polygon" and not geom.is_empty else None


def _triangle_angles(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    p0, p1, p2 = points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]]

    def angle(u, v):
        cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-12)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    return np.stack([angle(p1 - p0, p2 - p0), angle(p0 - p1, p2 - p1), angle(p0 - p2, p1 - p2)], 1)


def _chaikin(coords: np.ndarray, iterations: int) -> np.ndarray:
    for _ in range(iterations):
        nxt = np.roll(coords, -1, axis=0)
        smoothed = np.empty((len(coords) * 2, 2))
        smoothed[0::2] = 0.75 * coords + 0.25 * nxt
        smoothed[1::2] = 0.25 * coords + 0.75 * nxt
        coords = smoothed
    return coords


class _CellOutline:
    """Prune a cell's Delaunay triangulation to one concave outline."""

    def __init__(self, points: np.ndarray):
        self.tri = Delaunay(points)
        self.points = self.tri.points
        dist, _ = cKDTree(self.points).query(self.points, k=2)
        self.d_max = float(dist[:, 1].max())
        self.edges = self._build_edges()
        self.degree = np.bincount(
            np.array(list(self.edges), dtype=np.int64).ravel(), minlength=len(self.points)
        )

    @staticmethod
    def _simplex_edges(simplex):
        return [tuple(sorted((simplex[i], simplex[(i + 1) % 3]))) for i in range(3)]

    def _build_edges(self) -> dict:
        angles = _triangle_angles(self.points, self.tri.simplices)
        edges: dict = {}
        for ti, simplex in enumerate(self.tri.simplices):
            for k, edge in enumerate(self._simplex_edges(simplex)):
                if edge not in edges:
                    a, b = edge
                    edges[edge] = {"tri": {}, "length": float(np.linalg.norm(self.points[a] - self.points[b]))}
                edges[edge]["tri"][ti] = angles[ti][(k + 2) % 3]
        return edges

    def _drop_edge(self, edge) -> bool:
        a, b = edge
        if self.degree[a] <= 1 or self.degree[b] <= 1:
            return False
        del self.edges[edge]
        self.degree[a] -= 1
        self.degree[b] -= 1
        return True

    def _prune(self, predicate) -> None:
        boundary = [e for e in self.edges if len(self.edges[e]["tri"]) < 2]
        changed = True
        while changed:
            changed, nxt = False, []
            for edge in boundary:
                info = self.edges.get(edge)
                if info is None:
                    continue
                if not info["tri"]:
                    if not self._drop_edge(edge):
                        nxt.append(edge)
                    continue
                ti = next(iter(info["tri"]))
                if predicate(info, ti) and self._drop_edge(edge):
                    for other in self._simplex_edges(self.tri.simplices[ti]):
                        if other != edge and other in self.edges:
                            self.edges[other]["tri"].pop(ti, None)
                            nxt.append(other)
                    changed = True
                else:
                    nxt.append(edge)
            boundary = nxt

    def refine(self, connectivity: float = 2.0) -> "_CellOutline":
        d_max = self.d_max
        self._prune(lambda info, ti: info["length"] > 2 * connectivity * d_max)
        max_angle = 180 - (180 / 16) / connectivity
        self._prune(
            lambda info, ti: (info["length"] > 1.5 * connectivity * d_max and info["tri"][ti] > 90)
            or info["tri"][ti] > max_angle
        )
        return self

    def polygon(self) -> Polygon | None:
        lines = [
            LineString([self.points[a], self.points[b]])
            for a, b in self.edges
            if len(self.edges[(a, b)]["tri"]) < 2
        ]
        polys = list(polygonize(lines))
        return _as_polygon(max(polys, key=lambda p: p.area)) if polys else None


def cell_boundary(
    points: np.ndarray,
    kind: str = "delaunay",
    smoothing: int = 0,
    connectivity: float = 2.0,
) -> Polygon | None:
    """Outline of one cell from its transcript coordinates.

    Parameters
    ----------
    points
        ``(n, 2)`` transcript coordinates.
    kind
        ``"delaunay"`` (as ``segger export``), ``"concave"`` (the NSCLC cell geometry) or ``"convex_hull"``.
    smoothing
        Number of Chaikin corner-cutting iterations.
    connectivity
        Values above 1 keep more boundary edges (more convex outlines); ``"delaunay"`` only.

    Returns
    -------
    A polygon, or None for fewer than three distinct points.
    """
    if np.unique(points, axis=0).shape[0] < 3:
        return None
    if kind == "convex_hull":
        poly = _as_polygon(MultiPoint(points).convex_hull)
    elif kind == "delaunay":
        try:
            poly = _CellOutline(points).refine(connectivity).polygon()
        except Exception:
            poly = None
    elif kind == "concave":
        poly = _concave_outline(points)
    else:
        raise ValueError(f"Unknown outline kind: {kind!r}")
    if poly is not None and smoothing > 0:
        poly = _as_polygon(Polygon(_chaikin(np.asarray(poly.exterior.coords)[:-1], smoothing)).buffer(0))
    return poly


def _concave_outline(points: np.ndarray) -> Polygon | None:
    from scipy.spatial import QhullError

    try:
        outline = _ConcaveOutline(np.asarray(points, dtype=float))
        outline.calculate_part_1(plot=False)
        outline.calculate_part_2(plot=False)
        return _as_polygon(outline.find_cycles())
    except (QhullError, ValueError, KeyError, IndexError, RecursionError):
        return None


def _concave_outline_class():
    from .generate_boundaries import BoundaryIdentification

    class ConcaveOutline(BoundaryIdentification):
        """:class:`~sg_utils.tl.generate_boundaries.BoundaryIdentification` with vectorised geometry."""

        @staticmethod
        def calculate_d_max(points):
            dist, _ = cKDTree(points).query(points, k=2)
            return dist[:, 1].max()

        def generate_edges(self):
            d = self.d
            p0, p1, p2 = (d.points[d.simplices[:, k]] for k in range(3))

            def angle(u, v):
                with np.errstate(invalid="ignore", divide="ignore"):
                    cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
                return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

            angles = np.stack([angle(p1 - p0, p2 - p0), angle(p0 - p1, p2 - p1), angle(p0 - p2, p1 - p2)], 1)
            edges = {}
            for index, simplex in enumerate(d.simplices):
                for p in range(3):
                    edge = tuple(sorted((simplex[p], simplex[(p + 1) % 3])))
                    if edge not in edges:
                        edges[edge] = {"simplices": {}}
                    edges[edge]["simplices"][index] = angles[index][(p + 2) % 3]
            coords = d.points[np.array(list(edges))]
            lengths = ((coords[:, 1, 0] - coords[:, 0, 0]) ** 2 + (coords[:, 1, 1] - coords[:, 0, 1]) ** 2) ** 0.5
            for edge, xy, length in zip(edges, coords, lengths):
                edges[edge]["coords"] = xy
                edges[edge]["length"] = length
            self.edges = edges

    return ConcaveOutline


class _LazyConcave:
    cls = None

    def __call__(self, points):
        if _LazyConcave.cls is None:
            _LazyConcave.cls = _concave_outline_class()
        return _LazyConcave.cls(points)


_ConcaveOutline = _LazyConcave()


def _outline_chunk(chunk: list[tuple[str, np.ndarray]], kwargs: dict) -> list[tuple[str, Polygon | None]]:
    return [(cid, cell_boundary(points, **kwargs)) for cid, points in chunk]


def cell_boundaries(
    transcripts: pl.DataFrame,
    segmentation: str,
    cells: list[str] | None = None,
    smoothing: int = 2,
    n_jobs: int = 1,
    **kwargs,
) -> gpd.GeoSeries:
    """Outlines of the cells of one segmentation.

    Parameters
    ----------
    transcripts
        Transcripts with ``x``, ``y`` and the cell-id column ``segmentation``.
    segmentation
        Column with the cell ids.
    cells
        Cells to outline, all assigned cells by default.
    smoothing
        Chaikin iterations passed to :func:`cell_boundary`; ``segger export`` uses 0.
    n_jobs
        Worker processes.
    **kwargs
        ``kind`` and ``connectivity`` of :func:`cell_boundary`.

    Returns
    -------
    Polygons indexed by cell id (sorted); degenerate cells are dropped.
    """
    tx = transcripts.lazy().filter(pl.col(segmentation).is_not_null())
    if cells is not None:
        tx = tx.filter(pl.col(segmentation).is_in(list(cells)))
    grouped = (
        tx.group_by(pl.col(segmentation).cast(pl.Utf8), maintain_order=True)
        .agg(pl.col("x"), pl.col("y"))
        .sort(segmentation)
        .collect()
    )
    items = [(cid, np.column_stack([xs, ys])) for cid, xs, ys in grouped.iter_rows()]
    kwargs = {"smoothing": smoothing, **kwargs}
    if n_jobs > 1 and len(items) > 1000:
        size = -(-len(items) // (4 * n_jobs))
        chunks = [items[i : i + size] for i in range(0, len(items), size)]
        with ProcessPoolExecutor(n_jobs) as pool:
            results = [r for part in pool.map(_outline_chunk, chunks, [kwargs] * len(chunks)) for r in part]
    else:
        results = _outline_chunk(items, kwargs)
    results = [(cid, poly) for cid, poly in results if poly is not None]
    return gpd.GeoSeries([p for _, p in results], index=pd.Index([c for c, _ in results], name=segmentation))


def round_outline(outline, erode: float = 0.5, smooth: float = 0.7, iterations: int = 2) -> Polygon | None:
    """Rounded outline for display, as drawn in the cell-geometry panels of the paper.

    A morphological closing (radius ``3.4 * smooth`` µm) fills notches, an opening
    (``2.4 * smooth`` µm) trims spikes, an erosion by ``erode`` µm separates
    touching cells, and Chaikin corner cutting smooths the result. Steps that
    would remove the cell are skipped.
    """
    poly = _as_polygon(outline)
    if poly is None:
        return None
    close, open_ = 3.4 * smooth, 2.4 * smooth
    closed = _as_polygon(poly.simplify(0.8).buffer(close, join_style=1).buffer(-close, join_style=1))
    opened = _as_polygon(closed.buffer(-open_, join_style=1).buffer(open_, join_style=1)) if closed is not None else None
    base = next((g for g in (opened, closed, poly) if g is not None), poly)
    if erode:
        eroded = _as_polygon(base.buffer(-erode, join_style=1))
        if eroded is not None and eroded.area > 1.0:
            base = eroded
    ring = np.asarray(base.simplify(0.7).exterior.coords)[:-1]
    rounded = Polygon(_chaikin(ring, iterations)) if len(ring) >= 3 else base
    return rounded if rounded.is_valid and not rounded.is_empty else base
