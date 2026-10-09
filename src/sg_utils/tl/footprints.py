"""Compare segmentations by rasterized transcript footprints and neighbourhoods.

Each segmentation labels transcripts with a cell id. Binning the transcripts of
a cell onto a grid gives it a footprint that can be compared with a reference
segmentation (e.g. Cellpose); cells whose transcripts split into disconnected
components are bridged with a minimum spanning tree. Neighbourhoods are compared
by translating each method's neighbour edges into the reference label space.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
from scipy.ndimage import label as ndlabel
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

_GRID_AREA_THRESHOLD = 500_000


def get_bins_with_mst(
    tx: pl.DataFrame,
    column: str,
    bin_width: float = 1,
    x_column: str = 'x_location',
    y_column: str = 'y_location',
    n_workers: int = 8,
    chunk_size: int = 500,
) -> pl.DataFrame:
    """Rasterize each cell's transcripts to grid bins, bridging split cells.

    Args:
        tx: Transcript table with cell-id and coordinate columns.
        column: Cell-id column to rasterize.
        bin_width: Grid bin size in the coordinate units.
        x_column: Transcript x-coordinate column.
        y_column: Transcript y-coordinate column.
        n_workers: Worker processes for the bridging step.
        chunk_size: Cells per worker task.

    Returns:
        Long ``(column, x_bin, y_bin)`` frame of occupied bins per cell.
    """
    bins = (
        tx
        .select(
            pl.col(column),
            ((pl.col(x_column) / bin_width).floor().cast(pl.Int64)).alias('x_bin'),
            ((pl.col(y_column) / bin_width).floor().cast(pl.Int64)).alias('y_bin'),
        )
        .drop_nulls(column)
        .unique(subset=[column, 'x_bin', 'y_bin'])
    )

    grouped = (
        bins
        .group_by(column)
        .agg(pl.col('x_bin'), pl.col('y_bin'))
        .filter(pl.col('x_bin').list.len() > 1)
    )
    if grouped.height == 0:
        return bins

    flat = grouped.explode('x_bin', 'y_bin')
    cell_arr = flat[column].to_numpy()
    x_arr = flat['x_bin'].to_numpy()
    y_arr = flat['y_bin'].to_numpy()

    if cell_arr.dtype.kind in ('U', 'O', 'S'):
        changes = np.flatnonzero(cell_arr[:-1] != cell_arr[1:]) + 1
    else:
        changes = np.flatnonzero(np.diff(cell_arr)) + 1
    starts = np.empty(len(changes) + 1, dtype=np.intp)
    starts[0] = 0
    starts[1:] = changes
    ends = np.empty_like(starts)
    ends[:-1] = changes
    ends[-1] = len(cell_arr)

    cell_groups = [
        (cell_arr[starts[i]], x_arr[starts[i]:ends[i]], y_arr[starts[i]:ends[i]])
        for i in range(len(starts))
    ]
    chunks = [
        cell_groups[i:i + chunk_size]
        for i in range(0, len(cell_groups), chunk_size)
    ]

    bridge_col, bridge_x, bridge_y = [], [], []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_process_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            for cell_id, new_bins in future.result():
                if new_bins is not None and len(new_bins) > 0:
                    bridge_col.extend([cell_id] * len(new_bins))
                    bridge_x.extend(new_bins[:, 0].tolist())
                    bridge_y.extend(new_bins[:, 1].tolist())

    if bridge_col:
        bridges = pl.DataFrame({
            column: bridge_col,
            'x_bin': bridge_x,
            'y_bin': bridge_y,
        }).cast({column: bins[column].dtype, 'x_bin': pl.Int64, 'y_bin': pl.Int64})
        return pl.concat([bins, bridges])
    return bins


def map_labels(
    tx: pl.DataFrame,
    column: str,
    reference: str,
    threshold: float,
) -> pl.DataFrame:
    """Match each ``column`` cell to its dominant ``reference`` cell.

    Args:
        tx: Transcript table with both id columns.
        column: Cell-id column to match.
        reference: Reference cell-id column to match against.
        threshold: Minimum fraction of a cell's transcripts that must fall in
            the reference cell for the match to be kept.

    Returns:
        One row per matched ``column`` cell: ``(column, reference)``.
    """
    return (
        tx
        .filter(pl.col(column).is_not_null())
        .group_by([reference, column]).len('count')
        .with_columns(
            (pl.col('count') / pl.col('count').sum().over(column)).alias('fraction')
        )
        .filter(
            pl.col('fraction') > threshold,
            pl.col(reference).is_not_null(),
        )
        .sort('fraction')
        .group_by(column)
        .last()
        .select([column, reference])
    )


def compute_metrics(
    bins_col: pl.DataFrame,
    bins_ref: pl.DataFrame,
    mapping: pl.DataFrame,
    column: str,
    reference: str,
    min_overlap_fraction: float = 0.1,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Overlap metrics between matched ``column`` and ``reference`` footprints.

    Args:
        bins_col: Binned footprints for the ``column`` segmentation.
        bins_ref: Binned footprints for the ``reference`` segmentation.
        mapping: ``(column, reference)`` matches from :func:`map_labels`.
        column: Cell-id column being evaluated.
        reference: Reference cell-id column.
        min_overlap_fraction: Minimum footprint overlap for a ``column`` cell to
            count toward a reference cell's ``n_overlapping_cells``.

    Returns:
        A ``(iou, relative_area)`` frame per matched cell, and a
        ``(n_overlapping_cells)`` frame per reference cell.
    """
    mapped_col = bins_col.join(mapping.select(column), on=column, how='semi')
    mapped_ref = bins_ref.join(mapping.select(reference), on=reference, how='semi')

    sizes_col = mapped_col.group_by(column).len('size_col')
    sizes_ref = mapped_ref.group_by(reference).len('size_ref')

    intersection = (
        mapped_col
        .join(mapped_ref, on=['x_bin', 'y_bin'])
        .join(mapping, on=[column, reference], how='semi')
        .group_by([column, reference]).len('intersection')
    )

    column_metrics = (
        mapping
        .join(sizes_col, on=column)
        .join(sizes_ref, on=reference)
        .join(intersection, on=[column, reference], how='left')
        .with_columns(pl.col('intersection').fill_null(0))
        .with_columns(
            (pl.col('intersection')
             / (pl.col('size_col') + pl.col('size_ref') - pl.col('intersection')))
            .alias('iou'),
            (pl.col('size_col') / pl.col('size_ref')).alias('relative_area'),
        )
        .with_columns(pl.lit(column).alias('segmentation'))
        .select([column, reference, 'segmentation', 'iou', 'relative_area'])
    )

    overlap = (
        bins_ref
        .join(bins_col.select([column, 'x_bin', 'y_bin']), on=['x_bin', 'y_bin'])
        .group_by([reference, column]).len('overlap_bins')
        .join(sizes_col, on=column, how='left')
        .filter(pl.col('overlap_bins') / pl.col('size_col') > min_overlap_fraction)
        .group_by(reference)
        .agg(pl.col(column).n_unique().alias('n_overlapping_cells'))
        .with_columns(pl.lit(column).alias('segmentation'))
    )

    return column_metrics, overlap


def centroid_neighbors(
    tx: pl.DataFrame,
    column: str,
    radius: float = 15,
    x_column: str = 'x_location',
    y_column: str = 'y_location',
) -> pl.DataFrame:
    """Neighbouring cells whose transcript centroids lie within ``radius``.

    Args:
        tx: Transcript table with a cell-id and coordinate columns.
        column: Cell-id column.
        radius: Maximum centroid distance for two cells to be neighbours.
        x_column: Transcript x-coordinate column.
        y_column: Transcript y-coordinate column.

    Returns:
        Undirected ``(cell, neighbor)`` edges (both directions present).
    """
    centroids = (
        tx
        .filter(pl.col(column).is_not_null())
        .group_by(column)
        .agg(pl.col(x_column).mean().alias('x'), pl.col(y_column).mean().alias('y'))
    )
    ids = centroids[column].to_numpy()
    coords = centroids.select(['x', 'y']).to_numpy()
    pairs = cKDTree(coords).query_pairs(radius, output_type='ndarray')
    a, b = ids[pairs[:, 0]], ids[pairs[:, 1]]
    return pl.DataFrame({
        'cell': np.concatenate([a, b]),
        'neighbor': np.concatenate([b, a]),
    }).cast({'cell': centroids[column].dtype, 'neighbor': centroids[column].dtype})


def centroid_neighbor_diffs(
    tx: pl.DataFrame,
    segmentation_columns: list[str],
    reference: str,
    mappings: dict[str, pl.DataFrame],
    radii,
    x_column: str = 'x_location',
    y_column: str = 'y_location',
) -> pl.DataFrame:
    """Per-cell neighbour differences vs. the reference at each centroid radius.

    Args:
        tx: Transcript table.
        segmentation_columns: Method cell-id columns to evaluate.
        reference: Reference cell-id column.
        mappings: ``{column: map_labels(...)}`` matches per method.
        radii: Centroid radii to sweep.
        x_column: Transcript x-coordinate column.
        y_column: Transcript y-coordinate column.

    Returns:
        Tidy :func:`neighbor_difference` frame with ``segmentation`` and
        ``radius`` columns.
    """
    diffs = []
    for radius in radii:
        reference_edges = centroid_neighbors(tx, reference, radius, x_column, y_column)
        for column in segmentation_columns:
            diff = neighbor_difference(
                centroid_neighbors(tx, column, radius, x_column, y_column),
                reference_edges, mappings[column], column, reference,
            )
            diffs.append(diff.with_columns(
                pl.lit(column).alias('segmentation'),
                pl.lit(float(radius)).alias('radius'),
            ))
    return pl.concat(diffs)


def point_neighbor_edges(
    df: pl.DataFrame,
    column: str,
    radii,
    x_column: str = 'x_location',
    y_column: str = 'y_location',
    block_size: int = 500_000,
    workers: int = -1,
) -> pl.DataFrame:
    """Cells sharing a transcript within each radius.

    Transcripts are queried in blocks at the largest radius and every smaller
    radius is derived from the candidate distances, so redundant transcript
    pairs never accumulate in memory.

    Args:
        df: Transcript table with a cell-id and coordinate columns.
        column: Cell-id column.
        radii: Transcript radii to evaluate.
        x_column: Transcript x-coordinate column.
        y_column: Transcript y-coordinate column.
        block_size: Transcripts queried per block.
        workers: Threads for the radius query (``-1`` uses all cores).

    Returns:
        Distinct cross-cell ``(radius, cell, neighbor)`` edges.
    """
    sub = df.drop_nulls(column).select([column, x_column, y_column])
    ids = sub[column].to_numpy()
    tree = cKDTree(sub.select([x_column, y_column]).to_numpy())
    return pl.concat(
        _block_edges(tree, ids, start, start + block_size, radii, workers)
        for start in range(0, len(ids), block_size)
    ).unique().cast({'cell': sub[column].dtype, 'neighbor': sub[column].dtype})


def neighbor_difference(
    method_edges: pl.DataFrame,
    reference_edges: pl.DataFrame,
    mapping: pl.DataFrame,
    column: str,
    reference: str,
) -> pl.DataFrame:
    """Neighbours each matched method cell adds or drops relative to its reference cell.

    Args:
        method_edges: ``(cell, neighbor)`` edges of the method segmentation.
        reference_edges: ``(cell, neighbor)`` edges of the reference.
        mapping: ``(column, reference)`` matches from :func:`map_labels`.
        column: Method cell-id column in ``mapping``.
        reference: Reference cell-id column in ``mapping``.

    Returns:
        One row per matched method cell: ``extra``, ``missing``,
        ``intersection``, ``union``, ``symmetric_difference`` and the
        neighbour counts ``a`` (method) and ``b`` (reference).
    """
    matched = mapping.select(
        pl.col(column).alias('cell'), pl.col(reference).alias('C')
    ).unique()
    ref_ids = matched.select('C').unique()

    map_cell = mapping.rename({column: 'cell', reference: 'C'})
    map_nbr = mapping.rename({column: 'neighbor', reference: 'C_nbr'})

    method = (
        method_edges
        .join(map_cell, on='cell')
        .join(map_nbr, on='neighbor')
        .select(['cell', 'C', 'C_nbr'])
        .filter(pl.col('C') != pl.col('C_nbr'))
        .unique()
    )
    ref = (
        reference_edges
        .rename({'cell': 'C', 'neighbor': 'C_nbr'})
        .join(ref_ids, on='C', how='semi')
        .join(ref_ids.rename({'C': 'C_nbr'}), on='C_nbr', how='semi')
        .unique()
        .join(matched, on='C')
        .select(['cell', 'C', 'C_nbr'])
    )

    keys = ['cell', 'C_nbr']
    extra = method.join(ref, on=keys, how='anti').group_by('cell').len('extra')
    missing = ref.join(method, on=keys, how='anti').group_by('cell').len('missing')
    intersection = method.join(ref, on=keys, how='inner').group_by('cell').len('intersection')
    a = method.group_by('cell').len('a')
    b = ref.group_by('cell').len('b')

    return (
        matched
        .join(a, on='cell', how='left')
        .join(b, on='cell', how='left')
        .join(extra, on='cell', how='left')
        .join(missing, on='cell', how='left')
        .join(intersection, on='cell', how='left')
        .with_columns(pl.col('a', 'b', 'extra', 'missing', 'intersection').fill_null(0))
        .with_columns(
            union=pl.col('a') + pl.col('b') - pl.col('intersection'),
            symmetric_difference=pl.col('extra') + pl.col('missing'),
        )
        .select(
            'extra', 'missing', 'intersection', 'union', 'symmetric_difference', 'a', 'b',
        )
    )


def _block_edges(tree, ids, start, stop, radii, workers) -> pl.DataFrame:
    radii = sorted(float(r) for r in radii)
    data = np.asarray(tree.data)

    neighbor_lists = tree.query_ball_point(data[start:stop], radii[-1], workers=workers)
    counts = np.fromiter(map(len, neighbor_lists), np.intp, len(neighbor_lists))
    q_idx = np.repeat(np.arange(start, start + len(neighbor_lists)), counts)
    if counts.sum():
        n_idx = np.concatenate([np.asarray(n, np.intp) for n in neighbor_lists])
    else:
        n_idx = np.empty(0, np.intp)

    dist = np.linalg.norm(data[q_idx] - data[n_idx], axis=1)
    cell, neighbor = ids[q_idx], ids[n_idx]

    frames = []
    for radius in radii:
        keep = (dist <= radius) & (cell != neighbor)
        frames.append(pl.DataFrame({
            'radius': np.full(int(keep.sum()), radius),
            'cell': cell[keep],
            'neighbor': neighbor[keep],
        }).unique())
    return pl.concat(frames)


def _process_chunk(chunk: list[tuple]) -> list[tuple]:
    return [(cell_id, _connect_components(gx, gy)) for cell_id, gx, gy in chunk]


def _connect_components(xs: np.ndarray, ys: np.ndarray) -> np.ndarray | None:
    n_pts = len(xs)
    if n_pts <= 1:
        return None

    x_min, y_min = xs.min(), ys.min()
    lx = xs - x_min
    ly = ys - y_min
    w, h = int(lx.max()) + 1, int(ly.max()) + 1

    if w * h <= _GRID_AREA_THRESHOLD:
        comp_labels, n = _label_via_grid(lx, ly, w, h)
    else:
        comp_labels, n = _label_via_graph(lx, ly)
    if n <= 1:
        return None

    existing = set(zip(xs.tolist(), ys.tolist()))
    comp_points = {
        c: np.column_stack([xs[comp_labels == c], ys[comp_labels == c]])
        for c in range(1, n + 1)
    }

    comps = list(range(1, n + 1))
    nc = len(comps)
    dist_matrix = np.full((nc, nc), np.inf)
    nearest_pairs = {}
    for i in range(nc):
        pts_i = comp_points[comps[i]]
        for j in range(i + 1, nc):
            pts_j = comp_points[comps[j]]
            dists = cdist(pts_i, pts_j, metric='chebyshev')
            bi, bj = divmod(int(np.argmin(dists)), dists.shape[1])
            dist_matrix[i, j] = dists[bi, bj]
            nearest_pairs[(i, j)] = (tuple(pts_i[bi]), tuple(pts_j[bj]))

    mst = minimum_spanning_tree(csr_matrix(dist_matrix)).tocoo()
    bridge_list = []
    for i, j in zip(mst.row, mst.col):
        (x1, y1), (x2, y2) = nearest_pairs[(min(i, j), max(i, j))]
        path = _make_bridge_path(int(x1), int(y1), int(x2), int(y2), existing)
        if len(path) > 0:
            bridge_list.append(path)

    return np.concatenate(bridge_list, axis=0) if bridge_list else None


def _label_via_grid(lx, ly, w, h) -> tuple[np.ndarray, int]:
    grid = np.zeros((w, h), dtype=np.int32)
    grid[lx, ly] = 1
    labeled, n = ndlabel(grid, structure=np.ones((3, 3)))
    return labeled[lx, ly], n


def _label_via_graph(lx, ly) -> tuple[np.ndarray, int]:
    n_pts = len(lx)
    coords_to_idx = {(int(lx[i]), int(ly[i])): i for i in range(n_pts)}
    rows, cols = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            for i in range(n_pts):
                j = coords_to_idx.get((int(lx[i]) + dx, int(ly[i]) + dy))
                if j is not None:
                    rows.append(i)
                    cols.append(j)
    if not rows:
        return np.arange(1, n_pts + 1), n_pts
    adj = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n_pts, n_pts))
    n_components, labels = connected_components(adj, directed=False)
    return labels + 1, n_components


def _make_bridge_path(x1, y1, x2, y2, existing: set) -> np.ndarray:
    dx, dy = x2 - x1, y2 - y1
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return np.empty((0, 2), dtype=np.int64)

    sx = int(np.sign(dx))
    sy = int(np.sign(dy))
    path_x = np.empty(steps, dtype=np.int64)
    path_y = np.empty(steps, dtype=np.int64)
    px, py, idx = x1, y1, 0
    for _ in range(steps):
        px += sx if px != x2 else 0
        py += sy if py != y2 else 0
        path_x[idx], path_y[idx] = px, py
        idx += 1
    if idx > 0 and path_x[idx - 1] == x2 and path_y[idx - 1] == y2:
        idx -= 1
    if idx == 0:
        return np.empty((0, 2), dtype=np.int64)

    path_x, path_y = path_x[:idx], path_y[:idx]
    keep = np.array([
        (int(path_x[i]), int(path_y[i])) not in existing for i in range(idx)
    ])
    if not keep.any():
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack([path_x[keep], path_y[keep]])
