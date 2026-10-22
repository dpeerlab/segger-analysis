"""Tiling, batch packing and partitioning of IST graphs (ED Fig. 2, Supp. Fig. 2).

Segger tiles a dataset spatially before building its graph: an adaptive
(quadtree) tiling caps the transcripts per tile, a square tiling is the
fixed-grid baseline. Tiles are packed into batches under a memory budget, and
edges cut by tile boundaries truncate the receptive field of nearby nodes.
Requires torch, torch_geometric and segger.
"""

import gc
from pathlib import Path
from timeit import default_timer
from typing import Sequence

import numpy as np
import polars as pl
import torch
from scipy.optimize import minimize
from segger.data.partition.sampler import best_fit_decreasing, first_fit_decreasing_bucketed
from segger.data.tiling import QuadTreeTiling, SquareTiling
from segger.io import StandardTranscriptFields, get_preprocessor
from torch_geometric.utils import k_hop_subgraph


COSMX_MICRONS_PER_PIXEL = 0.12028


_MERSCOPE_FIELDS = ('global_x', 'global_y')

# Short platform names and the preprocessors segger registers for them
_SEGGER_PLATFORMS = {
    'xenium': '10x_xenium',
    'xenium_v1': '10x_xenium_v1',
    'cosmx': 'nanostring_cosmx',
}


def _read_merscope_positions(raw_dir: Path) -> pl.DataFrame:
    """Read MERSCOPE transcript coordinates directly from the flat files."""
    candidates = [
        *raw_dir.glob('detected_transcripts.parquet'),
        *raw_dir.glob('detected_transcripts.csv'),
        *raw_dir.glob('**/labeled_transcripts.parquet'),
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No MERSCOPE transcript table found under {raw_dir}."
        )
    path = candidates[0]
    reader = pl.read_parquet if path.suffix == '.parquet' else pl.read_csv
    return reader(path).select(_MERSCOPE_FIELDS)


def load_transcript_positions(
    raw_dir: Path,
    platform: str | None = None,
    device: str = 'cuda',
    microns_per_pixel: float | None = None,
) -> torch.Tensor:
    """Load every transcript coordinate of a dataset as an (N, 2) tensor.

    Args:
        raw_dir: Directory of the platform output (e.g. a Xenium output bundle).
        platform: One of ``xenium``, ``xenium_v1``, ``cosmx``, ``merscope``.
            Inferred from the directory contents when ``None``.
        device: Device to place the returned tensor on.
        microns_per_pixel: Scale factor applied to the coordinates. Defaults
            to :data:`COSMX_MICRONS_PER_PIXEL` for CosMx, whose preprocessor
            reports global pixel coordinates, and to 1 elsewhere.

    Returns:
        An (N, 2) float tensor of transcript positions in microns.
    """
    raw_dir = Path(raw_dir)
    platform = (platform or '').lower() or None
    platform = _SEGGER_PLATFORMS.get(platform, platform)

    if platform in ('merscope', 'vizgen_merscope'):
        positions = _read_merscope_positions(raw_dir)
    else:
        fields = StandardTranscriptFields()
        positions = (
            get_preprocessor(raw_dir, platform)
            .transcripts
            .select([fields.x, fields.y])
        )
        if platform is None:
            platform = _infer_platform_name(raw_dir)

    if microns_per_pixel is None:
        microns_per_pixel = (
            COSMX_MICRONS_PER_PIXEL if platform == 'nanostring_cosmx' else 1.0
        )
    if microns_per_pixel != 1.0:
        positions = positions.select(pl.all() * microns_per_pixel)

    return positions.to_torch().to(torch.float32).to(device)


def _infer_platform_name(raw_dir: Path) -> str:
    """Return the platform segger infers for a dataset directory."""
    from segger.io.preprocessor import _infer_platform

    return _infer_platform(raw_dir)


def tile_counts(
    positions: torch.Tensor,
    tiling,
    include_empty: bool = False,
) -> torch.Tensor:
    """Count the transcripts falling in each tile.

    Args:
        positions: (N, 2) transcript coordinates.
        tiling: A segger ``Tiling`` instance.
        include_empty: If True, return one count per tile in
            ``tiling.tiles`` order, including tiles holding no transcripts
            (needed to color a tiling map). If False, empty tiles are
            dropped, matching how a loader skips them.

    Returns:
        A 1-D tensor of transcript counts.
    """
    labels = tiling.label(positions)
    if include_empty:
        return torch.bincount(labels, minlength=len(tiling.tiles)).cpu()
    return torch.unique(labels, return_counts=True)[1].cpu()


def transcript_density(
    positions: torch.Tensor,
    probe_width: float = 50.0,
) -> float:
    """Estimate transcripts per square micron over tissue-covered area only.

    A square grid is laid over the field and only tiles containing at least
    one transcript contribute area, so the large empty margins of a section
    do not deflate the estimate.

    Args:
        positions: (N, 2) transcript coordinates in microns.
        probe_width: Side length of the probe grid, in microns.

    Returns:
        Transcripts per square micron of covered tissue.
    """
    tiling = SquareTiling(positions, probe_width)
    occupied = torch.unique(tiling.label(positions)).cpu().numpy()
    return positions.shape[0] / tiling.tiles.iloc[occupied].area.sum()


def matched_square_side_length(
    positions: torch.Tensor,
    nodes_per_tile: int,
    probe_width: float = 50.0,
) -> float:
    """Square-tile side length holding ``nodes_per_tile`` on average.

    This is the fair fixed-tiling baseline for an adaptive tiling of the same
    capacity: both tilings then have the same mean occupancy, and only the
    spread differs.

    Args:
        positions: (N, 2) transcript coordinates in microns.
        nodes_per_tile: Target transcripts per tile.
        probe_width: Side length of the density probe grid, in microns.

    Returns:
        The side length in microns.
    """
    density = transcript_density(positions, probe_width=probe_width)
    return float(np.sqrt(nodes_per_tile / density))


def matched_quadtree_capacity(
    positions: torch.Tensor,
    target_width: float,
    bounds: tuple[float, float] = (20_000, 50_000),
    x0: float = 40_000,
    tol: float = 1.0,
) -> float:
    """Quadtree capacity whose mean tile width matches ``target_width``.

    The inverse of :func:`matched_square_side_length`: used when the two
    tilings are compared visually and should share a mean tile size.

    Args:
        positions: (N, 2) transcript coordinates in microns.
        target_width: Mean tile width to match, in microns.
        bounds: Search interval for the capacity.
        x0: Starting capacity for the search.
        tol: Convergence tolerance passed to the optimizer.

    Returns:
        The transcripts-per-tile capacity.
    """
    def objective(x) -> float:
        mean_area = QuadTreeTiling(positions, x[0]).tiles.area.mean()
        return abs(np.sqrt(mean_area) - target_width)

    result = minimize(
        objective,
        x0=(x0,),
        method='Nelder-Mead',
        bounds=(bounds,),
        tol=tol,
    )
    return float(result.x[0])


def pack_fixed_size(
    items: Sequence[float],
    tiles_per_batch: int,
) -> list[list[int]]:
    """Group a fixed number of consecutive tiles into each batch.

    The strategy of a loader with a constant ``batch_size``: it ignores tile
    size entirely, so batches overflow wherever tiles happen to be large.

    Args:
        items: Tile sizes, in the order the loader would see them.
        tiles_per_batch: Number of tiles per batch.

    Returns:
        A list of batches, each a list of indices into ``items``.
    """
    if tiles_per_batch < 1:
        raise ValueError("tiles_per_batch must be at least 1.")
    return [
        list(range(i, min(i + tiles_per_batch, len(items))))
        for i in range(0, len(items), tiles_per_batch)
    ]


def pack_sequential(
    items: Sequence[float],
    capacity: float,
) -> list[list[int]]:
    """Accumulate tiles in order, closing a batch before it overflows.

    This is the greedy online strategy of the PyTorch Geometric
    ``DynamicBatchSampler``: it respects the budget but never reorders, so
    every batch ends with whatever headroom the next tile did not fit into.

    Args:
        items: Tile sizes, in the order the loader would see them.
        capacity: Maximum size of a batch.

    Returns:
        A list of batches, each a list of indices into ``items``.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    total = 0.0
    for i, size in enumerate(items):
        if current and total + size > capacity:
            batches.append(current)
            current, total = [], 0.0
        current.append(i)
        total += size
    if current:
        batches.append(current)
    return batches


def packing_efficiencies(
    items: Sequence[float],
    capacity: float,
    strategies: Sequence[str] = ('fixed', 'sequential', 'ffd', 'bfd'),
) -> pl.DataFrame:
    """Batch fill fractions for each packing strategy.

    Efficiency is the total size of a batch over the capacity: values above
    one overflow the budget (a prospective out-of-memory failure), values
    below it leave memory allocated but unused.

    Args:
        items: Tile sizes.
        capacity: Batch capacity, in the same units as ``items``.
        strategies: Any of ``fixed``, ``sequential``, ``ffd``, ``bfd``. The
            ``fixed`` baseline uses the mean tiles-per-bin of the ``bfd``
            packing, so its average batch size is matched.

    Returns:
        A frame with columns ``efficiency`` and ``strategy``.
    """
    items = list(items)
    bins_bfd = best_fit_decreasing(items, capacity)

    packings = {
        'bfd': bins_bfd,
        'ffd': first_fit_decreasing_bucketed(items, capacity),
        'sequential': pack_sequential(items, capacity),
        'fixed': pack_fixed_size(
            items, max(1, int(np.mean([len(b) for b in bins_bfd])))
        ),
    }

    frames = []
    for name in strategies:
        sizes = np.array(
            [sum(items[i] for i in batch) for batch in packings[name]],
            dtype=float,
        )
        frames.append(pl.DataFrame({
            'efficiency': sizes / capacity,
            'strategy': np.full(len(sizes), name),
        }))
    return pl.concat(frames)


def estimate_hop_margin(
    data,
    quantile: float = 0.75,
    n_sample: int = 10_000_000,
    device: str = 'cuda',
) -> float:
    """Margin wide enough to span roughly three hops of the graph.

    Taken as ``max(2 * d + r, 3 * d)`` where ``d`` and ``r`` are quantiles of
    the transcript-transcript and transcript-cell edge lengths, i.e. the
    distance covered by three hops under either edge type.

    Args:
        data: A segger ``HeteroData`` graph with ``tx`` and ``bd`` nodes.
        quantile: Edge-length quantile used for each hop.
        n_sample: Maximum edges sampled per edge type.
        device: Device to compute on.

    Returns:
        The margin in microns.
    """
    pdist = torch.nn.PairwiseDistance(p=2)
    pos_tx = data['tx'].pos.to(device)
    pos_bd = data['bd'].pos.to(device)

    def edge_quantile(edge_index, src_pos, dst_pos) -> torch.Tensor:
        src, dst = edge_index.to(device)
        idx = torch.randperm(src.shape[0], device=device)[:n_sample]
        return pdist(src_pos[src[idx]], dst_pos[dst[idx]]).quantile(quantile)

    d = edge_quantile(
        data['tx', 'neighbors', 'tx'].edge_index, pos_tx, pos_tx)
    r = edge_quantile(
        data['tx', 'belongs', 'bd'].edge_index, pos_tx, pos_bd)
    return max(2 * d + r, 3 * d).item()


def khop_impact(
    data,
    tiling,
    margin: float,
    k: int = 2,
    batch_size: int = 100,
    eps: float = 2.0,
    device: str = 'cuda',
) -> dict:
    """Count nodes and edges lost to tile boundaries at a given margin.

    A node is *interior* if it survives eroding its tile by ``margin``, and
    *impacted* if it lies within ``k`` hops of an edge cut by a tile
    boundary. Nodes that are both are retained for training yet have a
    truncated receptive field.

    Args:
        data: A segger ``HeteroData`` graph with ``pos`` on every node type.
        tiling: A segger ``Tiling`` instance.
        margin: Inward erosion applied to each tile, in microns.
        k: Number of hops expanded around each cut node.
        batch_size: Tiles processed per chunk, to bound peak memory.
        eps: Outward buffer applied when subsetting a chunk of tiles.
        device: Device to compute on.

    Returns:
        A dict with per-node-type and per-edge-type totals, and the
        aggregate keys ``node_all_total``, ``node_all_interior``,
        ``node_all_khop`` and ``node_all_khop_interior`` (and the ``edge_``
        equivalents), from which the reported fractions are derived:
        complete nodes = ``1 - node_all_khop_interior / node_all_total``,
        training data = ``node_all_interior / node_all_total``.
    """
    node_types, edge_types = data.node_types, data.edge_types
    totals = {
        f'{scope}_{key}': {t: 0 for t in types}
        for scope, types in (('node', node_types), ('edge', edge_types))
        for key in ('total', 'interior', 'khop', 'khop_interior')
    }

    for start in range(0, len(tiling.tiles), batch_size):
        batch = tiling.tiles[start:start + batch_size]

        # Subset the graph to the tiles of this chunk, with a small buffer
        # so that edges leaving a tile are still present to be counted.
        subset = {}
        for node_type in node_types:
            pos = data[node_type].pos.to(device)
            keep = torch.zeros(pos.shape[0], dtype=torch.bool, device=device)
            for tile in batch:
                xmin, ymin, xmax, ymax = tile.buffer(eps).bounds
                keep |= (
                    (pos[:, 0] > xmin) & (pos[:, 0] < xmax) &
                    (pos[:, 1] > ymin) & (pos[:, 1] < ymax)
                )
            subset[node_type] = keep
        subgraph = data.to(device).subgraph(subset)

        # Label each node with its tile (0 = buffer only) and mark the
        # nodes that survive eroding their tile by the margin.
        for node_type in subgraph.node_types:
            n = subgraph[node_type].num_nodes
            label = torch.zeros(n, dtype=torch.long, device=device)
            interior = torch.zeros(n, dtype=torch.bool, device=device)
            pos = subgraph[node_type].pos.to(device)
            for i, tile in enumerate(batch):
                xmin, ymin, xmax, ymax = tile.bounds
                label[
                    (pos[:, 0] > xmin) & (pos[:, 0] < xmax) &
                    (pos[:, 1] > ymin) & (pos[:, 1] < ymax)
                ] = i + 1
                xmin, ymin, xmax, ymax = tile.buffer(-margin).bounds
                interior |= (
                    (pos[:, 0] > xmin) & (pos[:, 0] < xmax) &
                    (pos[:, 1] > ymin) & (pos[:, 1] < ymax)
                )
            subgraph[node_type]['label'] = label
            subgraph[node_type]['interior'] = interior

        homo = subgraph.to_homogeneous()
        src, dst = homo.edge_index

        # Cut edges join nodes assigned to different tiles
        src_in_tile, dst_in_tile = homo.label[src] > 0, homo.label[dst] > 0
        cut = (homo.label[src] != homo.label[dst]) & (
            src_in_tile | dst_in_tile)
        cut_nodes = torch.cat([
            src[cut & src_in_tile], dst[cut & dst_in_tile]
        ]).unique()

        # Expand to the k-hop neighborhood of every cut node
        impacted = torch.zeros(
            homo.num_nodes, dtype=torch.bool, device=homo.edge_index.device)
        hop_edges = torch.zeros(
            homo.edge_index.size(1), dtype=torch.bool,
            device=homo.edge_index.device)
        if cut_nodes.numel() > 0:
            hop_nodes, _, _, hop_edges = k_hop_subgraph(
                node_idx=cut_nodes,
                num_hops=k,
                edge_index=homo.edge_index,
                num_nodes=homo.num_nodes,
                relabel_nodes=False,
                flow='target_to_source',
            )
            impacted[hop_nodes[homo.label[hop_nodes] > 0]] = True

        in_tile = homo.label > 0
        edge_in_tile = src_in_tile & dst_in_tile
        edge_interior = homo.interior[src] & homo.interior[dst]

        for i, node_type in enumerate(node_types):
            mask = homo.node_type == i
            totals['node_total'][node_type] += int(
                (mask & in_tile).sum())
            totals['node_interior'][node_type] += int(
                (mask & homo.interior).sum())
            totals['node_khop'][node_type] += int(
                (mask & impacted).sum())
            totals['node_khop_interior'][node_type] += int(
                (mask & impacted & homo.interior).sum())

        for i, edge_type in enumerate(edge_types):
            mask = homo.edge_type == i
            totals['edge_total'][edge_type] += int(
                (mask & edge_in_tile).sum())
            totals['edge_interior'][edge_type] += int(
                (mask & edge_interior).sum())
            totals['edge_khop'][edge_type] += int(
                (mask & hop_edges).sum())
            totals['edge_khop_interior'][edge_type] += int(
                (mask & hop_edges & edge_interior).sum())

        torch.cuda.empty_cache()

    out = {}
    for key, per_type in totals.items():
        scope, _, metric = key.partition('_')
        for type_name, value in per_type.items():
            if not isinstance(type_name, str):
                type_name = '_'.join(type_name)
            out[f'{scope}_{type_name}_{metric}'] = value
        out[f'{scope}_all_{metric}'] = sum(per_type.values())
    return out


def benchmark_loader(
    data_module,
    mode: str = 'partition',
    n_warmup: int = 1,
) -> dict:
    """Time one pass over a dataset's tiles and record its peak GPU memory.

    Setup (sorting the graph by tile, or moving it to the GPU) and warm-up
    passes are excluded; memory is the GPU memory in use at the start of the
    timed pass plus the peak allocated during it.

    Args:
        data_module: A configured segger ``ISTDataModule``.
        mode: ``partition`` times segger's index-slicing training loader,
            moving each batch to the GPU as in training; ``subgraph`` times per-tile extraction with the PyTorch Geometric
            ``subgraph`` method over the same tiles.
        n_warmup: Untimed passes run before the timed one (``partition``).

    Returns:
        A dict with ``time`` (seconds) and ``memory`` (peak bytes in use).
    """
    if hasattr(data_module, 'fit_dataset'):
        data_module.teardown('fit')
    if hasattr(data_module, 'predict_dataset'):
        data_module.teardown('predict')
    gc.collect()
    torch.cuda.empty_cache()

    if mode == 'partition':
        data_module.setup('fit')
        loader = data_module.train_dataloader()
        # Each batch is moved to the GPU, as Lightning does during training
        for _ in range(n_warmup):
            for batch in loader:
                batch = batch.to('cuda')
        del batch
        memory = _PeakMemory()
        start = default_timer()
        for batch in loader:
            batch = batch.to('cuda')
        torch.cuda.synchronize()
        elapsed = default_timer() - start
        del batch

    elif mode == 'subgraph':
        data_module.setup('predict')
        data = data_module.data
        tiles = data_module.predict_dataset.tiling.tiles
        memory = _PeakMemory()
        start = default_timer()
        for tile in tiles:
            xmin, ymin, xmax, ymax = tile.bounds
            subset = {}
            for node_type in data.node_types:
                pos = data[node_type]['pos']
                subset[node_type] = (
                    (pos[:, 0] >= xmin) & (pos[:, 0] < xmax) &
                    (pos[:, 1] >= ymin) & (pos[:, 1] < ymax)
                ).nonzero().flatten()
            data.subgraph(subset)
        elapsed = default_timer() - start

    else:
        raise ValueError(
            f"Unknown mode: '{mode}'. Use 'partition' or 'subgraph'."
        )

    return {'time': elapsed, 'memory': memory.stop()}


class _PeakMemory:
    """Peak GPU memory in use from construction to :meth:`stop`.

    ``import segger`` routes torch, CuPy and cuDF through one RMM pool, whose
    statistics replace torch's peak counters (unsupported by RMM).
    """

    def __init__(self):
        torch.cuda.synchronize()
        try:
            torch.cuda.reset_peak_memory_stats()
            self.rmm = None
        except RuntimeError:
            from rmm import statistics

            self.rmm = statistics
            self.baseline = statistics.get_statistics().current_bytes
            statistics.push_statistics()

    def stop(self) -> int:
        torch.cuda.synchronize()
        if self.rmm is None:
            return torch.cuda.max_memory_allocated()
        return self.baseline + self.rmm.pop_statistics().peak_bytes
