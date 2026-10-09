"""Compare two label images of the same tissue.

Nuclei are a well-defined image object; the extent of a cell is not. Running
an independent segmenter over the same morphology images and scoring how
closely it reproduces the vendor masks separates what the image determines
from what the algorithm decides.
"""

from pathlib import Path

import numpy as np
import polars as pl
from skimage.metrics import adapted_rand_error


def normalize_channel(
    channel: np.ndarray,
    lower: float = 0.001,
    upper: float = 0.999,
) -> np.ndarray:
    """Clip a morphology channel to a percentile range and rescale to [0, 1].

    Args:
        channel: 2-D intensity image.
        lower: Lower clipping quantile.
        upper: Upper clipping quantile.

    Returns:
        A float32 image scaled to [0, 1]. Whole-section morphology images
        are large, so the result is deliberately not float64.
    """
    vmin, vmax = np.quantile(channel, [lower, upper])
    scaled = (channel.clip(vmin, vmax) - vmin) / (vmax - vmin)
    return scaled.astype(np.float32)


def read_morphology_image(
    morphology_dir: Path,
    channels: tuple[int, ...] = (0, 1, 2),
    level: int = 0,
    selection: tuple[slice, slice] | None = None,
) -> np.ndarray:
    """Read Xenium morphology channels into one normalized stack.

    Channel 0 is the DAPI nuclear stain, 1 the cell boundary stain and 2 the
    interior stain.

    Args:
        morphology_dir: ``morphology_focus`` directory of a Xenium dataset.
        channels: Channel indices to read, in output order.
        level: Pyramid level of the OME-TIFF.
        selection: Optional ``(rows, cols)`` slices to read a crop.

    Returns:
        An (H, W, len(channels)) float32 image scaled to [0, 1].
    """
    from tifffile import imread

    morphology_dir = Path(morphology_dir)
    planes = []
    for c in channels:
        kwargs = dict(is_ome=False, level=level)
        if selection is not None:
            kwargs['selection'] = selection
        planes.append(normalize_channel(
            imread(morphology_dir / f'morphology_focus_000{c}.ome.tif',
                   **kwargs)
        ))
    return np.stack(planes, axis=-1)


def read_xenium_masks(zarr_path: Path, boundary: str = 'nucleus'):
    """Read the Xenium onboard label image from the ``cells.zarr.zip`` store.

    Args:
        zarr_path: Path to ``cells.zarr.zip``.
        boundary: ``nucleus`` or ``cell``.

    Returns:
        A 2-D integer label image.
    """
    import zarr

    index = {'nucleus': 0, 'cell': 1}[boundary]
    store = zarr.storage.ZipStore(Path(zarr_path))
    return zarr.load(store, path=f'masks/{index}')


def segment_tiled(
    image: np.ndarray,
    model,
    tile_size: int = 20_000,
    batch_size: int = 8,
) -> np.ndarray:
    """Segment a large image tile by tile, keeping labels unique.

    Args:
        image: 2-D or (H, W, C) image to segment.
        model: A Cellpose ``CellposeModel``; the default model of cellpose
            v4 is Cellpose-SAM.
        tile_size: Side length in pixels of each segmentation tile.
        batch_size: Batch size passed to the model.

    Returns:
        A 2-D integer label image the same size as ``image``.
    """
    height, width = image.shape[:2]
    labels = np.zeros((height, width), dtype=np.int32)
    offset = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = image[y:y + tile_size, x:x + tile_size]
            mask, _, _ = model.eval(tile, batch_size=batch_size)
            mask = mask.astype(np.int32)
            mask[mask > 0] += offset
            offset = max(offset, int(mask.max()))
            labels[y:y + tile_size, x:x + tile_size] = mask
    return labels


def tiled_rand_index(
    left: np.ndarray,
    right: np.ndarray,
    tile_size: int = 250,
) -> pl.DataFrame:
    """Adapted Rand index between two label images, computed per tile.

    A whole-section score would be dominated by the empty regions of the
    slide, so agreement is scored independently inside each tile.

    Args:
        left: First label image.
        right: Second label image, same shape as ``left``.
        tile_size: Side length in pixels of each scoring tile.

    Returns:
        A frame with one row per tile: ``rand_index`` (one minus the adapted
        Rand error), ``precision``, ``recall`` and the number of distinct
        labels each segmentation places in the tile (``n_labels_left``,
        ``n_labels_right``), used to drop near-empty tiles.
    """
    if left.shape != right.shape:
        raise ValueError("Label images must have the same shape.")

    rows = []
    height, width = left.shape
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            tile_left = left[i:i + tile_size, j:j + tile_size]
            tile_right = right[i:i + tile_size, j:j + tile_size]
            error, precision, recall = adapted_rand_error(
                tile_left, tile_right)
            rows.append({
                'tile_row': i // tile_size,
                'tile_col': j // tile_size,
                'rand_index': 1 - error,
                'precision': precision,
                'recall': recall,
                'n_labels_left': len(np.unique(tile_left)),
                'n_labels_right': len(np.unique(tile_right)),
            })
    return pl.DataFrame(rows)


def mask_agreement(left, right, buffer: float = 1.0):
    """Split two polygon sets into shared and exclusive area.

    Args:
        left: GeoDataFrame of polygons from one segmentation.
        right: GeoDataFrame of polygons from the other.
        buffer: Open-close buffer applied to each polygon to remove slivers
            introduced by rasterization.

    Returns:
        A ``(shared, exclusive)`` pair of geometries: area covered by both
        segmentations, and area covered by only one of them.
    """
    import geopandas as gpd
    import shapely

    left_geom = left.geometry.buffer(buffer).buffer(-buffer)
    right_geom = right.geometry.buffer(buffer).buffer(-buffer)

    pairs = gpd.GeoDataFrame(geometry=left_geom).sjoin(
        gpd.GeoDataFrame(geometry=right_geom))
    intersections = [
        geom for geom in (
            shapely.intersection(left_geom.loc[i], right_geom.loc[j])
            for i, j in pairs['index_right'].items()
        )
        if not geom.is_empty and geom.area > 0
    ]
    shared = shapely.union_all([
        geom for geom in intersections
        if geom.geom_type in ('Polygon', 'MultiPolygon')
    ])
    union = shapely.union(
        shapely.union_all(left_geom.values),
        shapely.union_all(right_geom.values),
    )
    exclusive = union.difference(shared).buffer(-0.1).buffer(0.1)
    return shared, exclusive
