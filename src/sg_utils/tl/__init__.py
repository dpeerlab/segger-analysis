from importlib.util import find_spec as _find_spec

from . import (
    boundaries,
    celltyping,
    comparison_metrics,
    footprints,
    generate_boundaries,
    get_group_markers,
    masks,
    metrics,
    neighbors,
    wta,
    xenium_utils,
)

# PhenoGraph and the CellTypist helpers run on RAPIDS (cudf, cugraph, cuml).
if _find_spec("cudf") is not None:
    try:
        from . import celltypist_utils, phenograph_rapids
    except ImportError:
        pass

# Tiling and partitioning of IST graphs need torch and segger.
if _find_spec("segger") is not None:
    try:
        from . import scaling
    except ImportError:
        pass
