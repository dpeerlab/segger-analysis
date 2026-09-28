from importlib.util import find_spec as _find_spec

from . import datasets, io, pl, tl

# The RAPIDS preprocessing helpers need a CUDA GPU (cupy, cuml).
if _find_spec("cupy") is not None:
    try:
        from . import pp
    except ImportError:
        pass
