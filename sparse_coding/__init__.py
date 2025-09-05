from .__about__ import __version__
from .sparse_coder import SparseCoder
try:
    from .sklearn_estimator import SparseCoderEstimator  # optional (sklearn)
except Exception:  # pragma: no cover
    SparseCoderEstimator = None

__all__ = ["__version__", "SparseCoder", "SparseCoderEstimator"]
