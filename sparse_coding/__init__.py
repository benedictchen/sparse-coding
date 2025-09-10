from .__about__ import __version__
from .sparse_coder import SparseCoder
from .dictionary_learner import DictionaryLearner
from .advanced_optimization import (
    AdvancedOptimizer, L1Proximal, ElasticNetProximal, 
    NonNegativeL1Proximal, create_advanced_sparse_coder
)
from .dashboard import TB, CSVDump, DashboardLogger
from . import visualization
from . import penalties

try:
    from .sklearn_estimator import SparseCoderEstimator  # optional (sklearn)
except Exception:  # pragma: no cover
    SparseCoderEstimator = None

__all__ = [
    "__version__", 
    "SparseCoder", 
    "DictionaryLearner",
    "AdvancedOptimizer",
    "L1Proximal",
    "ElasticNetProximal", 
    "NonNegativeL1Proximal",
    "create_advanced_sparse_coder",
    "TB",
    "CSVDump", 
    "DashboardLogger",
    "visualization",
    "penalties",
    "SparseCoderEstimator"
]
