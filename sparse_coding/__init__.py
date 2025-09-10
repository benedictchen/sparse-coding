from .__about__ import __version__

# Core implementations (backwards compatible)
from .sparse_coder import SparseCoder
from .dictionary_learner import DictionaryLearner

# Advanced optimization
from .advanced_optimization import (
    AdvancedOptimizer, L1Proximal, ElasticNetProximal, 
    NonNegativeL1Proximal, create_advanced_sparse_coder
)

# Visualization and logging
from .dashboard import TB, CSVDump, DashboardLogger
from . import visualization
from . import penalties

# New modular architecture
from . import core
from . import api
from . import components  # Auto-registers default components
from . import adapters
from . import streaming
from . import serialization
from . import algorithms
from . import factories
from . import examples

# Framework adapters - these will fail if dependencies missing
from .sklearn_estimator import SparseCoderEstimator
from .adapters.sklearn import SparseCoderEstimator as SparseCoderEstimatorV2
from .adapters.sklearn import DictionaryLearnerEstimator
from .adapters.torch import SparseCodingModule, DictionaryLearningModule

__all__ = [
    # Version
    "__version__", 
    
    # Core (backwards compatible)
    "SparseCoder", 
    "DictionaryLearner",
    
    # Advanced optimization
    "AdvancedOptimizer",
    "L1Proximal",
    "ElasticNetProximal", 
    "NonNegativeL1Proximal",
    "create_advanced_sparse_coder",
    
    # Visualization and logging
    "TB",
    "CSVDump", 
    "DashboardLogger",
    "visualization",
    "penalties",
    
    # New modular architecture
    "core",
    "api", 
    "adapters",
    "streaming",
    "serialization",
    "algorithms",
    "factories", 
    "examples",
    
    # Framework adapters (may be None if dependencies missing)
    "SparseCoderEstimator",           # Legacy sklearn adapter
    "SparseCoderEstimatorV2",         # New sklearn adapter
    "DictionaryLearnerEstimator",     # sklearn dictionary learner
    "SparseCodingModule",             # PyTorch module
    "DictionaryLearningModule",       # PyTorch dictionary learner
]
