from .__about__ import __version__

# Core implementations (backwards compatible)
from .sparse_coder import SparseCoder
from .dictionary_learner import DictionaryLearner

# Essential functionality only during restructuring
try:
    from . import penalties
except ImportError:
    penalties = None

try:
    from . import core
except ImportError:
    core = None

__all__ = [
    "__version__",
    "SparseCoder", 
    "DictionaryLearner",
    "penalties",
    "core",
]
