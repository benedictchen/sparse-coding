"""
Dictionary learning algorithms for sparse coding.

DEPRECATED: This directory contains legacy individual implementations.
Use the consolidated dict_updater_implementations module instead.

The consolidated implementation provides:
- Single source of truth with factory pattern
- Better maintainability and consistency
- Unified configuration system
"""

import warnings
warnings.warn(
    "sparse_coding.core.dictionary is deprecated. "
    "Use sparse_coding.core.dict_updater_implementations instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from consolidated implementation for backward compatibility
from ..dict_updater_implementations import (
    ModUpdater as MethodOptimalDirections,
    KsvdUpdater as KSVDDictionaryLearning,
    GradientUpdater as GradientDescentUpdate,
    OnlineUpdater as OnlineDictionaryLearning
)

__all__ = [
    'MethodOptimalDirections',
    'KSVDDictionaryLearning', 
    'GradientDescentUpdate',
    'OnlineDictionaryLearning'
]