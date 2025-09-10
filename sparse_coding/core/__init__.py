"""
Core backend-agnostic array operations and utilities.

Provides duck array support for NumPy, CuPy, PyTorch, JAX via Array API standard.
"""

from .array import xp, as_same, ensure_array, to_device
from .interfaces import Penalty, InferenceSolver, DictUpdater, Learner

__all__ = [
    'xp', 'as_same', 'ensure_array', 'to_device',
    'Penalty', 'InferenceSolver', 'DictUpdater', 'Learner'
]