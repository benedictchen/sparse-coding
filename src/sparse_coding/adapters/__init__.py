"""
Framework adapters for sklearn, PyTorch, JAX integration.

Provides seamless integration with major ML frameworks while maintaining
backend-agnostic core implementation.
"""

from .sklearn import SparseCoderEstimator, DictionaryLearnerEstimator
from .torch import SparseCodingModule, DictionaryLearningModule  
from .jax import sparse_encode_jit, dictionary_update_jit

__all__ = [
    # sklearn adapters
    'SparseCoderEstimator', 'DictionaryLearnerEstimator',
    # PyTorch adapters  
    'SparseCodingModule', 'DictionaryLearningModule',
    # JAX functions
    'sparse_encode_jit', 'dictionary_update_jit'
]