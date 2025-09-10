"""
Streaming/online learning for sparse coding.

Provides incremental dictionary learning with partial_fit interface,
memory mapping for large datasets, and thread-safe operations.
"""

from .online_learner import OnlineSparseCoderLearner, StreamingConfig
from .memory_map import MemoryMappedDataset, create_memory_mapped_loader
from .batch_processor import BatchProcessor, BatchConfig

__all__ = [
    'OnlineSparseCoderLearner', 'StreamingConfig',
    'MemoryMappedDataset', 'create_memory_mapped_loader', 
    'BatchProcessor', 'BatchConfig'
]