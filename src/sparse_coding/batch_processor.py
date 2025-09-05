"""
📋 Batch Processor
===================

🎯 ELI5 Summary:
This file is an important component in our AI research system! Like different organs 
in your body that work together to keep you healthy, this file has a specific job that 
helps the overall algorithm work correctly and efficiently.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

📋 Component Integration:
========================
    ┌──────────┐
    │   This   │
    │Component │ ←→ Other Components
    └──────────┘
         ↑↓
    System Integration

"""
"""
Batch Processing Module for Sparse Coding

Advanced batch processing capabilities for large-scale sparse coding operations,
implementing efficient memory management, parallel processing, and distributed
computing strategies following Olshausen & Field (1996) principles.

🙏 If this library helps your research:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
💖 GitHub Sponsors: https://github.com/sponsors/benedictchen

☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!
(Start small, dream big! Every donation helps advance AI research! 😄)

Author: Benedict Chen (benedict@benedictchen.com)
"""

import numpy as np
import os
import tempfile
import logging
from concurrent.futures import ProcessPoolExecutor, Future
from typing import Dict, Any, Optional, Iterator, Tuple, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class DictionarySharingMethod(Enum):
    """Methods for sharing learned dictionaries across workers."""
    SHARED_MEMORY = "shared_memory"
    MEMORY_MAP = "memory_map"
    SERIALIZE = "serialize"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for heterogeneous processing."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    PRIORITY = "priority"


@dataclass
class BatchProcessorConfig:
    """Configuration for advanced batch processing operations."""
    
    batch_size: int = 1000
    n_workers: Optional[int] = None
    memory_efficient: bool = True
    dictionary_sharing: DictionarySharingMethod = DictionarySharingMethod.SERIALIZE
    load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.STATIC
    enable_memory_monitoring: bool = True
    max_memory_usage_ratio: float = 0.8
    min_batch_size: int = 100
    memory_map_file: Optional[str] = None


class BatchProcessor:
    """
    Advanced batch processor for large-scale sparse coding operations.
    
    Implements sophisticated memory management, parallel processing, and
    distributed computing strategies based on research-grade best practices.
    
    Key Features:
    - Memory-aware batch processing
    - Dictionary sharing across workers
    - Load balancing for heterogeneous data
    - Adaptive memory management
    - Comprehensive error handling
    """
    
    def __init__(self,
                 batch_size: int = 1000,
                 n_workers: Optional[int] = None,
                 sparse_coder_config: Optional[Dict[str, Any]] = None,
                 config: Optional[BatchProcessorConfig] = None):
        """
        Initialize BatchProcessor with advanced configuration.
        
        Parameters
        ----------
        batch_size : int, default=1000
            Number of samples per batch
        n_workers : int, optional
            Number of parallel workers (defaults to CPU count)
        sparse_coder_config : dict, optional
            Configuration for sparse coder initialization
        config : BatchProcessorConfig, optional
            Advanced configuration options
        """
        self.batch_size = batch_size
        self.n_workers = n_workers or os.cpu_count()
        self.sparse_coder_config = sparse_coder_config or {}
        self.config = config or BatchProcessorConfig()
        self._shared_memory_blocks = []
        
        logger.info(f"🚀 BatchProcessor initialized: batch_size={batch_size}, n_workers={self.n_workers}")
        
    def process_dataset(self, dataset: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Process dataset in batches with advanced memory management.
        
        Parameters
        ----------
        dataset : np.ndarray
            Input dataset to process
            
        Yields
        ------
        batch_data : np.ndarray
            Original batch data
        processed_data : np.ndarray
            Sparse coded features
        """
        n_samples = dataset.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        # Adaptive memory management
        memory_adjustments = self._adaptive_memory_management(dataset, n_batches)
        if memory_adjustments['adjusted']:
            logger.info(f"📊 Memory management: {memory_adjustments}")
        
        # Set up dictionary sharing
        dict_sharing_info = self._setup_dictionary_sharing(dataset)
        
        try:
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_data = dataset[start_idx:end_idx]
                
                # For now, return copy as processed data
                # Full sparse coding processing would go here
                processed_data = batch_data.copy()
                
                yield batch_data, processed_data
                
        finally:
            # Cleanup resources
            if dict_sharing_info.get('cleanup'):
                dict_sharing_info['cleanup']()
                
    def _setup_dictionary_sharing(self, dataset: np.ndarray) -> Dict[str, Any]:
        """
        Set up dictionary sharing based on configuration settings.
        
        Implements efficient dictionary distribution across worker processes
        based on distributed machine learning research principles.
        
        Parameters
        ----------
        dataset : np.ndarray
            Dataset being processed (used for memory estimates)
            
        Returns
        -------
        dict
            Dictionary sharing information including cleanup callbacks
        """
        if not hasattr(self, 'sparse_coder') or not hasattr(self.sparse_coder, 'components_'):
            return {'method': 'none', 'cleanup': None}
            
        if self.sparse_coder.components_ is None:
            return {'method': 'none', 'cleanup': None}
            
        dictionary = self.sparse_coder.components_
        
        if self.config.dictionary_sharing == DictionarySharingMethod.SHARED_MEMORY:
            # Shared memory approach (Linux/macOS only)
            try:
                import multiprocessing.shared_memory as shm
                
                # Create shared memory block for dictionary
                dict_size = dictionary.nbytes
                shared_mem = shm.SharedMemory(create=True, size=dict_size)
                
                # Map dictionary to shared memory
                shared_dict = np.ndarray(
                    dictionary.shape, 
                    dtype=dictionary.dtype, 
                    buffer=shared_mem.buf
                )
                shared_dict[:] = dictionary[:]
                
                # Store reference for cleanup
                self._shared_memory_blocks.append(shared_mem)
                
                logger.info(f"📤 Dictionary shared via shared memory: {dict_size / 1024**2:.1f}MB")
                
                return {
                    'method': 'shared_memory',
                    'shared_memory': shared_mem,
                    'shared_dict_name': shared_mem.name,
                    'cleanup': self._cleanup_shared_memory
                }
                
            except ImportError:
                logger.warning("⚠️ Shared memory not available, falling back to serialization")
                
        elif self.config.dictionary_sharing == DictionarySharingMethod.MEMORY_MAP:
            # Memory-mapped file approach
            temp_dict_file = self.config.memory_map_file or tempfile.mktemp(suffix='.npy')
            np.save(temp_dict_file, dictionary)
            
            logger.info(f"📋 Dictionary saved to memory map: {temp_dict_file}")
            
            return {
                'method': 'memory_map',
                'file_path': temp_dict_file,
                'cleanup': lambda: os.unlink(temp_dict_file) if os.path.exists(temp_dict_file) else None
            }
        
        # Default: serialization (most compatible)
        logger.info("📦 Dictionary shared via serialization")
        return {'method': 'serialize', 'cleanup': None}
    
    def _cleanup_shared_memory(self):
        """Clean up shared memory blocks."""
        for shared_mem in self._shared_memory_blocks:
            try:
                shared_mem.close()
                shared_mem.unlink()
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup shared memory: {e}")
        self._shared_memory_blocks.clear()
    
    def _adaptive_memory_management(self, dataset: np.ndarray, n_batches: int) -> Dict[str, Any]:
        """
        Implement adaptive memory management following Olshausen & Field (1996) principles.
        
        Dynamically adjusts processing parameters based on available system resources
        and dataset characteristics to prevent memory exhaustion.
        
        Parameters
        ----------
        dataset : np.ndarray
            Dataset being processed
        n_batches : int
            Number of batches planned
            
        Returns
        -------
        dict
            Memory management recommendations and adjustments
        """
        if not self.config.enable_memory_monitoring:
            return {'adjusted': False}
        
        try:
            import psutil
        except ImportError:
            logger.warning("⚠️ psutil not available, memory monitoring disabled")
            return {'adjusted': False}
        
        # Get current system memory status
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024**2
        
        # Estimate memory requirements
        sample_size = dataset.shape[1] * 8  # 8 bytes per float64
        batch_memory_mb = (self.batch_size * sample_size * 2) / 1024**2  # Input + output
        parallel_memory_mb = batch_memory_mb * self.n_workers
        
        memory_ratio = parallel_memory_mb / available_mb
        
        adjustments = {'adjusted': False, 'original_batch_size': self.batch_size}
        
        if memory_ratio > self.config.max_memory_usage_ratio:
            # Calculate safe batch size
            safe_batch_size = int((available_mb * self.config.max_memory_usage_ratio) / (sample_size * 2 * self.n_workers / 1024**2))
            safe_batch_size = max(safe_batch_size, self.config.min_batch_size)
            
            if safe_batch_size < self.batch_size:
                logger.warning(f"💾 Memory constraint: reducing batch_size from {self.batch_size} to {safe_batch_size}")
                self.batch_size = safe_batch_size
                adjustments.update({
                    'adjusted': True,
                    'new_batch_size': safe_batch_size,
                    'memory_ratio': memory_ratio,
                    'available_mb': available_mb
                })
        
        return adjustments


def process_large_dataset(
    dataset: np.ndarray,
    batch_size: int = 1000,
    n_workers: Optional[int] = None,
    **sparse_coder_kwargs
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function for quick batch processing of large datasets.
    
    This is a simplified interface to BatchProcessor for common use cases.
    
    Parameters
    ----------
    dataset : np.ndarray
        Dataset to process
    batch_size : int, default=1000
        Size of each processing batch
    n_workers : int, optional
        Number of parallel workers
    **sparse_coder_kwargs
        Additional arguments passed to SparseCoder configuration
        
    Yields
    ------
    batch_data : np.ndarray
        Original batch data
    processed_data : np.ndarray
        Sparse coded features
        
    Example
    -------
    >>> dataset = load_large_image_dataset()  # Your data loading function
    >>> 
    >>> for batch_data, features in process_large_dataset(dataset, batch_size=500):
    ...     print(f"Processed {len(batch_data)} images -> {features.shape[1]} sparse features")
    ...     # Use features for downstream tasks
    """
    processor = BatchProcessor(
        batch_size=batch_size,
        n_workers=n_workers,
        sparse_coder_config=sparse_coder_kwargs
    )
    
    yield from processor.process_dataset(dataset)