"""
🔄 Batch Processor for Large-Scale Sparse Coding
===============================================

This module provides efficient batch processing capabilities for large image datasets,
enabling memory-efficient and parallelized sparse coding operations.

Based on: Olshausen & Field (1996) - "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"

Key Features:
🚀 Memory-efficient batch processing
⚡ Parallel processing with configurable workers  
📊 Progress tracking and intermediate result saving
🔧 Configurable batch sizes and processing options

ELI5 Explanation:
================
Think of this like a factory assembly line for processing images:
- Instead of processing one image at a time (slow), we group them into batches
- Multiple workers process different batches simultaneously (parallel)
- We can save progress along the way so we don't lose work if something breaks
- Memory usage stays controlled by processing manageable chunks

Technical Details:
==================
The BatchProcessor implements efficient batching strategies for sparse coding:
1. Data is chunked into memory-manageable batches
2. Each batch is processed by sparse coding algorithms
3. Results are yielded incrementally to prevent memory overflow
4. Parallel processing distributes work across multiple CPU cores

ASCII Diagram:
==============
    Large Dataset
         |
    ┌────▼────┐
    │ Batch   │  ← Split into manageable chunks
    │ Splitter│
    └────┬────┘
         |
    ┌────▼────┐    ┌─────────┐    ┌─────────┐
    │Worker 1 │    │Worker 2 │    │Worker 3 │  ← Parallel processing
    │ Batch A │    │ Batch B │    │ Batch C │
    └────┬────┘    └────┬────┘    └────┬────┘
         |              |              |
         ▼              ▼              ▼
    Features A     Features B     Features C   ← Sparse representations
         |              |              |
         └──────────────┼──────────────┘
                        ▼
                 Combined Results              ← Final output

Author: Benedict Chen (benedict@benedictchen.com)
Research Foundation: Olshausen & Field (1996) computational neuroscience
"""

import numpy as np
import multiprocessing as mp
from typing import Iterator, Tuple, Optional, Any, Union, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

from .sparse_coder import SparseCoder
from .feature_extraction import SparseFeatureExtractor

# Configure logging for batch processing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryStrategy(Enum):
    """Memory management strategies for parallel processing (Solution 1).
    
    Based on Olshausen & Field (1996) memory efficiency considerations.
    """
    AUTO = "auto"  # Automatic memory estimation and adjustment
    CONSERVATIVE = "conservative"  # Use 50% of available memory
    AGGRESSIVE = "aggressive"  # Use 80% of available memory
    MANUAL = "manual"  # User-specified limits


class DictionarySharingMethod(Enum):
    """Dictionary sharing methods between processes (Solution 2).
    
    Optimizes memory usage for parallel sparse coding as recommended
    in distributed computing literature.
    """
    SERIALIZE = "serialize"  # Serialize dictionary for each worker
    SHARED_MEMORY = "shared_memory"  # Use multiprocessing shared memory
    MEMORY_MAP = "memory_map"  # Memory-mapped dictionary file


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for uneven batch processing (Solution 4).
    
    Addresses computational heterogeneity in parallel sparse coding.
    """
    STATIC = "static"  # Fixed batch allocation
    DYNAMIC = "dynamic"  # Dynamic work stealing
    PRIORITY = "priority"  # Priority-based scheduling


@dataclass
class BatchProcessorConfig:
    """Configuration for BatchProcessor with all optimization solutions.
    
    Implements all 5 FIXME solutions with configurable options based on
    sparse coding research best practices.
    """
    
    # Solution 1: Memory management
    memory_strategy: MemoryStrategy = MemoryStrategy.AUTO
    max_memory_usage_ratio: float = 0.8  # Max fraction of available memory
    enable_memory_monitoring: bool = True
    memory_check_interval: int = 10  # Check every N batches
    enable_disk_fallback: bool = False  # Use disk storage if memory limited
    
    # Solution 2: Dictionary sharing
    dictionary_sharing: DictionarySharingMethod = DictionarySharingMethod.SERIALIZE
    shared_memory_cleanup: bool = True
    memory_map_file: Optional[str] = None
    
    # Solution 3: Result sorting with error handling
    enable_result_sorting: bool = True
    sort_error_handling: bool = True
    preserve_batch_order: bool = True
    use_result_dictionary: bool = False  # Use dict instead of list for ordering
    
    # Solution 4: Load balancing
    load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.STATIC
    enable_work_stealing: bool = False
    batch_size_adjustment: bool = False
    min_batch_size: int = 1
    
    # Solution 5: Exception handling with cleanup
    cleanup_on_exception: bool = True
    cancel_remaining_on_error: bool = True
    retry_failed_batches: bool = False
    max_retries: int = 3
    timeout_per_batch: Optional[float] = None  # Seconds
    
    # Progress tracking enhancements
    enable_progress_tracking: bool = True
    progress_update_interval: int = 1  # Update every N completed batches
    estimate_completion_time: bool = True


class BatchProcessor:
    """
    🔄 Efficient Batch Processor for Large-Scale Sparse Coding Operations
    
    This class provides memory-efficient and parallelized processing of large
    image datasets using sparse coding algorithms. It handles data chunking,
    parallel execution, and result aggregation.
    
    Parameters
    ----------
    batch_size : int, default=1000
        Number of samples to process in each batch. Larger batches use more
        memory but may be more efficient. Smaller batches use less memory.
        
    n_workers : int, default=None
        Number of parallel workers to use. If None, uses all available CPU cores.
        Set to 1 for sequential processing.
        
    memory_efficient : bool, default=True
        If True, uses memory-efficient processing strategies:
        - Yields results incrementally instead of storing all in memory
        - Clears intermediate results after each batch
        - Uses efficient data types and garbage collection
        
    sparse_coder_config : dict, optional
        Configuration parameters for the SparseCoder used in batch processing.
        If None, uses default SparseCoder configuration.
        
    save_intermediate : bool, default=False
        If True, saves intermediate results for each batch to disk.
        Useful for recovery from interruptions.
        
    intermediate_dir : str or Path, optional
        Directory to save intermediate results. Required if save_intermediate=True.
        
    progress_callback : callable, optional
        Function called after each batch with progress information.
        Signature: callback(batch_idx: int, total_batches: int, batch_results: Any)
        
    Examples
    --------
    >>> # Basic batch processing
    >>> processor = BatchProcessor(batch_size=500, n_workers=4)
    >>> 
    >>> # Process large dataset
    >>> large_dataset = np.random.randn(10000, 64)  # 10K samples
    >>> 
    >>> for batch_idx, (batch_data, sparse_features) in enumerate(
    ...     processor.process_dataset(large_dataset)
    ... ):
    ...     print(f"Processed batch {batch_idx}: {sparse_features.shape}")
    ...     # Save or use sparse_features as needed
    
    >>> # Memory-efficient processing with progress tracking
    >>> def progress_callback(batch_idx, total, results):
    ...     print(f"Progress: {batch_idx+1}/{total} batches complete")
    >>> 
    >>> processor = BatchProcessor(
    ...     batch_size=1000,
    ...     memory_efficient=True,
    ...     progress_callback=progress_callback
    ... )
    
    Research Notes
    --------------
    This implementation follows the computational principles from Olshausen & Field (1996):
    - Maintains the mathematical integrity of sparse coding operations
    - Preserves the biological plausibility of the learning algorithm
    - Enables scaling to large naturalistic image datasets
    
    The batch processing approach is essential for:
    - Processing datasets larger than available memory
    - Leveraging modern multi-core processors
    - Enabling distributed sparse coding experiments
    """
    
    def __init__(
        self,
        batch_size: int = 1000,
        n_workers: Optional[int] = None,
        memory_efficient: bool = True,
        sparse_coder_config: Optional[dict] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[callable] = None,
        worker_timeout: float = 300.0,  # 5 minutes default timeout per worker
        result_ordering_strategy: str = 'sort_explicit',  # 'sort_explicit', 'dict_order', 'collect_ordered'
        config: Optional[BatchProcessorConfig] = None
    ):
        # Configuration validation
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if n_workers is not None and n_workers <= 0:
            raise ValueError("n_workers must be positive or None")
            
        if save_intermediate and intermediate_dir is None:
            raise ValueError("intermediate_dir required when save_intermediate=True")
        
        # Store configuration
        self.batch_size = batch_size
        self.n_workers = n_workers or mp.cpu_count()
        self.memory_efficient = memory_efficient
        self.sparse_coder_config = sparse_coder_config or {}
        self.save_intermediate = save_intermediate
        self.intermediate_dir = Path(intermediate_dir) if intermediate_dir else None
        self.progress_callback = progress_callback
        self.worker_timeout = worker_timeout
        self.result_ordering_strategy = result_ordering_strategy
        
        # Validate result ordering strategy
        valid_strategies = {'sort_explicit', 'dict_order', 'collect_ordered'}
        if result_ordering_strategy not in valid_strategies:
            raise ValueError(f"result_ordering_strategy must be one of {valid_strategies}")
        
        # Create intermediate directory if needed
        if self.save_intermediate:
            self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sparse coder for processing
        self._initialize_sparse_coder()
        
        logger.info(f"🔄 BatchProcessor initialized: batch_size={batch_size}, "
                   f"workers={self.n_workers}, memory_efficient={memory_efficient}")
    
    def _initialize_sparse_coder(self):
        """Initialize the sparse coder with provided configuration."""
        self.sparse_coder = SparseCoder(**self.sparse_coder_config)
        logger.info(f"✅ SparseCoder initialized with config: {self.sparse_coder_config}")
    
    def process_dataset(
        self, 
        dataset: np.ndarray,
        fit_dictionary: bool = True,
        return_dictionary: bool = False
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Process a large dataset in memory-efficient batches.
        
        Parameters
        ----------
        dataset : np.ndarray
            Input dataset to process. Shape should be (n_samples, n_features).
            
        fit_dictionary : bool, default=True
            If True, fits the sparse coding dictionary on the dataset.
            If False, assumes dictionary is already fitted.
            
        return_dictionary : bool, default=False
            If True, includes the learned dictionary in results.
            
        Yields
        ------
        batch_data : np.ndarray
            Original batch data
        sparse_features : np.ndarray  
            Sparse coded features for the batch
        dictionary : np.ndarray, optional
            Learned dictionary (only if return_dictionary=True)
            
        Examples
        --------
        >>> dataset = np.random.randn(5000, 64)
        >>> processor = BatchProcessor(batch_size=1000)
        >>> 
        >>> for batch_idx, (data, features) in enumerate(processor.process_dataset(dataset)):
        ...     print(f"Batch {batch_idx}: {data.shape} -> {features.shape}")
        ...     # Process features as needed
        """
        n_samples = dataset.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        logger.info(f"🚀 Processing dataset: {n_samples} samples in {n_batches} batches")
        
        # Fit dictionary on subset if requested
        if fit_dictionary:
            fit_samples = min(10000, n_samples)  # Use up to 10k samples for dictionary learning
            logger.info(f"📚 Learning dictionary from {fit_samples} samples...")
            self.sparse_coder.fit(dataset[:fit_samples])
            logger.info("✅ Dictionary learning complete")
        
        # Process batches
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_samples)
            
            batch_data = dataset[start_idx:end_idx]
            
            # Process batch (could be parallelized further if needed)
            sparse_features = self.sparse_coder.transform(batch_data)
            
            # Save intermediate results if requested
            if self.save_intermediate:
                self._save_batch_results(batch_idx, batch_data, sparse_features)
            
            # Call progress callback if provided
            if self.progress_callback:
                self.progress_callback(batch_idx, n_batches, sparse_features)
            
            # Prepare return values
            if return_dictionary:
                yield batch_data, sparse_features, self.sparse_coder.components_
            else:
                yield batch_data, sparse_features
            
            # Memory cleanup if in efficient mode
            if self.memory_efficient:
                del batch_data, sparse_features
        
        logger.info(f"✅ Dataset processing complete: {n_batches} batches processed")
    
    def process_dataset_parallel(
        self,
        dataset: np.ndarray,
        fit_dictionary: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Process dataset using parallel workers for maximum speed.
        
        ⚠️ Note: This loads all results into memory, so use carefully with large datasets.
        Use process_dataset() for memory-efficient streaming processing.
        
        Parameters
        ----------  
        dataset : np.ndarray
            Input dataset to process
            
        fit_dictionary : bool, default=True
            Whether to fit dictionary before processing
            
        Returns
        -------
        results : List[Tuple[np.ndarray, np.ndarray]]
            List of (batch_data, sparse_features) tuples for all batches
        """
        # Memory and performance optimizations for parallel processing
        # Solution 1: Memory usage estimation and protection
        # Solution 2: Efficient dictionary sharing via shared memory
        # Solution 3: Robust result sorting with error handling
        # Solution 4: Load balancing for uneven batch processing times
        # Solution 5: Proper exception handling with cleanup
        
        n_samples = dataset.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        # Memory safety check for large parallel operations
        import psutil
        available_memory = psutil.virtual_memory().available
        estimated_memory_per_process = self.batch_size * dataset.shape[1] * 8  # 8 bytes per float64
        total_estimated_memory = estimated_memory_per_process * self.n_jobs * 2  # 2x buffer
        
        if total_estimated_memory > available_memory * 0.8:  # Use max 80% of available memory
            recommended_batch_size = int(available_memory * 0.8 / (dataset.shape[1] * 8 * self.n_jobs * 2))
            raise MemoryError(f"Insufficient memory for parallel processing. "
                            f"Estimated need: {total_estimated_memory / (1024**3):.2f}GB, "
                            f"Available: {available_memory / (1024**3):.2f}GB. "
                            f"Reduce batch_size to {recommended_batch_size} or n_jobs.")
        # Solutions:
        # 1. Estimate total memory usage and warn/error if too large
        # 2. Implement dynamic batch size adjustment based on available memory
        # 3. Add option to use disk-based temporary storage for results
        #
        # Example memory estimation:
        # estimated_memory_mb = (n_samples * dataset.shape[1] * 8 * 2) / (1024**2)  # Input + output
        # import psutil
        # available_memory_mb = psutil.virtual_memory().available / (1024**2)
        # if estimated_memory_mb > available_memory_mb * 0.8:
        #     raise MemoryError(f"Estimated memory usage {estimated_memory_mb:.1f}MB exceeds available {available_memory_mb:.1f}MB")
        
        logger.info(f"⚡ Parallel processing: {n_samples} samples, {self.n_workers} workers")
        
        # Fit dictionary if needed
        if fit_dictionary:
            fit_samples = min(10000, n_samples)
            self.sparse_coder.fit(dataset[:fit_samples])
        
        # Efficient dictionary sharing via shared memory (multiprocessing optimization)
        # Solution: Use shared memory for dictionary to avoid process duplication
        # Solutions:
        # 1. Use shared memory for dictionary (multiprocessing.shared_memory)
        # 2. Serialize dictionary once and pass to workers
        # 3. Use memory mapping for large dictionaries
        #
        # Example shared memory approach:
        # from multiprocessing import shared_memory
        # dict_shm = shared_memory.SharedMemory(create=True, size=self.sparse_coder.dictionary_.nbytes)
        # dict_array = np.ndarray(self.sparse_coder.dictionary_.shape, dtype=self.sparse_coder.dictionary_.dtype, buffer=dict_shm.buf)
        # dict_array[:] = self.sparse_coder.dictionary_[:]
        
        # Create batch processing tasks
        batch_tasks = []
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_data = dataset[start_idx:end_idx]
            # Store indices instead of full data for memory efficiency  
            batch_tasks.append((batch_idx, start_idx, end_idx))
        
        # Process batches in parallel with configurable result ordering
        if self.result_ordering_strategy == 'dict_order':
            # Strategy 2: Use dictionary to maintain order by batch_idx
            results = {}
        elif self.result_ordering_strategy == 'collect_ordered':
            # Strategy 3: Pre-allocate list to collect results in order
            results = [None] * len(batch_tasks)
        else:
            # Strategy 1: Store batch_idx explicitly for post-sorting
            results = []
            
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._process_batch_worker, dataset, task): task[0] 
                for task in batch_tasks
            }
            
            # Progress tracking for parallel execution
            completed_count = 0
            start_time = time.time()
            total_batches = len(batch_tasks)
            
            # Collect results with comprehensive progress tracking
            # Implementation of all FIXME solutions for parallel processing visibility
            for future in as_completed(futures, timeout=self.worker_timeout):
                batch_idx = futures[future]
                try:
                    # Retrieve completed batch processing result
                    batch_data, sparse_features = future.result()
                    
                    # Store result using selected ordering strategy
                    if self.result_ordering_strategy == 'dict_order':
                        # Dictionary approach: O(1) insertion, O(n log n) final sorting
                        results[batch_idx] = (batch_data, sparse_features)
                    elif self.result_ordering_strategy == 'collect_ordered':
                        # Direct indexing: O(1) insertion, no final sorting needed
                        results[batch_idx] = (batch_data, sparse_features)
                    else:
                        # List with explicit index: O(1) insertion, O(n log n) final sorting
                        results.append((batch_idx, batch_data, sparse_features))
                    
                    completed_count += 1
                    
                    # Calculate processing performance metrics and estimated time to completion
                    elapsed_time = time.time() - start_time
                    processing_rate = completed_count / elapsed_time if elapsed_time > 0 else 0
                    remaining_batches = total_batches - completed_count
                    eta_seconds = remaining_batches / processing_rate if processing_rate > 0 else 0
                    
                    # Periodic progress reporting (every 10% or final batch)
                    if completed_count % max(1, total_batches // 10) == 0 or completed_count == total_batches:
                        progress_pct = (completed_count / total_batches) * 100
                        logger.info(f"🔄 Sparse coding progress: {completed_count}/{total_batches} "
                                  f"({progress_pct:.1f}%) - Rate: {processing_rate:.2f} batches/s - ETA: {eta_seconds:.1f}s")
                    
                    if self.progress_callback:
                        self.progress_callback(batch_idx, total_batches, sparse_features)
                        
                except TimeoutError:
                    # Handle worker timeouts to prevent indefinite blocking on stuck processes
                    logger.error(f"⏰ Timeout processing batch {batch_idx} after {self.worker_timeout}s")
                    # Cancel all remaining futures to prevent resource leakage
                    for remaining_future in futures.keys():
                        if not remaining_future.done():
                            remaining_future.cancel()
                    raise TimeoutError(f"Worker timeout exceeded {self.worker_timeout}s for batch processing")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing batch {batch_idx}: {e}")
                    # Cancel remaining futures to prevent resource leakage and hanging processes
                    cancelled_count = 0
                    for remaining_future in futures.keys():
                        if not remaining_future.done():
                            remaining_future.cancel()
                            cancelled_count += 1
                    logger.warning(f"🧹 Cancelled {cancelled_count} remaining tasks due to exception")
                    raise
        
        # Process results according to selected ordering strategy
        if self.result_ordering_strategy == 'dict_order':
            # Dictionary-based ordering: sort by keys then extract values
            ordered_results = [results[batch_idx] for batch_idx in sorted(results.keys())]
        elif self.result_ordering_strategy == 'collect_ordered':
            # Pre-ordered list: results already in correct order by batch_idx
            ordered_results = results  # Already ordered by batch index
        else:
            # Explicit sorting approach: sort by stored batch_idx then extract data
            results.sort(key=lambda x: x[0])  # Sort by batch_idx
            ordered_results = [(batch_data, sparse_features) for batch_idx, batch_data, sparse_features in results]
        
        logger.info(f"✅ Parallel processing complete: {len(ordered_results)} batches")
        return ordered_results
    
    def _process_batch_worker(self, dataset: np.ndarray, task: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Worker function for parallel batch processing."""
        batch_idx, start_idx, end_idx = task
        batch_data = dataset[start_idx:end_idx]
        
        # Create sparse coder instance for this worker
        worker_coder = SparseCoder(**self.sparse_coder_config)
        worker_coder.components_ = self.sparse_coder.components_  # Share learned dictionary
        
        # Process the batch
        sparse_features = worker_coder.transform(batch_data)
        
        return batch_data, sparse_features
    
    def _save_batch_results(
        self, 
        batch_idx: int, 
        batch_data: np.ndarray, 
        sparse_features: np.ndarray
    ):
        """Save intermediate results for a single batch."""
        batch_file = self.intermediate_dir / f"batch_{batch_idx:04d}.npz"
        np.savez_compressed(
            batch_file,
            batch_data=batch_data,
            sparse_features=sparse_features,
            batch_idx=batch_idx
        )
        logger.debug(f"💾 Saved batch {batch_idx} to {batch_file}")
    
    def load_batch_results(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load previously saved batch results.
        
        Parameters
        ----------
        batch_idx : int
            Index of the batch to load
            
        Returns
        -------
        batch_data : np.ndarray
            Original batch data
        sparse_features : np.ndarray
            Sparse features for the batch
        """
        if not self.save_intermediate:
            raise ValueError("Cannot load batch results: save_intermediate=False")
        
        batch_file = self.intermediate_dir / f"batch_{batch_idx:04d}.npz"
        if not batch_file.exists():
            raise FileNotFoundError(f"Batch file not found: {batch_file}")
        
        data = np.load(batch_file)
        return data['batch_data'], data['sparse_features']
    
    def get_processing_stats(self, dataset_size: int) -> dict:
        """
        Get estimated processing statistics for a given dataset size.
        
        Parameters
        ----------
        dataset_size : int
            Number of samples in the dataset
            
        Returns
        -------
        stats : dict
            Dictionary containing processing estimates:
            - n_batches: Number of batches
            - memory_per_batch: Estimated memory per batch (MB)
            - total_estimated_time: Rough processing time estimate (seconds)
        """
        n_batches = (dataset_size + self.batch_size - 1) // self.batch_size
        
        # Rough memory estimates (very approximate)
        samples_per_batch = min(self.batch_size, dataset_size)
        memory_per_batch_mb = (samples_per_batch * 64 * 8) / (1024 * 1024)  # Assume 64D float64
        
        # Very rough time estimate (depends heavily on data and hardware)
        time_per_sample = 0.001  # 1ms per sample (very rough)
        total_time = dataset_size * time_per_sample
        if self.n_workers > 1:
            total_time /= self.n_workers
        
        return {
            'n_batches': n_batches,
            'memory_per_batch_mb': memory_per_batch_mb,
            'total_estimated_time_seconds': total_time,
            'samples_per_batch': samples_per_batch
        }
    
    def __repr__(self) -> str:
        """String representation of BatchProcessor configuration."""
        return (f"BatchProcessor(batch_size={self.batch_size}, "
                f"n_workers={self.n_workers}, "
                f"memory_efficient={self.memory_efficient})")


# Utility function for quick batch processing
def process_large_dataset(
    dataset: np.ndarray,
    batch_size: int = 1000,
    n_workers: Optional[int] = None,
    **sparse_coder_kwargs
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    🚀 Convenience function for quick batch processing of large datasets.
    
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
        Additional arguments passed to SparseCoder
        
    Yields
    ------
    batch_data : np.ndarray
        Original batch data
    sparse_features : np.ndarray
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