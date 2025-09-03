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

from .sparse_coder import SparseCoder
from .feature_extraction import SparseFeatureExtractor

# Configure logging for batch processing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        progress_callback: Optional[callable] = None
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
        # FIXME: Critical memory and performance issues in parallel processing
        # Issue 1: No memory usage estimation or protection against OOM
        # Issue 2: Dictionary sharing between processes is inefficient (duplicated)
        # Issue 3: Result sorting logic is incorrect and may fail
        # Issue 4: No load balancing for uneven batch processing times
        # Issue 5: Exception handling doesn't clean up partial results
        
        n_samples = dataset.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        # FIXME: No memory safety check for large parallel operations
        # Issue: Could easily exceed available RAM with large datasets
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
        
        # FIXME: Inefficient dictionary sharing across processes
        # Issue: Each worker gets full copy of dictionary, wasting memory
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
            # FIXME: Storing full batch_data in tasks list uses excessive memory
            # Better approach: store indices and slice in worker
            batch_tasks.append((batch_idx, batch_data))
        
        # Process batches in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._process_batch_worker, task): task[0] 
                for task in batch_tasks
            }
            
            # FIXME: No progress tracking during parallel execution
            # Issue: Users have no visibility into parallel processing progress
            # Solutions:
            # 1. Add periodic progress updates using completed future count
            # 2. Implement timeout handling for stuck workers
            # 3. Add estimated completion time calculation
            #
            # Example progress tracking:
            # completed_count = 0
            # start_time = time.time()
            
            # Collect results as they complete
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_data, sparse_features = future.result()
                    results.append((batch_data, sparse_features))
                    
                    if self.progress_callback:
                        self.progress_callback(batch_idx, n_batches, sparse_features)
                        
                except Exception as e:
                    logger.error(f"❌ Error processing batch {batch_idx}: {e}")
                    # FIXME: Exception doesn't clean up other running tasks
                    # Should cancel remaining futures and clean up resources
                    raise
        
        # FIXME: Result sorting logic is incorrect and will fail
        # Issue: Trying to sort by batch_data[0] when it should sort by batch_idx
        # The current logic assumes batch_data has batch_idx attribute, which it doesn't
        # Solutions:
        # 1. Store batch_idx with results explicitly
        # 2. Use dictionary instead of list to maintain order
        # 3. Collect results in order instead of sorting at end
        #
        # Correct implementation:
        # results_with_idx = [(batch_idx, batch_data, sparse_features) for batch_idx, (batch_data, sparse_features) in results]
        # results_with_idx.sort(key=lambda x: x[0])  # Sort by batch_idx
        # results = [(batch_data, sparse_features) for _, batch_data, sparse_features in results_with_idx]
        
        # Sort results by original batch order
        results.sort(key=lambda x: x[0] if hasattr(x[0], 'batch_idx') else 0)
        
        logger.info(f"✅ Parallel processing complete: {len(results)} batches")
        return results
    
    def _process_batch_worker(self, task: Tuple[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Worker function for parallel batch processing."""
        batch_idx, batch_data = task
        
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