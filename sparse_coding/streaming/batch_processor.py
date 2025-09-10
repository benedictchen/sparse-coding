"""
Batch processing utilities for streaming sparse coding.

Provides efficient batch processing, parallel execution, and
progress tracking for large-scale dictionary learning.
"""

import numpy as np
from typing import Iterator, Callable, Dict, Any, Optional, Union
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from ..core.array import ArrayLike, ensure_array


@dataclass
class BatchConfig:
    """
    Configuration for batch processing.
    
    Attributes
    ----------
    batch_size : int
        Number of samples per batch
    n_workers : int
        Number of parallel workers (0 = sequential)
    prefetch_batches : int
        Number of batches to prefetch for efficiency
    shuffle : bool
        Whether to shuffle data between epochs
    drop_last : bool
        Whether to drop incomplete final batch
    """
    batch_size: int = 256
    n_workers: int = 0
    prefetch_batches: int = 2
    shuffle: bool = True
    drop_last: bool = False


class BatchProcessor:
    """
    High-performance batch processor for streaming sparse coding.
    
    Provides parallel batch processing with prefetching, progress tracking,
    and automatic memory management for large-scale datasets.
    
    Parameters
    ----------
    config : BatchConfig
        Batch processing configuration
    verbose : bool, default=True
        Whether to show progress information
    """
    
    def __init__(self, config: BatchConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self._stats = {
            'batches_processed': 0,
            'total_samples': 0,
            'processing_time': 0.0,
            'throughput_samples_per_sec': 0.0
        }
    
    def process_dataset(
        self,
        dataset,  # MemoryMappedDataset or array-like
        process_fn: Callable[[ArrayLike], Any],
        n_epochs: int = 1
    ) -> Iterator[Any]:
        """
        Process dataset in batches using provided function.
        
        Parameters
        ----------
        dataset : MemoryMappedDataset or ArrayLike
            Dataset to process
        process_fn : callable
            Function to apply to each batch: batch -> result
        n_epochs : int, default=1
            Number of epochs to process
            
        Yields
        ------
        results : Any
            Results from process_fn applied to each batch
        """
        start_time = time.time()
        
        for epoch in range(n_epochs):
            if self.verbose and n_epochs > 1:
                print(f"Epoch {epoch + 1}/{n_epochs}")
            
            if self.config.n_workers > 0:
                # Parallel processing
                yield from self._process_parallel(dataset, process_fn)
            else:
                # Sequential processing
                yield from self._process_sequential(dataset, process_fn)
        
        # Update stats
        total_time = time.time() - start_time
        self._stats['processing_time'] += total_time
        if self._stats['total_samples'] > 0:
            self._stats['throughput_samples_per_sec'] = (
                self._stats['total_samples'] / self._stats['processing_time']
            )
    
    def _process_sequential(self, dataset, process_fn: Callable) -> Iterator[Any]:
        """Process batches sequentially."""
        for batch in self._create_batches(dataset):
            self._stats['batches_processed'] += 1
            self._stats['total_samples'] += batch.shape[0]
            
            result = process_fn(batch)
            yield result
            
            if self.verbose and self._stats['batches_processed'] % 10 == 0:
                self._print_progress()
    
    def _process_parallel(self, dataset, process_fn: Callable) -> Iterator[Any]:
        """Process batches in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            # Submit initial batch of futures
            futures = {}
            batch_iter = self._create_batches(dataset)
            
            # Fill initial queue
            for _ in range(self.config.prefetch_batches):
                try:
                    batch = next(batch_iter)
                    future = executor.submit(process_fn, batch)
                    futures[future] = batch.shape[0]
                except StopIteration:
                    break
            
            # Process and refill
            while futures:
                # Get completed futures
                for future in as_completed(futures):
                    batch_size = futures.pop(future)
                    
                    self._stats['batches_processed'] += 1
                    self._stats['total_samples'] += batch_size
                    
                    result = future.result()
                    yield result
                    
                    # Submit new batch if available
                    try:
                        batch = next(batch_iter)
                        new_future = executor.submit(process_fn, batch)
                        futures[new_future] = batch.shape[0]
                    except StopIteration:
                        pass
                    
                    if self.verbose and self._stats['batches_processed'] % 10 == 0:
                        self._print_progress()
                    
                    break  # Process one result at a time to maintain order
    
    def _create_batches(self, dataset) -> Iterator[ArrayLike]:
        """Create batches from dataset."""
        if hasattr(dataset, 'iter_chunks'):
            # Memory-mapped dataset
            chunk_size = getattr(dataset, 'chunk_size', self.config.batch_size)
            
            if chunk_size == self.config.batch_size:
                # Chunks match batch size exactly
                for chunk in dataset.iter_chunks():
                    yield ensure_array(chunk)
            else:
                # Need to rebatch chunks
                buffer = []
                buffer_size = 0
                
                for chunk in dataset.iter_chunks():
                    chunk_array = ensure_array(chunk)
                    buffer.append(chunk_array)
                    buffer_size += chunk_array.shape[0]
                    
                    # Yield complete batches from buffer
                    while buffer_size >= self.config.batch_size:
                        combined = np.vstack(buffer)
                        
                        # Extract batch
                        batch = combined[:self.config.batch_size]
                        yield batch
                        
                        # Update buffer
                        remaining = combined[self.config.batch_size:]
                        buffer = [remaining] if remaining.shape[0] > 0 else []
                        buffer_size = remaining.shape[0] if len(buffer) > 0 else 0
                
                # Handle final incomplete batch
                if buffer and not self.config.drop_last:
                    final_batch = np.vstack(buffer)
                    if final_batch.shape[0] > 0:
                        yield final_batch
        
        else:
            # Regular array-like dataset
            dataset_array = ensure_array(dataset)
            n_samples = dataset_array.shape[0]
            
            # Create indices
            indices = np.arange(n_samples)
            if self.config.shuffle:
                np.random.shuffle(indices)
            
            # Generate batches
            for i in range(0, n_samples, self.config.batch_size):
                end_idx = min(i + self.config.batch_size, n_samples)
                
                if self.config.drop_last and end_idx - i < self.config.batch_size:
                    break
                
                batch_indices = indices[i:end_idx]
                batch = dataset_array[batch_indices]
                yield ensure_array(batch)
    
    def _print_progress(self):
        """Print processing progress."""
        elapsed = self._stats['processing_time']
        throughput = self._stats['throughput_samples_per_sec']
        
        print(f"  Processed {self._stats['batches_processed']} batches, "
              f"{self._stats['total_samples']:,} samples "
              f"({throughput:.0f} samples/sec)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        self._stats = {
            'batches_processed': 0,
            'total_samples': 0,
            'processing_time': 0.0,
            'throughput_samples_per_sec': 0.0
        }


def process_in_batches(
    data: ArrayLike,
    process_fn: Callable[[ArrayLike], Any],
    batch_size: int = 256,
    n_workers: int = 0,
    verbose: bool = True
) -> list:
    """
    Simple batch processing function.
    
    Parameters
    ----------
    data : ArrayLike, shape (n_samples, n_features)
        Data to process
    process_fn : callable
        Function to apply to each batch
    batch_size : int, default=256
        Number of samples per batch
    n_workers : int, default=0
        Number of parallel workers (0 = sequential)
    verbose : bool, default=True
        Whether to show progress
        
    Returns
    -------
    results : list
        List of results from process_fn
        
    Examples
    --------
    >>> import numpy as np
    >>> from sparse_coding.streaming import process_in_batches
    >>> 
    >>> # Create data
    >>> X = np.random.randn(1000, 50)
    >>> 
    >>> # Define processing function
    >>> def compute_norms(batch):
    >>>     return np.linalg.norm(batch, axis=1)
    >>> 
    >>> # Process in batches
    >>> results = process_in_batches(X, compute_norms, batch_size=100)
    >>> all_norms = np.concatenate(results)
    """
    config = BatchConfig(
        batch_size=batch_size,
        n_workers=n_workers,
        shuffle=False,
        drop_last=False
    )
    
    processor = BatchProcessor(config, verbose=verbose)
    results = list(processor.process_dataset(data, process_fn, n_epochs=1))
    
    if verbose:
        stats = processor.get_stats()
        print(f"Completed processing: {stats['total_samples']:,} samples in "
              f"{stats['processing_time']:.2f}s ({stats['throughput_samples_per_sec']:.0f} samples/sec)")
    
    return results


class OnlineMetrics:
    """
    Online computation of dataset metrics during batch processing.
    
    Computes running statistics (mean, variance, etc.) without storing
    all data in memory, useful for large datasets.
    """
    
    def __init__(self):
        self.n_samples = 0
        self.mean = None
        self.m2 = None  # Sum of squared differences from mean
        self.min_vals = None
        self.max_vals = None
    
    def update(self, batch: ArrayLike):
        """Update statistics with new batch."""
        batch_array = ensure_array(batch)
        batch_size, n_features = batch_array.shape
        
        if self.mean is None:
            # Initialize
            self.mean = np.zeros(n_features)
            self.m2 = np.zeros(n_features)
            self.min_vals = np.full(n_features, np.inf)
            self.max_vals = np.full(n_features, -np.inf)
        
        # Update sample count
        old_n = self.n_samples
        self.n_samples += batch_size
        
        # Update mean using Welford's online algorithm
        batch_mean = np.mean(batch_array, axis=0)
        delta = batch_mean - self.mean
        self.mean += delta * batch_size / self.n_samples
        
        # Update M2 (for variance calculation)
        delta2 = batch_mean - self.mean
        self.m2 += np.sum((batch_array - batch_mean)**2, axis=0) + old_n * delta * delta2
        
        # Update min/max
        self.min_vals = np.minimum(self.min_vals, np.min(batch_array, axis=0))
        self.max_vals = np.maximum(self.max_vals, np.max(batch_array, axis=0))
    
    def get_stats(self) -> Dict[str, np.ndarray]:
        """Get current statistics."""
        if self.n_samples == 0:
            return {}
        
        variance = self.m2 / (self.n_samples - 1) if self.n_samples > 1 else np.zeros_like(self.mean)
        std = np.sqrt(variance)
        
        return {
            'n_samples': self.n_samples,
            'mean': self.mean.copy(),
            'std': std,
            'variance': variance,
            'min': self.min_vals.copy(),
            'max': self.max_vals.copy()
        }


def compute_dataset_stats(
    dataset,
    batch_size: int = 1000,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute statistics for large dataset using online algorithms.
    
    Parameters
    ----------
    dataset : MemoryMappedDataset or ArrayLike
        Dataset to analyze
    batch_size : int, default=1000
        Batch size for processing
    verbose : bool, default=True
        Whether to show progress
        
    Returns
    -------
    stats : dict
        Dictionary with mean, std, min, max, etc.
    """
    metrics = OnlineMetrics()
    
    def update_metrics(batch):
        metrics.update(batch)
        return None  # Don't need to return anything
    
    # Process dataset
    config = BatchConfig(batch_size=batch_size, shuffle=False)
    processor = BatchProcessor(config, verbose=verbose)
    
    list(processor.process_dataset(dataset, update_metrics, n_epochs=1))
    
    return metrics.get_stats()