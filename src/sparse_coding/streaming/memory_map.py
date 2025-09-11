"""
Memory-mapped dataset utilities for large-scale sparse coding.

Provides efficient access to datasets that don't fit in memory,
with support for lazy loading and chunked processing.
"""

import numpy as np
from typing import Iterator, Tuple, Optional, Union
from pathlib import Path
import warnings

from ..core.array import ArrayLike, ensure_array


class MemoryMappedDataset:
    """
    Memory-mapped dataset for efficient large-scale data access.
    
    Provides numpy.memmap-based access to large datasets stored on disk
    with automatic chunking and lazy loading capabilities.
    
    Parameters
    ----------
    data_path : str or Path
        Path to the memory-mapped data file (.npy or .dat)
    shape : tuple
        Shape of the full dataset (n_samples, n_features)
    dtype : numpy.dtype, default=np.float32
        Data type of the stored data
    mode : str, default='r'
        File access mode ('r', 'r+', 'w+', 'c')
    chunk_size : int, default=1000
        Number of samples to load per chunk
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        shape: Tuple[int, int],
        dtype: np.dtype = np.float32,
        mode: str = 'r',
        chunk_size: int = 1000
    ):
        self.data_path = Path(data_path)
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.chunk_size = chunk_size
        
        # Create memory map
        try:
            self.data = np.memmap(
                self.data_path,
                dtype=dtype,
                mode=mode,
                shape=shape
            )
        except Exception as e:
            raise IOError(f"Failed to create memory map for {data_path}: {e}")
        
        self.n_samples, self.n_features = shape
        self.n_chunks = int(np.ceil(self.n_samples / chunk_size))
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples
    
    def __getitem__(self, idx: Union[int, slice, np.ndarray]) -> ArrayLike:
        """Get samples by index."""
        return ensure_array(self.data[idx])
    
    def get_chunk(self, chunk_idx: int) -> ArrayLike:
        """
        Get a specific chunk of data.
        
        Parameters
        ----------
        chunk_idx : int
            Index of chunk to retrieve (0-based)
            
        Returns
        -------
        chunk : ArrayLike, shape (chunk_size, n_features)
            Data chunk (may be smaller for last chunk)
        """
        if chunk_idx >= self.n_chunks:
            raise IndexError(f"Chunk index {chunk_idx} out of range [0, {self.n_chunks})")
        
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.n_samples)
        
        return ensure_array(self.data[start_idx:end_idx])
    
    def iter_chunks(self) -> Iterator[ArrayLike]:
        """
        Iterator over data chunks.
        
        Yields
        ------
        chunk : ArrayLike, shape (chunk_size, n_features)
            Sequential data chunks
        """
        for chunk_idx in range(self.n_chunks):
            yield self.get_chunk(chunk_idx)
    
    def get_sample(self, idx: int) -> ArrayLike:
        """Get a single sample."""
        if idx >= self.n_samples:
            raise IndexError(f"Sample index {idx} out of range [0, {self.n_samples})")
        
        return ensure_array(self.data[idx])
    
    def get_batch(self, indices: np.ndarray) -> ArrayLike:
        """
        Get multiple samples by indices.
        
        Parameters
        ----------
        indices : np.ndarray
            Array of sample indices
            
        Returns
        -------
        batch : ArrayLike, shape (len(indices), n_features)
            Selected samples
        """
        return ensure_array(self.data[indices])
    
    def random_batch(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> ArrayLike:
        """
        Get random batch of samples.
        
        Parameters
        ----------
        batch_size : int
            Number of samples to return
        rng : np.random.Generator, optional
            Random number generator. If None, uses default.
            
        Returns
        -------
        batch : ArrayLike, shape (batch_size, n_features)
            Random samples
        """
        if rng is None:
            rng = np.random.default_rng()
        
        indices = rng.choice(self.n_samples, size=batch_size, replace=False)
        return self.get_batch(indices)
    
    def compute_stats(self, n_samples: Optional[int] = None) -> dict:
        """
        Compute dataset statistics using chunked processing.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to use for statistics. If None, uses all data.
            
        Returns
        -------
        stats : dict
            Dictionary with 'mean', 'std', 'min', 'max' statistics
        """
        if n_samples is not None and n_samples < self.n_samples:
            # Use random sampling
            rng = np.random.default_rng(42)
            sample_indices = rng.choice(self.n_samples, size=n_samples, replace=False)
            sample_data = self.get_batch(sample_indices)
            
            return {
                'mean': np.mean(sample_data, axis=0),
                'std': np.std(sample_data, axis=0),
                'min': np.min(sample_data, axis=0),
                'max': np.max(sample_data, axis=0)
            }
        
        # Compute statistics incrementally over chunks
        running_sum = np.zeros(self.n_features)
        running_sum_sq = np.zeros(self.n_features)
        running_min = np.full(self.n_features, np.inf)
        running_max = np.full(self.n_features, -np.inf)
        total_samples = 0
        
        for chunk in self.iter_chunks():
            chunk_array = np.asarray(chunk)
            chunk_size = chunk_array.shape[0]
            
            running_sum += np.sum(chunk_array, axis=0)
            running_sum_sq += np.sum(chunk_array**2, axis=0)
            running_min = np.minimum(running_min, np.min(chunk_array, axis=0))
            running_max = np.maximum(running_max, np.max(chunk_array, axis=0))
            total_samples += chunk_size
        
        mean = running_sum / total_samples
        variance = (running_sum_sq / total_samples) - mean**2
        std = np.sqrt(np.maximum(variance, 0))  # Avoid negative variance due to numerical errors
        
        return {
            'mean': mean,
            'std': std,
            'min': running_min,
            'max': running_max
        }
    
    def flush(self):
        """Flush changes to disk (if opened in write mode)."""
        if hasattr(self.data, 'flush'):
            self.data.flush()
    
    def close(self):
        """Close the memory-mapped file."""
        # Memory maps are automatically closed when the object is deleted
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_memory_mapped_loader(
    data: ArrayLike,
    save_path: Union[str, Path],
    chunk_size: int = 1000,
    dtype: Optional[np.dtype] = None
) -> MemoryMappedDataset:
    """
    Create memory-mapped dataset from in-memory data.
    
    Parameters
    ----------
    data : ArrayLike, shape (n_samples, n_features)
        Data to save to memory-mapped file
    save_path : str or Path
        Path where to save the memory-mapped file
    chunk_size : int, default=1000
        Chunk size for the created dataset
    dtype : np.dtype, optional
        Data type for storage. If None, uses data's dtype.
        
    Returns
    -------
    dataset : MemoryMappedDataset
        Memory-mapped dataset instance
        
    Examples
    --------
    >>> import numpy as np
    >>> from sparse_coding.streaming import create_memory_mapped_loader
    >>> 
    >>> # Create large synthetic dataset
    >>> X = np.random.randn(10000, 512).astype(np.float32)
    >>> 
    >>> # Save as memory-mapped file
    >>> dataset = create_memory_mapped_loader(X, 'large_dataset.dat', chunk_size=500)
    >>> 
    >>> # Use for chunked processing
    >>> for chunk in dataset.iter_chunks():
    >>>     print(f"Processing chunk: {chunk.shape}")
    """
    save_path = Path(save_path)
    data_array = ensure_array(data)
    
    if dtype is None:
        dtype = data_array.dtype
    
    # Create memory-mapped file
    n_samples, n_features = data_array.shape
    
    mmap_data = np.memmap(
        save_path,
        dtype=dtype,
        mode='w+',
        shape=(n_samples, n_features)
    )
    
    # Copy data in chunks to avoid memory issues
    chunk_size_write = min(chunk_size, 1000)  # Limit write chunk size
    
    for i in range(0, n_samples, chunk_size_write):
        end_idx = min(i + chunk_size_write, n_samples)
        mmap_data[i:end_idx] = data_array[i:end_idx].astype(dtype)
    
    # Flush to disk
    mmap_data.flush()
    del mmap_data  # Close the memmap
    
    # Return dataset instance
    return MemoryMappedDataset(
        save_path,
        shape=(n_samples, n_features),
        dtype=dtype,
        mode='r',
        chunk_size=chunk_size
    )


def create_from_files(
    file_paths: list,
    save_path: Union[str, Path],
    chunk_size: int = 1000,
    dtype: np.dtype = np.float32
) -> MemoryMappedDataset:
    """
    Create memory-mapped dataset by concatenating multiple files.
    
    Parameters
    ----------
    file_paths : list
        List of paths to .npy files to concatenate
    save_path : str or Path
        Path for output memory-mapped file
    chunk_size : int, default=1000
        Chunk size for processing
    dtype : np.dtype, default=np.float32
        Data type for storage
        
    Returns
    -------
    dataset : MemoryMappedDataset
        Concatenated memory-mapped dataset
    """
    save_path = Path(save_path)
    
    # First pass: determine total shape
    total_samples = 0
    n_features = None
    
    for file_path in file_paths:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load just shape information
        data = np.load(file_path, mmap_mode='r')
        
        if n_features is None:
            n_features = data.shape[1]
        elif data.shape[1] != n_features:
            raise ValueError(f"Inconsistent n_features: expected {n_features}, got {data.shape[1]} in {file_path}")
        
        total_samples += data.shape[0]
    
    # Create output memory map
    mmap_data = np.memmap(
        save_path,
        dtype=dtype,
        mode='w+',
        shape=(total_samples, n_features)
    )
    
    # Second pass: copy data
    current_idx = 0
    
    for file_path in file_paths:
        data = np.load(file_path)
        end_idx = current_idx + data.shape[0]
        
        mmap_data[current_idx:end_idx] = data.astype(dtype)
        current_idx = end_idx
        
        print(f"Processed {file_path}: {data.shape[0]} samples")
    
    # Flush and close
    mmap_data.flush()
    del mmap_data
    
    print(f"Created memory-mapped dataset: {total_samples} samples, {n_features} features")
    
    return MemoryMappedDataset(
        save_path,
        shape=(total_samples, n_features),
        dtype=dtype,
        mode='r',
        chunk_size=chunk_size
    )