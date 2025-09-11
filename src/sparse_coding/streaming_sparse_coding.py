"""
Streaming sparse coding for large-scale data processing.

Implements memory-efficient sparse coding for datasets too large to fit in RAM.
Essential for real-world applications with millions of natural image patches or large-scale
feature learning on limited-memory systems.

Memory-Mapped Processing:
Uses NumPy's memory-mapping to access data directly from disk without loading
into RAM. This enables processing of TB-scale datasets on GB-scale machines.

Streaming Algorithm:
For large data matrix X = [x₁, x₂, ..., xₙ] stored on disk:
1. Load small batch Xᵦ = [xᵢ, ..., xᵢ₊ᵦ] into memory
2. Solve sparse coding: Aᵦ = argmin ||Xᵦ - DAᵦ||² + λ||Aᵦ||₁  
3. Write Aᵦ directly to disk using memory-mapped output file
4. Repeat for next batch until all data processed

Applications in Sparse Coding Literature:
- Olshausen & Field (1996): Processing van Hateren natural image database
- Mairal et al. (2010): Online dictionary learning for streaming data
- Coates & Ng (2011): Learning features from millions of images
- Le et al. (2012): Building high-level features using large scale unsupervised learning

Memory Efficiency Benefits:
- Constant memory usage regardless of dataset size
- Enables processing of datasets larger than available RAM
- Parallelizable across multiple machines (map-reduce style)
- Suitable for real-time/online learning applications

References:
    Olshausen & Field (1996). Original sparse coding formulation.
    Mairal et al. (2010). Online dictionary learning for sparse coding.
    Coates & Ng (2011). The importance of encoding versus training with sparse coding.
    Beck & Teboulle (2009). FISTA algorithm for efficient optimization.
"""

from __future__ import annotations
import os
import numpy as np
from typing import Iterator, Optional, Union
from pathlib import Path


def stream_columns(
    X_path: Union[str, Path], 
    batch_size: int = 10000,
    start_col: int = 0,
    end_col: Optional[int] = None
) -> Iterator[np.ndarray]:
    """
    Stream data columns in batches from memory-mapped NumPy file.
    
    Memory-efficient iterator for processing large datasets column-wise without
    loading the entire array into memory. Uses NumPy's memory-mapping for
    direct disk access.
    
    Args:
        X_path: Path to .npy file containing data matrix (n_features, n_samples)
        batch_size: Number of columns per batch (default: 10000)
        start_col: Starting column index (default: 0)
        end_col: Ending column index (default: file size)
        
    Yields:
        Data batches as (n_features, batch_size) arrays
        
    Memory Usage:
        O(n_features × batch_size) regardless of total dataset size
        
    Example:
        >>> # Process 1TB dataset with only 100MB RAM usage per batch
        >>> for batch in stream_columns("huge_dataset.npy", batch_size=1000):
        ...     # batch.shape = (256, 1000) uses ~2MB RAM
        ...     codes = sparse_coder.encode(batch)
        ...     process_batch(codes)
        
    Research Context:
        Essential for large-scale sparse coding experiments:
        - ImageNet: 14M images → ~1TB of patches
        - Audio datasets: Hours of spectrograms → hundreds of GB
        - Neuroscience: Multi-electrode recordings → TB-scale timeseries
        
    Technical Details:
        - Uses mmap_mode='r' for read-only memory mapping
        - Yields actual numpy arrays (not memory-mapped views) for safety
        - Handles file boundary conditions gracefully
        - Compatible with all NumPy data types
    """
    X_path = Path(X_path)
    
    if not X_path.exists():
        raise FileNotFoundError(f"Data file not found: {X_path}")
    
    # Open file in memory-mapped mode (read-only)
    X = np.load(X_path, mmap_mode='r')
    
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    
    n_features, total_samples = X.shape
    
    # Handle end_col boundary
    if end_col is None:
        end_col = total_samples
    else:
        end_col = min(end_col, total_samples)
    
    if start_col >= end_col:
        return  # Empty range
    
    # Stream batches
    for i in range(start_col, end_col, batch_size):
        end_idx = min(i + batch_size, end_col)
        # Convert to regular array to avoid memory-mapping issues downstream
        batch = np.asarray(X[:, i:end_idx])
        yield batch


def encode_stream(
    D: np.ndarray,
    X_path: Union[str, Path], 
    lam: float,
    batch_size: int = 10000,
    out_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    dtype: np.dtype = np.float32
) -> str:
    """
    Encode large dataset using streaming sparse coding with FISTA.
    
    Processes data too large for memory by streaming batches from disk,
    applying FISTA sparse coding, and writing results directly to disk
    using memory-mapped output files.
    
    Args:
        D: Dictionary matrix (n_features, n_atoms)  
        X_path: Path to input data (.npy file with shape n_features, n_samples)
        lam: Sparsity penalty strength (L1 regularization parameter)
        batch_size: Number of samples per batch (default: 10000)
        out_path: Output path for sparse codes (default: X_path.with_suffix('.codes.npy'))
        verbose: Print progress information
        dtype: Output data type (default: float32 for memory efficiency)
        
    Returns:
        Path to output file containing sparse codes
        
    Output Format:
        Sparse codes saved as (n_atoms, n_samples) NumPy array
        
    Memory Usage:
        Peak memory ≈ batch_size × (n_features + n_atoms) × 8 bytes
        Constant regardless of total dataset size
        
    Algorithm:
        For each batch Xᵦ of input data:
        1. Load Xᵦ into memory (size: n_features × batch_size)
        2. Solve: Aᵦ = argmin ||Xᵦ - DAᵦ||² + λ||Aᵦ||₁ using FISTA
        3. Write Aᵦ directly to memory-mapped output file
        4. Clear batch from memory and proceed to next
        
    Performance Characteristics:
        - Linear scaling with dataset size
        - Constant memory usage
        - I/O bound (limited by disk read/write speed)
        - Parallelizable across different data ranges
        
    Example:
        >>> # Learn dictionary on subset
        >>> D = learn_dictionary_subset(data[:, :10000])  
        >>> # Encode entire dataset (e.g., 100M samples)
        >>> codes_path = encode_stream(D, "massive_dataset.npy", lam=0.1)
        >>> # Load results in chunks as needed
        >>> codes = np.load(codes_path, mmap_mode='r')
        
    Research Applications:
        - Large-scale feature learning (Coates & Ng 2011)
        - Natural image statistics (van Hateren database processing)
        - Audio sparse coding (spectrograms from large audio corpora)
        - Real-time encoding in production systems
        
    References:
        - Beck & Teboulle (2009): FISTA optimization algorithm
        - Mairal et al. (2010): Online dictionary learning principles
        - Coates & Ng (2011): Scaling sparse coding to millions of images
    """
    from .fista_batch import fista_batch
    
    X_path = Path(X_path)
    
    if not X_path.exists():
        raise FileNotFoundError(f"Input file not found: {X_path}")
    
    # Load data header to get dimensions
    X_header = np.load(X_path, mmap_mode='r')
    if X_header.ndim != 2:
        raise ValueError(f"Expected 2D input array, got shape {X_header.shape}")
    
    n_features, total_samples = X_header.shape
    n_atoms = D.shape[1]
    
    # Validate dictionary dimensions
    if D.shape[0] != n_features:
        raise ValueError(f"Dictionary shape {D.shape} incompatible with data shape {X_header.shape}")
    
    # Determine output path
    if out_path is None:
        out_path = X_path.with_suffix('.codes.npy')
    else:
        out_path = Path(out_path)
    
    # Pre-compute FISTA parameters for efficiency
    # Lipschitz constant L for step size
    # Compute Lipschitz constant for FISTA step size
    try:
        from .fista_batch import power_iter_L
        L = power_iter_L(D)  # More accurate than ||D||² approximation
    except ImportError as e:
        # Module missing - this is a critical dependency error
        raise ImportError(f"Required module 'fista_batch' not found: {e}. "
                         f"Cannot compute Lipschitz constant for FISTA algorithm.") from e
    
    # Ensure L is a proper scalar
    if not np.isscalar(L) or not np.isfinite(L) or L <= 0:
        # Fallback to spectral norm approximation
        L = float(np.linalg.norm(D, ord=2)**2)
    
    if verbose:
        print(f"Initializing streaming sparse coding pipeline:")
        print(f"   Input dataset: {X_path} ({n_features} features × {total_samples} samples)")
        print(f"   Dictionary matrix: {D.shape}")
        print(f"   Sparsity parameter: λ={lam}")
        print(f"   Processing batch size: {batch_size}")
        print(f"   Output file: {out_path}")
        print(f"   Lipschitz constant: L={L:.4f}")
    
    # Create memory-mapped output file
    # Using dtype for memory efficiency (float32 often sufficient for codes)
    codes_mmap = np.memmap(
        out_path, 
        dtype=dtype, 
        mode='w+', 
        shape=(n_atoms, total_samples)
    )
    
    try:
        # Process data in streaming fashion
        sample_offset = 0
        
        for batch_idx, X_batch in enumerate(stream_columns(X_path, batch_size)):
            batch_samples = X_batch.shape[1]
            
            if verbose and batch_idx % 10 == 0:
                progress = (sample_offset / total_samples) * 100
                print(f"   Processing batch {batch_idx}: {progress:.1f}% complete")
            
            # Solve sparse coding for current batch
            # A_batch shape: (n_atoms, batch_samples)
            A_batch = fista_batch(D, X_batch, lam, L=L, max_iter=200, tol=1e-6)
            
            # Write batch results to memory-mapped file
            end_offset = sample_offset + batch_samples
            codes_mmap[:, sample_offset:end_offset] = A_batch.astype(dtype)
            
            # Ensure data is written to disk (important for large files)
            codes_mmap.flush()
            
            sample_offset += batch_samples
        
        if verbose:
            print(f"Streaming sparse coding completed: {sample_offset} samples processed")
            
            # Compute and report sparsity statistics
            sparsity_ratio = np.mean(np.abs(codes_mmap[:, :min(1000, total_samples)]) < 1e-6)
            print(f"   Coefficient sparsity: {sparsity_ratio:.3f} (fraction of near-zero elements)")
            print(f"   Output file size: {out_path.stat().st_size / (1024**3):.2f} GB")
        
    finally:
        # Ensure memory-mapped file is properly closed
        del codes_mmap
    
    return str(out_path)


def stream_batches(
    *file_paths: Union[str, Path],
    batch_size: int = 1000,
    align_samples: bool = True
) -> Iterator[tuple]:
    """
    Stream multiple aligned datasets simultaneously.
    
    Useful for processing paired datasets (e.g., original data and ground truth codes)
    where corresponding samples must be processed together.
    
    Args:
        *file_paths: Paths to .npy files to stream together
        batch_size: Samples per batch
        align_samples: Ensure all files have same number of samples
        
    Yields:
        Tuples of aligned batches from each file
        
    Example:
        >>> # Process data and labels together
        >>> for X_batch, Y_batch in stream_batches("data.npy", "labels.npy"):
        ...     process_aligned_batch(X_batch, Y_batch)
    """
    file_paths = [Path(p) for p in file_paths]
    
    # Validate files and get dimensions
    file_shapes = []
    for path in file_paths:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        data = np.load(path, mmap_mode='r')
        file_shapes.append(data.shape)
    
    if align_samples:
        n_samples_list = [shape[1] if len(shape) == 2 else shape[0] for shape in file_shapes]
        if len(set(n_samples_list)) > 1:
            raise ValueError(f"Sample counts don't match: {n_samples_list}")
        n_samples = n_samples_list[0]
    else:
        n_samples = min(shape[1] if len(shape) == 2 else shape[0] for shape in file_shapes)
    
    # Stream aligned batches
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        batches = []
        for path in file_paths:
            data = np.load(path, mmap_mode='r')
            if data.ndim == 2:
                batch = np.asarray(data[:, start:end])
            else:
                batch = np.asarray(data[start:end])
            batches.append(batch)
        
        yield tuple(batches)


def estimate_memory_usage(
    data_shape: tuple,
    dict_shape: tuple, 
    batch_size: int,
    dtype: np.dtype = np.float32
) -> dict:
    """
    Estimate memory usage for streaming sparse coding.
    
    Helps users choose appropriate batch sizes for their available memory.
    
    Args:
        data_shape: Shape of input data (n_features, n_samples)
        dict_shape: Shape of dictionary (n_features, n_atoms)
        batch_size: Proposed batch size
        dtype: Data type for calculations
        
    Returns:
        Dictionary with memory usage estimates in MB
    """
    n_features, n_samples = data_shape
    _, n_atoms = dict_shape
    bytes_per_element = np.dtype(dtype).itemsize
    
    # Memory components
    input_batch = n_features * batch_size * bytes_per_element
    output_batch = n_atoms * batch_size * bytes_per_element
    dictionary = n_features * n_atoms * bytes_per_element
    working_memory = input_batch  # FISTA working arrays
    
    total_mb = (input_batch + output_batch + dictionary + working_memory) / (1024**2)
    
    return {
        'input_batch_mb': input_batch / (1024**2),
        'output_batch_mb': output_batch / (1024**2), 
        'dictionary_mb': dictionary / (1024**2),
        'working_memory_mb': working_memory / (1024**2),
        'total_mb': total_mb,
        'recommended_max_mb': 1024,  # Reasonable limit for most systems
        'batch_size': batch_size,
        'total_batches': (n_samples + batch_size - 1) // batch_size
    }
