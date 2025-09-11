"""
Deterministic execution utilities for reproducible sparse coding research.

Provides functions to control random number generation and threading behavior
for consistent results across different computing environments.

Reproducibility is fundamental to sparse coding research (Olshausen & Field 1996,
Aharon et al. 2006) where identical initialization and random sampling are crucial
for algorithm comparison and theoretical validation.

Threading Control:
Controls BLAS/LAPACK threading to prevent non-deterministic parallel execution:
- OpenBLAS: Matrix operations threading control
- MKL (Intel Math Kernel Library): High-performance linear algebra threading
- OMP (OpenMP): General parallel processing control

Random State Management:
Sets deterministic seeds for all major random number generators:
- Python's built-in random module
- NumPy's random number generator
- Ensures consistent patch sampling, dictionary initialization, and noise generation

References:
    Peng (2011). Reproducible Research in Computational Science.
    LeVeque et al. (2012). Reproducible Research for Scientific Computing.
    Sandve et al. (2013). Ten Simple Rules for Reproducible Computational Research.
    Olshausen & Field (1996). Emergence of simple-cell receptive field properties.
"""

import os
import random 
import numpy as np


def set_deterministic(seed: int = 0) -> None:
    """
    Set deterministic execution for reproducible sparse coding experiments.
    
    Configures the computing environment for consistent, reproducible results across
    different machines and runs. Essential for scientific validation and benchmarking.
    
    Args:
        seed: Random seed for all generators (default: 0)
        
    Side Effects:
        Sets environment variables controlling BLAS threading, initializes random 
        number generators with fixed seed, and configures Intel MKL if available.
        
    Note:
        Sparse coding algorithms are highly sensitive to dictionary initialization,
        patch sampling order, noise injection, and parallel matrix operations.
        
    Example:
        >>> set_deterministic(42)
        >>> coder = SparseCoder(n_atoms=100, seed=42)
        >>> # Training will produce identical dictionaries across runs
    """
    # Control BLAS/LAPACK threading for deterministic matrix operations
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    
    # Configure Intel MKL if available
    try:
        import mkl  # type: ignore
        mkl.set_num_threads(1)
    except (ImportError, AttributeError):
        # MKL not available - OpenBLAS/standard BLAS will be used
        pass
    
    # Set random seeds for all major generators
    random.seed(seed)
    np.random.seed(seed)


def is_deterministic() -> bool:
    """
    Check if deterministic mode is currently enabled.
    
    Returns:
        True if environment is configured for deterministic execution
        
    Example:
        >>> set_deterministic(42)
        >>> is_deterministic()
        True
    """
    # Check if threading is limited to single thread
    threading_vars = ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]
    single_threaded = all(os.environ.get(var) == "1" for var in threading_vars)
    
    # Note: Can't easily check if random seeds were set, so we check threading only
    return single_threaded


def get_reproducibility_info() -> dict:
    """
    Get current reproducibility configuration for research documentation.
    
    Returns:
        Dictionary with current threading and environment settings
        
    Example:
        >>> info = get_reproducibility_info()
        >>> print(f"Threading: {info['threading']}")
        >>> print(f"Environment: {info['environment']}")
    """
    return {
        'threading': {
            'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS', 'unset'),
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', 'unset'),
            'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS', 'unset'),
        },
        'environment': {
            'numpy_version': np.__version__,
            'deterministic_enabled': is_deterministic(),
        },
        'recommendations': [
            "Call set_deterministic() before any sparse coding operations",
            "Use consistent seeds across experiments for comparison",
            "Document seed values in research papers for reproducibility",
            "Verify single-threaded execution for exact reproducibility"
        ]
    }
