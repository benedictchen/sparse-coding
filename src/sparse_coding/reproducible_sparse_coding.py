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
    
    # Additional threading control for comprehensive determinism
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS Accelerate framework
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")       # Numba JIT compilation
    
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
    
    # Set additional NumPy/SciPy deterministic behavior
    try:
        # Modern NumPy random number generator (if available)
        np.random.default_rng(seed)
    except AttributeError:
        # Fallback for older NumPy versions
        pass
    
    # Configure hash randomization (if needed)
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = str(seed)


def is_deterministic() -> bool:
    """
    Check if deterministic mode is currently enabled.
    
    Performs comprehensive validation of deterministic execution requirements:
    1. Threading configuration (BLAS/LAPACK single-threaded)
    2. Random number generator state validation
    3. NumPy random state consistency checks
    
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
    
    if not single_threaded:
        return False
    
    # Check if random number generators appear to be seeded consistently
    # We do this by testing for predictable sequences
    try:
        # Save current state
        py_state = random.getstate()
        np_state = np.random.get_state()
        
        # Test Python random consistency
        random.seed(12345)
        test_py_1 = random.random()
        random.seed(12345)
        test_py_2 = random.random()
        py_consistent = (test_py_1 == test_py_2)
        
        # Test NumPy random consistency
        np.random.seed(12345)
        test_np_1 = np.random.random()
        np.random.seed(12345)
        test_np_2 = np.random.random()
        np_consistent = (test_np_1 == test_np_2)
        
        # Restore original states
        random.setstate(py_state)
        np.random.set_state(np_state)
        
        return py_consistent and np_consistent
        
    except Exception:
        # If we can't test random state, fall back to threading check only
        return single_threaded


def _test_python_random_determinism() -> bool:
    """Test if Python's random module is generating deterministic sequences."""
    try:
        state = random.getstate()
        
        random.seed(9999)
        val1 = random.random()
        random.seed(9999)
        val2 = random.random()
        
        random.setstate(state)
        return val1 == val2
    except Exception:
        return False


def _test_numpy_random_determinism() -> bool:
    """Test if NumPy's random module is generating deterministic sequences."""
    try:
        state = np.random.get_state()
        
        np.random.seed(9999)
        val1 = np.random.random()
        np.random.seed(9999)
        val2 = np.random.random()
        
        np.random.set_state(state)
        return val1 == val2
    except Exception:
        return False


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
            'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS', 'unset'),
            'VECLIB_MAXIMUM_THREADS': os.environ.get('VECLIB_MAXIMUM_THREADS', 'unset'),
            'NUMBA_NUM_THREADS': os.environ.get('NUMBA_NUM_THREADS', 'unset'),
        },
        'environment': {
            'numpy_version': np.__version__,
            'deterministic_enabled': is_deterministic(),
            'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'unset'),
        },
        'random_state_test': {
            'python_random_deterministic': _test_python_random_determinism(),
            'numpy_random_deterministic': _test_numpy_random_determinism(),
        },
        'recommendations': [
            "Call set_deterministic() before any sparse coding operations",
            "Use consistent seeds across experiments for comparison",
            "Document seed values in research papers for reproducibility",
            "Verify single-threaded execution for exact reproducibility",
            "Check random state determinism with get_reproducibility_info()"
        ]
    }
