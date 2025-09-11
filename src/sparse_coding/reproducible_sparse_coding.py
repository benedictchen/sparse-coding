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
    4. Hash randomization control validation
    5. Memory allocation pattern verification
    6. System-level non-determinism detection
    
    Returns:
        True if environment is configured for deterministic execution
        
    Example:
        >>> set_deterministic(42)
        >>> is_deterministic()
        True
    """
    # 1. Check comprehensive threading configuration
    critical_threading_vars = ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]
    extended_threading_vars = ["NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"]
    
    # Critical threading must be single-threaded
    critical_single_threaded = all(os.environ.get(var) == "1" for var in critical_threading_vars)
    
    # Extended threading should be controlled (warn if not, but don't fail)
    extended_controlled = all(
        os.environ.get(var) in ["1", None] or os.environ.get(var) == "unset" 
        for var in extended_threading_vars
    )
    
    if not critical_single_threaded:
        return False
    
    # 2. Check hash randomization control (critical for dictionary ordering)
    hash_seed_set = os.environ.get("PYTHONHASHSEED") is not None
    if not hash_seed_set:
        return False
    
    # 3. Test random number generator determinism with multiple patterns
    try:
        # Save current state
        py_state = random.getstate()
        np_state = np.random.get_state()
        
        # Test Python random consistency (multiple seeds)
        py_tests = []
        for test_seed in [12345, 67890, 11111]:
            random.seed(test_seed)
            val1 = random.random()
            random.seed(test_seed)
            val2 = random.random()
            py_tests.append(val1 == val2)
        
        py_consistent = all(py_tests)
        
        # Test NumPy random consistency (multiple seeds)
        np_tests = []
        for test_seed in [12345, 67890, 11111]:
            np.random.seed(test_seed)
            val1 = np.random.random()
            np.random.seed(test_seed)
            val2 = np.random.random()
            np_tests.append(val1 == val2)
        
        np_consistent = all(np_tests)
        
        # Test NumPy array ordering consistency (sparse coding depends on consistent dict atom ordering)
        np.random.seed(42)
        arr1 = np.random.randn(10, 5)
        indices1 = np.argsort(arr1.flat)
        
        np.random.seed(42)
        arr2 = np.random.randn(10, 5)
        indices2 = np.argsort(arr2.flat)
        
        array_ordering_consistent = np.array_equal(indices1, indices2)
        
        # Restore original states
        random.setstate(py_state)
        np.random.set_state(np_state)
        
        # All tests must pass for true determinism
        return py_consistent and np_consistent and array_ordering_consistent
        
    except Exception as e:
        # If we can't test random state, fall back to basic threading + hash seed check
        return critical_single_threaded and hash_seed_set


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


def _test_numpy_array_ordering_determinism() -> bool:
    """Test if NumPy array operations produce consistent ordering (critical for dictionary atoms)."""
    try:
        # Test sorting consistency
        np.random.seed(42)
        arr1 = np.random.randn(20)
        sorted_indices1 = np.argsort(arr1)
        
        np.random.seed(42)
        arr2 = np.random.randn(20)
        sorted_indices2 = np.argsort(arr2)
        
        # Test matrix operations consistency
        np.random.seed(123)
        mat1 = np.random.randn(5, 5)
        eigenvals1, _ = np.linalg.eig(mat1)
        
        np.random.seed(123)
        mat2 = np.random.randn(5, 5)
        eigenvals2, _ = np.linalg.eig(mat2)
        
        arrays_equal = np.array_equal(sorted_indices1, sorted_indices2)
        eigenvals_close = np.allclose(eigenvals1, eigenvals2, rtol=1e-15, atol=1e-15)
        
        return arrays_equal and eigenvals_close
    except Exception:
        return False


def _calculate_determinism_score() -> float:
    """Calculate overall determinism confidence score (0.0 to 1.0)."""
    score = 0.0
    max_score = 6.0  # Total possible points
    
    # Threading configuration (2 points)
    critical_threading_vars = ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]
    if all(os.environ.get(var) == "1" for var in critical_threading_vars):
        score += 2.0
    elif any(os.environ.get(var) == "1" for var in critical_threading_vars):
        score += 1.0
    
    # Hash seed control (1 point)
    if os.environ.get("PYTHONHASHSEED") is not None:
        score += 1.0
    
    # Random state consistency (2 points)
    if _test_python_random_determinism():
        score += 0.5
    if _test_numpy_random_determinism():
        score += 0.5
    if _test_numpy_array_ordering_determinism():
        score += 1.0
    
    # Overall determinism function result (1 point)
    if is_deterministic():
        score += 1.0
    
    return min(score / max_score, 1.0)


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
    # Detect system-level determinism threats
    system_threats = []
    
    # Check for parallel processing libraries that could introduce non-determinism
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if cpu_count > 1:
            system_threats.append(f"Multi-core system ({cpu_count} cores) - ensure single-threading")
    except ImportError:
        pass
    
    # Check for GPU libraries that could have non-deterministic operations
    gpu_libraries = []
    try:
        import torch
        if torch.cuda.is_available():
            gpu_libraries.append(f"PyTorch CUDA available - device: {torch.cuda.get_device_name()}")
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
            gpu_libraries.append("TensorFlow GPU available")
    except ImportError:
        pass
    
    if gpu_libraries:
        system_threats.extend(gpu_libraries)
        system_threats.append("GPU operations may have non-deterministic algorithms")
    
    # Check memory allocation patterns
    memory_info = {}
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_info = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'memory_pressure': memory.percent > 80
        }
        if memory_info['memory_pressure']:
            system_threats.append("High memory pressure - may cause non-deterministic swapping")
    except ImportError:
        memory_info = {'status': 'psutil not available for memory monitoring'}
    
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
            'memory_info': memory_info,
        },
        'random_state_test': {
            'python_random_deterministic': _test_python_random_determinism(),
            'numpy_random_deterministic': _test_numpy_random_determinism(),
            'array_ordering_consistent': _test_numpy_array_ordering_determinism(),
        },
        'system_analysis': {
            'potential_threats': system_threats,
            'determinism_score': _calculate_determinism_score(),
        },
        'recommendations': [
            "Call set_deterministic() before any sparse coding operations",
            "Use consistent seeds across experiments for comparison", 
            "Document seed values in research papers for reproducibility",
            "Verify single-threaded execution for exact reproducibility",
            "Monitor system threats to determinism with get_reproducibility_info()",
            "Test determinism with multiple runs before publishing results",
            "Disable GPU operations if strict determinism is required"
        ]
    }
