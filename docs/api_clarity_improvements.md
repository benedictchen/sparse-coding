# API Clarity Improvements for Sparse Coding Library

## Overview

This document outlines comprehensive API clarity improvements for the sparse coding library to address documentation issues identified in the comprehensive analysis. The improvements focus on making the library more accessible to both researchers and practitioners.

## Core API Documentation Improvements

### 1. SparseCoder Class - Enhanced Documentation

```python
class SparseCoder:
    """
    Main interface for sparse coding inference with research-accurate algorithms.
    
    Implements multiple sparse coding algorithms from foundational research papers:
    - FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for L1 penalties
    - ISTA (Iterative Shrinkage-Thresholding Algorithm) for L1 penalties  
    - OMP (Orthogonal Matching Pursuit) for exact sparsity constraints
    - NCG (Nonlinear Conjugate Gradient) for smooth differentiable penalties
    
    Research Foundation:
        Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding 
        algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1), 183-202.
        
        Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive 
        field properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.
    
    Examples:
        Basic L1 sparse coding:
        >>> import numpy as np
        >>> from sparse_coding import SparseCoder
        >>> 
        >>> # Create dictionary and signal
        >>> D = np.random.randn(64, 32)
        >>> D /= np.linalg.norm(D, axis=0)  # Normalize atoms
        >>> signal = np.random.randn(64, 1)
        >>> 
        >>> # Initialize sparse coder
        >>> coder = SparseCoder(n_atoms=32, mode='l1', lam=0.1, max_iter=100)
        >>> coder.dictionary = D
        >>> 
        >>> # Encode signal
        >>> codes = coder.encode(signal)
        >>> print(f"Sparsity level: {np.mean(np.abs(codes) < 1e-3):.2f}")
        
        Research-accurate Olshausen & Field (1996) reproduction:
        >>> # Use "paper" mode for log-Cauchy prior with NCG optimization
        >>> coder = SparseCoder(n_atoms=32, mode='paper', lam=0.1, max_iter=50)
        >>> coder.dictionary = D
        >>> codes = coder.encode(signal)  # Uses NCG with log-Cauchy penalty
        
        Different solver algorithms:
        >>> # FISTA (fastest for L1)
        >>> coder_fista = SparseCoder(n_atoms=32, mode='l1', solver='fista', lam=0.1)
        >>> 
        >>> # OMP (exact sparsity level)
        >>> coder_omp = SparseCoder(n_atoms=32, mode='omp', sparsity=5)  # Exactly 5 non-zero coefficients
        >>> 
        >>> # NCG (for smooth penalties)
        >>> from sparse_coding.core.penalties.implementations import L2Penalty
        >>> penalty = L2Penalty(lam=0.1)
        >>> codes = coder.solve_with_penalty(signal, penalty)  # Custom penalty function
    
    Parameters:
        n_atoms : int
            Number of dictionary atoms (columns in dictionary matrix).
            Should match the second dimension of your dictionary.
            
        mode : str, default='l1'
            Penalty/algorithm mode. Options:
            - 'l1': L1 penalty with FISTA solver (most common)
            - 'l2': L2 penalty with analytical solution
            - 'elastic_net': Combined L1+L2 penalty
            - 'omp': Orthogonal Matching Pursuit (exact sparsity)
            - 'paper': Olshausen & Field (1996) log-Cauchy prior with NCG
            
        lam : float, default=0.1
            Regularization strength. Higher values → sparser solutions.
            Typical range: [0.01, 1.0] for L1/L2 penalties.
            
        max_iter : int, default=1000
            Maximum number of optimization iterations.
            FISTA: 100-1000 typical
            NCG: 50-200 typical
            OMP: Automatically determined by sparsity level
            
        tol : float, default=1e-6
            Convergence tolerance. Algorithm stops when gradient norm < tol.
            Stricter tolerances (1e-8) for research applications.
            
        solver : str, optional
            Override automatic solver selection:
            - 'fista': Fast Iterative Shrinkage-Thresholding (for L1)
            - 'ista': Basic Iterative Shrinkage-Thresholding (for L1)
            - 'ncg': Nonlinear Conjugate Gradient (for smooth penalties)
            - 'omp': Orthogonal Matching Pursuit
            
        sparsity : int, optional
            For OMP mode: exact number of non-zero coefficients to find.
            
        seed : int, optional
            Random seed for reproducible results.
    
    Attributes:
        dictionary : np.ndarray, shape (n_features, n_atoms)
            Dictionary matrix. Each column is a normalized atom.
            Set this before calling encode().
            
        n_atoms : int
            Number of dictionary atoms.
            
        mode : str
            Current penalty/algorithm mode.
            
        lam : float
            Current regularization parameter.
    
    Methods:
        encode(X) : Encode signals using current dictionary and parameters
        fit_dictionary(D) : Set dictionary matrix with validation
        decode(A) : Reconstruct signals from sparse codes
        solve_with_penalty(X, penalty) : Advanced: use custom penalty function
    
    Notes:
        Algorithm Selection Guidelines:
        - Use FISTA ('l1' mode) for standard sparse coding with L1 penalty
        - Use OMP when you need exactly k non-zero coefficients
        - Use NCG ('paper' mode) for research reproduction of Olshausen & Field
        - Use L2/ElasticNet for problems requiring smooth penalties
        
        Performance Tips:
        - Normalize dictionary atoms: D /= np.linalg.norm(D, axis=0)
        - Start with lam=0.1, adjust based on desired sparsity
        - Use max_iter=100 for most problems, increase for high accuracy
        
        Common Issues:
        - If solutions are too sparse: decrease lam
        - If solutions are too dense: increase lam  
        - If convergence is slow: check dictionary conditioning
        - If results vary: set seed parameter for reproducibility
    """
    
    def __init__(self, n_atoms: int, mode: str = 'l1', lam: float = 0.1, 
                 max_iter: int = 1000, tol: float = 1e-6, solver: str = None,
                 sparsity: int = None, seed: int = None):
        # Implementation...
        
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode signals into sparse representations using the current dictionary.
        
        This is the main method for sparse coding inference. It solves the optimization problem:
            minimize ||X - D*A||_F^2 + λ * penalty(A)
        where D is the dictionary, A are the sparse codes, and penalty depends on mode.
        
        Parameters:
            X : np.ndarray, shape (n_features,) or (n_features, n_samples)
                Input signal(s) to encode. Can be:
                - Single signal: shape (n_features,)
                - Multiple signals: shape (n_features, n_samples)
                - Must match dictionary's first dimension (n_features)
        
        Returns:
            A : np.ndarray, shape (n_atoms,) or (n_atoms, n_samples)
                Sparse codes with same number of columns as input.
                Each column contains the sparse representation of corresponding input signal.
        
        Raises:
            ValueError: If dictionary is not set or has wrong dimensions
            RuntimeError: If optimization fails to converge
        
        Examples:
            Single signal encoding:
            >>> signal = np.random.randn(64)  # Single 64-D signal
            >>> codes = coder.encode(signal)  # Returns (32,) codes
            >>> reconstruction = coder.dictionary @ codes
            >>> error = np.linalg.norm(signal - reconstruction)
            
            Batch encoding:
            >>> signals = np.random.randn(64, 100)  # 100 signals
            >>> codes = coder.encode(signals)  # Returns (32, 100) codes
            >>> reconstructions = coder.dictionary @ codes
            
            Sparsity validation:
            >>> codes = coder.encode(signal)
            >>> sparsity_ratio = np.mean(np.abs(codes) < 1e-6)
            >>> print(f"Proportion of near-zero coefficients: {sparsity_ratio:.2f}")
        
        Notes:
            Algorithm Details by Mode:
            - 'l1': Uses FISTA with L1 penalty ||A||_1
            - 'l2': Uses analytical solution with L2 penalty ||A||_2^2
            - 'omp': Uses Orthogonal Matching Pursuit with exact sparsity
            - 'paper': Uses NCG with log-Cauchy penalty from Olshausen & Field
            
            Performance Characteristics:
            - FISTA: O(1/k^2) convergence rate, best for L1
            - NCG: Superlinear convergence for smooth problems
            - OMP: Greedy algorithm, fast for low sparsity levels
            
            Memory Usage:
            - Single signal: O(n_features + n_atoms)
            - Batch signals: O(n_features * n_samples + n_atoms * n_samples)
        """
        # Implementation...
```

### 2. Dictionary Learning Classes - Enhanced Documentation

```python
class DictionaryLearner:
    """
    Dictionary learning for sparse coding using alternating optimization.
    
    Learns an overcomplete dictionary D and sparse codes A such that X ≈ DA.
    Implements classical dictionary learning algorithms from the literature:
    - K-SVD (Aharon et al., 2006): SVD-based atom updates
    - MOD (Engan et al., 1999): Method of Optimal Directions
    - Online learning (Mairal et al., 2010): Stochastic optimization
    
    Research Foundation:
        Aharon, M., Elad, M., & Bruckstein, A. (2006). K-SVD: An algorithm for 
        designing overcomplete dictionaries for sparse representation. IEEE 
        Transactions on Signal Processing, 54(11), 4311-4322.
    
    Examples:
        Basic dictionary learning:
        >>> import numpy as np
        >>> from sparse_coding import DictionaryLearner
        >>> 
        >>> # Generate synthetic data
        >>> n_features, n_samples = 64, 1000
        >>> X = np.random.randn(n_features, n_samples)
        >>> 
        >>> # Learn dictionary with 32 atoms
        >>> learner = DictionaryLearner(n_atoms=32, max_iter=20, method='ksvd')
        >>> D, A = learner.fit_transform(X)
        >>> 
        >>> # Validate dictionary
        >>> atom_norms = np.linalg.norm(D, axis=0)
        >>> print(f"Dictionary atoms normalized: {np.allclose(atom_norms, 1.0)}")
        >>> 
        >>> # Check reconstruction quality
        >>> reconstruction_error = np.linalg.norm(X - D @ A, 'fro')
        >>> relative_error = reconstruction_error / np.linalg.norm(X, 'fro')
        >>> print(f"Relative reconstruction error: {relative_error:.4f}")
        
        Natural image patch learning (Olshausen & Field setup):
        >>> # Typical setup for 8x8 natural image patches
        >>> patch_size = 8
        >>> n_patches = 10000
        >>> overcomplete_factor = 2  # 128 atoms for 64-D patches
        >>> 
        >>> learner = DictionaryLearner(
        ...     n_atoms=patch_size**2 * overcomplete_factor,  # 128 atoms
        ...     max_iter=100,
        ...     method='ksvd',
        ...     sparse_coder_params={'lam': 0.1, 'max_iter': 100}
        ... )
        >>> 
        >>> # X should be shape (64, 10000) - flattened patches
        >>> D, A = learner.fit_transform(patches)
        >>> 
        >>> # Visualize learned atoms as 8x8 patches
        >>> for i in range(16):  # Show first 16 atoms
        ...     atom_patch = D[:, i].reshape(8, 8)
        ...     # Display atom_patch using matplotlib
    
    Parameters:
        n_atoms : int
            Number of dictionary atoms to learn.
            Should be >= n_features for overcomplete dictionaries.
            Common choice: n_atoms = 2 * n_features (2x overcomplete)
            
        max_iter : int, default=20
            Maximum number of dictionary learning iterations.
            Each iteration: (1) sparse coding step, (2) dictionary update step.
            
        method : str, default='ksvd'
            Dictionary update algorithm:
            - 'ksvd': K-SVD with SVD-based atom updates
            - 'mod': Method of Optimal Directions (closed-form)
            - 'online': Online dictionary learning
            
        sparse_coder_params : dict, optional
            Parameters passed to sparse coding step.
            Common parameters:
            - 'lam': regularization strength (default: 0.1)
            - 'max_iter': sparse coding iterations (default: 100)
            - 'mode': penalty type (default: 'l1')
            
        tol : float, default=1e-4
            Convergence tolerance. Stops when dictionary change < tol.
            
        random_state : int, optional
            Random seed for reproducible dictionary initialization.
    
    Attributes:
        dictionary_ : np.ndarray, shape (n_features, n_atoms)
            Learned dictionary matrix after fitting.
            Each column is a unit-normalized atom.
            
        sparse_codes_ : np.ndarray, shape (n_atoms, n_samples)
            Final sparse codes from last iteration.
            
        objective_history_ : list
            Reconstruction error at each iteration for monitoring convergence.
    
    Methods:
        fit(X) : Learn dictionary from training data
        transform(X) : Encode new data using learned dictionary
        fit_transform(X) : Learn dictionary and return codes for training data
        score(X) : Compute reconstruction error on test data
    
    Notes:
        Algorithm Comparison:
        - K-SVD: Best reconstruction quality, slower
        - MOD: Fast, good for well-conditioned problems  
        - Online: Memory efficient, good for large datasets
        
        Memory Requirements:
        - Dictionary: O(n_features * n_atoms)
        - Sparse codes: O(n_atoms * n_samples)
        - Total: O(max(n_features * n_atoms, n_atoms * n_samples))
        
        Typical Convergence:
        - K-SVD: 20-100 iterations
        - MOD: 10-50 iterations
        - Online: Streaming, no fixed iteration count
        
        Common Issues:
        - Poor initialization: Use random_state for reproducibility
        - Slow convergence: Reduce sparse_coder_params['lam']
        - Memory issues: Use online method or reduce n_samples
    """
```

### 3. Penalty Functions - Enhanced Documentation

```python
class L1Penalty:
    """
    L1 (Lasso) penalty for sparse coding optimization.
    
    Implements the L1 norm penalty ||x||_1 = Σ|x_i| that promotes sparsity
    by driving small coefficients to exactly zero.
    
    Mathematical formulation:
        penalty(x) = λ * ||x||_1 = λ * Σ|x_i|
        
    The L1 penalty is non-differentiable at zero, requiring specialized 
    optimization algorithms like FISTA or ISTA with proximal operators.
    
    Research Foundation:
        Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
        Journal of the Royal Statistical Society, 58(1), 267-288.
    
    Examples:
        Basic L1 penalty:
        >>> from sparse_coding.core.penalties.implementations import L1Penalty
        >>> penalty = L1Penalty(lam=0.1)
        >>> 
        >>> # Evaluate penalty
        >>> x = np.array([0.5, -0.3, 0.0, 0.8])
        >>> penalty_value = penalty.value(x)
        >>> print(f"L1 penalty: {penalty_value}")  # 0.1 * (0.5 + 0.3 + 0.0 + 0.8)
        >>> 
        >>> # Proximal operator (soft thresholding)
        >>> x_prox = penalty.prox(x, step_size=1.0)
        >>> print(f"After soft thresholding: {x_prox}")
        
        Integration with optimization:
        >>> from sparse_coding import SparseCoder
        >>> coder = SparseCoder(n_atoms=32, mode='l1', lam=0.1)
        >>> # Internally uses L1Penalty with FISTA optimizer
    
    Parameters:
        lam : float, default=0.1
            Regularization strength (λ). Controls sparsity level:
            - Higher λ → sparser solutions (more zeros)
            - Lower λ → denser solutions (fewer zeros)
            - Typical range: [0.01, 1.0]
    
    Properties:
        is_differentiable : bool
            Always False for L1 penalty (non-differentiable at zero)
        is_prox_friendly : bool
            Always True (has efficient proximal operator)
    
    Methods:
        value(x) : Compute penalty value λ * ||x||_1
        prox(x, step_size) : Proximal operator (soft thresholding)
        
    Notes:
        Proximal Operator (Soft Thresholding):
            prox(x, t) = sign(x) * max(|x| - λt, 0)
            
        This is the key operation in FISTA/ISTA algorithms that handles
        the non-differentiability of the L1 norm.
        
        Sparsity Properties:
        - Exact zeros: L1 penalty drives coefficients to exactly zero
        - Unbiased: Large coefficients shrunk by constant amount λt
        - Scale invariant: Penalty scales with coefficient magnitude
        
        Computational Complexity:
        - value(): O(n) where n is vector length
        - prox(): O(n) element-wise soft thresholding
    """
    
    def value(self, x: np.ndarray) -> float:
        """
        Compute L1 penalty value.
        
        Parameters:
            x : np.ndarray
                Input vector/matrix
                
        Returns:
            float : λ * ||x||_1
        """
        
    def prox(self, x: np.ndarray, step_size: float) -> np.ndarray:
        """
        Proximal operator (soft thresholding).
        
        Computes: prox(x, t) = argmin_z [0.5 * ||z - x||_2^2 + λt * ||z||_1]
        
        Parameters:
            x : np.ndarray
                Input vector/matrix
            step_size : float
                Step size parameter (t in the formula)
                
        Returns:
            np.ndarray : Soft-thresholded result
        """
```

## Usage Examples and Tutorials

### Tutorial 1: Getting Started with Sparse Coding

```python
"""
Getting Started with Sparse Coding
==================================

This tutorial demonstrates basic sparse coding workflows for common use cases.
"""

import numpy as np
import matplotlib.pyplot as plt
from sparse_coding import SparseCoder, DictionaryLearner

# Example 1: Basic sparse coding with known dictionary
def basic_sparse_coding_example():
    """Demonstrate basic sparse coding with a simple example."""
    
    # Create a simple 1D dictionary (basis functions)
    n_features = 50
    n_atoms = 20
    
    # Dictionary with sinusoidal and step function atoms
    D = np.zeros((n_features, n_atoms))
    for i in range(n_atoms//2):
        # Sine waves of different frequencies
        t = np.linspace(0, 4*np.pi, n_features)
        D[:, i] = np.sin((i+1) * t)
    
    for i in range(n_atoms//2, n_atoms):
        # Step functions at different positions
        step_pos = int((i - n_atoms//2) * n_features // (n_atoms//2))
        D[:step_pos, i] = -1
        D[step_pos:, i] = 1
    
    # Normalize dictionary atoms
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    
    # Create a test signal (combination of 3 atoms)
    true_codes = np.zeros(n_atoms)
    true_codes[2] = 0.8   # Sine component
    true_codes[15] = -0.5 # Step component  
    true_codes[7] = 0.3   # Another sine component
    
    signal = D @ true_codes + 0.05 * np.random.randn(n_features)
    
    # Sparse coding
    coder = SparseCoder(n_atoms=n_atoms, mode='l1', lam=0.1, max_iter=100)
    coder.dictionary = D
    
    recovered_codes = coder.encode(signal)
    reconstruction = D @ recovered_codes
    
    # Analysis
    print("Basic Sparse Coding Results:")
    print(f"Original sparsity: {np.sum(np.abs(true_codes) > 1e-6)}/{n_atoms}")
    print(f"Recovered sparsity: {np.sum(np.abs(recovered_codes) > 1e-6)}/{n_atoms}")
    print(f"Reconstruction error: {np.linalg.norm(signal - reconstruction):.4f}")
    print(f"Signal norm: {np.linalg.norm(signal):.4f}")
    
    return D, signal, true_codes, recovered_codes

# Example 2: Dictionary learning from data
def dictionary_learning_example():
    """Demonstrate dictionary learning from synthetic data."""
    
    # Generate synthetic sparse-coded data
    n_features = 30
    n_atoms = 40  # Overcomplete
    n_samples = 200
    
    # Create ground truth dictionary
    true_dict = np.random.randn(n_features, n_atoms)
    true_dict /= np.linalg.norm(true_dict, axis=0, keepdims=True)
    
    # Generate sparse codes (only 3-5 non-zero per sample)
    codes = np.zeros((n_atoms, n_samples))
    for i in range(n_samples):
        # Random sparsity level
        sparsity = np.random.randint(3, 6)
        active_atoms = np.random.choice(n_atoms, sparsity, replace=False)
        codes[active_atoms, i] = np.random.randn(sparsity)
    
    # Generate training data
    X = true_dict @ codes + 0.05 * np.random.randn(n_features, n_samples)
    
    # Learn dictionary
    learner = DictionaryLearner(
        n_atoms=n_atoms, 
        max_iter=50,
        method='ksvd',
        sparse_coder_params={'lam': 0.15, 'max_iter': 50}
    )
    
    learned_dict, learned_codes = learner.fit_transform(X)
    
    # Evaluate quality
    reconstruction = learned_dict @ learned_codes
    reconstruction_error = np.linalg.norm(X - reconstruction, 'fro')
    relative_error = reconstruction_error / np.linalg.norm(X, 'fro')
    
    print("\nDictionary Learning Results:")
    print(f"Reconstruction error: {reconstruction_error:.4f}")
    print(f"Relative error: {relative_error:.4f}")
    print(f"Average sparsity: {np.mean(np.sum(np.abs(learned_codes) > 1e-6, axis=0)):.1f} atoms/sample")
    
    return true_dict, learned_dict, X, learned_codes

# Run examples
if __name__ == "__main__":
    # Run basic example
    D, signal, true_codes, recovered_codes = basic_sparse_coding_example()
    
    # Run dictionary learning example  
    true_dict, learned_dict, X, learned_codes = dictionary_learning_example()
```

### Tutorial 2: Advanced Usage and Algorithm Selection

```python
"""
Advanced Sparse Coding: Algorithm Selection and Performance Optimization
=======================================================================

This tutorial covers advanced topics for research and production use.
"""

def algorithm_comparison_example():
    """Compare different sparse coding algorithms."""
    
    from sparse_coding import SparseCoder
    from sparse_coding.core.penalties.implementations import L2Penalty
    import time
    
    # Create test problem
    np.random.seed(42)
    n_features, n_atoms = 100, 50
    D = np.random.randn(n_features, n_atoms)
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    
    # Test signal
    true_codes = np.zeros(n_atoms)
    true_codes[np.random.choice(n_atoms, 8, replace=False)] = np.random.randn(8)
    signal = D @ true_codes + 0.02 * np.random.randn(n_features)
    
    algorithms = [
        ('FISTA (L1)', {'mode': 'l1', 'solver': 'fista', 'lam': 0.1}),
        ('ISTA (L1)', {'mode': 'l1', 'solver': 'ista', 'lam': 0.1}),
        ('OMP (Exact)', {'mode': 'omp', 'sparsity': 8}),
        ('NCG (L2)', {'mode': 'l2', 'solver': 'ncg', 'lam': 0.1}),
    ]
    
    print("Algorithm Comparison Results:")
    print("-" * 60)
    
    for name, params in algorithms:
        coder = SparseCoder(n_atoms=n_atoms, max_iter=200, **params)
        coder.dictionary = D
        
        # Time the encoding
        start_time = time.time()
        codes = coder.encode(signal)
        encode_time = time.time() - start_time
        
        # Compute metrics
        reconstruction = D @ codes
        mse = np.mean((signal - reconstruction)**2)
        sparsity = np.sum(np.abs(codes) > 1e-6)
        
        print(f"{name:15} | MSE: {mse:.2e} | Sparsity: {sparsity:2d} | Time: {encode_time:.3f}s")

def research_reproduction_example():
    """Reproduce results from Olshausen & Field (1996)."""
    
    print("\nOlshausen & Field (1996) Reproduction:")
    print("-" * 40)
    
    # Simulate natural image patches (normally you'd use real patches)
    np.random.seed(42)
    patch_size = 8
    n_features = patch_size * patch_size  # 64
    n_atoms = 128  # 2x overcomplete
    n_patches = 1000
    
    # Generate patches with natural image statistics (simplified)
    patches = np.random.laplace(0, 1, (n_features, n_patches))
    # Add spatial correlation
    for i in range(n_patches):
        patch_2d = patches[:, i].reshape(patch_size, patch_size)
        # Simple blur to add correlation
        for j in range(1, patch_size-1):
            for k in range(1, patch_size-1):
                patch_2d[j, k] = 0.6 * patch_2d[j, k] + 0.1 * (
                    patch_2d[j-1, k] + patch_2d[j+1, k] + 
                    patch_2d[j, k-1] + patch_2d[j, k+1]
                )
        patches[:, i] = patch_2d.flatten()
    
    # Olshausen & Field setup: log-Cauchy prior with NCG
    from sparse_coding import DictionaryLearner
    
    learner = DictionaryLearner(
        n_atoms=n_atoms,
        max_iter=20,  # They used many more iterations
        method='ksvd',
        sparse_coder_params={
            'mode': 'paper',  # Uses log-Cauchy prior + NCG
            'lam': 0.01,
            'max_iter': 50
        }
    )
    
    print("Learning dictionary (this may take a while)...")
    learned_dict, codes = learner.fit_transform(patches)
    
    # Analyze learned features
    sparsity_levels = np.sum(np.abs(codes) > 0.1, axis=0)  # Non-negligible coefficients
    avg_sparsity = np.mean(sparsity_levels)
    
    print(f"Average sparsity level: {avg_sparsity:.1f} atoms per patch")
    print(f"Sparsity ratio: {avg_sparsity/n_atoms:.3f}")
    
    # Check if atoms are localized and oriented (hallmarks of O&F results)
    atom_localizations = []
    for i in range(min(20, n_atoms)):  # Check first 20 atoms
        atom_2d = learned_dict[:, i].reshape(patch_size, patch_size)
        # Measure localization using norm concentration
        center_mass = np.sum(np.abs(atom_2d)) / np.max(np.abs(atom_2d))
        atom_localizations.append(center_mass)
    
    avg_localization = np.mean(atom_localizations)
    print(f"Average localization measure: {avg_localization:.2f} (higher = more localized)")
    
    return learned_dict, codes

# Performance optimization tips
def performance_optimization_example():
    """Demonstrate performance optimization techniques."""
    
    print("\nPerformance Optimization Tips:")
    print("-" * 35)
    
    n_features, n_atoms = 200, 100
    n_signals = 500
    
    # Create test data
    D = np.random.randn(n_features, n_atoms)
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    signals = np.random.randn(n_features, n_signals)
    
    # Tip 1: Batch processing vs individual signals
    coder = SparseCoder(n_atoms=n_atoms, mode='l1', lam=0.1, max_iter=100)
    coder.dictionary = D
    
    # Individual encoding (slower)
    start_time = time.time()
    codes_individual = np.zeros((n_atoms, n_signals))
    for i in range(n_signals):
        codes_individual[:, i] = coder.encode(signals[:, i])
    individual_time = time.time() - start_time
    
    # Batch encoding (faster)
    start_time = time.time()
    codes_batch = coder.encode(signals)  # Process all at once
    batch_time = time.time() - start_time
    
    print(f"Individual processing: {individual_time:.2f}s")
    print(f"Batch processing: {batch_time:.2f}s")
    print(f"Speedup: {individual_time/batch_time:.1f}x")
    
    # Tip 2: Algorithm selection for different scenarios
    scenarios = [
        ("High accuracy needed", "Use mode='l1' with high max_iter (500+)"),
        ("Speed critical", "Use mode='omp' with low sparsity level"),
        ("Smooth penalties", "Use mode='l2' or custom penalties with NCG"),
        ("Research reproduction", "Use mode='paper' for Olshausen & Field"),
    ]
    
    print("\nScenario-Based Algorithm Selection:")
    for scenario, recommendation in scenarios:
        print(f"  {scenario}: {recommendation}")

if __name__ == "__main__":
    algorithm_comparison_example()
    research_reproduction_example()
    performance_optimization_example()
```

## Error Handling and Troubleshooting Guide

### Common Issues and Solutions

```python
"""
Troubleshooting Guide for Common Sparse Coding Issues
====================================================
"""

def troubleshooting_examples():
    """Demonstrate solutions to common problems."""
    
    import numpy as np
    from sparse_coding import SparseCoder, DictionaryLearner
    
    print("Common Issues and Solutions:")
    print("=" * 50)
    
    # Issue 1: Dictionary not normalized
    print("\n1. Dictionary Normalization Issues:")
    D_unnormalized = np.random.randn(50, 25) * 10  # Unnormalized atoms
    signal = np.random.randn(50)
    
    coder = SparseCoder(n_atoms=25, mode='l1', lam=0.1)
    coder.dictionary = D_unnormalized
    
    try:
        codes = coder.encode(signal)
        print("   ⚠️  Warning: Using unnormalized dictionary may cause issues")
        
        # Check atom norms
        atom_norms = np.linalg.norm(D_unnormalized, axis=0)
        print(f"   Atom norms range: [{np.min(atom_norms):.2f}, {np.max(atom_norms):.2f}]")
        
        # Solution: Normalize dictionary
        D_normalized = D_unnormalized / atom_norms[np.newaxis, :]
        coder.dictionary = D_normalized
        codes_fixed = coder.encode(signal)
        print("   ✅ Solution: Normalize dictionary atoms to unit norm")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Issue 2: Poor convergence due to lambda selection
    print("\n2. Lambda Selection Issues:")
    D = np.random.randn(30, 15)
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    signal = np.random.randn(30)
    
    lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    for lam in lambda_values:
        coder = SparseCoder(n_atoms=15, mode='l1', lam=lam, max_iter=100)
        coder.dictionary = D
        codes = coder.encode(signal)
        sparsity = np.mean(np.abs(codes) < 1e-6)
        reconstruction_error = np.linalg.norm(signal - D @ codes)
        
        status = "✅ Good" if 0.3 <= sparsity <= 0.8 else "❌ Poor"
        print(f"   λ={lam:6.3f}: sparsity={sparsity:.3f}, error={reconstruction_error:.3f} {status}")
    
    print("   💡 Tip: Start with λ=0.1, adjust based on desired sparsity")
    
    # Issue 3: Dimension mismatches
    print("\n3. Dimension Mismatch Issues:")
    D = np.random.randn(40, 20)
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    
    # Wrong signal dimension
    wrong_signal = np.random.randn(30)  # Should be 40-D
    
    coder = SparseCoder(n_atoms=20)
    coder.dictionary = D
    
    try:
        codes = coder.encode(wrong_signal)
    except ValueError as e:
        print(f"   ❌ Error: {e}")
        print("   ✅ Solution: Ensure signal matches dictionary's first dimension")
        
        correct_signal = np.random.randn(40)
        codes = coder.encode(correct_signal)
        print(f"   Correct encoding: signal {correct_signal.shape} → codes {codes.shape}")
    
    # Issue 4: Convergence problems
    print("\n4. Convergence Issues:")
    
    # Create ill-conditioned dictionary
    D_ill = np.random.randn(20, 15)
    # Make atoms nearly parallel
    for i in range(1, 5):
        D_ill[:, i] = 0.99 * D_ill[:, 0] + 0.01 * np.random.randn(20)
    D_ill /= np.linalg.norm(D_ill, axis=0, keepdims=True)
    
    signal = np.random.randn(20)
    
    # This might converge slowly or poorly
    coder = SparseCoder(n_atoms=15, mode='l1', lam=0.1, max_iter=50, tol=1e-6)
    coder.dictionary = D_ill
    
    try:
        codes = coder.encode(signal)
        print("   ⚠️  Warning: Ill-conditioned dictionary detected")
        
        # Check condition number
        condition_number = np.linalg.cond(D_ill)
        print(f"   Dictionary condition number: {condition_number:.1e}")
        
        if condition_number > 1e12:
            print("   ❌ Dictionary is numerically singular")
        elif condition_number > 1e6:
            print("   ⚠️  Dictionary is ill-conditioned")
        
        print("   💡 Solutions:")
        print("     - Use more diverse training data")
        print("     - Increase regularization (higher λ)")
        print("     - Use different dictionary learning method")
        print("     - Add noise to break exact dependencies")
        
    except RuntimeError as e:
        print(f"   ❌ Convergence failed: {e}")
        
        # Solution: Increase tolerance or iterations
        coder_relaxed = SparseCoder(n_atoms=15, mode='l1', lam=0.1, 
                                  max_iter=200, tol=1e-4)
        coder_relaxed.dictionary = D_ill
        
        try:
            codes_relaxed = coder_relaxed.encode(signal)
            print("   ✅ Solution: Relaxed tolerance and increased iterations")
        except Exception as e2:
            print(f"   Still failing: {e2}")
    
    # Issue 5: Memory problems with large datasets
    print("\n5. Memory Management:")
    
    # Simulate large dataset scenario
    n_large_samples = 50000  # Large number of samples
    n_features = 100
    print(f"   Large dataset: {n_features} features × {n_large_samples} samples")
    
    estimated_memory_gb = (n_features * n_large_samples * 8) / (1024**3)  # 8 bytes per float64
    print(f"   Estimated memory for signals: {estimated_memory_gb:.2f} GB")
    
    if estimated_memory_gb > 4:  # More than 4GB
        print("   ⚠️  Large memory requirement detected")
        print("   💡 Solutions:")
        print("     - Process in batches:")
        print("       batch_size = 1000")
        print("       for i in range(0, n_samples, batch_size):")
        print("           batch = X[:, i:i+batch_size]")
        print("           codes_batch = coder.encode(batch)")
        print("     - Use online dictionary learning")
        print("     - Reduce precision (float32 instead of float64)")

if __name__ == "__main__":
    troubleshooting_examples()
```

## API Reference Summary

This comprehensive API documentation addresses the key issues identified in the analysis:

1. **Clear parameter documentation** with types, defaults, and valid ranges
2. **Research context** linking algorithms to foundational papers
3. **Practical examples** showing real-world usage patterns
4. **Performance guidance** for algorithm selection and optimization  
5. **Error handling** with common issues and solutions
6. **Mathematical background** explaining the theory behind algorithms

The documentation follows Python documentation best practices with detailed docstrings, type hints, and comprehensive examples that bridge the gap between theoretical understanding and practical implementation.