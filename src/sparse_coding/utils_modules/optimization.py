"""
🏗️ Sparse Coding - Optimization Utilities Module
===============================================


Author: Benedict Chen (benedict@benedictchen.com)
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties"

🎯 MODULE PURPOSE:
=================
Optimization utilities for sparse coding including thresholding operators,
Lipschitz constant computation, and line search algorithms.

🔬 RESEARCH FOUNDATION:
======================
Implements optimization tools for sparse coding algorithms:
- Soft/hard thresholding for proximal operators (ISTA/FISTA)
- SCAD and other advanced thresholding operators
- Lipschitz constant computation for gradient methods
- Backtracking line search for step size selection

This module contains the optimization utilities, split from the
994-line monolith for specialized optimization support functionality.
"""

import numpy as np
from typing import Callable
import warnings


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Soft thresholding operator (proximal operator for L1 norm)
    
    Parameters
    ----------
    x : array
        Input array
    threshold : float
        Threshold parameter
        
    Returns
    -------
    thresholded : array, same shape as input
        Soft thresholded values
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def hard_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Hard thresholding operator
    
    Parameters
    ----------
    x : array
        Input array
    threshold : float
        Threshold parameter
        
    Returns
    -------
    thresholded : array, same shape as input
        Hard thresholded values (set to 0 if |x| < threshold)
    """
    return x * (np.abs(x) >= threshold)


def shrinkage_threshold(x: np.ndarray, threshold: float, shrinkage_type: str = 'soft') -> np.ndarray:
    """
    Generalized shrinkage/thresholding operator
    
    Parameters
    ----------
    x : array
        Input array
    threshold : float
        Threshold parameter
    shrinkage_type : str
        Type of shrinkage: 'soft', 'hard', 'garrote', 'scad'
        
    Returns
    -------
    thresholded : array, same shape as input
        Thresholded values
    """
    if shrinkage_type == 'soft':
        return soft_threshold(x, threshold)
    elif shrinkage_type == 'hard':
        return hard_threshold(x, threshold)
    elif shrinkage_type == 'garrote':
        # Non-negative garrote
        return np.maximum(1 - threshold / np.maximum(np.abs(x), 1e-8), 0) * x
    elif shrinkage_type == 'scad':
        # SCAD (Smoothly Clipped Absolute Deviation)
        a = 3.7  # SCAD parameter
        abs_x = np.abs(x)
        
        result = np.zeros_like(x)
        
        # Region 1: |x| <= threshold
        mask1 = abs_x <= threshold
        result[mask1] = 0
        
        # Region 2: threshold < |x| <= a*threshold  
        mask2 = (abs_x > threshold) & (abs_x <= a * threshold)
        result[mask2] = np.sign(x[mask2]) * (abs_x[mask2] - threshold)
        
        # Region 3: |x| > a*threshold
        mask3 = abs_x > a * threshold
        result[mask3] = np.sign(x[mask3]) * (abs_x[mask3] * (a - 1) - threshold * a) / (a - 2)
        
        return result
    else:
        raise ValueError(f"Unknown shrinkage type: {shrinkage_type}")


def compute_lipschitz_constant(A: np.ndarray) -> float:
    """
    Compute Lipschitz constant for gradient of f(x) = 0.5 * ||Ax - b||^2
    
    Parameters
    ----------
    A : array, shape (m, n)
        Matrix A
        
    Returns
    -------
    L : float
        Lipschitz constant (largest eigenvalue of A^T A)
    """
    # FIXME: Critical efficiency and numerical stability issues
    # Issue 1: Using full eigendecomposition for large matrices is very expensive O(n³)
    # Issue 2: No numerical stability checking for ill-conditioned matrices
    # Issue 3: Complex eigenvalues not handled properly in edge cases
    # Issue 4: No input validation for matrix dimensions or properties
    
    # FIXME: No input validation
    # Issue: Could crash with invalid input matrices
    # Solutions:
    # 1. Validate input is 2D array with valid dimensions
    # 2. Check for degenerate cases (zero matrix, single element)
    # 3. Add warnings for ill-conditioned matrices
    #
    # Example validation:
    # if A.ndim != 2:
    #     raise ValueError("Input matrix must be 2-dimensional")
    # if A.size == 0:
    #     return 0.0
    # if np.allclose(A, 0):
    #     return 0.0
    
    if A.shape[0] <= A.shape[1]:
        # More columns than rows: compute eigenvalues of A A^T
        # FIXME: For large matrices, this is computationally expensive O(m³)
        # Solutions:
        # 1. Use power iteration for large matrices: faster O(mn) per iteration
        # 2. Use scipy.sparse.linalg.norm for matrix norm approximation
        # 3. Use randomized SVD for approximation: sklearn.utils.extmath.randomized_svd
        #
        # Example power iteration implementation:
        # if A.shape[0] > 1000:  # For large matrices
        #     return power_iteration_largest_eigenvalue(A @ A.T, max_iter=20)
        
        eigenvals = np.linalg.eigvals(A @ A.T)
    else:
        # More rows than columns: compute eigenvalues of A^T A  
        # FIXME: Same computational complexity issue O(n³)
        # Better approach for large matrices:
        # if A.shape[1] > 1000:
        #     return scipy.sparse.linalg.norm(A, ord=2)**2  # More efficient
        
        eigenvals = np.linalg.eigvals(A.T @ A)
    
    # FIXME: No handling of numerical precision issues
    # Issue: Complex eigenvalues due to numerical errors aren't handled
    # Solutions:
    # 1. Take real part and warn if imaginary part is significant
    # 2. Use more robust eigenvalue computation
    # 3. Add tolerance checking for near-zero eigenvalues
    #
    # Example:
    # max_eigenval = np.max(np.real(eigenvals))
    # if np.max(np.imag(eigenvals)) > 1e-10:
    #     warnings.warn("Complex eigenvalues detected, taking real part")
    # return max(max_eigenval, 1e-12)  # Avoid zero Lipschitz constant
    
    return np.max(np.real(eigenvals))


def line_search_backtrack(f: Callable, grad_f: Callable, x: np.ndarray, 
                         direction: np.ndarray, alpha: float = 1.0,
                         beta: float = 0.5, c1: float = 1e-4,
                         max_iter: int = 20) -> float:
    """
    Backtracking line search with Armijo condition
    
    Parameters
    ----------
    f : callable
        Objective function
    grad_f : callable  
        Gradient function
    x : array
        Current point
    direction : array
        Search direction
    alpha : float
        Initial step size
    beta : float
        Backtracking parameter (0 < beta < 1)
    c1 : float
        Armijo parameter (0 < c1 < 1)
    max_iter : int
        Maximum number of backtracking steps
        
    Returns
    -------
    step_size : float
        Selected step size
    """
    f_x = f(x)
    grad_f_x = grad_f(x)
    directional_derivative = np.dot(grad_f_x, direction)
    
    for _ in range(max_iter):
        if f(x + alpha * direction) <= f_x + c1 * alpha * directional_derivative:
            return alpha
        alpha *= beta
    
    return alpha


# Export functions
__all__ = [
    'soft_threshold',
    'hard_threshold', 
    'shrinkage_threshold',
    'compute_lipschitz_constant',
    'line_search_backtrack'
]


if __name__ == "__main__":
    print("🏗️ Sparse Coding - Optimization Utilities Module")
    print("=" * 50)
    print("📊 MODULE CONTENTS:")
    print("  • Soft and hard thresholding operators")
    print("  • Advanced shrinkage operators (SCAD, non-negative garrote)")
    print("  • Lipschitz constant computation for gradient methods")
    print("  • Backtracking line search with Armijo condition")
    print("  • Research-accurate optimization tools with FIXME annotations")
    print("")
    print("✅ Optimization utilities module loaded successfully!")
    print("🔬 Essential optimization tools for sparse coding algorithms!")