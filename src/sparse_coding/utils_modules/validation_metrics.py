"""
📋 Validation Metrics
======================

🎯 ELI5 Summary:
This file is an important component in our AI research system! Like different organs 
in your body that work together to keep you healthy, this file has a specific job that 
helps the overall algorithm work correctly and efficiently.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

📋 Component Integration:
========================
    ┌──────────┐
    │   This   │
    │Component │ ←→ Other Components
    └──────────┘
         ↑↓
    System Integration

"""
"""
🏗️ Sparse Coding - Validation and Metrics Module
===============================================


Author: Benedict Chen (benedict@benedictchen.com)
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties"

🎯 MODULE PURPOSE:
=================
Validation and metrics utilities for sparse coding including data validation,
dictionary coherence computation, convergence analysis, and quality metrics.

🔬 RESEARCH FOUNDATION:
======================
Implements validation methods for sparse coding quality assessment:
- Reconstruction error and relative metrics for algorithm performance
- Dictionary coherence and spark computation for theoretical analysis
- Convergence validation for optimization algorithms
- Comprehensive sparsity and quality metrics

This module contains the validation and metrics components, split from the
994-line monolith for specialized quality assessment functionality.
"""

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, List
import warnings


def validate_sparse_coding_data(X: np.ndarray, dictionary: np.ndarray, 
                               codes: np.ndarray) -> Dict[str, Any]:
    """
    Comprehensive validation of sparse coding results
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Original data
    dictionary : array, shape (n_components, n_features)  
        Learned dictionary
    codes : array, shape (n_samples, n_components)
        Sparse codes
        
    Returns
    -------
    metrics : dict
        Dictionary of validation metrics
    """
    # Comprehensive input validation for robust sparse coding validation
    
    # Validate array inputs are proper numpy arrays
    if not all(isinstance(arr, np.ndarray) for arr in [X, dictionary, codes]):
        raise ValueError("All inputs must be numpy arrays")
    
    # Validate 2D arrays
    if not all(arr.ndim == 2 for arr in [X, dictionary, codes]):
        raise ValueError("All inputs must be 2D arrays")
    
    # Comprehensive shape validation for matrix compatibility
    if X.shape[0] != codes.shape[0]:
        raise ValueError(f"X and codes must have same number of samples: {X.shape[0]} vs {codes.shape[0]}")
    if dictionary.shape[0] != codes.shape[1]:
        raise ValueError(f"Dictionary components {dictionary.shape[0]} != codes components {codes.shape[1]}")
    if dictionary.shape[1] != X.shape[1]:
        raise ValueError(f"Dictionary features {dictionary.shape[1]} != X features {X.shape[1]}")
    
    # Validate non-empty arrays
    if X.size == 0 or dictionary.size == 0 or codes.size == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Numerical quality validation to prevent corrupted results
    for name, arr in [("X", X), ("dictionary", dictionary), ("codes", codes)]:
        if np.any(np.isnan(arr)):
            raise ValueError(f"NaN values detected in {name} - indicates numerical instability")
        if np.any(np.isinf(arr)):
            raise ValueError(f"Inf values detected in {name} - indicates overflow or division by zero")
    
    # Validate dictionary mathematical constraints
    dict_norms_check = np.linalg.norm(dictionary, axis=1)
    zero_norm_atoms = np.sum(dict_norms_check < 1e-12)
    if zero_norm_atoms > 0:
        warnings.warn(f"{zero_norm_atoms} dictionary atoms have near-zero norm (dead atoms)")
    
    # Check for degenerate solutions
    if np.all(codes == 0):
        warnings.warn("All sparse codes are zero - indicates convergence failure or excessive regularization")
    
    # Reconstruction
    X_reconstructed = codes @ dictionary
    
    # Basic metrics
    reconstruction_error = mean_squared_error(X, X_reconstructed)
    
    # Robust relative error calculation handling constant data
    var_X = np.var(X)
    if var_X < 1e-12:  # Constant or near-constant data
        if reconstruction_error > 1e-12:
            relative_error = np.inf  # Perfect reconstruction impossible
            warnings.warn("Input data has zero variance - relative error is infinite")
        else:
            relative_error = 0.0  # Perfect reconstruction of constant data
    else:
        relative_error = reconstruction_error / var_X
    
    # Sparsity metrics
    sparsity = np.mean(np.sum(codes != 0, axis=1))  # Average non-zeros per sample
    sparsity_ratio = sparsity / codes.shape[1]  # Fraction of non-zero coefficients
    
    # Dictionary properties
    dict_norms = np.linalg.norm(dictionary, axis=1)
    dict_coherence = compute_dictionary_coherence(dictionary)
    
    # Robust code statistics computation handling edge cases
    nonzero_codes = codes[codes != 0]
    if len(nonzero_codes) == 0:
        code_mean = 0.0
        code_std = 0.0
        warnings.warn("All sparse codes are zero - possible convergence failure or excessive sparsity")
    else:
        code_mean = np.mean(np.abs(nonzero_codes))
        if len(nonzero_codes) > 1:
            code_std = np.std(nonzero_codes)
        else:
            code_std = 0.0  # Single nonzero value has zero variance
            warnings.warn("Only one nonzero coefficient - insufficient statistics")
    
    # Additional comprehensive validation metrics for research quality assessment
    dictionary_condition_number = np.linalg.cond(dictionary)
    dead_atoms = np.sum(np.all(codes == 0, axis=0))  # Unused dictionary atoms
    
    # Effective sparsity (Hoyer's measure): (√n - ||x||₁/||x||₂) / (√n - 1)
    def compute_effective_sparsity(x):
        if np.linalg.norm(x) < 1e-12:
            return 1.0  # All zeros = maximally sparse
        n = len(x)
        l1_norm = np.sum(np.abs(x))
        l2_norm = np.linalg.norm(x)
        return (np.sqrt(n) - l1_norm / l2_norm) / (np.sqrt(n) - 1)
    
    effective_sparsity_mean = np.mean([compute_effective_sparsity(codes[i]) for i in range(codes.shape[0])])
    
    # Residual analysis (should be white noise for good reconstruction)
    residual = X - X_reconstructed
    residual_mean = np.mean(residual)
    residual_autocorr = np.corrcoef(residual.flatten()[:-1], residual.flatten()[1:])[0, 1] if residual.size > 1 else 0
    
    # Sparsity level validation
    if sparsity_ratio < 0.01:
        warnings.warn(f"Very sparse codes ({sparsity_ratio*100:.2f}%) - may indicate over-regularization")
    elif sparsity_ratio > 0.5:
        warnings.warn(f"Dense codes ({sparsity_ratio*100:.2f}%) - may indicate under-regularization")
    
    if dead_atoms > 0:
        warnings.warn(f"{dead_atoms}/{dictionary.shape[0]} dictionary atoms are never used (dead atoms)")
    
    if dictionary_condition_number > 1e12:
        warnings.warn(f"Dictionary is poorly conditioned (κ = {dictionary_condition_number:.2e})")
    
    # Return comprehensive validation metrics with all FIXME solutions implemented
    return {
        # Basic reconstruction metrics
        'reconstruction_error': reconstruction_error,
        'relative_reconstruction_error': relative_error,
        'snr_db': 10 * np.log10(np.var(X) / reconstruction_error) if reconstruction_error > 0 else np.inf,
        
        # Sparsity analysis
        'mean_sparsity': sparsity,
        'sparsity_ratio': sparsity_ratio,
        'effective_sparsity': effective_sparsity_mean,
        
        # Dictionary properties
        'dictionary_coherence': dict_coherence,
        'dictionary_condition_number': dictionary_condition_number,
        'dictionary_norm_mean': np.mean(dict_norms),
        'dictionary_norm_std': np.std(dict_norms),
        'dead_atoms_count': dead_atoms,
        'dead_atoms_ratio': dead_atoms / dictionary.shape[0],
        
        # Code statistics
        'code_magnitude_mean': code_mean,
        'code_magnitude_std': code_std,
        'nonzero_codes_count': len(nonzero_codes),
        
        # Residual analysis
        'residual_mean': residual_mean,
        'residual_autocorr': residual_autocorr,
        'residual_variance': np.var(residual)
    }


def compute_dictionary_coherence(dictionary: np.ndarray) -> float:
    """
    Compute mutual coherence of dictionary (maximum off-diagonal correlation)
    
    Parameters
    ----------
    dictionary : array, shape (n_components, n_features)
        Dictionary matrix (each row is an atom)
        
    Returns
    -------
    coherence : float
        Maximum absolute correlation between distinct dictionary atoms
    """
    # Normalize dictionary atoms
    normalized_dict = normalize(dictionary, axis=1)
    
    # Compute Gram matrix
    gram = normalized_dict @ normalized_dict.T
    
    # Set diagonal to zero and find maximum
    np.fill_diagonal(gram, 0)
    
    return np.max(np.abs(gram))


def compute_spark(dictionary: np.ndarray, tolerance: float = 1e-6) -> int:
    """
    Estimate the spark of a dictionary (smallest linearly dependent subset)
    
    Parameters
    ----------
    dictionary : array, shape (n_components, n_features)
        Dictionary matrix
    tolerance : float
        Numerical tolerance for linear dependence
        
    Returns
    -------
    spark : int
        Estimated spark of the dictionary
    """
    # For efficiency, we use SVD to estimate rather than exact computation
    U, s, Vt = np.linalg.svd(dictionary.T, full_matrices=False)
    
    # Count significant singular values
    rank = np.sum(s > tolerance)
    
    # Spark is at least rank + 1 for overdetermined dictionaries
    return rank + 1


def validate_convergence(history: List[float], window: int = 10, 
                        tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Validate convergence of optimization algorithm
    
    Parameters
    ----------
    history : list
        History of objective function values
    window : int
        Window size for convergence checking
    tolerance : float
        Convergence tolerance
        
    Returns
    -------
    convergence_info : dict
        Convergence analysis results
    """
    if len(history) < window + 1:
        return {
            'converged': False,
            'reason': 'Insufficient iterations',
            'relative_change': np.inf,
            'monotonic': None
        }
    
    recent = history[-window:]
    
    # Check relative change
    relative_change = abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-8)
    converged = relative_change < tolerance
    
    # Check monotonicity
    differences = np.diff(history)
    monotonic_decreasing = np.all(differences <= 0)
    monotonic_increasing = np.all(differences >= 0)
    
    if monotonic_decreasing:
        monotonic = 'decreasing'
    elif monotonic_increasing:
        monotonic = 'increasing'
    else:
        monotonic = 'non-monotonic'
    
    return {
        'converged': converged,
        'reason': 'Converged' if converged else 'Not converged',
        'relative_change': relative_change,
        'monotonic': monotonic,
        'final_value': history[-1],
        'total_decrease': history[0] - history[-1] if len(history) > 1 else 0
    }


# Export functions
__all__ = [
    'validate_sparse_coding_data',
    'compute_dictionary_coherence',
    'compute_spark',
    'validate_convergence'
]


if __name__ == "__main__":
    # print("🏗️ Sparse Coding - Validation and Metrics Module")
    print("=" * 50)
    # Removed print spam: "...
    print("  • Comprehensive sparse coding data validation")
    print("  • Dictionary coherence and spark computation")
    print("  • Convergence analysis for optimization algorithms")
    print("  • Quality metrics with extensive FIXME research accuracy notes")
    print("  • SNR, sparsity, and reconstruction error analysis")
    print("")
    # # Removed print spam: "...
    print("🔬 Comprehensive quality assessment for sparse coding!")