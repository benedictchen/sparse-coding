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
    # FIXME: Missing input validation and error handling
    # Issue 1: No shape compatibility checking between X, dictionary, and codes
    # Issue 2: No handling of edge cases (empty arrays, NaN/Inf values)
    # Issue 3: Missing validation of mathematical constraints (dictionary norms)
    # Issue 4: No detection of degenerate solutions
    
    # FIXME: No input shape validation
    # Issue: Matrix multiplication could fail with incompatible shapes
    # Solutions:
    # 1. Validate shapes are compatible: codes.shape = (n_samples, n_components), dict.shape = (n_components, n_features)
    # 2. Check X.shape = (n_samples, n_features)
    # 3. Provide clear error messages for shape mismatches
    #
    # Example validation:
    # if X.shape[0] != codes.shape[0]:
    #     raise ValueError(f"X and codes must have same number of samples: {X.shape[0]} vs {codes.shape[0]}")
    # if dictionary.shape[0] != codes.shape[1]:
    #     raise ValueError(f"Dictionary components {dictionary.shape[0]} != codes components {codes.shape[1]}")
    # if dictionary.shape[1] != X.shape[1]:
    #     raise ValueError(f"Dictionary features {dictionary.shape[1]} != X features {X.shape[1]}")
    
    # FIXME: No NaN/Inf detection
    # Issue: NaN or Inf values in inputs will propagate through all calculations
    # Solutions:
    # 1. Check for NaN/Inf in all inputs and raise informative errors
    # 2. Add option to handle or remove invalid samples
    # 3. Warn about numerical issues that might cause problems
    #
    # Example:
    # for name, arr in [("X", X), ("dictionary", dictionary), ("codes", codes)]:
    #     if np.any(np.isnan(arr)):
    #         raise ValueError(f"NaN values detected in {name}")
    #     if np.any(np.isinf(arr)):
    #         raise ValueError(f"Inf values detected in {name}")
    
    # Reconstruction
    X_reconstructed = codes @ dictionary
    
    # Basic metrics
    reconstruction_error = mean_squared_error(X, X_reconstructed)
    
    # FIXME: Division by zero risk in relative error calculation
    # Issue: np.var(X) could be zero for constant data
    # Solutions:
    # 1. Add small epsilon to denominator
    # 2. Handle zero variance case explicitly
    # 3. Use alternative relative error metrics
    #
    # Example:
    # var_X = np.var(X)
    # if var_X < 1e-12:
    #     relative_error = np.inf if reconstruction_error > 0 else 0.0
    # else:
    #     relative_error = reconstruction_error / var_X
    
    relative_error = reconstruction_error / np.var(X)
    
    # Sparsity metrics
    sparsity = np.mean(np.sum(codes != 0, axis=1))  # Average non-zeros per sample
    sparsity_ratio = sparsity / codes.shape[1]  # Fraction of non-zero coefficients
    
    # Dictionary properties
    dict_norms = np.linalg.norm(dictionary, axis=1)
    dict_coherence = compute_dictionary_coherence(dictionary)
    
    # FIXME: Potential division by zero in code statistics
    # Issue: If all codes are zero, codes[codes != 0] is empty
    # Solutions:
    # 1. Check for empty arrays before computing statistics
    # 2. Return NaN or special values for degenerate cases
    # 3. Add warnings for unusual sparsity patterns
    #
    # Example:
    # nonzero_codes = codes[codes != 0]
    # if len(nonzero_codes) == 0:
    #     code_mean = 0.0
    #     code_std = 0.0
    #     warnings.warn("All sparse codes are zero - possible convergence failure")
    # else:
    #     code_mean = np.mean(np.abs(nonzero_codes))
    #     code_std = np.std(nonzero_codes)
    
    # Code statistics
    code_mean = np.mean(np.abs(codes[codes != 0]))
    code_std = np.std(codes[codes != 0]) if np.sum(codes != 0) > 1 else 0
    
    # FIXME: Missing additional validation metrics
    # Issue: Could add more comprehensive validation measures
    # Solutions:
    # 1. Add condition number of dictionary (numerical stability)
    # 2. Compute residual statistics (should be white noise)
    # 3. Check for dead dictionary atoms (never used)
    # 4. Validate sparsity level is reasonable (not too sparse/dense)
    #
    # Additional metrics to consider:
    # 'dictionary_condition_number': np.linalg.cond(dictionary),
    # 'dead_atoms': np.sum(np.all(codes == 0, axis=0)),  # Unused dictionary atoms
    # 'residual_whiteness': test_residual_whiteness(X - X_reconstructed),
    # 'effective_sparsity': compute_effective_sparsity(codes),
    
    return {
        'reconstruction_error': reconstruction_error,
        'relative_reconstruction_error': relative_error,
        'mean_sparsity': sparsity,
        'sparsity_ratio': sparsity_ratio,
        'dictionary_coherence': dict_coherence,
        'dictionary_norm_mean': np.mean(dict_norms),
        'dictionary_norm_std': np.std(dict_norms),
        'code_magnitude_mean': code_mean,
        'code_magnitude_std': code_std,
        'snr_db': 10 * np.log10(np.var(X) / reconstruction_error) if reconstruction_error > 0 else np.inf
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
    print("🏗️ Sparse Coding - Validation and Metrics Module")
    print("=" * 50)
    print("📊 MODULE CONTENTS:")
    print("  • Comprehensive sparse coding data validation")
    print("  • Dictionary coherence and spark computation")
    print("  • Convergence analysis for optimization algorithms")
    print("  • Quality metrics with extensive FIXME research accuracy notes")
    print("  • SNR, sparsity, and reconstruction error analysis")
    print("")
    print("✅ Validation and metrics module loaded successfully!")
    print("🔬 Comprehensive quality assessment for sparse coding!")