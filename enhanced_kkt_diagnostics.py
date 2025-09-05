"""
Enhanced KKT Diagnostics for Sparse Coding
==========================================

Research-accurate KKT (Karush-Kuhn-Tucker) condition checking for L1 sparse coding optimization.
Provides comprehensive violation analysis and debugging tools for optimization validation.

Based on:
- Boyd, S., & Vandenberghe, L. (2004). Convex optimization. Chapter 5.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm.
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties.

Author: Benedict Chen
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings

def kkt_violation_comprehensive(D: np.ndarray, X: np.ndarray, A: np.ndarray, 
                              lam: float, tol: float = 1e-12, 
                              detailed: bool = True) -> Dict[str, Any]:
    """
    Comprehensive KKT (Karush-Kuhn-Tucker) condition analysis for L1 sparse coding.
    
    For the optimization problem:
        min_A ½‖X - DA‖²_F + λ‖A‖_1
        
    The KKT conditions are:
    1. For zero coefficients A[i,j] = 0: |G[i,j]| ≤ λ 
    2. For nonzero coefficients A[i,j] ≠ 0: G[i,j] = λ·sign(A[i,j])
    
    Where G = D^T(X - DA) is the dual gradient.
    
    Args:
        D: Dictionary matrix [n_features, n_components]
        X: Data matrix [n_features, n_samples]
        A: Coefficient matrix [n_components, n_samples]
        lam: L1 regularization parameter λ
        tol: Threshold for considering coefficients as zero
        detailed: Whether to return detailed violation analysis
        
    Returns:
        Dict containing comprehensive KKT analysis results
    """
    # Input validation
    if not isinstance(D, np.ndarray) or not isinstance(X, np.ndarray) or not isinstance(A, np.ndarray):
        raise ValueError("D, X, A must be numpy arrays")
    
    if D.shape[1] != A.shape[0]:
        raise ValueError(f"Dictionary columns ({D.shape[1]}) must match coefficient rows ({A.shape[0]})")
    
    if D.shape[0] != X.shape[0]:
        raise ValueError(f"Dictionary rows ({D.shape[0]}) must match data rows ({X.shape[0]})")
        
    if X.shape[1] != A.shape[1]:
        raise ValueError(f"Data samples ({X.shape[1]}) must match coefficient samples ({A.shape[1]})")
    
    if lam <= 0:
        raise ValueError(f"Regularization parameter must be positive, got {lam}")
    
    # Compute dual gradient: G = D^T(X - DA) = D^T*residual
    residual = X - D @ A
    dual_gradient = D.T @ residual
    
    # Identify zero and nonzero coefficient masks
    zero_mask = np.abs(A) < tol
    nonzero_mask = ~zero_mask
    
    n_zero = np.sum(zero_mask)
    n_nonzero = np.sum(nonzero_mask)
    n_total = A.size
    
    # Compute KKT violations
    
    # 1. Zero coefficients: |G| ≤ λ
    violations_zero = np.zeros_like(A)
    if n_zero > 0:
        violations_zero[zero_mask] = np.maximum(0.0, np.abs(dual_gradient[zero_mask]) - lam)
    
    # 2. Nonzero coefficients: G = λ·sign(A)
    violations_nonzero = np.zeros_like(A)
    if n_nonzero > 0:
        expected_gradient = lam * np.sign(A[nonzero_mask])
        violations_nonzero[nonzero_mask] = np.abs(dual_gradient[nonzero_mask] - expected_gradient)
    
    # Aggregate violation statistics
    max_violation_zero = np.max(violations_zero) if n_zero > 0 else 0.0
    max_violation_nonzero = np.max(violations_nonzero) if n_nonzero > 0 else 0.0
    max_violation_total = max(max_violation_zero, max_violation_nonzero)
    
    mean_violation_zero = np.mean(violations_zero[zero_mask]) if n_zero > 0 else 0.0
    mean_violation_nonzero = np.mean(violations_nonzero[nonzero_mask]) if n_nonzero > 0 else 0.0
    
    # Overall KKT satisfaction check
    kkt_tolerance = 1e-3  # Standard tolerance for optimization
    kkt_satisfied = max_violation_total <= kkt_tolerance
    
    # Build results dictionary
    results = {
        'max_violation': float(max_violation_total),
        'max_violation_zero': float(max_violation_zero),
        'max_violation_nonzero': float(max_violation_nonzero),
        'mean_violation_zero': float(mean_violation_zero),
        'mean_violation_nonzero': float(mean_violation_nonzero),
        'kkt_satisfied': kkt_satisfied,
        'kkt_tolerance': kkt_tolerance,
        'n_zero_coeffs': int(n_zero),
        'n_nonzero_coeffs': int(n_nonzero),
        'sparsity_level': float(n_zero / n_total),
        'regularization_param': float(lam),
        'zero_threshold': float(tol)
    }
    
    if detailed:
        # Add detailed violation maps and statistics
        results.update({
            'violations_zero': violations_zero,
            'violations_nonzero': violations_nonzero,
            'dual_gradient': dual_gradient,
            'zero_mask': zero_mask,
            'nonzero_mask': nonzero_mask,
            'residual_norm': float(np.linalg.norm(residual)),
            'coefficient_norm': float(np.linalg.norm(A)),
            'gradient_norm': float(np.linalg.norm(dual_gradient)),
            'reconstruction_error': float(np.linalg.norm(residual) ** 2 / (2 * A.shape[1]))
        })
        
        # Per-sample violation statistics
        sample_violations = []
        for i in range(A.shape[1]):
            sample_viol_zero = np.max(violations_zero[:, i]) if np.any(zero_mask[:, i]) else 0.0
            sample_viol_nonzero = np.max(violations_nonzero[:, i]) if np.any(nonzero_mask[:, i]) else 0.0
            sample_violations.append(max(sample_viol_zero, sample_viol_nonzero))
        
        results['per_sample_violations'] = np.array(sample_violations)
        results['worst_sample_idx'] = int(np.argmax(sample_violations))
        
        # Dictionary condition analysis
        if D.shape[1] > D.shape[0]:  # Overcomplete
            # Compute dictionary coherence (mutual coherence)
            D_norm = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
            gram_matrix = D_norm.T @ D_norm
            np.fill_diagonal(gram_matrix, 0)
            mutual_coherence = np.max(np.abs(gram_matrix))
            results['dictionary_coherence'] = float(mutual_coherence)
            
            # Check if dictionary is well-conditioned for sparse recovery
            # Rule of thumb: mutual coherence < 1/(2k-1) where k is sparsity level
            avg_sparsity = np.mean(np.sum(nonzero_mask, axis=0))
            coherence_bound = 1.0 / (2 * avg_sparsity - 1) if avg_sparsity > 0.5 else 1.0
            results['coherence_bound'] = float(coherence_bound)
            results['coherence_satisfied'] = mutual_coherence < coherence_bound
    
    return results

def diagnose_kkt_violations(kkt_results: Dict[str, Any], verbose: bool = True) -> None:
    """
    Diagnose and provide guidance on KKT violations.
    
    Args:
        kkt_results: Results from kkt_violation_comprehensive
        verbose: Whether to print detailed diagnostics
    """
    if not verbose:
        return
        
    print("=" * 60)
    print("🔍 KKT CONDITION ANALYSIS")
    print("=" * 60)
    
    # Overall status
    status_emoji = "✅" if kkt_results['kkt_satisfied'] else "❌"
    print(f"{status_emoji} KKT Conditions: {'SATISFIED' if kkt_results['kkt_satisfied'] else 'VIOLATED'}")
    print(f"   Maximum violation: {kkt_results['max_violation']:.2e}")
    print(f"   Tolerance: {kkt_results['kkt_tolerance']:.2e}")
    
    # Sparsity analysis
    print(f"\n📊 SPARSITY ANALYSIS:")
    print(f"   Zero coefficients: {kkt_results['n_zero_coeffs']:,} ({kkt_results['sparsity_level']:.1%})")
    print(f"   Nonzero coefficients: {kkt_results['n_nonzero_coeffs']:,}")
    print(f"   L1 penalty: λ = {kkt_results['regularization_param']:.4f}")
    
    # Violation breakdown
    print(f"\n🔍 VIOLATION BREAKDOWN:")
    print(f"   Zero coeffs max violation: {kkt_results['max_violation_zero']:.2e}")
    print(f"   Nonzero coeffs max violation: {kkt_results['max_violation_nonzero']:.2e}")
    print(f"   Zero coeffs mean violation: {kkt_results['mean_violation_zero']:.2e}")
    print(f"   Nonzero coeffs mean violation: {kkt_results['mean_violation_nonzero']:.2e}")
    
    # Detailed analysis if available
    if 'per_sample_violations' in kkt_results:
        worst_idx = kkt_results['worst_sample_idx']
        worst_violation = kkt_results['per_sample_violations'][worst_idx]
        print(f"\n🎯 PER-SAMPLE ANALYSIS:")
        print(f"   Worst sample index: {worst_idx}")
        print(f"   Worst violation: {worst_violation:.2e}")
        print(f"   Reconstruction error: {kkt_results['reconstruction_error']:.4f}")
    
    # Dictionary analysis if available
    if 'dictionary_coherence' in kkt_results:
        coherence_ok = "✅" if kkt_results['coherence_satisfied'] else "⚠️"
        print(f"\n🔧 DICTIONARY ANALYSIS:")
        print(f"   {coherence_ok} Mutual coherence: {kkt_results['dictionary_coherence']:.4f}")
        print(f"   Coherence bound: {kkt_results['coherence_bound']:.4f}")
        print(f"   Well-conditioned: {'Yes' if kkt_results['coherence_satisfied'] else 'No'}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if kkt_results['kkt_satisfied']:
        print("   ✅ Optimization converged to KKT-optimal solution")
        print("   ✅ Solution satisfies necessary optimality conditions")
    else:
        print("   ❌ Optimization did not converge to optimal solution")
        
        if kkt_results['max_violation_zero'] > kkt_results['max_violation_nonzero']:
            print("   🎯 Primary issue: Zero coefficient violations")
            print("      → Increase max iterations")
            print("      → Check gradient computation accuracy")
            print("      → Reduce step size for better convergence")
        else:
            print("   🎯 Primary issue: Nonzero coefficient violations")
            print("      → Check proximal operator implementation")
            print("      → Verify soft thresholding accuracy")
            print("      → Consider reducing regularization parameter")
            
        if 'dictionary_coherence' in kkt_results and not kkt_results['coherence_satisfied']:
            print("   ⚠️  Dictionary has high mutual coherence")
            print("      → Consider dictionary learning to reduce coherence")
            print("      → Increase regularization for more stable recovery")
    
    print("=" * 60)

def kkt_violation_simple(D: np.ndarray, X: np.ndarray, A: np.ndarray, 
                        lam: float, tol: float = 1e-12) -> float:
    """
    Simple KKT violation computation (backward compatible).
    
    Returns maximum KKT violation as a single scalar.
    """
    results = kkt_violation_comprehensive(D, X, A, lam, tol, detailed=False)
    return results['max_violation']

def test_kkt_implementation():
    """Test the enhanced KKT implementation with synthetic data."""
    print("Testing Enhanced KKT Implementation")
    print("=" * 40)
    
    # Create synthetic sparse coding problem
    np.random.seed(42)
    n_features, n_components, n_samples = 20, 30, 10
    
    # Create dictionary
    D = np.random.randn(n_features, n_components)
    D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    
    # Create sparse coefficients
    A = np.random.randn(n_components, n_samples)
    A[np.abs(A) < 1.0] = 0  # Make sparse
    
    # Generate data
    X = D @ A + 0.01 * np.random.randn(n_features, n_samples)
    
    # Test KKT analysis
    print(f"Problem size: D{D.shape}, X{X.shape}, A{A.shape}")
    print(f"True sparsity: {np.mean(np.abs(A) < 1e-12):.2%}")
    
    # Comprehensive KKT analysis
    kkt_results = kkt_violation_comprehensive(D, X, A, lam=0.1, detailed=True)
    diagnose_kkt_violations(kkt_results, verbose=True)
    
    # Test simple interface
    simple_violation = kkt_violation_simple(D, X, A, lam=0.1)
    print(f"\nSimple violation: {simple_violation:.2e}")
    
    return kkt_results

if __name__ == "__main__":
    test_kkt_implementation()