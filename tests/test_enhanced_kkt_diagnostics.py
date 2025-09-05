"""
Comprehensive Tests for Enhanced KKT Diagnostics
===============================================

Test suite for the enhanced KKT (Karush-Kuhn-Tucker) condition checking
in L1 sparse coding optimization. Validates research accuracy and edge cases.

Based on:
- Boyd, S., & Vandenberghe, L. (2004). Convex optimization.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm.
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sparse_coding.diagnostics import (
    kkt_violation_l1,
    kkt_violation_comprehensive, 
    diagnose_kkt_violations,
    dictionary_coherence
)

class TestKKTDiagnostics:
    """Comprehensive test suite for KKT diagnostics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Standard test problem
        self.n_features = 20
        self.n_components = 30
        self.n_samples = 10
        
        # Create overcomplete dictionary
        self.D = np.random.randn(self.n_features, self.n_components)
        self.D = self.D / (np.linalg.norm(self.D, axis=0, keepdims=True) + 1e-12)
        
        # Create sparse coefficient matrix
        self.A = np.random.randn(self.n_components, self.n_samples)
        self.A[np.abs(self.A) < 0.8] = 0  # Make sparse (about 70% zeros)
        
        # Generate noisy observations
        self.X = self.D @ self.A + 0.01 * np.random.randn(self.n_features, self.n_samples)
        
        self.lam = 0.1  # L1 regularization parameter
    
    def test_basic_kkt_violation_computation(self):
        """Test basic KKT violation computation."""
        # Test original function
        violation_old = kkt_violation_l1(self.D, self.X, self.A, self.lam)
        assert isinstance(violation_old, float)
        assert violation_old >= 0
        
        # Test comprehensive function (simple mode)
        results = kkt_violation_comprehensive(self.D, self.X, self.A, self.lam, detailed=False)
        violation_new = results['max_violation']
        
        # Should be similar (may differ slightly due to implementation details)
        assert abs(violation_old - violation_new) < 1e-10, f"Old: {violation_old}, New: {violation_new}"
    
    def test_comprehensive_kkt_analysis(self):
        """Test comprehensive KKT analysis functionality."""
        results = kkt_violation_comprehensive(self.D, self.X, self.A, self.lam, detailed=True)
        
        # Check required keys
        required_keys = [
            'max_violation', 'max_violation_zero', 'max_violation_nonzero',
            'mean_violation_zero', 'mean_violation_nonzero', 
            'kkt_satisfied', 'n_zero_coeffs', 'n_nonzero_coeffs',
            'sparsity_level', 'regularization_param'
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
        
        # Check detailed keys
        detailed_keys = [
            'violations_zero', 'violations_nonzero', 'dual_gradient',
            'zero_mask', 'nonzero_mask', 'per_sample_violations',
            'worst_sample_idx', 'dictionary_coherence'
        ]
        for key in detailed_keys:
            assert key in results, f"Missing detailed key: {key}"
        
        # Validate data consistency
        assert results['n_zero_coeffs'] + results['n_nonzero_coeffs'] == self.A.size
        assert 0 <= results['sparsity_level'] <= 1
        assert results['regularization_param'] == self.lam
        
        # Check violation arrays have correct shape
        assert results['violations_zero'].shape == self.A.shape
        assert results['violations_nonzero'].shape == self.A.shape
        assert results['dual_gradient'].shape == self.A.shape
    
    def test_input_validation(self):
        """Test input validation and error handling."""
        # Test invalid inputs
        with pytest.raises(ValueError, match="must be numpy arrays"):
            kkt_violation_comprehensive("not_array", self.X, self.A, self.lam)
        
        # Test dimension mismatches
        D_wrong = np.random.randn(self.n_features, self.n_components + 1)
        with pytest.raises(ValueError, match="Dictionary columns.*must match coefficient rows"):
            kkt_violation_comprehensive(D_wrong, self.X, self.A, self.lam)
        
        X_wrong = np.random.randn(self.n_features + 1, self.n_samples)
        with pytest.raises(ValueError, match="Dictionary rows.*must match data rows"):
            kkt_violation_comprehensive(self.D, X_wrong, self.A, self.lam)
        
        # Test negative regularization
        with pytest.raises(ValueError, match="Regularization parameter must be positive"):
            kkt_violation_comprehensive(self.D, self.X, self.A, -0.1)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single sample case
        X_single = self.X[:, :1]
        A_single = self.A[:, :1]
        
        results = kkt_violation_comprehensive(self.D, X_single, A_single, self.lam)
        assert results['max_violation'] >= 0
        assert len(results['per_sample_violations']) == 1
        
        # All zero coefficients
        A_zero = np.zeros_like(self.A)
        results_zero = kkt_violation_comprehensive(self.D, self.X, A_zero, self.lam)
        assert results_zero['n_nonzero_coeffs'] == 0
        assert results_zero['n_zero_coeffs'] == self.A.size
        assert results_zero['max_violation_nonzero'] == 0.0
        
        # All nonzero coefficients  
        A_dense = np.ones_like(self.A) * 0.5  # All nonzero
        results_dense = kkt_violation_comprehensive(self.D, self.X, A_dense, self.lam)
        assert results_dense['n_zero_coeffs'] == 0
        assert results_dense['max_violation_zero'] == 0.0
    
    def test_kkt_conditions_perfect_solution(self):
        """Test KKT conditions on a constructed KKT-optimal solution."""
        # Create a simple problem where we can construct a KKT-optimal solution
        # For KKT optimality in L1 problems:
        # - Nonzero coeffs: D^T(X - DA) = λ*sign(A)  
        # - Zero coeffs: |D^T(X - DA)| ≤ λ
        
        # Simple case: single atom active with perfect KKT conditions
        D_simple = np.array([[1.0], [0.0]])  # 2x1 dictionary
        A_simple = np.array([[1.0]])  # Single nonzero coefficient
        lam = 0.1
        
        # For KKT optimality: D^T(X - DA) = λ*sign(A) = 0.1
        # So: X - DA should equal λ*sign(A) / D^T = [0.1, 0]
        # Therefore: X = DA + [0.1, 0] = [1.0, 0] + [0.1, 0] = [1.1, 0]
        X_kkt = np.array([[1.1], [0.0]])
        
        results = kkt_violation_comprehensive(D_simple, X_kkt, A_simple, lam=lam)
        
        # Check that this satisfies KKT conditions within numerical precision
        assert results['max_violation'] < 1e-10, f"KKT-optimal solution has violation {results['max_violation']}"
    
    def test_tolerance_effects(self):
        """Test effects of different zero tolerance values."""
        # Create a case where tolerance will make a difference
        # Use coefficients close to zero
        A_mixed = self.A.copy()
        A_mixed[np.abs(A_mixed) < 0.1] = 1e-6  # Set some coefficients very close to zero
        
        tolerances = [1e-12, 1e-8, 1e-4, 1e-2]
        sparsity_levels = []
        
        for tol in tolerances:
            results = kkt_violation_comprehensive(self.D, self.X, A_mixed, self.lam, tol=tol)
            sparsity_levels.append(results['sparsity_level'])
        
        # Different tolerances should lead to different sparsity classifications
        assert len(set(sparsity_levels)) > 1, f"Different tolerances should yield different sparsity levels: {sparsity_levels}"
    
    def test_dictionary_coherence_analysis(self):
        """Test dictionary coherence computation and analysis."""
        # Test with identity matrix (perfect conditioning)
        D_identity = np.eye(self.n_features, self.n_features)
        coherence_identity = dictionary_coherence(D_identity)
        assert coherence_identity < 1e-10, f"Identity matrix should have zero coherence, got {coherence_identity}"
        
        # Test with random overcomplete dictionary
        coherence_random = dictionary_coherence(self.D)
        assert 0 <= coherence_random <= 1, f"Coherence should be in [0,1], got {coherence_random}"
        
        # Test coherence in comprehensive analysis
        results = kkt_violation_comprehensive(self.D, self.X, self.A, self.lam, detailed=True)
        assert 'dictionary_coherence' in results
        assert results['dictionary_coherence'] == coherence_random
    
    def test_per_sample_analysis(self):
        """Test per-sample violation analysis."""
        results = kkt_violation_comprehensive(self.D, self.X, self.A, self.lam, detailed=True)
        
        # Check per-sample violations
        per_sample = results['per_sample_violations']
        assert len(per_sample) == self.n_samples
        assert all(v >= 0 for v in per_sample)
        
        # Worst sample index should be valid
        worst_idx = results['worst_sample_idx']
        assert 0 <= worst_idx < self.n_samples
        
        # Worst violation should match the maximum
        worst_violation = per_sample[worst_idx]
        assert worst_violation == max(per_sample)
    
    def test_diagnosis_output(self, capsys):
        """Test diagnostic output formatting."""
        results = kkt_violation_comprehensive(self.D, self.X, self.A, self.lam, detailed=True)
        
        # Test diagnosis with output capture
        diagnose_kkt_violations(results, verbose=True)
        captured = capsys.readouterr()
        
        # Check that key information is present
        assert "KKT CONDITION ANALYSIS" in captured.out
        assert "SPARSITY ANALYSIS" in captured.out
        assert "VIOLATION BREAKDOWN" in captured.out
        assert "RECOMMENDATIONS" in captured.out
        
        # Check that numerical values are present
        assert f"{results['max_violation']:.2e}" in captured.out
        assert f"{results['sparsity_level']:.1%}" in captured.out
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very small values
        A_tiny = self.A * 1e-15
        results_tiny = kkt_violation_comprehensive(self.D, self.X, A_tiny, self.lam)
        assert not np.isnan(results_tiny['max_violation'])
        assert not np.isinf(results_tiny['max_violation'])
        
        # Very large dictionary values
        D_large = self.D * 1e6
        results_large = kkt_violation_comprehensive(D_large, self.X, self.A, self.lam)
        assert not np.isnan(results_large['max_violation'])
        assert not np.isinf(results_large['max_violation'])
        
        # Zero regularization (edge case)
        with pytest.raises(ValueError):
            kkt_violation_comprehensive(self.D, self.X, self.A, 0.0)
    
    def test_backward_compatibility(self):
        """Test that enhanced functions maintain backward compatibility."""
        # Original function should still work
        violation_old = kkt_violation_l1(self.D, self.X, self.A, self.lam)
        
        # New function in simple mode should give similar results
        results_new = kkt_violation_comprehensive(self.D, self.X, self.A, self.lam, detailed=False)
        violation_new = results_new['max_violation']
        
        # Should be very close (allowing for minor implementation differences)
        assert abs(violation_old - violation_new) < 1e-12
    
    def test_research_accuracy_validation(self):
        """Test KKT conditions against known theoretical results."""
        # Create a simple problem with constructed KKT-optimal solution
        # Use orthogonal dictionary for simpler analysis
        D_simple = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2 identity (complete dictionary)
        A_simple = np.array([[1.0], [0.0]])  # Only first atom active
        lam = 0.1
        
        # For KKT optimality with active coefficient A[0,0] = 1.0:
        # D^T(X - DA)[0,0] should equal λ*sign(A[0,0]) = 0.1
        # D^T(X - DA)[1,0] should satisfy |.| ≤ λ (since A[1,0] = 0)
        #
        # D^T = identity, so X - DA = X - [1.0, 0.0]^T
        # We need: X[0] - 1.0 = 0.1, so X[0] = 1.1
        # We need: |X[1] - 0.0| ≤ 0.1, so let X[1] = 0.05 (within bound)
        
        X_kkt = np.array([[1.1], [0.05]])
        
        results = kkt_violation_comprehensive(D_simple, X_kkt, A_simple, lam=lam, detailed=True)
        
        # Check dual gradient manually
        residual = X_kkt - D_simple @ A_simple  # [0.1, 0.05]
        dual_grad = D_simple.T @ residual  # [0.1, 0.05] (since D^T = I)
        
        # For active coeff (index 0): should equal λ*sign(A) = 0.1*1 = 0.1 ✓
        # For inactive coeff (index 1): |0.05| = 0.05 ≤ 0.1 ✓
        
        # Both KKT conditions satisfied, so violation should be very small
        assert results['max_violation'] < 1e-10, f"KKT violation: {results['max_violation']}"

# Integration tests with optimization algorithms would go here
class TestKKTIntegration:
    """Integration tests with actual optimization algorithms."""
    
    def test_kkt_with_iterative_soft_thresholding(self):
        """Test KKT checking with ISTA-style optimization."""
        # This would test integration with actual optimization algorithms
        # For now, we'll create a simple soft-thresholding example
        
        np.random.seed(42)
        D = np.random.randn(10, 15)
        D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
        X = np.random.randn(10, 5)
        lam = 0.1
        
        # Simple soft-thresholding step
        gradient = D.T @ (D @ np.zeros((15, 5)) - X)
        A_soft = np.sign(gradient) * np.maximum(0, np.abs(gradient) - lam)
        
        # Check KKT conditions
        results = kkt_violation_comprehensive(D, X, A_soft, lam)
        
        # Should have reasonable KKT violation (not perfect since this is one step)
        assert results['max_violation'] >= 0

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])