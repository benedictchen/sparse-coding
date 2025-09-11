"""
Comprehensive numerical stability and convergence tests.

Tests for ill-conditioned dictionaries, convergence rate validation,
and edge case handling as identified in the audit report.
"""

import numpy as np
import pytest
from sparse_coding import SparseCoder
from sparse_coding import L1Penalty, ElasticNetPenalty, L1Proximal


class TestNumericalStability:
    """Test numerical stability across various challenging conditions."""
    
    def test_ill_conditioned_dictionary(self):
        """Test performance with highly ill-conditioned dictionaries."""
        np.random.seed(42)
        
        # Create ill-conditioned dictionary (high condition number)
        n_features, n_atoms = 50, 100
        U, _, V = np.linalg.svd(np.random.randn(n_features, n_atoms), full_matrices=False)
        
        # Create condition numbers spanning several orders of magnitude
        condition_numbers = [1e2, 1e6, 1e12]
        
        for condition_num in condition_numbers:
            # Create dictionary with specific condition number
            s = np.logspace(0, -np.log10(condition_num), min(n_features, n_atoms))
            D = U @ np.diag(s) @ V
            D = D / np.linalg.norm(D, axis=0, keepdims=True)  # Normalize columns
            
            # Test signal
            x = np.random.randn(n_features, 10) * 0.1
            
            # Create coder with MOD update for stability
            coder = SparseCoder(n_atoms=n_atoms, mode='l1', lam=0.1, max_iter=10)
            coder.dictionary = D.copy()
            
            # Should not crash or produce NaN/inf
            codes = coder.encode(x)
            
            assert np.isfinite(codes).all(), f"Non-finite codes with condition number {condition_num}"
            assert not np.isnan(codes).any(), f"NaN codes with condition number {condition_num}"
            
            # Reconstruction should be reasonable
            x_recon = coder.decode(codes)
            mse = np.mean((x - x_recon)**2)
            assert mse < 1.0, f"Poor reconstruction (MSE={mse}) with condition number {condition_num}"
    
    def test_extreme_regularization_parameters(self):
        """Test stability with extreme lambda values."""
        np.random.seed(42)
        
        D = np.random.randn(50, 100)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        x = np.random.randn(50, 5) * 0.1
        
        # Test extreme regularization values
        extreme_lambdas = [1e-12, 1e-6, 1e6, 1e12]
        
        for lam in extreme_lambdas:
            penalty = L1(lam=lam)
            
            # Should handle extreme values gracefully
            codes = penalty.prox(x, 1.0)
            penalty_value = penalty.value(codes)
            
            assert np.isfinite(codes).all(), f"Non-finite codes with lambda={lam}"
            assert np.isfinite(penalty_value), f"Non-finite penalty with lambda={lam}"
            
            # Very high lambda should give near-zero codes
            if lam > 1e3:
                assert np.max(np.abs(codes)) < 1e-3, f"Codes not sparse enough with lambda={lam}"
    
    def test_convergence_rate_validation(self):
        """Validate O(1/k²) convergence for FISTA vs O(1/k) for ISTA."""
        np.random.seed(42)
        
        # Create well-conditioned problem for clear convergence comparison
        n_features, n_atoms = 64, 128
        D = np.random.randn(n_features, n_atoms)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        
        # Create test signal with known sparse solution
        true_codes = np.zeros(n_atoms)
        true_codes[:5] = np.random.randn(5)  # 5-sparse solution
        x = D @ true_codes + np.random.randn(n_features) * 0.01  # Small noise
        
        # Test both algorithms
        penalty_params = {'lam': 0.1}
        max_iter = 200
        
        fista_coder = create_advanced_sparse_coder(
            D, penalty_type='l1', penalty_params=penalty_params, max_iter=max_iter
        )
        
        # Run FISTA
        fista_result = fista_coder.fista(x)
        fista_objectives = fista_result['history']['objectives']
        
        # Run ISTA
        ista_result = fista_coder.ista(x)
        ista_objectives = ista_result['history']['objectives']
        
        # Analyze convergence rates after initial iterations
        start_iter = 20  # Skip initial transient
        end_iter = min(100, len(fista_objectives), len(ista_objectives))
        
        if end_iter > start_iter:
            # Compute empirical convergence rates
            fista_log_errors = np.log(np.array(fista_objectives[start_iter:end_iter]))
            ista_log_errors = np.log(np.array(ista_objectives[start_iter:end_iter]))
            
            iterations = np.arange(start_iter, end_iter)
            
            # Fit linear regression to log(error) vs log(iteration)
            # For FISTA: log(error) ~ -2*log(k) + const (O(1/k²))
            # For ISTA: log(error) ~ -1*log(k) + const (O(1/k))
            log_iterations = np.log(iterations + 1)
            
            # Fit slopes
            fista_slope = np.polyfit(log_iterations, fista_log_errors, 1)[0]
            ista_slope = np.polyfit(log_iterations, ista_log_errors, 1)[0]
            
            # FISTA should converge faster (more negative slope)
            assert fista_slope < ista_slope - 0.1, (
                f"FISTA convergence rate ({fista_slope:.3f}) not significantly "
                f"faster than ISTA ({ista_slope:.3f})"
            )
            
            # Rough check for theoretical rates (allowing for numerical variation)
            assert fista_slope < -1.2, f"FISTA slope {fista_slope:.3f} not close to -2"
            assert ista_slope > -1.5 and ista_slope < -0.3, f"ISTA slope {ista_slope:.3f} not close to -1"
    
    def test_coordinate_descent_stability(self):
        """Test coordinate descent with nearly singular systems."""
        np.random.seed(42)
        
        # Create nearly rank-deficient dictionary
        n_features, n_atoms = 20, 50
        base_dict = np.random.randn(n_features, n_atoms // 2)
        # Add some nearly dependent columns
        noise_cols = base_dict + np.random.randn(n_features, n_atoms // 2) * 1e-6
        D = np.column_stack([base_dict, noise_cols])
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        
        x = np.random.randn(n_features) * 0.1
        
        # Test coordinate descent
        coder = create_advanced_sparse_coder(
            D, penalty_type='l1', penalty_params={'lam': 0.1}, max_iter=100
        )
        
        result = coder.coordinate_descent(x)
        
        # Should converge without crashing
        assert result['converged'] or result['iterations'] == 100
        assert np.isfinite(result['solution']).all()
        assert result['final_objective'] > 0  # Should be finite positive value
    
    def test_elastic_net_proximal_stability(self):
        """Test elastic net proximal operator numerical stability."""
        np.random.seed(42)
        
        # Test with various parameter combinations
        test_cases = [
            {'l1': 0.1, 'l2': 0.1},    # Balanced
            {'l1': 1e-8, 'l2': 1.0},   # L2 dominated
            {'l1': 1.0, 'l2': 1e-8},   # L1 dominated  
            {'l1': 1e6, 'l2': 1e-6},   # Extreme ratio
        ]
        
        for params in test_cases:
            penalty = ElasticNet(l1=params['l1'], l2=params['l2'])
            
            # Test on various inputs
            test_inputs = [
                np.array([0.0, 1e-10, 1e10, -1e10]),  # Extreme values
                np.random.randn(100),                   # Random
                np.zeros(50),                           # All zeros
            ]
            
            for x in test_inputs:
                prox_result = penalty.prox(x, 1.0)
                penalty_value = penalty.value(prox_result)
                
                assert np.isfinite(prox_result).all(), (
                    f"Non-finite proximal result with params {params}"
                )
                assert np.isfinite(penalty_value), (
                    f"Non-finite penalty value with params {params}"
                )
    
    def test_gradient_validation_finite_differences(self):
        """Validate gradients using finite differences for complex penalties."""
        np.random.seed(42)
        
        # Test parameters
        n_features = 20
        D = np.random.randn(n_features, n_features)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        x = np.random.randn(n_features) * 0.1
        a = np.random.randn(n_features) * 0.5
        
        penalty = L1(lam=0.1)
        
        # Compute analytical gradient (for L1: lam * sign(a))
        grad_analytical = penalty.lam * np.sign(a)
        
        # Compute finite difference gradient
        eps = 1e-8
        grad_fd = np.zeros_like(a)
        
        for i in range(len(a)):
            a_plus = a.copy()
            a_minus = a.copy()
            a_plus[i] += eps
            a_minus[i] -= eps
            
            # Skip if we're at exactly zero (sign discontinuity)
            if abs(a[i]) < eps:
                continue
                
            grad_fd[i] = (penalty.value(a_plus) - penalty.value(a_minus)) / (2 * eps)
        
        # Compare non-zero elements
        nonzero_mask = np.abs(a) > eps
        if np.any(nonzero_mask):
            rel_error = np.abs(grad_analytical[nonzero_mask] - grad_fd[nonzero_mask])
            rel_error = rel_error / (np.abs(grad_analytical[nonzero_mask]) + eps)
            
            assert np.max(rel_error) < 1e-5, (
                f"Gradient validation failed: max relative error {np.max(rel_error):.2e}"
            )
    
    def test_memory_and_overflow_protection(self):
        """Test protection against memory issues and numerical overflow."""
        # Test with very large arrays (but not too large for CI)
        try:
            n_features, n_atoms = 1000, 2000
            D = np.random.randn(n_features, n_atoms).astype(np.float32)
            D = D / np.linalg.norm(D, axis=0, keepdims=True)
            
            x = np.random.randn(n_features, 100).astype(np.float32) * 0.1
            
            coder = SparseCoder(n_atoms=n_atoms, mode='l1', lam=0.1, max_iter=5)
            coder.dictionary = D
            
            # Should handle large arrays without memory issues
            codes = coder.encode(x)
            
            assert codes.shape == (n_atoms, 100)
            assert np.isfinite(codes).all()
            
        except MemoryError:
            pytest.skip("Insufficient memory for large array test")


class TestEdgeCases:
    """Test various edge cases for robustness."""
    
    def test_zero_input_signals(self):
        """Test behavior with zero input signals."""
        np.random.seed(42)
        
        D = np.random.randn(50, 100)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        
        coder = SparseCoder(n_atoms=100, mode='l1', lam=0.1)
        coder.dictionary = D
        
        # All-zero input
        x_zero = np.zeros((50, 5))
        codes = coder.encode(x_zero)
        
        # Should return zero codes
        assert np.allclose(codes, 0, atol=1e-10)
    
    def test_single_atom_dictionary(self):
        """Test degenerate case with single atom."""
        np.random.seed(42)
        
        D = np.random.randn(50, 1)
        D = D / np.linalg.norm(D)
        
        coder = SparseCoder(n_atoms=1, mode='l1', lam=0.1)
        coder.dictionary = D
        
        x = np.random.randn(50, 3) * 0.1
        codes = coder.encode(x)
        
        assert codes.shape == (1, 3)
        assert np.isfinite(codes).all()
    
    def test_overcomplete_vs_undercomplete(self):
        """Test both overcomplete and undercomplete dictionaries."""
        np.random.seed(42)
        
        n_features = 50
        test_cases = [
            25,   # Undercomplete
            50,   # Complete
            100,  # Overcomplete
            200,  # Highly overcomplete
        ]
        
        for n_atoms in test_cases:
            D = np.random.randn(n_features, n_atoms)
            D = D / np.linalg.norm(D, axis=0, keepdims=True)
            
            coder = SparseCoder(n_atoms=n_atoms, mode='l1', lam=0.1, max_iter=5)
            coder.dictionary = D
            
            x = np.random.randn(n_features, 10) * 0.1
            codes = coder.encode(x)
            
            assert codes.shape == (n_atoms, 10)
            assert np.isfinite(codes).all()
            
            # Test sparsity increases with overcompleteness and regularization
            sparsity = np.mean(np.abs(codes) < 1e-10)
            if n_atoms > n_features:
                assert sparsity > 0.1, f"Expected some sparsity with {n_atoms} atoms, got {sparsity:.3f}"


if __name__ == '__main__':
    pytest.main([__file__])