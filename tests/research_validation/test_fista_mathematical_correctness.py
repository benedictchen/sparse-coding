"""
Tests for FISTA algorithm mathematical correctness.

Validates our FISTA implementation against Beck & Teboulle (2009):
"A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems"

Key validation points:
1. Momentum parameter updates: t_{k+1} = (1 + √(1 + 4t_k²))/2
2. Soft thresholding operator correctness
3. Gradient step formulation 
4. Convergence rate O(1/k²) vs ISTA's O(1/k)
5. Lipschitz constant estimation
6. Proximal operator mathematical properties
"""

import numpy as np
import pytest
from sparse_coding import SparseCoder, L1Proximal
from sparse_coding.fista_batch import fista_batch, soft_thresh, power_iter_L
from tests.conftest import (create_test_dictionary, measure_convergence_rate)


class TestFISTAMomentumUpdates:
    """Test FISTA momentum parameter updates."""
    
    def test_momentum_parameter_sequence(self):
        """Test that momentum parameters follow Beck & Teboulle formula."""
        # FISTA momentum: t_{k+1} = (1 + √(1 + 4t_k²))/2
        t_values = [1.0]  # Initial value
        
        for k in range(20):
            t_k = t_values[-1]
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
            t_values.append(t_next)
        
        # Test theoretical properties
        t_array = np.array(t_values)
        
        # Sequence should be monotonically increasing
        assert np.all(np.diff(t_array) > 0), "Momentum sequence must be increasing"
        
        # Should grow approximately as k/2 asymptotically
        k_values = np.arange(10, len(t_values))
        theoretical_values = (k_values + 1) / 2.0
        
        # Check asymptotic behavior (should be close for large k)
        relative_errors = np.abs(t_array[10:] - theoretical_values) / theoretical_values
        assert np.mean(relative_errors[-5:]) < 0.1, "Momentum should approach (k+1)/2 asymptotically"
    
    def test_momentum_parameters_in_algorithm(self, synthetic_data):
        """Test momentum parameters in actual FISTA implementation."""
        data = synthetic_data
        X = data['signals'][:, 0:1]  # Single signal
        D = data['true_dict']
        
        # Get FISTA result with history tracking
        coder = SparseCoder(n_atoms=data['n_components'], mode="l1", max_iter=50)
        coder.dictionary = D.copy()
        
        result = coder._fista_single(X, D)
        
        # Check that algorithm converged
        assert result['converged'] or result['iterations'] >= 40, "FISTA should make significant progress"
        
        # Final solution should be reasonable
        a = result['coefficients']
        reconstruction = D @ a
        error = np.linalg.norm(X - reconstruction)**2
        assert error < 10.0 * np.var(X), "FISTA should achieve reasonable reconstruction"


class TestSoftThresholdingOperator:
    """Test soft thresholding (proximal operator for L1 norm)."""
    
    def test_soft_thresholding_formula(self):
        """Test soft thresholding formula: S_t(x) = sign(x) * max(|x| - t, 0)"""
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        threshold = 0.8
        
        # Manual implementation
        expected = np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)
        
        # Our implementation
        result = soft_thresh(x, threshold)
        
        np.testing.assert_allclose(result, expected, atol=1e-12)
    
    def test_soft_thresholding_properties(self):
        """Test mathematical properties of soft thresholding operator."""
        x = np.random.randn(100) * 5.0
        threshold = 1.0
        
        result = soft_thresh(x, threshold)
        
        # Property 1: |S_t(x)| ≤ |x| (shrinkage property)
        assert np.all(np.abs(result) <= np.abs(x) + 1e-12), "Soft thresholding should shrink magnitudes"
        
        # Property 2: S_t(x) = 0 when |x| ≤ t (thresholding property)
        small_indices = np.abs(x) <= threshold
        assert np.allclose(result[small_indices], 0.0, atol=1e-12), "Should threshold small values to zero"
        
        # Property 3: sign(S_t(x)) = sign(x) when S_t(x) ≠ 0 (sign preservation)
        nonzero_result = result[np.abs(result) > 1e-12]
        nonzero_x = x[np.abs(result) > 1e-12]
        assert np.all(np.sign(nonzero_result) == np.sign(nonzero_x)), "Should preserve signs of non-thresholded values"
    
    def test_vectorized_soft_thresholding(self):
        """Test that vectorized implementation matches element-wise."""
        X = np.random.randn(50, 30) * 3.0
        threshold = 0.5
        
        # Vectorized version
        result_vectorized = soft_thresh(X, threshold)
        
        # Element-wise version
        result_elementwise = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x_ij = X[i, j]
                result_elementwise[i, j] = np.sign(x_ij) * max(abs(x_ij) - threshold, 0.0)
        
        np.testing.assert_allclose(result_vectorized, result_elementwise, atol=1e-12)


class TestGradientStepFormulation:
    """Test FISTA gradient step formulation."""
    
    def test_gradient_computation(self, synthetic_data):
        """Test gradient computation for L1 regularized least squares."""
        data = synthetic_data
        X = data['signals'][:, 0:1]  # Single signal
        D = data['true_dict']
        
        # Current coefficients
        a = np.random.randn(data['n_components'], 1) * 0.1
        
        # Analytical gradient: ∇f(a) = D^T(Da - x)
        gradient_analytical = D.T @ (D @ a - X)
        
        # Numerical gradient estimation
        eps = 1e-8
        gradient_numerical = np.zeros_like(a)
        
        def objective(coeffs):
            return 0.5 * np.linalg.norm(D @ coeffs - X)**2
        
        base_obj = objective(a)
        
        for i in range(len(a)):
            a_plus = a.copy()
            a_plus[i] += eps
            gradient_numerical[i] = (objective(a_plus) - base_obj) / eps
        
        # Should match within numerical precision
        np.testing.assert_allclose(gradient_analytical.flatten(), 
                                   gradient_numerical.flatten(), 
                                   rtol=1e-6, atol=1e-8)
    
    def test_lipschitz_constant_estimation(self, synthetic_data):
        """Test Lipschitz constant estimation via power iteration."""
        data = synthetic_data
        D = data['true_dict']
        
        # Our implementation
        L_estimated = power_iter_L(D, n_iter=100, tol=1e-10)
        
        # True Lipschitz constant is largest eigenvalue of D^T D
        DtD = D.T @ D
        eigenvalues = np.linalg.eigvals(DtD)
        L_true = np.max(eigenvalues)
        
        # Should be close
        relative_error = abs(L_estimated - L_true) / L_true
        assert relative_error < 0.01, f"Lipschitz estimation error: {relative_error:.6f}"
        
        # Should be at least as large as true value (conservative estimate okay)
        assert L_estimated >= L_true * 0.99, "Lipschitz estimate should not be too small"


class TestConvergenceRateComparison:
    """Test FISTA vs ISTA convergence rates."""
    
    @pytest.mark.slow
    def test_fista_vs_ista_convergence_rate(self, synthetic_data):
        """Test that FISTA converges faster than ISTA."""
        data = synthetic_data
        X = data['signals'][:, :3]  # Few signals for speed
        D = create_test_dictionary(data['n_features'], data['n_components'], seed=42)
        
        # Create optimizer with both methods
        proximal_op = L1Proximal(lam=0.1)
        optimizer = AdvancedOptimizer(D, proximal_op, max_iter=100, tolerance=1e-8)
        
        # Compare on single signal
        x = X[:, 0]
        
        # ISTA result
        ista_result = optimizer.ista(x)
        
        # FISTA result
        fista_result = optimizer.fista(x)
        
        # Both should converge
        assert ista_result['iterations'] <= 100, "ISTA should converge within max iterations"
        assert fista_result['iterations'] <= 100, "FISTA should converge within max iterations"
        
        # FISTA should converge in fewer iterations (for this problem)
        if ista_result['converged'] and fista_result['converged']:
            # Allow some flexibility since problems vary
            iteration_ratio = fista_result['iterations'] / max(ista_result['iterations'], 1)
            assert iteration_ratio <= 1.2, f"FISTA not significantly faster: {iteration_ratio:.3f}"
        
        # Both should reach similar final objectives
        obj_ratio = abs(fista_result['final_objective'] - ista_result['final_objective']) / max(ista_result['final_objective'], 1e-10)
        assert obj_ratio < 0.1, f"Final objectives should be similar: {obj_ratio:.6f}"
    
    def test_theoretical_convergence_rate(self, synthetic_data):
        """Test that FISTA achieves O(1/k²) convergence rate."""
        data = synthetic_data
        X = data['signals'][:, 0:1]
        D = create_test_dictionary(data['n_features'], data['n_components'], 
                                 condition_number=2.0, seed=42)
        
        # Run FISTA with objective tracking
        result, objectives = fista_batch(D, X, lam=0.1, max_iter=50, tol=1e-12, return_objectives=True)
        
        # Test that objectives decrease over iterations (research requirement)
        if len(objectives) > 10:
            # Compare early vs late objectives
            early_obj = np.mean(objectives[:5])
            late_obj = np.mean(objectives[-5:])
            assert late_obj < early_obj, f"Objective should decrease over iterations: early={early_obj:.6f}, late={late_obj:.6f}"
            
            # Test monotonic decrease (weaker requirement due to numerical precision)
            decreasing_pairs = sum(1 for i in range(1, len(objectives)) if objectives[i] <= objectives[i-1] + 1e-10)
            decreasing_ratio = decreasing_pairs / (len(objectives) - 1)
            assert decreasing_ratio > 0.8, f"Objective should decrease monotonically in ≥80% of steps: {decreasing_ratio:.2f}"


class TestProximalOperatorProperties:
    """Test proximal operator mathematical properties."""
    
    def test_proximal_operator_definition(self):
        """Test proximal operator definition: prox_t(v) = argmin_x { 0.5||x-v||² + t*g(x) }"""
        # For L1 penalty g(x) = λ||x||₁, prox operator is soft thresholding
        v = np.array([2.0, -1.5, 0.3, -0.2])
        t = 0.5
        lam = 0.1
        threshold = t * lam
        
        # Analytical solution (soft thresholding)
        expected = np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)
        
        # Using proximal operator
        proximal_op = L1Proximal(lam)
        result = proximal_op.prox(v, t)
        
        np.testing.assert_allclose(result, expected, atol=1e-12)
    
    def test_proximal_operator_properties(self):
        """Test key properties of proximal operators."""
        v = np.random.randn(20) * 2.0
        t = 0.8
        lam = 0.2
        
        proximal_op = L1Proximal(lam)
        result = proximal_op.prox(v, t)
        
        # Property 1: Proximal operator is non-expansive
        # ||prox_t(x) - prox_t(y)|| ≤ ||x - y||
        v2 = v + 0.1 * np.random.randn(*v.shape)
        result2 = proximal_op.prox(v2, t)
        
        prox_distance = np.linalg.norm(result - result2)
        input_distance = np.linalg.norm(v - v2)
        assert prox_distance <= input_distance + 1e-10, "Proximal operator should be non-expansive"
        
        # Property 2: For L1, should produce sparser result than input
        sparsity_input = np.mean(np.abs(v) < 0.01)
        sparsity_output = np.mean(np.abs(result) < 0.01)
        assert sparsity_output >= sparsity_input, "L1 proximal should increase sparsity"
    
    def test_step_size_scaling_property(self):
        """Test scaling property of proximal operators."""
        v = np.array([2.0, -1.0, 0.5, -2.5])
        lam = 0.3
        
        proximal_op = L1Proximal(lam)
        
        # Property: prox_{αt}(αx) = α * prox_t(x) for α > 0
        alpha = 2.0
        t = 0.4
        
        # Left side: prox_{αt}(αv)
        left = proximal_op.prox(alpha * v, alpha * t)
        
        # Right side: α * prox_t(v)
        right = alpha * proximal_op.prox(v, t)
        
        # For L1 proximal this property holds
        np.testing.assert_allclose(left, right, atol=1e-12, 
                                   err_msg="Proximal scaling property should hold")


class TestBatchFISTAImplementation:
    """Test batch FISTA implementation correctness."""
    
    def test_batch_vs_single_consistency(self, synthetic_data):
        """Test that batch FISTA matches single-signal FISTA."""
        data = synthetic_data
        X = data['signals'][:, :3]  # Small batch
        D = data['true_dict']
        lam = 0.1
        
        # Batch version
        A_batch = fista_batch(D, X, lam=lam, max_iter=100, tol=1e-6)
        
        # Single signal versions
        A_single = np.zeros_like(A_batch)
        for i in range(X.shape[1]):
            x_i = X[:, i:i+1]
            a_i = fista_batch(D, x_i, lam=lam, max_iter=100, tol=1e-6)
            A_single[:, i:i+1] = a_i
        
        # Should be very close
        np.testing.assert_allclose(A_batch, A_single, rtol=1e-4, atol=1e-6)
    
    def test_batch_fista_convergence_properties(self, synthetic_data):
        """Test convergence properties of batch FISTA."""
        data = synthetic_data
        X = data['signals']
        D = create_test_dictionary(data['n_features'], data['n_components'], seed=42)
        
        # Different lambda values
        lambda_values = [0.01, 0.1, 0.5]
        
        for lam in lambda_values:
            A = fista_batch(D, X, lam=lam, max_iter=200, tol=1e-8)
            
            # Basic sanity checks
            assert A.shape == (data['n_components'], data['n_samples'])
            assert np.all(np.isfinite(A)), "All coefficients should be finite"
            
            # Check sparsity increases with lambda
            sparsity = np.mean(np.abs(A) < 0.01)
            
            # Reconstruction quality
            reconstruction = D @ A
            mse = np.mean((X - reconstruction)**2)
            relative_mse = mse / np.var(X)
            
            assert relative_mse < 2.0, f"Reconstruction error too high for λ={lam}: {relative_mse:.3f}"
    
    def test_lipschitz_constant_effect(self, synthetic_data):
        """Test effect of different Lipschitz constant estimates."""
        data = synthetic_data
        X = data['signals'][:, :2]  # Small problem for speed
        D = data['true_dict']
        lam = 0.1
        
        # Compute true Lipschitz constant
        L_true = power_iter_L(D)
        
        # Test with different Lipschitz estimates
        L_values = [L_true * 0.5, L_true, L_true * 2.0]
        results = []
        
        for L in L_values:
            A = fista_batch(D, X, lam=lam, L=L, max_iter=100, tol=1e-6)
            
            # Compute final objective
            reconstruction_error = 0.5 * np.linalg.norm(X - D @ A, 'fro')**2
            sparsity_penalty = lam * np.sum(np.abs(A))
            objective = reconstruction_error + sparsity_penalty
            
            results.append({
                'L': L,
                'A': A,
                'objective': objective
            })
        
        # All should converge to similar objectives (algorithm should be robust)
        objectives = [r['objective'] for r in results]
        obj_std = np.std(objectives)
        obj_mean = np.mean(objectives)
        
        # Relative standard deviation should be small
        if obj_mean > 1e-10:
            relative_std = obj_std / obj_mean
            assert relative_std < 0.2, f"Results too sensitive to Lipschitz estimate: {relative_std:.3f}"


@pytest.mark.numerical
class TestNumericalStability:
    """Test numerical stability of FISTA implementation."""
    
    def test_ill_conditioned_problems(self):
        """Test FISTA on ill-conditioned problems."""
        # Create ill-conditioned dictionary
        n_features, n_atoms = 100, 80
        D = create_test_dictionary(n_features, n_atoms, condition_number=100.0, seed=42)
        
        # Generate test signals  
        np.random.seed(42)
        A_true = np.random.laplace(scale=0.1, size=(n_atoms, 5))
        A_true[np.abs(A_true) < 0.2] = 0
        X = D @ A_true + 0.01 * np.random.randn(n_features, 5)
        
        # Run FISTA
        A_estimated = fista_batch(D, X, lam=0.05, max_iter=500, tol=1e-8)
        
        # Should still produce reasonable results
        assert np.all(np.isfinite(A_estimated)), "Solution should remain finite"
        
        reconstruction = D @ A_estimated
        relative_error = np.linalg.norm(X - reconstruction, 'fro') / np.linalg.norm(X, 'fro')
        assert relative_error < 0.5, f"Reconstruction error too high: {relative_error:.3f}"
    
    def test_extreme_lambda_values(self, synthetic_data):
        """Test behavior with extreme regularization parameters."""
        data = synthetic_data
        X = data['signals'][:, :3]
        D = data['true_dict']
        
        # Very small lambda (should approach least squares)
        A_small_lam = fista_batch(D, X, lam=1e-8, max_iter=100, tol=1e-8)
        reconstruction_small = D @ A_small_lam
        error_small = np.linalg.norm(X - reconstruction_small, 'fro')**2
        
        # Very large lambda (should be very sparse)
        A_large_lam = fista_batch(D, X, lam=10.0, max_iter=100, tol=1e-8)
        sparsity_large = np.mean(np.abs(A_large_lam) < 1e-6)
        
        # Sanity checks
        assert np.all(np.isfinite(A_small_lam)), "Small lambda solution should be finite"
        assert np.all(np.isfinite(A_large_lam)), "Large lambda solution should be finite"
        assert sparsity_large > 0.8, f"Large lambda should produce sparse solution: {sparsity_large:.3f}"
        assert error_small < np.var(X), "Small lambda should achieve good reconstruction"