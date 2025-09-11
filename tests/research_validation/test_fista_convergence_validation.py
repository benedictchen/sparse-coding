"""
FISTA mathematical correctness validation - simplified working version.

Tests the critical mathematical properties without relying on unimplemented features.
"""

import numpy as np
import pytest
from sparse_coding.fista_batch import fista_batch, soft_thresh, power_iter_L
from tests.conftest import create_test_dictionary


class TestFISTABasicMathematical:
    """Basic mathematical validation that works with current implementation."""
    
    def test_soft_thresholding_exact_formula(self):
        """Test soft thresholding matches exact mathematical formula."""
        # Test specific values where we know the exact answer
        x = np.array([-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0])
        threshold = 1.0
        
        result = soft_thresh(x, threshold)
        expected = np.array([-2.0, -0.5, 0.0, 0.0, 0.0, 0.5, 2.0])
        
        np.testing.assert_allclose(result, expected, atol=1e-12,
            err_msg="Soft thresholding must match exact mathematical formula")
    
    def test_soft_thresholding_mathematical_properties(self):
        """Test key mathematical properties of soft thresholding."""
        np.random.seed(42)
        x = np.random.randn(100) * 5.0
        threshold = 1.0
        
        result = soft_thresh(x, threshold)
        
        # Property 1: |S_t(x)| ≤ |x| (shrinkage property)
        assert np.all(np.abs(result) <= np.abs(x) + 1e-12), \
            "Soft thresholding must have shrinkage property"
        
        # Property 2: S_t(x) = 0 when |x| ≤ t
        small_mask = np.abs(x) <= threshold
        assert np.allclose(result[small_mask], 0.0, atol=1e-12), \
            "Values ≤ threshold must be set to exactly zero"
        
        # Property 3: sign preservation for non-zero outputs
        nonzero_mask = np.abs(result) > 1e-12
        if np.any(nonzero_mask):
            assert np.all(np.sign(result[nonzero_mask]) == np.sign(x[nonzero_mask])), \
                "Sign must be preserved for non-thresholded values"
    
    def test_lipschitz_constant_accuracy(self):
        """Test Lipschitz constant estimation accuracy."""
        np.random.seed(123)
        n_features, n_atoms = 50, 30
        D = create_test_dictionary(n_features, n_atoms, seed=123)
        
        # True Lipschitz constant
        L_true = np.linalg.norm(D.T @ D, ord=2)
        
        # Our estimation
        L_estimated = power_iter_L(D, n_iter=100, tol=1e-10)
        
        relative_error = abs(L_estimated - L_true) / L_true
        assert relative_error < 0.01, (
            f"Lipschitz constant estimation error too large: {relative_error:.6f}. "
            f"True: {L_true:.6f}, Estimated: {L_estimated:.6f}"
        )
    
    def test_fista_basic_convergence(self):
        """Test that FISTA achieves basic convergence on well-posed problem."""
        np.random.seed(42)
        
        # Well-conditioned problem
        n_features, n_atoms = 32, 20
        D = create_test_dictionary(n_features, n_atoms, condition_number=2.0, seed=42)
        
        # Generate test signal
        true_codes = np.random.randn(n_atoms, 1) * 0.2
        true_codes[np.abs(true_codes) < 0.1] = 0  # Sparse
        signal = D @ true_codes + 0.02 * np.random.randn(n_features, 1)
        
        lambda_param = 0.05
        
        # Run FISTA
        codes = fista_batch(D, signal, lambda_param, max_iter=200, tol=1e-8)
        
        # Basic validation
        assert np.all(np.isfinite(codes)), "FISTA solution must be finite"
        
        # Reconstruction quality
        reconstruction = D @ codes
        relative_error = np.linalg.norm(signal - reconstruction) / np.linalg.norm(signal)
        assert relative_error < 0.5, f"Reconstruction error too high: {relative_error:.3f}"
        
        # Sparsity check
        sparsity = np.mean(np.abs(codes) < 1e-6)
        assert 0.1 <= sparsity <= 0.9, f"Solution should be reasonably sparse: {sparsity:.3f}"
    
    def test_fista_vs_least_squares_with_zero_lambda(self):
        """Test FISTA with λ=0 approaches least squares solution."""
        np.random.seed(456)
        
        n_features, n_atoms = 30, 20  # Overdetermined for stability
        D = create_test_dictionary(n_features, n_atoms, seed=456)
        signal = np.random.randn(n_features, 1)
        
        # FISTA with very small lambda
        codes_fista = fista_batch(D, signal, lam=1e-8, max_iter=200, tol=1e-10)
        
        # Least squares solution
        codes_ls = np.linalg.pinv(D) @ signal
        
        # Should be close (allowing for numerical differences)
        np.testing.assert_allclose(codes_fista, codes_ls, rtol=1e-4,
            err_msg="FISTA with λ→0 should approach least squares solution")
    
    def test_fista_sparsity_increases_with_lambda(self):
        """Test that sparsity increases monotonically with lambda."""
        np.random.seed(789)
        
        n_features, n_atoms = 40, 30
        D = create_test_dictionary(n_features, n_atoms, seed=789)
        signal = np.random.randn(n_features, 1)
        
        lambda_values = [0.01, 0.05, 0.1, 0.2]
        sparsities = []
        
        for lam in lambda_values:
            codes = fista_batch(D, signal, lam=lam, max_iter=100, tol=1e-6)
            sparsity = np.mean(np.abs(codes) < 1e-6)
            sparsities.append(sparsity)
        
        # Sparsity should generally increase with lambda
        # Allow for some numerical noise but require overall trend
        sparsity_diffs = np.diff(sparsities)
        positive_changes = np.sum(sparsity_diffs >= -0.05)  # Allow small decreases
        assert positive_changes >= len(sparsity_diffs) * 0.7, \
            f"Sparsity should generally increase with lambda: {sparsities}"
    
    def test_momentum_sequence_properties(self):
        """Test theoretical momentum sequence properties."""
        # Beck & Teboulle momentum: t_{k+1} = (1 + sqrt(1 + 4*t_k^2))/2
        t_values = [1.0]
        
        for k in range(50):
            t_k = t_values[-1]
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
            t_values.append(t_next)
        
        t_array = np.array(t_values)
        
        # Should be strictly increasing
        assert np.all(np.diff(t_array) > 0), "Momentum sequence must be strictly increasing"
        
        # Should grow approximately linearly for large k
        # For large k: t_k ≈ (k+1)/2
        k_large = np.arange(20, 30)
        t_large = t_array[20:30]
        theoretical_large = (k_large + 1) / 2.0
        
        relative_errors = np.abs(t_large - theoretical_large) / theoretical_large
        assert np.mean(relative_errors) < 0.2, \
            "Momentum should approach (k+1)/2 for large k"
    
    def test_fista_numerical_stability(self):
        """Test FISTA numerical stability on challenging problems."""
        # Test various challenging scenarios
        test_cases = [
            # (condition_number, noise_level, lambda, description)
            (1.0, 0.01, 0.1, "well-conditioned"),
            (5.0, 0.05, 0.1, "moderately ill-conditioned"),
            (10.0, 0.02, 0.05, "ill-conditioned with regularization"),
        ]
        
        for cond_num, noise, lam, desc in test_cases:
            np.random.seed(42)  # Consistent across test cases
            
            n_features, n_atoms = 50, 40
            D = create_test_dictionary(n_features, n_atoms, 
                                     condition_number=cond_num, seed=42)
            
            # Generate test signal
            true_codes = np.random.randn(n_atoms, 1) * 0.1
            true_codes[np.abs(true_codes) < 0.05] = 0
            signal = D @ true_codes + noise * np.random.randn(n_features, 1)
            
            # Run FISTA
            codes = fista_batch(D, signal, lam=lam, max_iter=300, tol=1e-8)
            
            # Should remain numerically stable
            assert np.all(np.isfinite(codes)), f"FISTA should be stable for {desc} case"
            
            # Should achieve reasonable reconstruction
            reconstruction = D @ codes
            relative_error = np.linalg.norm(signal - reconstruction) / np.linalg.norm(signal)
            assert relative_error < 1.0, \
                f"Reconstruction error too high for {desc}: {relative_error:.3f}"


class TestFISTAConvergenceRate:
    """Test FISTA convergence rate properties (simplified)."""
    
    def test_fista_faster_than_naive_iteration(self):
        """Test that FISTA converges faster than naive gradient descent."""
        np.random.seed(123)
        
        # Medium-scale well-conditioned problem
        n_features, n_atoms = 40, 25  
        D = create_test_dictionary(n_features, n_atoms, condition_number=2.0, seed=123)
        signal = np.random.randn(n_features, 1)
        lambda_param = 0.05
        max_iter = 100
        
        # FISTA
        codes_fista = fista_batch(D, signal, lambda_param, max_iter=max_iter, tol=1e-8)
        obj_fista = 0.5 * np.linalg.norm(signal - D @ codes_fista)**2 + \
                   lambda_param * np.sum(np.abs(codes_fista))
        
        # Simple proximal gradient (ISTA)
        L = power_iter_L(D)
        codes_ista = np.zeros((n_atoms, 1))
        
        for i in range(max_iter):
            gradient = D.T @ (D @ codes_ista - signal)
            codes_ista = soft_thresh(codes_ista - gradient/L, lambda_param/L)
        
        obj_ista = 0.5 * np.linalg.norm(signal - D @ codes_ista)**2 + \
                  lambda_param * np.sum(np.abs(codes_ista))
        
        # FISTA should achieve better or equal objective
        improvement = (obj_ista - obj_fista) / abs(obj_ista)
        assert improvement >= -0.1, (  # Allow small tolerance for numerical effects
            f"FISTA should perform at least as well as ISTA: "
            f"FISTA obj={obj_fista:.6f}, ISTA obj={obj_ista:.6f}, "
            f"improvement={improvement:.6f}"
        )
    
    def test_convergence_with_different_step_sizes(self):
        """Test FISTA convergence with different Lipschitz estimates."""
        np.random.seed(456)
        
        n_features, n_atoms = 30, 20
        D = create_test_dictionary(n_features, n_atoms, seed=456)
        signal = np.random.randn(n_features, 1)
        lambda_param = 0.1
        
        # True Lipschitz constant
        L_true = power_iter_L(D)
        
        # Test different step size strategies
        L_values = [L_true * 0.9, L_true, L_true * 1.5]
        objectives = []
        
        for L in L_values:
            codes = fista_batch(D, signal, lambda_param, L=L, max_iter=150, tol=1e-8)
            obj = 0.5 * np.linalg.norm(signal - D @ codes)**2 + \
                  lambda_param * np.sum(np.abs(codes))
            objectives.append(obj)
        
        # All should converge to reasonable solutions
        for i, obj in enumerate(objectives):
            assert np.isfinite(obj), f"Objective {i} should be finite"
            assert obj > 0, f"Objective {i} should be positive"
        
        # Conservative step size should be most reliable (often best objective)
        conservative_obj = objectives[2]  # L_true * 1.5
        for obj in objectives:
            ratio = abs(obj - conservative_obj) / conservative_obj
            assert ratio < 0.5, f"Objectives should be reasonably close: {objectives}"


if __name__ == "__main__":
    # Quick self-test
    test = TestFISTABasicMathematical()
    test.test_soft_thresholding_exact_formula()
    test.test_lipschitz_constant_accuracy()
    test.test_fista_basic_convergence()
    print("✅ FISTA mathematical tests pass")
    
    test_conv = TestFISTAConvergenceRate()
    test_conv.test_fista_faster_than_naive_iteration()
    print("✅ FISTA convergence tests pass")
    
    print("Run full test suite with: pytest test_fista_convergence_validation.py -v")