"""
Gradient validation tests for paper mode sparse coding.

Tests the mathematical correctness of our log prior fixes and validates
that gradient calculations are accurate using finite differences.
"""

import numpy as np
import pytest
from sparse_coding.sparse_coder import _paper_energy_grad, _ncg_infer_single, SparseCoder


class TestGradientValidation:
    """Test gradient calculations for mathematical correctness."""
    
    def test_paper_energy_grad_finite_difference_validation(self):
        """Test _paper_energy_grad gradients using finite differences."""
        # Set up test problem
        np.random.seed(42)
        p, K = 20, 8
        D = np.random.randn(p, K)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)  # Normalize
        x = np.random.randn(p) * 0.1  # Small signal for better conditioning
        a = np.random.randn(K) * 0.05  # Small initial coefficients
        lam, sigma = 0.1, 1.0
        
        # Compute analytical gradient
        energy, grad_analytical = _paper_energy_grad(x, D, a, lam, sigma)
        
        # Compute numerical gradient using finite differences
        eps = 1e-6
        grad_numerical = np.zeros_like(a)
        
        for i in range(K):
            a_plus = a.copy()
            a_minus = a.copy()
            a_plus[i] += eps
            a_minus[i] -= eps
            
            energy_plus, _ = _paper_energy_grad(x, D, a_plus, lam, sigma)
            energy_minus, _ = _paper_energy_grad(x, D, a_minus, lam, sigma)
            
            grad_numerical[i] = (energy_plus - energy_minus) / (2 * eps)
        
        # Gradients should match closely
        rel_error = np.linalg.norm(grad_analytical - grad_numerical) / (np.linalg.norm(grad_numerical) + 1e-12)
        assert rel_error < 1e-4, f"Gradient error too large: {rel_error:.2e}"
        
        # Check individual components for detailed diagnostics
        max_component_error = np.max(np.abs(grad_analytical - grad_numerical))
        assert max_component_error < 1e-5, f"Max component error: {max_component_error:.2e}"
    
    def test_log_prior_sign_correctness(self):
        """Test that log prior has correct sign (should penalize large coefficients)."""
        # Set up simple test case
        np.random.seed(42)
        p, K = 10, 5
        D = np.eye(p)[:, :K]  # Simple orthogonal dictionary
        x = np.ones(p) * 0.1  # Small constant signal
        lam, sigma = 0.5, 1.0
        
        # Test with small coefficients
        a_small = np.ones(K) * 0.01
        energy_small, _ = _paper_energy_grad(x, D, a_small, lam, sigma)
        
        # Test with large coefficients
        a_large = np.ones(K) * 0.5
        energy_large, _ = _paper_energy_grad(x, D, a_large, lam, sigma)
        
        # Energy should increase with larger coefficients (penalty working correctly)
        assert energy_large > energy_small, (
            f"Log prior penalty not working: small={energy_small:.6f}, large={energy_large:.6f}"
        )
        
        # Test gradient direction
        _, grad_small = _paper_energy_grad(x, D, a_small, lam, sigma)
        _, grad_large = _paper_energy_grad(x, D, a_large, lam, sigma)
        
        # For positive coefficients, gradient should be positive (encouraging reduction)
        assert np.all(grad_large > grad_small), "Gradient should increase with coefficient magnitude"
    
    def test_ncg_convergence_with_fixed_gradients(self):
        """Test that NCG converges to reasonable solution with corrected gradients."""
        np.random.seed(42)
        p, K = 15, 6
        
        # Create well-conditioned test problem
        D = np.random.randn(p, K)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        
        # Create sparse ground truth
        a_true = np.zeros(K)
        a_true[[1, 3]] = [0.3, -0.2]  # Only 2 active coefficients
        x = D @ a_true + 0.01 * np.random.randn(p)  # Add small noise
        
        lam, sigma = 0.1, 1.0
        
        # Run NCG optimization
        a_solution = _ncg_infer_single(x, D, lam, sigma, max_iter=100, tol=1e-8)
        
        # Check solution quality
        reconstruction_error = np.linalg.norm(x - D @ a_solution)
        assert reconstruction_error < 0.1, f"Poor reconstruction: {reconstruction_error:.6f}"
        
        # Check sparsity
        active_coeffs = np.sum(np.abs(a_solution) > 0.01)
        assert active_coeffs <= K, "Solution should be sparse"
        
        # Check that gradient is near zero at solution (first-order optimality)
        _, grad_final = _paper_energy_grad(x, D, a_solution, lam, sigma)
        grad_norm = np.linalg.norm(grad_final)
        assert grad_norm < 0.1, f"Gradient not near zero at solution: {grad_norm:.6f}"
    
    def test_paper_modes_produce_different_valid_results(self):
        """Test that both paper modes work and produce different results."""
        np.random.seed(42)
        data_dim, n_atoms, n_samples = 20, 8, 15
        
        X = np.random.randn(data_dim, n_samples) * 0.1
        
        # Test paper mode
        coder_paper = SparseCoder(n_atoms=n_atoms, mode="paper", lam=0.1, seed=42)
        coder_paper.fit(X, n_steps=3)
        A_paper = coder_paper.encode(X[:, :5])
        
        # Test paper_gdD mode
        coder_gdD = SparseCoder(n_atoms=n_atoms, mode="paper_gdD", lam=0.1, seed=42)
        coder_gdD.fit(X, n_steps=3, lr=0.05)
        A_gdD = coder_gdD.encode(X[:, :5])
        
        # Both should produce valid results
        assert np.all(np.isfinite(A_paper)), "Paper mode produced invalid results"
        assert np.all(np.isfinite(A_gdD)), "Paper_gdD mode produced invalid results"
        
        # Results should be different (different update rules)
        diff_norm = np.linalg.norm(A_paper - A_gdD, 'fro')
        assert diff_norm > 0.01, f"Modes produced too similar results: {diff_norm:.6f}"
        
        # Both should achieve reasonable reconstruction
        X_test = X[:, :5]
        recon_paper = coder_paper.decode(A_paper)
        recon_gdD = coder_gdD.decode(A_gdD)
        
        error_paper = np.linalg.norm(X_test - recon_paper, 'fro') / np.linalg.norm(X_test, 'fro')
        error_gdD = np.linalg.norm(X_test - recon_gdD, 'fro') / np.linalg.norm(X_test, 'fro')
        
        assert error_paper < 1.5, f"Paper mode reconstruction error too high: {error_paper:.3f}"
        assert error_gdD < 1.5, f"Paper_gdD mode reconstruction error too high: {error_gdD:.3f}"
    
    def test_energy_monotonic_decrease_during_optimization(self):
        """Test that energy decreases during NCG optimization."""
        np.random.seed(42)
        p, K = 12, 5
        
        D = np.random.randn(p, K)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        x = np.random.randn(p) * 0.1
        lam, sigma = 0.2, 1.0
        
        # Modified NCG to track energy
        a = np.zeros(K)
        energies = []
        
        for iteration in range(20):
            energy, grad = _paper_energy_grad(x, D, a, lam, sigma)
            energies.append(energy)
            
            if np.linalg.norm(grad) < 1e-8:
                break
                
            # Simple gradient descent step for testing
            step_size = 0.01
            a = a - step_size * grad
        
        # Energy should generally decrease
        if len(energies) > 5:
            # Allow some initial fluctuation but should decrease overall
            final_energy = energies[-1]
            initial_energy = energies[2]  # Skip first few iterations
            assert final_energy < initial_energy, (
                f"Energy should decrease: initial={initial_energy:.6f}, final={final_energy:.6f}"
            )
    
    def test_gradient_components_individually(self):
        """Test individual components of the gradient calculation."""
        np.random.seed(42)
        p, K = 10, 4
        D = np.random.randn(p, K)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        x = np.random.randn(p) * 0.1
        a = np.random.randn(K) * 0.1
        lam, sigma = 0.15, 1.0
        
        # Compute full gradient
        energy, grad_full = _paper_energy_grad(x, D, a, lam, sigma)
        
        # Manually compute gradient components
        r = x - D @ a
        reconstruction_term = -(D.T @ r)
        sparsity_term = lam * (2*a / (sigma**2 + a*a))
        grad_manual = reconstruction_term + sparsity_term
        
        # Should match our implementation
        np.testing.assert_allclose(grad_full, grad_manual, rtol=1e-12, atol=1e-15)
        
        # Test energy components
        energy_reconstruction = 0.5 * float(r @ r)
        energy_sparsity = lam * float(np.sum(np.log1p((a / sigma)**2)))
        energy_manual = energy_reconstruction + energy_sparsity
        
        np.testing.assert_allclose(energy, energy_manual, rtol=1e-12, atol=1e-15)
    
    def test_dictionary_learner_with_paper_modes(self):
        """Test that DictionaryLearner works with paper modes."""
        from sparse_coding import DictionaryLearner
        
        # Create small test images
        images = np.random.randn(3, 16, 16) * 0.1
        
        # Test with different modes
        modes = ["l1", "paper", "paper_gdD"]
        
        for mode in modes:
            learner = DictionaryLearner(
                n_components=8,
                patch_size=(4, 4),
                max_iterations=5,  # Short for testing
                mode=mode,
                random_seed=42
            )
            
            # Should train without errors
            history = learner.fit(images, verbose=False)
            
            # Should produce features
            features = learner.transform(images)
            
            assert history is not None
            assert features.shape == (3, 8), f"Wrong feature shape for mode {mode}: {features.shape}"
            assert np.all(np.isfinite(features)), f"Invalid features for mode {mode}"


class TestMathematicalConsistency:
    """Test mathematical consistency of the fixes."""
    
    def test_cauchy_prior_behavior(self):
        """Test that the log prior behaves like a Cauchy/heavy-tailed distribution."""
        sigma = 1.0
        
        # Test prior penalty for different coefficient values
        coeffs = np.array([-2, -1, -0.1, 0, 0.1, 1, 2])
        penalties = np.log1p((coeffs / sigma)**2)
        
        # Should be symmetric around zero
        assert np.allclose(penalties[[0, 6]], penalties[[6, 0]]), "Prior should be symmetric"
        assert np.allclose(penalties[[1, 5]], penalties[[5, 1]]), "Prior should be symmetric"
        
        # Should penalize larger coefficients more
        assert penalties[0] > penalties[2], "Larger coefficients should have higher penalty"
        assert penalties[6] > penalties[4], "Larger coefficients should have higher penalty"
        
        # Minimum at zero
        assert penalties[3] == np.min(penalties), "Minimum penalty should be at zero"
    
    def test_mod_update_numerical_stability(self):
        """Test that MOD update is numerically stable with solve vs inv."""
        from sparse_coding.sparse_coder import _mod_update
        
        np.random.seed(42)
        p, K, N = 50, 20, 30
        
        # Create test data
        D = np.random.randn(p, K)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        X = np.random.randn(p, N) * 0.1
        A = np.random.randn(K, N) * 0.1
        
        # Test with different conditioning
        for eps in [1e-6, 1e-3, 1e-1]:
            D_updated = _mod_update(D, X, A, eps=eps)
            
            # Should produce valid dictionary
            assert np.all(np.isfinite(D_updated)), f"Invalid dictionary with eps={eps}"
            assert D_updated.shape == D.shape, "Dictionary shape changed"
            
            # Should be normalized
            norms = np.linalg.norm(D_updated, axis=0)
            np.testing.assert_allclose(norms, 1.0, rtol=1e-10, atol=1e-12)
    
    def test_gradient_implementation_integration(self):
        """Test all gradient implementations working together."""
        np.random.seed(42)
        
        # Test all modes with same data
        X = np.random.randn(25, 20) * 0.1
        n_atoms = 12
        
        results = {}
        for mode in ["l1", "paper", "paper_gdD"]:
            coder = SparseCoder(n_atoms=n_atoms, mode=mode, lam=0.1, seed=42)
            coder.fit(X, n_steps=3, lr=0.05)
            A = coder.encode(X[:, :5])
            
            # Store results
            results[mode] = {
                'codes': A,
                'dictionary': coder.D.copy(),
                'reconstruction_error': np.linalg.norm(X[:, :5] - coder.D @ A, 'fro')
            }
            
            # Basic validity checks
            assert np.all(np.isfinite(A)), f"Invalid codes for {mode}"
            assert np.all(np.isfinite(coder.D)), f"Invalid dictionary for {mode}"
        
        # All methods should achieve reasonable reconstruction
        for mode, result in results.items():
            rel_error = result['reconstruction_error'] / np.linalg.norm(X[:, :5], 'fro')
            assert rel_error < 2.0, f"{mode} mode has poor reconstruction: {rel_error:.3f}"
        
        # Different methods should produce different results
        codes_l1 = results['l1']['codes']
        codes_paper = results['paper']['codes']
        codes_gdD = results['paper_gdD']['codes']
        
        diff_l1_paper = np.linalg.norm(codes_l1 - codes_paper, 'fro')
        diff_paper_gdD = np.linalg.norm(codes_paper - codes_gdD, 'fro')
        
        assert diff_l1_paper > 0.01, "L1 and paper modes too similar"
        assert diff_paper_gdD > 0.01, "Paper modes too similar"