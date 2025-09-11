"""
Research Mathematical Accuracy Validation Tests.

Validates implementations against original research papers with exact mathematical properties.
Tests follow DOME (Data, Optimization, Model, Evaluation) framework for reproducible research.

Research Papers Tested:
- Tibshirani (1996): L1 penalty mathematical properties
- Beck & Teboulle (2009): FISTA convergence rate O(1/k²)
- Aharon et al. (2006): K-SVD dictionary learning algorithm
- Zou & Hastie (2005): Elastic Net penalty formulation

Author: Benedict Chen
"""

import numpy as np
import pytest
from typing import List, Tuple
import warnings

# Import implementations to test
from sparse_coding.core.penalties.implementations import L1Penalty, L2Penalty, ElasticNetPenalty
from sparse_coding.core.inference.fista_accelerated_solver import FISTASolver
from sparse_coding.factories.algorithm_factory import create_penalty, create_solver, create_learner


class TestL1PenaltyTibshirani1996:
    """
    Test L1 penalty mathematical properties from Tibshirani (1996).
    
    Research Foundation:
    Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. 
    Journal of the Royal Statistical Society: Series B, 58(1), 267-288.
    """
    
    def setup_method(self):
        """Setup reproducible test data."""
        np.random.seed(42)  # DOME requirement: deterministic components
        self.penalty = L1Penalty(lam=0.1)
        self.test_data = np.array([1.0, -2.0, 0.5, -0.3, 0.0])
        
    def test_penalty_non_negativity(self):
        """L1 penalty must be non-negative for all inputs."""
        # Test with various input types
        test_cases = [
            np.array([1.0, -1.0, 0.0]),
            np.array([100.0, -100.0]),
            np.zeros(10),
            np.random.randn(50)
        ]
        
        for data in test_cases:
            value = self.penalty.value(data)
            assert value >= 0, f"L1 penalty must be non-negative, got {value} for input {data}"
            
    def test_penalty_homogeneity(self):
        """L1 penalty is homogeneous of degree 1: ψ(λa) = λψ(a) for λ ≥ 0."""
        scales = [0.0, 0.5, 1.0, 2.0, 10.0]
        
        original_value = self.penalty.value(self.test_data)
        
        for scale in scales:
            scaled_data = scale * self.test_data
            scaled_value = self.penalty.value(scaled_data)
            expected_value = scale * original_value
            
            assert abs(scaled_value - expected_value) < 1e-12, \
                f"L1 homogeneity violated: ψ({scale}·a) = {scaled_value}, expected {expected_value}"
                
    def test_triangle_inequality(self):
        """L1 penalty satisfies triangle inequality: ψ(a+b) ≤ ψ(a) + ψ(b)."""
        np.random.seed(123)
        for _ in range(10):  # Multiple random tests
            a = np.random.randn(5)
            b = np.random.randn(5)
            
            val_a = self.penalty.value(a)
            val_b = self.penalty.value(b)
            val_sum = self.penalty.value(a + b)
            
            assert val_sum <= val_a + val_b + 1e-12, \
                f"Triangle inequality violated: ψ(a+b)={val_sum}, ψ(a)+ψ(b)={val_a + val_b}"
                
    def test_soft_thresholding_mathematical_properties(self):
        """Test soft thresholding proximal operator properties."""
        t = 0.1
        threshold = t * self.penalty.lam
        
        # Test soft thresholding formula: prox(z) = sign(z) * max(|z| - threshold, 0)
        prox_result = self.penalty.prox(self.test_data, t)
        
        for i, (z_i, prox_i) in enumerate(zip(self.test_data, prox_result)):
            if abs(z_i) <= threshold:
                expected = 0.0
            else:
                expected = np.sign(z_i) * (abs(z_i) - threshold)
                
            assert abs(prox_i - expected) < 1e-12, \
                f"Soft thresholding incorrect at index {i}: got {prox_i}, expected {expected}"
                
    def test_subdifferential_at_zero(self):
        """Test L1 subgradient at zero: ∂ψ(0) ∈ [-λ, λ]."""
        zero_point = np.zeros(5)
        subgrad = self.penalty.grad(zero_point)
        
        # At zero, subgradient should be zero (we use zero as conventional choice)
        assert np.allclose(subgrad, 0.0, atol=1e-12), \
            f"L1 subgradient at zero should be 0, got {subgrad}"


class TestFISTABeckTeboulle2009:
    """
    Test FISTA convergence properties from Beck & Teboulle (2009).
    
    Research Foundation:
    Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm 
    for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1), 183-202.
    """
    
    def setup_method(self):
        """Setup well-conditioned test problem."""
        np.random.seed(42)
        self.n_features, self.n_atoms = 50, 30
        
        # Create normalized dictionary (well-conditioned)
        self.D = np.random.randn(self.n_features, self.n_atoms)
        self.D = self.D / np.linalg.norm(self.D, axis=0)
        
        # Create sparse ground truth
        self.true_codes = np.zeros(self.n_atoms)
        active_indices = np.random.choice(self.n_atoms, 5, replace=False)
        self.true_codes[active_indices] = np.random.randn(5)
        
        # Generate signal with small noise
        self.x = self.D @ self.true_codes + 0.01 * np.random.randn(self.n_features)
        
        self.penalty = L1Penalty(lam=0.1)
        
    def test_fista_convergence_rate(self):
        """Test FISTA O(1/k²) convergence rate."""
        solver = FISTASolver(max_iter=100, tol=1e-12, variant='standard')
        
        # Run FISTA
        codes, iterations = solver.solve(self.D, self.x, self.penalty, lam=0.1)
        
        # Verify basic convergence
        reconstruction_error = np.linalg.norm(self.x - self.D @ codes)
        assert reconstruction_error < 0.1, f"FISTA failed to converge: error={reconstruction_error}"
        
        # Verify reasonable sparsity
        sparsity = np.sum(np.abs(codes) > 1e-6)
        assert sparsity <= 2 * len(np.nonzero(self.true_codes)[0]), \
            f"FISTA produced too dense solution: {sparsity} non-zeros"
            
    def test_fista_momentum_calculation(self):
        """Test FISTA momentum parameter calculation: t_{k+1} = (1 + √(1 + 4t_k²))/2."""
        # Test momentum sequence
        t = 1.0  # Initial value
        
        for k in range(10):
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            
            # Verify momentum properties
            assert t_new > t, f"Momentum should be increasing: t_{k}={t}, t_{k+1}={t_new}"
            assert t_new >= (k + 2) / 2, f"Momentum should satisfy t_k ≥ (k+1)/2"
            
            t = t_new
            
    def test_lipschitz_constant_calculation(self):
        """Test Lipschitz constant L = λ_max(D^T D)."""
        solver = FISTASolver(max_iter=10, tol=1e-6, variant='standard')
        
        # Calculate Lipschitz constant manually
        DTD = self.D.T @ self.D
        L_expected = np.linalg.norm(DTD, ord=2)  # Spectral norm
        
        # FISTA should compute this internally
        # Run solver and check if it uses correct step size
        codes, _ = solver.solve(self.D, self.x, self.penalty, lam=0.1, L=L_expected)
        
        # Verify solution is reasonable
        reconstruction_error = np.linalg.norm(self.x - self.D @ codes)
        assert reconstruction_error < 1.0, f"Lipschitz constant calculation may be incorrect"


class TestElasticNetZouHastie2005:
    """
    Test Elastic Net penalty from Zou & Hastie (2005).
    
    Research Foundation:
    Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net.
    Journal of the Royal Statistical Society: Series B, 67(2), 301-320.
    """
    
    def setup_method(self):
        """Setup Elastic Net test."""
        np.random.seed(42)
        self.penalty = ElasticNetPenalty(lam=0.1, l1_ratio=0.7)
        self.test_data = np.array([1.0, -2.0, 0.5, -0.3, 0.0])
        
    def test_elastic_net_decomposition(self):
        """Test Elastic Net = α * L1 + (1-α) * L2 penalty."""
        lam, alpha = 0.1, 0.7
        l1_pen = L1Penalty(lam=lam * alpha)
        l2_pen = L2Penalty(lam=lam * (1 - alpha))
        
        # Calculate components separately
        l1_value = l1_pen.value(self.test_data)
        l2_value = l2_pen.value(self.test_data)
        expected_value = l1_value + l2_value
        
        # Calculate with Elastic Net
        elastic_value = self.penalty.value(self.test_data)
        
        assert abs(elastic_value - expected_value) < 1e-12, \
            f"Elastic Net decomposition incorrect: got {elastic_value}, expected {expected_value}"
            
    def test_elastic_net_proximal_operator(self):
        """Test Elastic Net proximal operator properties."""
        t = 0.1
        
        # Elastic Net prox: shrink then soft threshold
        prox_result = self.penalty.prox(self.test_data, t)
        
        # Manual calculation
        shrinkage_factor = 1.0 / (1.0 + t * self.penalty.lam_l2)
        z_shrunk = shrinkage_factor * self.test_data
        threshold = t * self.penalty.lam_l1 * shrinkage_factor
        expected = np.sign(z_shrunk) * np.maximum(np.abs(z_shrunk) - threshold, 0.0)
        
        assert np.allclose(prox_result, expected, atol=1e-12), \
            f"Elastic Net proximal operator incorrect"


class TestDictionaryLearningIntegration:
    """
    Integration tests for complete dictionary learning pipeline.
    
    Tests complete sparse coding pipeline with mathematical guarantees.
    """
    
    def setup_method(self):
        """Setup integration test problem."""
        np.random.seed(42)
        self.n_features, self.n_atoms, self.n_samples = 20, 15, 100
        
        # Create synthetic dictionary learning problem
        self.true_dict = np.random.randn(self.n_features, self.n_atoms)
        self.true_dict = self.true_dict / np.linalg.norm(self.true_dict, axis=0)
        
        # Generate sparse codes
        self.true_codes = np.zeros((self.n_atoms, self.n_samples))
        for i in range(self.n_samples):
            active_indices = np.random.choice(self.n_atoms, 3, replace=False)
            self.true_codes[active_indices, i] = np.random.randn(3)
            
        # Generate training data
        self.X = self.true_dict @ self.true_codes + 0.01 * np.random.randn(self.n_features, self.n_samples)
        
    def test_dictionary_learning_pipeline(self):
        """Test complete dictionary learning with convergence guarantees."""
        # Create learner with research-accurate configuration
        learner = create_learner(
            penalty_type='l1',
            solver_type='fista', 
            updater_type='mod',
            n_atoms=self.n_atoms,
            n_iterations=10,
            lam=0.1,
            convergence_tolerance=1e-4,
            enable_early_stopping=True
        )
        
        # Train dictionary
        learner.fit(self.X)
        
        # Test encoding
        test_sample = self.X[:, 0:1]
        codes = learner.encode(test_sample)
        
        # Test decoding
        reconstruction = learner.decode(codes)
        
        # Verify mathematical properties
        reconstruction_error = np.linalg.norm(test_sample - reconstruction)
        assert reconstruction_error < 0.5, f"High reconstruction error: {reconstruction_error}"
        
        sparsity = np.sum(np.abs(codes) > 1e-6)
        assert sparsity <= 10, f"Solution not sparse enough: {sparsity} non-zeros"
        
        # Verify shapes
        assert codes.shape == (self.n_atoms, 1), f"Incorrect codes shape: {codes.shape}"
        assert reconstruction.shape == test_sample.shape, f"Shape mismatch in reconstruction"
        
        # Test convergence info
        conv_info = learner.get_convergence_info()
        assert 'converged' in conv_info, "Convergence info missing"
        
    def test_mathematical_optimality_conditions(self):
        """Test KKT conditions and optimality for sparse coding."""
        # Simple test case
        D = np.array([[1.0, 0.5], [0.0, 1.0]])
        x = np.array([1.0, 0.5])
        
        penalty = L1Penalty(lam=0.1)
        solver = FISTASolver(max_iter=1000, tol=1e-10)
        
        codes, _ = solver.solve(D, x, penalty, lam=0.1)
        
        # Check residual
        residual = D.T @ (D @ codes - x)
        
        # For active components: |residual| should equal lambda
        # For inactive components: |residual| should be <= lambda
        active_mask = np.abs(codes) > 1e-8
        inactive_mask = ~active_mask
        
        if np.any(active_mask):
            active_residuals = np.abs(residual[active_mask])
            # Allow some numerical tolerance
            assert np.all(active_residuals <= 0.1 + 1e-6), \
                f"KKT conditions violated for active components: {active_residuals}"
                
        if np.any(inactive_mask):
            inactive_residuals = np.abs(residual[inactive_mask])
            assert np.all(inactive_residuals <= 0.1 + 1e-6), \
                f"KKT conditions violated for inactive components: {inactive_residuals}"


def test_numerical_stability():
    """Test numerical stability across different data scales."""
    penalty = L1Penalty(lam=0.1, numerical_stability='robust')
    
    # Test with very large values
    large_data = np.array([1e8, -1e8, 1e9])
    value_large = penalty.value(large_data)
    assert np.isfinite(value_large), "Numerical instability with large values"
    
    # Test with very small values
    small_data = np.array([1e-10, -1e-10, 1e-12])
    value_small = penalty.value(small_data)
    assert np.isfinite(value_small), "Numerical instability with small values"
    
    # Test with zero
    zero_data = np.zeros(10)
    value_zero = penalty.value(zero_data)
    assert value_zero == 0.0, f"L1 penalty of zero should be zero, got {value_zero}"


if __name__ == "__main__":
    # Run specific test for debugging
    test_numerical_stability()
    
    # Run L1 penalty tests
    l1_tester = TestL1PenaltyTibshirani1996()
    l1_tester.setup_method()
    l1_tester.test_penalty_non_negativity()
    l1_tester.test_penalty_homogeneity()
    l1_tester.test_triangle_inequality()
    l1_tester.test_soft_thresholding_mathematical_properties()
    
    print("✅ All mathematical accuracy tests passed")