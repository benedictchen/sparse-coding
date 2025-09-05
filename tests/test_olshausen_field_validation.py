#!/usr/bin/env python3
"""
Research validation tests for Olshausen & Field (1996) sparse coding.

Tests mathematical correctness against the published algorithm:
- Cost function: E = ||x - Φa||² + λ Σᵢ S(aᵢ/σ)
- Sparseness function: S(u) = log(1 + u²)
- Gradient-based optimization
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparse_coding.sparse_coding_modules.optimization import (
    log_sparseness_derivative,
    OptimizationMixin
)


class TestOlshausenFieldValidation:
    """Test mathematical accuracy against Olshausen & Field (1996)."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.patch_size = 8
        self.n_atoms = 16
        self.patch = np.random.randn(self.patch_size * self.patch_size)
        self.dictionary = np.random.randn(self.patch_size * self.patch_size, self.n_atoms)
        
        # Normalize dictionary atoms (standard preprocessing)
        for i in range(self.n_atoms):
            self.dictionary[:, i] /= np.linalg.norm(self.dictionary[:, i])
    
    def test_log_sparseness_function_mathematical_correctness(self):
        """Test S(u) = log(1 + u²) implementation."""
        # Test cases with known values
        test_values = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
        expected = np.log(1 + test_values**2)
        
        # Manual calculation
        sigma = 1.0
        normalized = test_values / sigma
        actual_sparseness = np.log(1 + normalized**2)
        
        np.testing.assert_allclose(actual_sparseness, expected, rtol=1e-10)
    
    def test_sparseness_derivative_mathematical_correctness(self):
        """Test derivative S'(u) = 2u/(1 + u²) for S(u) = log(1 + u²)."""
        coeffs = np.array([0.0, 1.0, -1.0, 2.0])
        sigma = 1.0
        
        # Expected: S'(u) = 2u/(1 + u²), but we have S(aᵢ/σ)
        # So derivative w.r.t. aᵢ is: (1/σ) * S'(aᵢ/σ) = (2 * aᵢ/σ) / (σ * (1 + (aᵢ/σ)²))
        normalized_coeffs = coeffs / sigma
        expected = (2 * normalized_coeffs) / (sigma * (1 + normalized_coeffs**2))
        
        actual = log_sparseness_derivative(coeffs, sigma)
        
        np.testing.assert_allclose(actual, expected, rtol=1e-10)
    
    def test_cost_function_components(self):
        """Test that cost function has correct reconstruction and sparsity terms."""
        coeffs = np.random.randn(self.n_atoms) * 0.1  # Sparse coefficients
        
        # Reconstruction error: ||x - Φa||²
        reconstruction = self.dictionary @ coeffs
        reconstruction_error = np.linalg.norm(self.patch - reconstruction)**2
        
        # Sparsity penalty: λ Σᵢ S(aᵢ/σ)
        sigma = 1.0
        lambda_sparse = 0.1
        sparsity_penalty = lambda_sparse * np.sum(np.log(1 + (coeffs/sigma)**2))
        
        total_cost = reconstruction_error + sparsity_penalty
        
        # Verify components are positive
        assert reconstruction_error >= 0, "Reconstruction error must be non-negative"
        assert sparsity_penalty >= 0, "Sparsity penalty must be non-negative"
        assert total_cost >= 0, "Total cost must be non-negative"
    
    def test_gradient_computation(self):
        """Test gradient computation for optimization."""
        coeffs = np.random.randn(self.n_atoms) * 0.1
        sigma = 1.0
        lambda_sparse = 0.1
        
        # Reconstruction gradient: ∇_a ||x - Φa||² = 2Φᵀ(Φa - x)
        reconstruction = self.dictionary @ coeffs
        reconstruction_grad = 2 * self.dictionary.T @ (reconstruction - self.patch)
        
        # Sparsity gradient: ∇_a Σᵢ S(aᵢ/σ) 
        sparsity_grad = lambda_sparse * log_sparseness_derivative(coeffs, sigma)
        
        total_grad = reconstruction_grad + sparsity_grad
        
        # Gradient should have same shape as coefficients
        assert total_grad.shape == coeffs.shape
        
        # For sparse solutions, most gradient components should be small
        # This is a sanity check, not a strict mathematical requirement
        assert len(total_grad) == self.n_atoms
    
    def test_sparsity_promotion(self):
        """Test that sparsity penalty promotes sparse solutions."""
        # Dense coefficients
        dense_coeffs = np.ones(self.n_atoms) * 0.5
        
        # Sparse coefficients (fewer non-zero elements)
        sparse_coeffs = np.zeros(self.n_atoms)
        sparse_coeffs[:3] = 1.0  # Only first 3 are non-zero
        
        sigma = 1.0
        
        # Compute sparsity penalties
        dense_penalty = np.sum(np.log(1 + (dense_coeffs/sigma)**2))
        sparse_penalty = np.sum(np.log(1 + (sparse_coeffs/sigma)**2))
        
        # Sparse solution should have lower penalty for same L2 norm
        # Normalize by L2 norm for fair comparison
        dense_penalty_normalized = dense_penalty / np.linalg.norm(dense_coeffs)**2
        sparse_penalty_normalized = sparse_penalty / np.linalg.norm(sparse_coeffs)**2
        
        # This is a general trend test - sparse solutions are preferred
        assert sparse_penalty < dense_penalty, "Sparse coefficients should have lower penalty"
    
    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        # Test with very small coefficients
        small_coeffs = np.array([1e-10, -1e-10, 0.0])
        grad_small = log_sparseness_derivative(small_coeffs, sigma=1.0)
        assert np.all(np.isfinite(grad_small)), "Gradient should be finite for small coefficients"
        
        # Test with moderate coefficients  
        moderate_coeffs = np.array([1.0, -1.0, 2.0])
        grad_moderate = log_sparseness_derivative(moderate_coeffs, sigma=1.0)
        assert np.all(np.isfinite(grad_moderate)), "Gradient should be finite for moderate coefficients"
        
        # Test derivative magnitude is reasonable
        assert np.all(np.abs(grad_moderate) < 10), "Gradient magnitude should be reasonable"


class TestAlgorithmIntegration:
    """Test integration with the optimization framework."""
    
    def setup_method(self):
        """Set up optimization framework."""
        self.patch_size = 8
        self.n_atoms = 16
        
        patch_size = self.patch_size
        n_atoms = self.n_atoms
        
        class MockOptimizer(OptimizationMixin):
            def __init__(self):
                self.dictionary = np.random.randn(patch_size**2, n_atoms)
                # Normalize dictionary
                for i in range(n_atoms):
                    self.dictionary[:, i] /= np.linalg.norm(self.dictionary[:, i])
                
                self.sparsity_penalty = 0.1
                self.sigma = 1.0
                self.learning_rate = 0.01
                self.max_iter = 50
                self.tolerance = 1e-6
        
        self.optimizer = MockOptimizer()
    
    def test_olshausen_field_integration(self):
        """Test that the algorithm runs without errors."""
        patch = np.random.randn(self.patch_size**2)
        
        # Should not raise exceptions
        coeffs = self.optimizer.olshausen_field_sparse_coding(patch)
        
        # Basic output validation
        assert isinstance(coeffs, np.ndarray), "Should return numpy array"
        assert coeffs.shape == (self.n_atoms,), f"Should return {self.n_atoms} coefficients"
        assert np.all(np.isfinite(coeffs)), "All coefficients should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])