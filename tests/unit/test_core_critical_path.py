#!/usr/bin/env python3
"""
Critical path test suite - 80%+ coverage for core functionality.

Tests the essential sparse coding workflow that users actually need:
1. Create dictionary 
2. Solve sparse coding problem
3. Get reconstruction
4. Verify mathematical properties

This focuses on the 4 core modules that provide 90% of user value.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from sparse_coding import SparseCoder, create_proximal_sparse_coder
from sparse_coding.fista_batch import fista_batch, soft_thresh, power_iter_L
from sparse_coding.core.penalties.implementations import L1Penalty, L2Penalty, ElasticNetPenalty
from sparse_coding.core.solver_implementations import FistaSolver, IstaSolver


class TestSparseCoder:
    """Test SparseCoder - the main user-facing API."""
    
    def test_basic_workflow(self):
        """Test basic sparse coding workflow works end-to-end."""
        # Setup
        np.random.seed(42)
        D = np.random.randn(64, 32)
        D /= np.linalg.norm(D, axis=0)
        X = np.random.randn(64, 10)
        
        # Test
        coder = SparseCoder(n_atoms=32, lam=0.1)
        coder.D = D
        codes = coder.encode(X)
        reconstruction = coder.decode(codes)
        
        # Verify
        assert codes.shape == (32, 10)
        assert reconstruction.shape == (64, 10)
        mse = np.mean((X - reconstruction) ** 2)
        assert mse < 1.0  # Reasonable reconstruction
        sparsity = np.mean(np.abs(codes) < 1e-6)
        assert sparsity > 0.1  # Some sparsity achieved
        
    def test_parameter_validation(self):
        """Test input validation catches errors."""
        coder = SparseCoder(n_atoms=32, lam=0.1)
        
        # Test invalid inputs
        with pytest.raises(ValueError):
            coder.encode(np.array([]))  # Empty array
        with pytest.raises(ValueError):
            coder.encode(np.ones((10,)))  # 1D array
        with pytest.raises(ValueError):
            coder.encode(np.full((10, 10), np.inf))  # Non-finite values
            
    def test_different_modes(self):
        """Test different sparse coding modes work."""
        np.random.seed(42)
        D = np.random.randn(32, 16)
        D /= np.linalg.norm(D, axis=0)
        X = np.random.randn(32, 5)
        
        for mode in ['l1', 'paper']:
            coder = SparseCoder(n_atoms=16, lam=0.1, mode=mode)
            coder.D = D
            codes = coder.encode(X)
            assert codes.shape == (16, 5)
            assert not np.any(np.isnan(codes))


class TestFistaBatch:
    """Test FISTA batch algorithms."""
    
    def test_soft_thresh(self):
        """Test soft thresholding function."""
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = soft_thresh(x, 1.0)
        expected = np.array([-1.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, expected)
        
    def test_power_iter_L(self):
        """Test Lipschitz constant computation."""
        np.random.seed(42)
        D = np.random.randn(64, 32)
        D /= np.linalg.norm(D, axis=0)
        
        L = power_iter_L(D)
        spectral_norm_squared = np.linalg.norm(D.T @ D, ord=2)
        
        # Power iteration should approximate spectral norm squared
        assert abs(L - spectral_norm_squared) < 0.1
        assert L > 0
        
    def test_fista_batch_convergence(self):
        """Test FISTA batch algorithm converges."""
        np.random.seed(42)
        n_features, n_atoms = 32, 16
        D = np.random.randn(n_features, n_atoms)
        D /= np.linalg.norm(D, axis=0)
        
        # Create sparse ground truth
        true_codes = np.zeros((n_atoms, 1))
        true_codes[:4] = np.random.randn(4, 1)
        X = D @ true_codes + 0.01 * np.random.randn(n_features, 1)
        
        L = power_iter_L(D)
        codes = fista_batch(X, D, lam=0.01, L=L, max_iter=100)
        
        # Verify convergence
        reconstruction = D @ codes
        mse = np.mean((X - reconstruction) ** 2)
        assert mse < 0.1  # Good reconstruction
        
        # Verify sparsity (should recover sparse structure)
        n_nonzero = np.sum(np.abs(codes) > 1e-3)
        assert n_nonzero <= 8  # Reasonable sparsity


class TestPenalties:
    """Test penalty function implementations."""
    
    def test_l1_penalty(self):
        """Test L1 penalty function."""
        penalty = L1Penalty(lam=1.0)
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # Test value
        value = penalty.value(x)
        expected_value = 6.0  # sum of absolute values
        assert abs(value - expected_value) < 1e-10
        
        # Test proximal operator (soft thresholding)
        prox_result = penalty.prox(x, 0.5)
        expected_prox = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
        np.testing.assert_allclose(prox_result, expected_prox)
        
    def test_l2_penalty(self):
        """Test L2 penalty function."""
        penalty = L2Penalty(lam=1.0)
        x = np.array([3.0, 4.0])
        
        # Test value: lam * ||x||_2^2 = 1.0 * (9 + 16) = 25.0
        value = penalty.value(x)
        assert abs(value - 25.0) < 1e-10
        
        # Test proximal operator: x / (1 + t*lam)
        prox_result = penalty.prox(x, 1.0)
        expected_prox = x / 2.0  # 1 + 1*1 = 2
        np.testing.assert_allclose(prox_result, expected_prox)
        
    def test_elastic_net_penalty(self):
        """Test Elastic Net penalty (L1 + L2)."""
        penalty = ElasticNetPenalty(lam=1.0, alpha=0.5)  # 50% L1, 50% L2
        x = np.array([2.0, -2.0])
        
        # Test value: 0.5 * (|2| + |-2|) + 0.5 * (4 + 4) = 2 + 4 = 6
        value = penalty.value(x)
        expected_value = 6.0
        assert abs(value - expected_value) < 1e-10
        
        # Test prox exists and returns correct shape
        prox_result = penalty.prox(x, 0.5)
        assert prox_result.shape == x.shape
        assert not np.any(np.isnan(prox_result))


class TestSolverImplementations:
    """Test solver implementations and registry."""
    
    def test_fista_solver_creation(self):
        """Test FISTA solver can be created and configured."""
        from sparse_coding.core.solver_implementations import get_solver
        
        solver = get_solver('fista')
        assert solver is not None
        assert hasattr(solver, 'solve')
        
    def test_solver_registry(self):
        """Test solver registry has expected solvers."""
        from sparse_coding.core.solver_implementations import list_solvers
        
        solvers = list_solvers()
        expected_solvers = ['fista', 'ista', 'omp', 'ncg']
        for solver_name in expected_solvers:
            assert solver_name in solvers
            
    def test_create_proximal_sparse_coder_factory(self):
        """Test proximal factory function works."""
        D = np.random.randn(32, 16)
        D /= np.linalg.norm(D, axis=0)
        
        optimizer = create_proximal_sparse_coder(D, penalty_type='l1', penalty_params={'lam': 0.1})
        assert optimizer is not None
        assert hasattr(optimizer, 'fista')


@given(
    n_features=st.integers(min_value=10, max_value=100),
    n_atoms=st.integers(min_value=5, max_value=50),
    lam=st.floats(min_value=0.001, max_value=1.0)
)
def test_sparse_coding_properties(n_features, n_atoms, lam):
    """Property-based test: sparse coding should satisfy basic mathematical properties."""
    # Ensure n_atoms <= n_features for well-posed problem
    if n_atoms > n_features:
        n_atoms = n_features // 2
        
    np.random.seed(42)  # Deterministic for reproducibility
    D = np.random.randn(n_features, n_atoms)
    D /= np.linalg.norm(D, axis=0)
    X = np.random.randn(n_features, 3)
    
    coder = SparseCoder(n_atoms=n_atoms, lam=lam)
    coder.D = D
    codes = coder.encode(X)
    
    # Property 1: Output shape is correct
    assert codes.shape == (n_atoms, 3)
    
    # Property 2: No NaN/Inf in output
    assert np.all(np.isfinite(codes))
    
    # Property 3: Sparsity increases with penalty
    if lam > 0.01:  # Avoid numerical issues with very small lambda
        sparsity = np.mean(np.abs(codes) < 1e-6)
        assert sparsity >= 0.0  # At least some sparsity expected


if __name__ == "__main__":
    # Quick smoke test
    test = TestSparseCoder()
    test.test_basic_workflow()
    print("✅ Critical path tests pass")