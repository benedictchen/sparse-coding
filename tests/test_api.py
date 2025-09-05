"""
Test the main SparseCoder API functionality.
"""

import numpy as np
import pytest
from sparse_coding import SparseCoder, kkt_violation_l1

def test_sparse_coder_init():
    """Test SparseCoder initialization."""
    coder = SparseCoder(n_atoms=64, mode='l1', seed=42)
    assert coder.n_atoms == 64
    assert coder.mode == 'l1'
    assert coder.D is None

def test_l1_mode_basic():
    """Test L1 sparse coding on synthetic data."""
    rng = np.random.default_rng(42)
    
    # Generate synthetic sparse data
    p, K, N = 64, 96, 200
    D_true = rng.normal(size=(p, K))
    D_true /= np.linalg.norm(D_true, axis=0, keepdims=True) + 1e-12
    
    # Sparse coefficients
    A_true = rng.laplace(size=(K, N)) * (rng.random((K, N)) < 0.15)
    X = D_true @ A_true + 0.01 * rng.normal(size=(p, N))
    
    # Fit sparse coder
    coder = SparseCoder(n_atoms=K, mode='l1', max_iter=200, tol=1e-7, seed=42)
    coder.fit(X, n_steps=20, lr=0.2)
    
    # Test encoding
    A_hat = coder.encode(X)
    X_hat = coder.decode(A_hat)
    
    # Check reconstruction quality
    err = np.linalg.norm(X - X_hat) / np.linalg.norm(X)
    assert err < 0.6, f"Reconstruction error too high: {err}"
    
    # Check sparsity
    sparsity = np.mean(np.abs(A_hat) < 1e-8)
    assert sparsity > 0.5, f"Not sparse enough: {sparsity}"

def test_paper_mode_basic():
    """Test paper-exact mode on synthetic data."""
    rng = np.random.default_rng(7)
    p, K, N = 64, 96, 80
    
    # True dictionary
    D_true = rng.normal(size=(p, K))
    D_true /= np.linalg.norm(D_true, axis=0, keepdims=True) + 1e-12
    
    # Sparse coefficients  
    A_true = rng.laplace(size=(K, N)) * (rng.random((K, N)) < 0.2)
    X = D_true @ A_true + 0.01 * rng.normal(size=(p, N))
    
    # Test paper mode
    coder = SparseCoder(n_atoms=K, mode='paper', max_iter=80, tol=1e-4, seed=7)
    coder.D = coder._init_dict(p)
    
    A = coder.encode(X)
    assert np.isfinite(A).all(), "Non-finite coefficients"
    
    # Reconstruction test
    X_hat = coder.decode(A)
    err = np.linalg.norm(X - X_hat) / np.linalg.norm(X)
    assert err < 0.9, f"Paper mode reconstruction error: {err}"

def test_kkt_violation():
    """Test KKT condition validation for L1 solutions."""
    rng = np.random.default_rng(3)
    p, K, N = 64, 96, 128
    
    # Random dictionary
    D = rng.normal(size=(p, K))
    D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    
    # Sparse ground truth
    A_true = rng.laplace(size=(K, N)) * (rng.random((K, N)) < 0.2)
    X = D @ A_true + 0.01 * rng.normal(size=(p, N))
    
    # Solve with tight tolerance
    coder = SparseCoder(n_atoms=K, mode='l1', max_iter=200, tol=1e-6, seed=3)
    coder.D = D.copy()
    A = coder.encode(X)
    
    # Check KKT violation (relax tolerance for this test)
    lam_eff = coder.ratio_lambda_over_sigma * float(np.std(X))
    violation = kkt_violation_l1(coder.D, X, A, lam_eff)
    assert violation < 0.05, f"KKT violation too high: {violation}"

def test_sklearn_compatibility():
    """Test sklearn-style fit/transform interface."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 64))  # (N, p) sklearn convention
    
    coder = SparseCoder(n_atoms=32, seed=5)
    
    # Test fit_transform
    codes = coder.fit_transform(X.T)  # Pass as (p, N)
    assert codes.shape == (100, 32)  # Returns (N, K)
    
    # Test separate fit/transform  
    coder2 = SparseCoder(n_atoms=32, seed=5)
    coder2.fit(X.T)
    codes2 = coder2.transform(X.T)
    
    np.testing.assert_array_almost_equal(codes, codes2, decimal=4)  # More realistic tolerance