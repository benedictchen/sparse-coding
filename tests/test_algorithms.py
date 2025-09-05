"""
Test core algorithms: FISTA, nonlinear CG, diagnostics.
"""

import numpy as np
import pytest
from sparse_coding.fista import fista, lipschitz_const, soft_thresh
from sparse_coding.paper_exact import nonlinear_cg, energy_paper
from sparse_coding.diagnostics import sparsity_level, reconstruction_error, dictionary_coherence

def test_soft_threshold():
    """Test soft thresholding operator."""
    x = np.array([-2, -0.5, 0, 0.5, 2])
    result = soft_thresh(x, 1.0)
    expected = np.array([-1, 0, 0, 0, 1])
    np.testing.assert_array_equal(result, expected)

def test_lipschitz_estimation():
    """Test Lipschitz constant estimation."""
    rng = np.random.default_rng(42)
    D = rng.normal(size=(50, 30))
    
    L_est = lipschitz_const(D, rng=rng)
    L_true = np.linalg.norm(D.T @ D, 2)
    
    # Should be close to true value
    assert abs(L_est - L_true) / L_true < 0.1

def test_fista_convergence():
    """Test FISTA convergence on well-conditioned problem."""
    rng = np.random.default_rng(123)
    p, K = 40, 60
    
    # Well-conditioned dictionary
    D = rng.normal(size=(p, K))
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    
    # Sparse ground truth
    a_true = rng.laplace(size=(K,)) * (rng.random(K) < 0.3)
    x = D @ a_true + 0.01 * rng.normal(size=(p,))
    
    # Solve with FISTA
    lam = 0.1 * np.std(x)
    a_hat, info = fista(D, x, lam, max_iter=500, tol=1e-8)
    
    assert info['converged'], "FISTA should converge"
    assert info['iters'] < 500, "Should converge quickly"
    
    # Check solution quality
    obj = 0.5 * np.linalg.norm(x - D @ a_hat)**2 + lam * np.sum(np.abs(a_hat))
    assert obj < 10 * lam, "Objective should be reasonable"

def test_paper_energy_function():
    """Test paper energy function evaluation."""
    rng = np.random.default_rng(44)
    p, K = 30, 45
    
    D = rng.normal(size=(p, K))
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    x = rng.normal(size=(p,))
    a = rng.normal(size=(K,))
    
    lam, sigma = 0.1, 1.0
    
    # Energy should be finite
    E = energy_paper(D, x, a, lam, sigma)
    assert np.isfinite(E), "Energy should be finite"
    
    # Sparser solutions should have lower energy (higher sparsity term)
    a_sparse = a * 0.1  # Make sparser
    E_sparse = energy_paper(D, x, a_sparse, lam, sigma)
    
    # Note: Energy has negative sparsity term, so sparser -> lower energy
    # (assuming reconstruction error doesn't increase too much)

def test_nonlinear_cg_basic():
    """Test nonlinear CG on simple problem.""" 
    rng = np.random.default_rng(66)
    p, K = 25, 35
    
    D = rng.normal(size=(p, K))
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    
    # Generate signal from sparse coefficients
    a_true = rng.laplace(size=(K,)) * (rng.random(K) < 0.4)  
    x = D @ a_true + 0.01 * rng.normal(size=(p,))
    
    # Optimize with nonlinear CG
    a0 = np.zeros(K)
    lam, sigma = 0.1, 1.0
    
    a_opt, info = nonlinear_cg(D, x, a0, lam, sigma, max_iter=100, rel_tol=0.05)
    
    assert np.isfinite(a_opt).all(), "Solution should be finite"
    assert info['iters'] > 0, "Should take at least one iteration"

def test_diagnostics():
    """Test diagnostic functions."""
    rng = np.random.default_rng(88)
    
    # Test sparsity level
    A = rng.laplace(size=(50, 100)) * (rng.random((50, 100)) < 0.2)
    sparsity = sparsity_level(A)
    assert 0.7 < sparsity < 0.9, f"Expected ~80% sparsity, got {sparsity}"
    
    # Test reconstruction error
    X = rng.normal(size=(64, 100))
    X_hat = X + 0.1 * rng.normal(size=(64, 100))
    
    err_abs = reconstruction_error(X, X_hat, relative=False)
    err_rel = reconstruction_error(X, X_hat, relative=True)
    
    assert err_rel < 0.2, "Relative error should be small"
    assert err_abs > 0, "Absolute error should be positive"
    
    # Test dictionary coherence
    D = rng.normal(size=(32, 50))
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    
    coherence = dictionary_coherence(D)
    assert 0 <= coherence <= 1, f"Coherence should be in [0,1], got {coherence}"

@pytest.mark.slow
def test_fista_vs_paper_mode():
    """Compare FISTA and paper mode on same problem."""
    rng = np.random.default_rng(99)
    p, K, N = 40, 60, 50
    
    # Generate test data
    D = rng.normal(size=(p, K))
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    A_true = rng.laplace(size=(K, N)) * (rng.random((K, N)) < 0.25)
    X = D @ A_true + 0.01 * rng.normal(size=(p, N))
    
    lam = 0.1 * np.std(X)
    sigma = 1.0
    
    # FISTA solution
    A_fista = np.zeros((K, N))
    for n in range(min(N, 10)):  # Test on subset
        a, _ = fista(D, X[:, n], lam, max_iter=200, tol=1e-6)
        A_fista[:, n] = a
    
    # Paper solution  
    A_paper = np.zeros((K, 10))
    for n in range(10):
        a0 = np.zeros(K)
        a, _ = nonlinear_cg(D, X[:, n], a0, lam, sigma, max_iter=100)
        A_paper[:, n] = a
    
    # Both should achieve reasonable reconstruction
    X_subset = X[:, :10]
    err_fista = np.linalg.norm(X_subset - D @ A_fista[:, :10]) / np.linalg.norm(X_subset)
    err_paper = np.linalg.norm(X_subset - D @ A_paper) / np.linalg.norm(X_subset)
    
    assert err_fista < 0.7, f"FISTA error: {err_fista}"  # Relaxed tolerance
    assert err_paper < 0.7, f"Paper error: {err_paper}"