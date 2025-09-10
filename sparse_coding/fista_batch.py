"""
Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) implementation.

High-performance batch FISTA implementation for sparse coding optimization.
Implements the accelerated proximal gradient method from Beck & Teboulle (2009).

Core optimization problem:
    minimize: ||X - DA||²_F + λ||A||₁

Where:
    X: Data matrix (n_features, n_samples)  
    D: Dictionary matrix (n_features, n_atoms)
    A: Sparse codes (n_atoms, n_samples)
    λ: Sparsity regularization parameter

Algorithm Properties:
- Convergence rate: O(1/k²) where k is iteration count
- Optimal first-order method for composite optimization
- Handles non-smooth L1 penalty through proximal operator (soft thresholding)

References:
    Beck & Teboulle (2009). A Fast Iterative Shrinkage-Thresholding Algorithm 
    for Linear Inverse Problems. SIAM Journal on Imaging Sciences.
    
    Olshausen & Field (1996). Emergence of simple-cell receptive field properties 
    by learning a sparse code for natural images. Nature.
"""

import numpy as np

def soft_thresh(X, t):
    """
    Soft thresholding operator (proximal operator for L1 norm).
    
    Implements the proximal operator for L1 penalty: prox_{t||·||₁}(x) = S_t(x)
    where S_t(x) = sign(x) * max(|x| - t, 0) is the soft thresholding function.
    
    Args:
        X: Input array
        t: Threshold parameter (must be non-negative)
        
    Returns:
        Soft-thresholded array with same shape as X
        
    Mathematical Properties:
        - Shrinks small values toward zero
        - Preserves sign of large values
        - Sets values with |x| ≤ t to exactly zero (sparsity-inducing)
    """
    return np.sign(X) * np.maximum(np.abs(X) - t, 0.0)

def power_iter_L(D, n_iter=50, tol=1e-7, rng=None):
    """
    Compute Lipschitz constant L = ||D^T D||₂ using power iteration.
    
    The Lipschitz constant determines the FISTA step size and is crucial
    for convergence. Uses power iteration to efficiently compute the largest
    eigenvalue of D^T D without full eigendecomposition.
    
    Args:
        D: Dictionary matrix (n_features, n_atoms)
        n_iter: Maximum power iterations (default: 50)
        tol: Convergence tolerance (default: 1e-7)
        rng: Random number generator (default: None)
        
    Returns:
        Lipschitz constant L ≥ ||D^T D||₂
        
    Algorithm:
        1. Initialize random vector v₀
        2. Iterate: v_{k+1} = (D^T D v_k) / ||D^T D v_k||
        3. Converged eigenvalue: λ = v^T (D^T D) v
    """
    rng = np.random.default_rng(None if rng is None else rng)
    K = D.shape[1]
    v = rng.normal(size=(K,))
    v /= (np.linalg.norm(v) + 1e-12)
    last = 0.0
    DtD = D.T @ D
    for _ in range(n_iter):
        v = DtD @ v
        nrm = np.linalg.norm(v) + 1e-12
        v = v / nrm
        lam = float(v @ (DtD @ v))
        if abs(lam - last) < tol * max(1.0, last):
            break
        last = lam
    return max(lam, 1e-12)

def fista_batch(D, X, lam, L=None, max_iter=200, tol=1e-6):
    """
    Batch FISTA optimization for sparse coding.
    
    Solves the sparse coding optimization problem:
        minimize: (1/2)||X - DA||²_F + λ||A||₁
        
    Using the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
    with optimal O(1/k²) convergence rate.
    
    Args:
        D: Dictionary matrix (n_features, n_atoms)
        X: Data matrix (n_features, n_samples)
        lam: L1 regularization parameter λ > 0
        L: Lipschitz constant (auto-computed if None)
        max_iter: Maximum iterations (default: 200)
        tol: Convergence tolerance (default: 1e-6)
        
    Returns:
        A: Sparse codes matrix (n_atoms, n_samples)
        
    Algorithm Steps:
        1. Initialize: A₀ = 0, Y₀ = A₀, t₀ = 1
        2. Gradient step: A_{k+1} = prox_{λ/L}(Y_k - ∇f(Y_k)/L)
        3. Acceleration: Y_{k+1} = A_{k+1} + ((t_k-1)/t_{k+1})(A_{k+1} - A_k)
        4. Update: t_{k+1} = (1 + √(1 + 4t_k²))/2
        
    where f(A) = (1/2)||X - DA||²_F and prox is soft thresholding.
    """
    D = np.asarray(D, float); X = np.asarray(X, float)
    p, K = D.shape; N = X.shape[1]
    DtD = D.T @ D
    DtX = D.T @ X
    if L is None:
        L = power_iter_L(D)
    A = np.zeros((K, N)); Y = np.zeros_like(A); t = 1.0
    for _ in range(max_iter):
        G = DtD @ Y - DtX
        A_next = soft_thresh(Y - G / L, lam / L)
        t_next = (1 + np.sqrt(1 + 4*t*t)) / 2.0
        Y = A_next + ((t - 1) / t_next) * (A_next - A)
        if np.linalg.norm(A_next - A) <= tol * max(1.0, np.linalg.norm(A)):
            return A_next
        A, t = A_next, t_next
    return A
