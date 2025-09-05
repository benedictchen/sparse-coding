"""
FISTA (Fast Iterative Soft-Thresholding Algorithm) implementation
with backtracking line search for L1 sparse coding.
"""

import numpy as np

def lipschitz_const(D, n_iter=50, tol=1e-7, rng=None):
    """Estimate Lipschitz constant ||D^T D||_2 using power iteration."""
    if rng is None:
        rng = np.random.default_rng()
    elif not hasattr(rng, 'normal'):
        rng = np.random.default_rng(rng)
        
    k = D.shape[1]
    v = rng.normal(size=(k,))
    v /= (np.linalg.norm(v) + 1e-12)
    
    last = 0.0
    for _ in range(n_iter):
        v = D.T @ (D @ v)
        nrm = np.linalg.norm(v) + 1e-12
        v = v / nrm
        lam = float(v @ (D.T @ (D @ v)))
        if abs(lam - last) < tol * max(1.0, abs(last)):
            break
        last = lam
    return max(lam, 1e-12)

def soft_thresh(x, t):
    """Soft thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

def fista(D, x, lam, a0=None, L=None, backtracking=True, bt_eta=1.5, 
          max_iter=500, tol=1e-6):
    """
    Solve: min_a 0.5*||x - D a||^2 + lam * ||a||_1
    
    Args:
        D: Dictionary (p, K) - atoms as columns
        x: Signal vector (p,)
        lam: L1 penalty parameter
        a0: Initial guess (K,) or None
        L: Lipschitz constant or None (auto-estimate)
        backtracking: Use backtracking line search
        bt_eta: Backtracking step factor
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        a: Sparse coefficients (K,)
        info: Convergence info dict
    """
    p, K = D.shape
    
    # Initialize
    if a0 is None:
        a = np.zeros(K, dtype=float)
    else:
        a = a0.copy()
    y = a.copy()
    t = 1.0
    
    # Estimate Lipschitz constant if not provided
    if L is None:
        L = lipschitz_const(D)
    
    # Precompute for efficiency
    Dt = D.T
    
    # Initial objective
    Dx = D @ y
    fy = 0.5 * np.linalg.norm(x - Dx)**2
    prev_obj = fy + lam * np.sum(np.abs(y))
    
    for it in range(1, max_iter + 1):
        # Gradient of smooth part at y
        grad = Dt @ (D @ y - x)
        
        if backtracking:
            # Beck-Teboulle backtracking line search
            Lk = L
            while True:
                a_next = soft_thresh(y - grad / Lk, lam / Lk)
                diff = a_next - y
                rhs = fy + diff @ grad + 0.5 * Lk * np.sum(diff**2)
                obj_smooth = 0.5 * np.linalg.norm(x - D @ a_next)**2
                if obj_smooth <= rhs + 1e-12:
                    L = Lk
                    break
                Lk *= bt_eta
        else:
            a_next = soft_thresh(y - grad / L, lam / L)
        
        # Compute objective
        obj = 0.5 * np.linalg.norm(x - D @ a_next)**2 + lam * np.sum(np.abs(a_next))
        
        # Momentum update
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2.0
        y = a_next + ((t - 1) / t_next) * (a_next - a)
        
        # Check convergence
        if abs(obj - prev_obj) <= tol * max(1.0, abs(prev_obj)):
            return a_next, {"iters": it, "obj": obj, "L": L, "converged": True}
            
        # Update for next iteration
        a, t, prev_obj = a_next, t_next, obj
        fy = 0.5 * np.linalg.norm(x - D @ y)**2
    
    return a, {"iters": max_iter, "obj": prev_obj, "L": L, "converged": False}