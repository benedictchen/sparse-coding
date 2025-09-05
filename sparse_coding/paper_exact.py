"""
Paper-exact Olshausen & Field (1996) sparse coding with log sparsity penalty
and nonlinear conjugate gradient optimization.
"""

import numpy as np

def S_log(x):
    """Log sparsity penalty: S(x) = log(1 + x^2)"""
    return np.log1p(x * x)

def dS_log(x):
    """Derivative: d/dx log(1+x^2) = 2x/(1+x^2)"""
    return 2.0 * x / (1.0 + x * x)

def energy_paper(D, x, a, lam, sigma):
    """
    Olshausen & Field energy function:
    E = 0.5 * ||x - D a||^2 - lam * sum(S(a/sigma))
    
    Note: Negative sign on sparsity term because we MINIMIZE energy
    but MAXIMIZE sparsity (paper formulation).
    """
    r = x - D @ a
    pen = S_log(a / sigma)
    return 0.5 * float(r @ r) - lam * float(np.sum(pen))

def grad_paper(D, x, a, lam, sigma):
    """
    Gradient of paper energy w.r.t. coefficients a:
    ∇E = -D^T(x - Da) - lam * (1/sigma) * dS(a/sigma)
    """
    r = x - D @ a
    g_data = -D.T @ r  # Data term gradient
    g_pen = -lam * (dS_log(a / sigma) / sigma)  # Sparsity term gradient
    return g_data + g_pen

def nonlinear_cg(D, x, a0, lam, sigma, max_iter=200, tol=1e-4, 
                ls_beta=0.5, ls_c=1e-4, rel_tol=0.01):
    """
    Nonlinear conjugate gradient (Polak-Ribière) with backtracking
    line search for the paper's energy function.
    
    Args:
        D: Dictionary (p, K)
        x: Signal vector (p,)
        a0: Initial coefficients (K,)
        lam: Sparsity penalty weight
        sigma: Sparsity scaling parameter
        max_iter: Maximum iterations
        tol: Absolute tolerance
        ls_beta: Line search reduction factor
        ls_c: Armijo condition parameter
        rel_tol: Relative tolerance (paper stopping criterion)
        
    Returns:
        a: Optimized coefficients (K,)
        info: Optimization info dict
    """
    a = a0.copy()
    g = grad_paper(D, x, a, lam, sigma)
    d = -g  # Initial search direction (steepest descent)
    E_prev = energy_paper(D, x, a, lam, sigma)
    
    for it in range(1, max_iter + 1):
        # Backtracking line search satisfying Armijo condition
        t = 1.0
        g_dot_d = float(g @ d)
        
        while True:
            a_try = a + t * d
            E_try = energy_paper(D, x, a_try, lam, sigma)
            
            # Armijo condition: sufficient decrease
            if E_try <= E_prev + ls_c * t * g_dot_d:
                break
                
            t *= ls_beta
            if t < 1e-12:
                # Fallback: take tiny step
                a_try = a + 1e-6 * d
                E_try = energy_paper(D, x, a_try, lam, sigma)
                break
        
        # Update state
        a_next = a_try
        g_next = grad_paper(D, x, a_next, lam, sigma)
        
        # Polak-Ribière beta computation
        y = g_next - g
        beta_pr = float(max(0.0, (g_next @ y) / (g @ g + 1e-18)))
        d_next = -g_next + beta_pr * d
        
        # Paper stopping criterion: 1% relative energy change
        rel_change = abs(E_prev - E_try) / max(1.0, abs(E_prev))
        if rel_change <= rel_tol:
            return a_next, {"iters": it, "obj": E_try, "converged": True}
        
        # Prepare for next iteration
        a, g, d, E_prev = a_next, g_next, d_next, E_try
    
    return a, {"iters": max_iter, "obj": E_prev, "converged": False}