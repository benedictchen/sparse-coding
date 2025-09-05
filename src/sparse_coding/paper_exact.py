"""
Paper-Exact Olshausen & Field (1996) Implementation
=================================================

Implements the exact log-penalty energy function and nonlinear conjugate gradient
optimization as described in:

Olshausen, B. A., & Field, D. J. (1996). 
"Emergence of simple-cell receptive field properties by learning a sparse code for natural images."
Nature, 381(6583), 607-609.

Energy function: E = 1/2 ||x - Da||² - λ Σ log(1 + (a_i/σ)²)

This addresses the critical gap identified in ChatGPT's patch.
"""

import numpy as np


def S_log(x):
    """
    S(x) = log(1 + x²)
    
    Sparsity penalty function from Olshausen & Field (1996).
    """
    return np.log1p(x * x)


def dS_log(x):
    """
    Derivative: d/dx log(1 + x²) = 2x/(1 + x²)
    """
    return 2.0 * x / (1.0 + x * x)


def energy_paper(D, x, a, lam, sigma):
    """
    Exact Olshausen & Field (1996) energy function.
    
    E = 1/2 ||x - Da||² - λ Σ log(1 + (a_i/σ)²)
    
    Args:
        D: Dictionary matrix (p, K) - atoms as columns
        x: Data vector (p,)
        a: Coefficient vector (K,)
        lam: Sparsity parameter λ
        sigma: Scale parameter σ
        
    Returns:
        Energy value (scalar)
    """
    # Reconstruction error term
    residual = x - D @ a
    reconstruction_error = 0.5 * float(residual @ residual)
    
    # Log-penalty sparsity term
    sparsity_penalty = S_log(a / sigma)
    sparsity_term = -lam * float(np.sum(sparsity_penalty))
    
    return reconstruction_error + sparsity_term


def grad_paper(D, x, a, lam, sigma):
    """
    Gradient of Olshausen & Field energy w.r.t. coefficients a.
    
    ∇_a E = -D^T(x - Da) - λ/σ * dS/da(a/σ)
    
    Args:
        D: Dictionary matrix (p, K)
        x: Data vector (p,)
        a: Coefficient vector (K,)
        lam: Sparsity parameter
        sigma: Scale parameter
        
    Returns:
        Gradient vector (K,)
    """
    # Reconstruction gradient: -D^T * residual
    residual = x - D @ a
    grad_reconstruction = -D.T @ residual
    
    # Sparsity gradient: -λ * (1/σ) * dS/da(a/σ)
    grad_sparsity = -lam * (dS_log(a / sigma) / sigma)
    
    return grad_reconstruction + grad_sparsity


def nonlinear_cg(D, x, a0, lam, sigma, max_iter=200, tol=1e-4, 
                 ls_beta=0.5, ls_c=1e-4, rel_tol=0.01):
    """
    Nonlinear conjugate gradient optimization for Olshausen & Field energy.
    
    Uses Polak-Ribière formula with Armijo backtracking line search.
    
    Args:
        D: Dictionary matrix (p, K)
        x: Data vector (p,)
        a0: Initial coefficients (K,)
        lam: Sparsity parameter
        sigma: Scale parameter  
        max_iter: Maximum iterations
        tol: Gradient tolerance
        ls_beta: Line search backtracking factor
        ls_c: Armijo constant
        rel_tol: Relative energy change stopping criterion (1% as in paper)
        
    Returns:
        a: Optimized coefficients
        info: Optimization info dict
    """
    a = a0.copy()
    g = grad_paper(D, x, a, lam, sigma)
    d = -g  # Initial search direction
    E_prev = energy_paper(D, x, a, lam, sigma)
    
    for it in range(1, max_iter + 1):
        # Armijo backtracking line search
        t = 1.0
        g_dot_d = float(g @ d)
        
        # Ensure descent direction
        if g_dot_d >= 0:
            # Reset to steepest descent
            d = -g
            g_dot_d = float(g @ d)
        
        while True:
            a_try = a + t * d
            E_try = energy_paper(D, x, a_try, lam, sigma)
            
            # Armijo condition
            if E_try <= E_prev + ls_c * t * g_dot_d:
                break
                
            t *= ls_beta
            if t < 1e-12:
                # Fallback: tiny step in gradient direction
                a_try = a - 1e-6 * g
                E_try = energy_paper(D, x, a_try, lam, sigma)
                break
        
        a_next = a_try
        g_next = grad_paper(D, x, a_next, lam, sigma)
        
        # Check convergence (1% relative energy change as in paper)
        rel_change = abs(E_prev - E_try) / max(1.0, abs(E_prev))
        if rel_change <= rel_tol:
            return a_next, {
                "iters": it, 
                "obj": E_try, 
                "converged": True,
                "rel_change": rel_change,
                "grad_norm": np.linalg.norm(g_next)
            }
        
        # Polak-Ribière conjugate gradient update
        y = g_next - g
        beta_pr = float(max(0.0, (g_next @ y) / (g @ g + 1e-18)))
        d_next = -g_next + beta_pr * d
        
        # Update for next iteration
        a, g, d, E_prev = a_next, g_next, d_next, E_try
    
    return a, {
        "iters": max_iter, 
        "obj": E_prev, 
        "converged": False,
        "rel_change": rel_change,
        "grad_norm": np.linalg.norm(g)
    }


def estimate_sigma(X):
    """
    Estimate scale parameter σ from data distribution.
    
    Args:
        X: Data matrix (p, N) or vector (p,)
        
    Returns:
        sigma: Estimated scale parameter
    """
    return float(np.std(X))


def lambda_from_sigma(sigma, ratio=0.14):
    """
    Compute λ from σ using ratio from Olshausen & Field.
    
    Paper uses λ/σ ≈ 0.14 for natural images.
    
    Args:
        sigma: Scale parameter
        ratio: λ/σ ratio (default 0.14 from paper)
        
    Returns:
        lambda: Sparsity parameter
    """
    return ratio * sigma