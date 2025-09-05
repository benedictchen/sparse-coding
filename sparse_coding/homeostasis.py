"""
Homeostatic gain control for coefficient variance equalization,
following Olshausen & Field (1996) paper methods.
"""

import numpy as np

def equalize_variance(A, eps=1e-8, target=1.0):
    """
    Compute homeostatic scaling to equalize coefficient variances.
    
    Args:
        A: Coefficient matrix (K, N) or (N, K)
        eps: Numerical stability epsilon  
        target: Target variance level
        
    Returns:
        g: Scaling vector of length K
    """
    # Infer dimensions: prefer K x N (atoms x samples)
    if A.shape[0] <= A.shape[1]:
        # Assume K x N (preferred layout)
        variances = np.var(A, axis=1) + eps
    else:
        # N x K layout
        variances = np.var(A, axis=0) + eps
    
    # Compute scaling to achieve target variance
    g = np.sqrt(target / variances)
    return g

def apply_gain(D, g):
    """
    Apply homeostatic gains to dictionary and compute renormalized gains.
    
    Dictionary atoms are scaled by gains and then renormalized to unit norm.
    Returns updated dictionary and the effective gains to apply to coefficients.
    
    Args:
        D: Dictionary (p, K) - atoms as columns
        g: Gains vector (K,)
        
    Returns:
        D_new: Updated dictionary with unit-norm atoms (p, K)  
        g_eff: Effective gains for coefficients (K,)
    """
    # Scale atoms by gains
    D_scaled = D * g[np.newaxis, :]
    
    # Renormalize atoms to unit norm
    norms = np.linalg.norm(D_scaled, axis=0) + 1e-12
    D_new = D_scaled / norms
    
    # Effective gains account for renormalization
    g_eff = g / norms
    
    return D_new, g_eff

def lambda_from_sigma(sigma, ratio=0.14):
    """
    Compute sparsity penalty from data statistics.
    
    Following paper: λ/σ = 0.14 (typical value)
    """
    return ratio * sigma