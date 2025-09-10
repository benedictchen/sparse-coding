"""
FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) solver.

Implements Beck & Teboulle (2009) accelerated proximal gradient method for 
sparse coding inference with O(1/k²) convergence rate.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from ..penalties.penalty_protocol import PenaltyProtocol


class FISTASolver:
    """Fast Iterative Shrinkage-Thresholding Algorithm.
    
    Reference: Beck & Teboulle (2009). A fast iterative shrinkage-thresholding 
    algorithm for linear inverse problems.
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
    
    def solve(self, 
              D: np.ndarray, 
              x: np.ndarray, 
              penalty: PenaltyProtocol,
              lam: float,
              L: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Solve sparse coding problem using FISTA.
        
        Args:
            D: Dictionary matrix (n_features, n_atoms)
            x: Signal vector (n_features,)
            penalty: Penalty function implementing PenaltyProtocol
            lam: Regularization parameter
            L: Lipschitz constant (computed if None)
            
        Returns:
            Sparse codes and iteration count
        """
        if L is None:
            L = float(np.linalg.norm(D.T @ D, ord=2))
        
        step_size = 1.0 / L
        n_atoms = D.shape[1]
        
        a_prev = np.zeros(n_atoms)
        y = a_prev.copy()
        t_prev = 1.0
        
        DTx = D.T @ x
        DTD = D.T @ D
        
        for k in range(self.max_iter):
            grad = DTD @ y - DTx
            z = y - step_size * grad
            a_new = penalty.prox(z, lam * step_size)
            
            if np.linalg.norm(a_new - a_prev) < self.tol:
                return a_new, k + 1
            
            t_new = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
            beta = (t_prev - 1) / t_new
            y = a_new + beta * (a_new - a_prev)
            
            a_prev = a_new
            t_prev = t_new
        
        return a_new, self.max_iter