"""
ISTA (Iterative Shrinkage-Thresholding Algorithm) solver.

Implements Daubechies et al. (2004) proximal gradient method for sparse coding
inference with O(1/k) convergence rate.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from ..penalties.penalty_protocol import PenaltyProtocol


class ISTASolver:
    """Iterative Shrinkage-Thresholding Algorithm.
    
    Reference: Daubechies et al. (2004). An iterative thresholding algorithm for 
    linear inverse problems with a sparsity constraint.
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
        """Solve sparse coding problem using ISTA.
        
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
        
        a = np.zeros(n_atoms)
        DTx = D.T @ x
        DTD = D.T @ D
        
        for k in range(self.max_iter):
            a_prev = a.copy()
            grad = DTD @ a - DTx
            z = a - step_size * grad
            a = penalty.prox(z, lam * step_size)
            
            if np.linalg.norm(a - a_prev) < self.tol:
                return a, k + 1
        
        return a, self.max_iter