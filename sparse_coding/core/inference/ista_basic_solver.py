"""
ISTA (Iterative Shrinkage-Thresholding Algorithm) solver.

Implements Daubechies et al. (2004) proximal gradient method for sparse coding
inference with O(1/k) convergence rate.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Literal
from ..penalties.penalty_protocol import PenaltyProtocol


class ISTASolver:
    """Daubechies et al. (2004) Iterative Shrinkage-Thresholding Algorithm.
    
    Implements all research variants from "An iterative thresholding algorithm for 
    linear inverse problems with a sparsity constraint" and extensions.
    """
    
    def __init__(self, 
                 max_iter: int = 1000, 
                 tol: float = 1e-6,
                 variant: Literal['basic', 'backtracking', 'accelerated', 'monotone'] = 'basic',
                 backtrack_factor: float = 0.5,
                 restart_threshold: float = 0.0):
        self.max_iter = max_iter
        self.tol = tol
        self.variant = variant
        self.backtrack_factor = backtrack_factor
        self.restart_threshold = restart_threshold
    
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
        
        t = 1.0
        y = a.copy()
        
        for k in range(self.max_iter):
            a_prev = a.copy()
            
            if self.variant == 'basic':
                grad = DTD @ a - DTx
                z = a - step_size * grad
                a = penalty.prox(z, lam * step_size)
                
            elif self.variant == 'backtracking':
                grad = DTD @ a - DTx
                L_trial = L
                while L_trial < 1e12:
                    step_trial = 1.0 / L_trial
                    z = a - step_trial * grad
                    a_trial = penalty.prox(z, lam * step_trial)
                    
                    f_trial = 0.5 * np.linalg.norm(D @ a_trial - x)**2
                    Q_L = 0.5 * np.linalg.norm(D @ a - x)**2 + np.dot(grad, a_trial - a) + \
                          (L_trial / 2) * np.linalg.norm(a_trial - a)**2
                    
                    if f_trial <= Q_L:
                        a = a_trial
                        break
                    L_trial *= 2
                else:
                    a = penalty.prox(a - step_size * grad, lam * step_size)
                    
            elif self.variant == 'accelerated':
                grad = DTD @ y - DTx
                z = y - step_size * grad
                a = penalty.prox(z, lam * step_size)
                
                t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
                beta = (t - 1) / t_new
                y = a + beta * (a - a_prev)
                t = t_new
                
            elif self.variant == 'monotone':
                grad = DTD @ a - DTx
                z = a - step_size * grad
                a_new = penalty.prox(z, lam * step_size)
                
                objective_new = 0.5 * np.linalg.norm(D @ a_new - x)**2
                if hasattr(penalty, 'value'):
                    objective_new += lam * penalty.value(a_new)
                
                if k > 0:
                    objective_prev = 0.5 * np.linalg.norm(D @ a_prev - x)**2
                    if hasattr(penalty, 'value'):
                        objective_prev += lam * penalty.value(a_prev)
                    
                    if objective_new > objective_prev + self.restart_threshold:
                        a = a_prev
                        continue
                
                a = a_new
            
            if np.linalg.norm(a - a_prev) < self.tol:
                return a, k + 1
        
        return a, self.max_iter