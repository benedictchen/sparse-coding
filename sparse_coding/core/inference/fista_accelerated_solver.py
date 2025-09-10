"""
FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) solver.

Implements Beck & Teboulle (2009) accelerated proximal gradient method for 
sparse coding inference with O(1/k²) convergence rate.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Literal
from ..penalties.penalty_protocol import PenaltyProtocol


class FISTASolver:
    """Beck & Teboulle (2009) Fast Iterative Shrinkage-Thresholding Algorithm.
    
    Implements all research variants from "A fast iterative shrinkage-thresholding 
    algorithm for linear inverse problems" with configurable acceleration strategies.
    """
    
    def __init__(self, 
                 max_iter: int = 1000, 
                 tol: float = 1e-6,
                 variant: Literal['standard', 'backtracking', 'monotone'] = 'standard',
                 backtrack_factor: float = 0.5,
                 backtrack_c: float = 1e-4,
                 restart_threshold: float = 0.0):
        self.max_iter = max_iter
        self.tol = tol
        self.variant = variant
        self.backtrack_factor = backtrack_factor
        self.backtrack_c = backtrack_c
        self.restart_threshold = restart_threshold
    
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
            if self.variant == 'standard':
                grad = DTD @ y - DTx
                z = y - step_size * grad
                a_new = penalty.prox(z, lam * step_size)
                
            elif self.variant == 'backtracking':
                grad = DTD @ y - DTx
                L_trial = L
                while L_trial < 1e12:
                    step_trial = 1.0 / L_trial
                    z = y - step_trial * grad
                    a_trial = penalty.prox(z, lam * step_trial)
                    
                    f_trial = 0.5 * np.linalg.norm(D @ a_trial - x)**2
                    Q_L = 0.5 * np.linalg.norm(D @ y - x)**2 + np.dot(grad, a_trial - y) + \
                          (L_trial / 2) * np.linalg.norm(a_trial - y)**2
                    
                    if f_trial <= Q_L:
                        a_new = a_trial
                        step_size = step_trial
                        break
                    L_trial *= 2
                else:
                    a_new = penalty.prox(y - step_size * grad, lam * step_size)
                    
            elif self.variant == 'monotone':
                grad = DTD @ y - DTx
                z = y - step_size * grad
                a_new = penalty.prox(z, lam * step_size)
                
                f_new = 0.5 * np.linalg.norm(D @ a_new - x)**2
                if hasattr(penalty, 'value'):
                    f_new += lam * penalty.value(a_new)
                    
                if k > 0:
                    f_prev = 0.5 * np.linalg.norm(D @ a_prev - x)**2
                    if hasattr(penalty, 'value'):
                        f_prev += lam * penalty.value(a_prev)
                    
                    if f_new > f_prev + self.restart_threshold:
                        t_prev = 1.0
                        y = a_prev
                        continue
            
            if np.linalg.norm(a_new - a_prev) < self.tol:
                return a_new, k + 1
            
            t_new = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
            beta = (t_prev - 1) / t_new
            y = a_new + beta * (a_new - a_prev)
            
            a_prev = a_new
            t_prev = t_new
        
        return a_new, self.max_iter