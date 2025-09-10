"""
Nonlinear Conjugate Gradient solver for smooth penalties.

Implements Polak & Ribière (1969) conjugate gradient method for differentiable
penalty functions in sparse coding optimization.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple
from ..penalties.penalty_protocol import PenaltyProtocol


class NonlinearConjugateGradient:
    """Nonlinear Conjugate Gradient with Polak-Ribière formula.
    
    Reference: Polak & Ribière (1969). Note sur la convergence de méthodes de 
    directions conjuguées.
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
    
    def solve(self, 
              D: np.ndarray, 
              x: np.ndarray, 
              penalty: PenaltyProtocol,
              lam: float) -> Tuple[np.ndarray, int]:
        """Solve sparse coding problem using nonlinear CG.
        
        Args:
            D: Dictionary matrix (n_features, n_atoms)
            x: Signal vector (n_features,)
            penalty: Penalty function (must be differentiable)
            lam: Regularization parameter
            
        Returns:
            Sparse codes and iteration count
        """
        if not penalty.is_differentiable:
            raise ValueError("NCG requires differentiable penalty function")
        
        n_atoms = D.shape[1]
        a = np.zeros(n_atoms)
        
        DTx = D.T @ x
        DTD = D.T @ D
        
        def objective_grad(a_val):
            residual_grad = DTD @ a_val - DTx
            penalty_grad = penalty.grad(a_val)
            return residual_grad + lam * penalty_grad
        
        grad = objective_grad(a)
        search_dir = -grad.copy()
        
        for k in range(self.max_iter):
            if np.linalg.norm(grad) < self.tol:
                return a, k
            
            # Armijo backtracking line search
            alpha = 1.0
            c1 = 1e-4
            grad_dot_dir = np.dot(grad, search_dir)
            
            while alpha > 1e-12:
                a_new = a + alpha * search_dir
                grad_new = objective_grad(a_new)
                
                # Sufficient decrease condition
                obj_decrease = 0.5 * np.linalg.norm(grad_new)**2 - 0.5 * np.linalg.norm(grad)**2
                if obj_decrease <= c1 * alpha * grad_dot_dir:
                    break
                alpha *= 0.5
            
            if alpha <= 1e-12:
                return a, k
            
            a = a_new
            grad_prev = grad
            grad = grad_new
            
            # Polak-Ribière beta
            y = grad - grad_prev
            beta_pr = max(0.0, np.dot(grad, y) / (np.linalg.norm(grad_prev)**2 + 1e-12))
            
            search_dir = -grad + beta_pr * search_dir
            
            # Reset to steepest descent if not descent direction
            if np.dot(grad, search_dir) > 0:
                search_dir = -grad
        
        return a, self.max_iter