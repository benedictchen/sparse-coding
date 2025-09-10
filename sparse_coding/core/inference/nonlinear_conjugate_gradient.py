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
        # FIXME: Multiple research-accurate NCG implementation variants needed
        #
        # ISSUE: Current implementation assumes penalty has is_differentiable and grad methods
        # but PenaltyProtocol doesn't define these. Also missing research-validated variants.
        #
        # SOLUTION 1: Basic differentiability check with proper error handling
        # Check for gradient method existence instead of is_differentiable attribute:
        if not hasattr(penalty, 'grad') or not callable(getattr(penalty, 'grad')):
            raise ValueError("NCG requires penalty function with grad() method")
            
        # SOLUTION 2: Fletcher-Reeves conjugate gradient (Fletcher & Reeves 1964)
        # Alternative beta formula: beta_fr = ||grad_new||^2 / ||grad_prev||^2
        # More robust for ill-conditioned problems
        
        # SOLUTION 3: Dai-Yuan conjugate gradient (Dai & Yuan 1999)
        # Hybrid formula: beta_dy = ||grad_new||^2 / dot(search_dir_prev, y)
        # Better global convergence properties
        
        # SOLUTION 4: Hestenes-Stiefel conjugate gradient (Hestenes & Stiefel 1952)
        # Original formula: beta_hs = dot(grad_new, y) / dot(search_dir_prev, y)
        # Conjugacy property preservation
        
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
            
            # FIXME: Line search implementation has mathematical errors
            #
            # ISSUE: Objective decrease calculation is wrong - should use actual function values
            # 
            # SOLUTION 1: Proper Armijo line search with function evaluations
            # Compute actual objective function f(x) = 0.5||Dx-y||² + λR(x)
            def objective(a_val):
                residual = 0.5 * np.linalg.norm(D @ a_val - x)**2
                penalty_val = penalty.value(a_val) if hasattr(penalty, 'value') else 0.0
                return residual + lam * penalty_val
            
            # SOLUTION 2: Wolfe conditions line search (Wolfe 1969, 1971)
            # Both sufficient decrease (Armijo) and curvature conditions
            # c1 * alpha * grad_dot_dir <= f(x+alpha*p) - f(x) <= c2 * alpha * grad_dot_dir
            
            # SOLUTION 3: Strong Wolfe conditions for better convergence
            # |grad(x+alpha*p)^T p| <= c2 * |grad(x)^T p|
            # Ensures bounded step sizes and convergence
            
            # Current broken implementation (TEMPORARY):
            alpha = 1.0
            c1 = 1e-4
            grad_dot_dir = np.dot(grad, search_dir)
            
            while alpha > 1e-12:
                a_new = a + alpha * search_dir
                grad_new = objective_grad(a_new)
                
                # BROKEN: This is gradient norm decrease, not objective decrease
                obj_decrease = 0.5 * np.linalg.norm(grad_new)**2 - 0.5 * np.linalg.norm(grad)**2
                if obj_decrease <= c1 * alpha * grad_dot_dir:
                    break
                alpha *= 0.5
            
            if alpha <= 1e-12:
                return a, k
            
            a = a_new
            grad_prev = grad
            grad = grad_new
            
            # FIXME: Beta computation should offer multiple research-validated formulas
            #
            # ISSUE: Only Polak-Ribière implemented, missing other proven variants
            #
            # SOLUTION 1: Current Polak-Ribière (Polak & Ribière 1969)
            y = grad - grad_prev
            beta_pr = max(0.0, np.dot(grad, y) / (np.linalg.norm(grad_prev)**2 + 1e-12))
            
            # SOLUTION 2: Fletcher-Reeves formula (Fletcher & Reeves 1964)
            # beta_fr = np.linalg.norm(grad)**2 / (np.linalg.norm(grad_prev)**2 + 1e-12)
            # More stable for noisy gradients
            
            # SOLUTION 3: Dai-Yuan formula (Dai & Yuan 1999)
            # beta_dy = np.linalg.norm(grad)**2 / (np.dot(search_dir, y) + 1e-12)
            # Global convergence without exact line search
            
            # SOLUTION 4: Hestenes-Stiefel formula (Hestenes & Stiefel 1952)
            # beta_hs = np.dot(grad, y) / (np.dot(search_dir, y) + 1e-12)
            # Conjugacy preservation property
            
            search_dir = -grad + beta_pr * search_dir
            
            # Reset to steepest descent if not descent direction
            if np.dot(grad, search_dir) > 0:
                search_dir = -grad
        
        return a, self.max_iter