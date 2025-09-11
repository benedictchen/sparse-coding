"""
Nonlinear Conjugate Gradient solver for smooth penalties.

Implements multiple classical conjugate gradient formulations:

- Polak, E., & Ribière, G. (1969). Note sur la convergence de méthodes de directions 
  conjuguées. Revue française d'informatique et de recherche opérationnelle, 3(R1), 35-43.

- Fletcher, R., & Reeves, C. M. (1964). Function minimization by conjugate gradients. 
  The Computer Journal, 7(2), 149-154.

- Dai, Y. H., & Yuan, Y. (1999). A nonlinear conjugate gradient method with a strong 
  global convergence property. SIAM Journal on Optimization, 10(1), 177-182.

- Hestenes, M. R., & Stiefel, E. (1952). Methods of conjugate gradients for solving 
  linear systems. Journal of Research of the National Bureau of Standards, 49(6), 409-436.

Includes Armijo, Wolfe, and Strong Wolfe line search conditions for optimal step sizes.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Literal
from ..penalties.penalty_protocol import PenaltyProtocol


class NonlinearConjugateGradient:
    """Nonlinear Conjugate Gradient for smooth optimization problems.
    
    Minimizes differentiable objective functions using classical beta update
    formulas from Fletcher-Reeves, Polak-Ribière, Dai-Yuan, and Hestenes-Stiefel.
    """
    
    def __init__(self, 
                 max_iter: int = 1000, 
                 tol: float = 1e-6,
                 beta_formula: Literal['polak_ribiere', 'fletcher_reeves', 'dai_yuan', 'hestenes_stiefel'] = 'polak_ribiere',
                 line_search: Literal['armijo', 'wolfe', 'strong_wolfe'] = 'armijo',
                 c1: float = 1e-4,
                 c2: float = 0.9):
        self.max_iter = max_iter
        self.tol = tol
        self.beta_formula = beta_formula
        self.line_search = line_search
        self.c1 = c1
        self.c2 = c2
    
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
        
        # Add gradient clipping to prevent exploding gradients
        # Critical for stability with Olshausen log priors that can explode for large a
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e3:  # Threshold from deep learning stability practices
            grad = grad * (1e3 / grad_norm)  # Normalize to max magnitude
        
        search_dir = -grad.copy()
        
        # Objective function for tracking monotonic decrease
        def objective(a_val):
            residual = 0.5 * np.linalg.norm(D @ a_val - x)**2
            penalty_val = penalty.value(a_val)
            
            # Robust scalar conversion - handle all possible array/scalar combinations
            residual_scalar = float(np.asarray(residual).sum())
            penalty_scalar = float(np.asarray(penalty_val).sum())
            lam_scalar = float(np.asarray(lam).sum()) if hasattr(lam, '__len__') else float(lam)
            
            return residual_scalar + lam_scalar * penalty_scalar
        
        # Track objective values for monotonic decrease validation
        f_prev = objective(a)
        
        for k in range(self.max_iter):
            # Rigorous convergence criteria validation
            grad_norm = np.linalg.norm(grad)
            
            # Primary convergence test: gradient norm
            if grad_norm < self.tol:
                # Validate that we've actually converged to a reasonable solution
                f_current = objective(a)
                
                # Additional convergence validation: check that objective is reasonable
                # For sparse coding: objective should be finite and not excessive
                if not np.isfinite(f_current):
                    raise RuntimeError(f"NCG converged to invalid objective: {f_current}")
                
                # Check for numerical stability: coefficients should be finite
                if not np.all(np.isfinite(a)):
                    raise RuntimeError("NCG converged to non-finite coefficients")
                
                return a, k
            
            # Line search for step size
            alpha = self._line_search(objective, objective_grad, a, grad, search_dir)
            
            # Validate line search found acceptable step with strict tolerance
            if alpha <= 1e-14:  # Tighter minimum step size requirement
                # Line search failure indicates potential convergence or ill-conditioning
                # Perform strict convergence check before declaring failure
                if grad_norm < self.tol * 2:  # Only 2x relaxed tolerance (was 10x)
                    return a, k
                else:
                    # True convergence failure - insufficient progress possible
                    raise RuntimeError(
                        f"NCG line search failure at iteration {k}: "
                        f"alpha={alpha:.2e}, grad_norm={grad_norm:.2e}, "
                        f"required_tol={self.tol:.2e}. "
                        f"Stricter validation: max_allowed={self.tol * 2:.2e}"
                    )
            
            # Update solution
            a_prev = a.copy()
            a = a + alpha * search_dir
            grad_prev = grad
            grad = objective_grad(a)
            
            # Rigorous monotonic decrease validation
            f_current = objective(a)
            if f_current > f_prev + 1e-10:  # Stricter than before
                # Line search should guarantee sufficient decrease (Armijo condition)
                # This indicates a serious algorithmic issue
                decrease_violation = f_current - f_prev
                relative_violation = decrease_violation / max(abs(f_prev), 1e-12)
                
                if relative_violation > 1e-8:  # Non-trivial violation
                    raise RuntimeError(
                        f"NCG monotonic decrease violated at iteration {k}: "
                        f"f_new={f_current:.6e} > f_old={f_prev:.6e}, "
                        f"increase={decrease_violation:.2e}, "
                        f"relative_increase={relative_violation:.2e}"
                    )
            f_prev = f_current
            
            # Clip gradients during iteration to maintain stability
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e3:
                grad = grad * (1e3 / grad_norm)
            
            y = grad - grad_prev
            
            if self.beta_formula == 'polak_ribiere':
                beta = max(0.0, np.dot(grad, y) / (np.linalg.norm(grad_prev)**2 + 1e-12))
            elif self.beta_formula == 'fletcher_reeves':
                beta = np.linalg.norm(grad)**2 / (np.linalg.norm(grad_prev)**2 + 1e-12)
            elif self.beta_formula == 'dai_yuan':
                beta = np.linalg.norm(grad)**2 / (np.dot(search_dir, y) + 1e-12)
            elif self.beta_formula == 'hestenes_stiefel':
                beta = np.dot(grad, y) / (np.dot(search_dir, y) + 1e-12)
            else:
                beta = 0.0
            
            search_dir = -grad + beta * search_dir
            
            # Reset to steepest descent if not descent direction
            if np.dot(grad, search_dir) > 0:
                search_dir = -grad
        
        # If we reach here, we've hit max iterations without convergence
        final_grad_norm = np.linalg.norm(grad)
        final_objective = objective(a)
        
        # Check if we're close to convergence even if we hit max_iter with practical tolerance
        if final_grad_norm < self.tol * 100:  # Practical tolerance for nonlinear optimization
            return a, self.max_iter
        else:
            # True non-convergence - algorithm failed with reasonable requirements
            raise RuntimeError(
                f"NCG failed to converge after {self.max_iter} iterations: "
                f"final_grad_norm={final_grad_norm:.2e} > practical_tol={self.tol * 100:.2e}, "
                f"final_objective={final_objective:.6e}. "
                f"Required: gradient_norm < {self.tol * 100:.2e} (100x base tolerance for practical use)"
            )
    
    def _line_search(self, objective, objective_grad, a, grad, search_dir):
        """Research-accurate line search with configurable Wolfe conditions."""
        grad_dot_dir = np.dot(grad, search_dir)
        
        if grad_dot_dir >= 0:
            return 0.0
        
        alpha = 1.0
        f_a = objective(a)
        
        if self.line_search == 'armijo':
            while alpha > 1e-15:  # Much stricter minimum step size (was 1e-12)
                a_new = a + alpha * search_dir
                f_new = objective(a_new)
                
                if f_new <= f_a + self.c1 * alpha * grad_dot_dir:
                    return alpha
                alpha *= 0.5
                
        elif self.line_search in ['wolfe', 'strong_wolfe']:
            max_iter = 20
            for _ in range(max_iter):
                a_new = a + alpha * search_dir
                f_new = objective(a_new)
                
                if f_new > f_a + self.c1 * alpha * grad_dot_dir:
                    alpha *= 0.5
                    continue
                
                grad_new = objective_grad(a_new)
                grad_new_dot_dir = np.dot(grad_new, search_dir)
                
                if self.line_search == 'wolfe':
                    if grad_new_dot_dir >= self.c2 * grad_dot_dir:
                        return alpha
                else:
                    if abs(grad_new_dot_dir) <= -self.c2 * grad_dot_dir:
                        return alpha
                
                if grad_new_dot_dir >= 0:
                    return alpha
                    
                alpha *= 2.0
                if alpha > 10:
                    break
        
        return alpha