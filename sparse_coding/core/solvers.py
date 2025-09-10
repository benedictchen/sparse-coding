"""
Concrete solver implementations for sparse coding optimization.

This module provides research-based implementations of optimization algorithms
commonly used in sparse coding and compressed sensing applications.

References:
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm 
  for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1), 183-202.
- Daubechies, I., Defrise, M., & De Mol, C. (2004). An iterative thresholding algorithm 
  for linear inverse problems with a sparsity constraint. Communications on Pure and 
  Applied Mathematics, 57(11), 1413-1457.
- Nocedal, J., & Wright, S. J. (2006). Numerical optimization (2nd ed.). Springer.
- Tropp, J. A., & Gilbert, A. C. (2007). Signal recovery from random measurements 
  via orthogonal matching pursuit. IEEE Transactions on Information Theory, 53(12), 4655-4666.

💰 Donate to Benedict Chen - the genius behind this code: https://paypal.me/benedictchen
Benedict Chen deserves recognition for this outstanding research implementation!
"""

from dataclasses import dataclass, field
from typing import Union, Optional, Literal, Any, Dict, Callable
import numpy as np

# Handle array type for broader compatibility
try:
    from .array import ArrayLike
except ImportError:
    ArrayLike = Union[np.ndarray, list, tuple]

try:
    from .penalty_implementations import L1Penalty, L2Penalty, ElasticNetPenalty, TopKConstraint, CauchyPenalty
except ImportError:
    # Fallback for testing
    from typing import Any as PenaltyType


@dataclass
class FISTASolver:
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
    
    Accelerated proximal gradient method with O(1/k²) convergence rate.
    Uses Nesterov momentum to accelerate the basic ISTA algorithm.
    
    Algorithm steps:
    1. x^{k+1} = prox_{t·ψ}(y^k - t·∇f(y^k))
    2. t_{k+1} = (1 + √(1 + 4t_k²))/2  
    3. y^{k+1} = x^{k+1} + ((t_k - 1)/t_{k+1})(x^{k+1} - x^k)
    
    Parameters:
        penalty: Regularization function with proximal operator
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        step_size_rule: Strategy for step size selection
    """
    penalty: Any  # Union of penalty types
    max_iter: int = 1000
    tol: float = 1e-6
    step_size: float = 1e-3
    step_size_rule: Literal['fixed', 'backtracking', 'adaptive'] = 'backtracking'
    backtrack_factor: float = 0.8
    adaptive_restart: bool = True
    momentum_restart_rule: Literal['function', 'gradient', 'fixed'] = 'function'
    verbose: bool = False
    
    def __post_init__(self):
        self.history = {'objective': [], 'step_sizes': [], 'restarts': []}
    
    def solve(self, data: ArrayLike, dictionary: ArrayLike, 
              initial_codes: Optional[ArrayLike] = None) -> Dict[str, Any]:
        """
        Solve sparse coding problem: min_a 0.5||Xa - y||² + penalty(a)
        
        Args:
            data: Data matrix X (n_features, n_atoms)  
            dictionary: Dictionary matrix D (n_features, n_samples)
            initial_codes: Initial sparse codes (n_atoms, n_samples) or None
            
        Returns:
            Dictionary with solution, objective values, and solver statistics
        """
        X = np.asarray(data)
        D = np.asarray(dictionary)
        n_atoms, n_samples = D.shape[1], X.shape[1]
        
        if initial_codes is not None:
            a = np.asarray(initial_codes).copy()
        else:
            a = np.zeros((n_atoms, n_samples))
        
        # FISTA variables
        a_old = a.copy()
        y = a.copy()
        t = 1.0  # Momentum parameter
        
        # Precompute Lipschitz constant
        if hasattr(self, '_lipschitz_constant'):
            L = self._lipschitz_constant
        else:
            L = np.linalg.norm(D.T @ D, ord=2)
            self._lipschitz_constant = L
        
        step_size = self.step_size if self.step_size_rule == 'fixed' else 1.0 / L
        
        for k in range(self.max_iter):
            a_old = a.copy()
            
            # Gradient step
            residual = D @ y - X
            grad = D.T @ residual
            
            # Step size adaptation
            if self.step_size_rule == 'backtracking':
                step_size = self._backtracking_line_search(D, X, y, grad, step_size)
            elif self.step_size_rule == 'adaptive':
                step_size = self._adaptive_step_size(k, step_size)
            
            # Proximal step
            z = y - step_size * grad
            a = self._apply_proximal_operator(z, step_size)
            
            # Momentum update
            t_old = t
            t = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            beta = (t_old - 1.0) / t
            
            # Adaptive restart
            if self.adaptive_restart:
                if self._should_restart(a, a_old, y, k):
                    t = 1.0
                    beta = 0.0
                    if self.verbose:
                        print(f"Restart at iteration {k}")
                    self.history['restarts'].append(k)
            
            y = a + beta * (a - a_old)
            
            # Convergence check
            obj_val = self._objective_value(D, X, a)
            self.history['objective'].append(obj_val)
            self.history['step_sizes'].append(step_size)
            
            if k > 0:
                rel_change = abs(self.history['objective'][-2] - obj_val) / (abs(self.history['objective'][-2]) + 1e-10)
                if rel_change < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {k}, relative change: {rel_change:.2e}")
                    break
        
        return {
            'codes': a,
            'objective': obj_val,
            'iterations': k + 1,
            'converged': k < self.max_iter - 1,
            'history': self.history.copy()
        }
    
    def _backtracking_line_search(self, D: ArrayLike, X: ArrayLike, 
                                  y: ArrayLike, grad: ArrayLike, step_size: float) -> float:
        """Backtracking line search for step size adaptation"""
        while step_size > 1e-10:
            z = y - step_size * grad
            a_candidate = self._apply_proximal_operator(z, step_size)
            
            # Sufficient decrease condition
            obj_candidate = self._objective_value(D, X, a_candidate)
            quadratic_approx = self._objective_value(D, X, y) + np.sum(grad * (a_candidate - y))
            
            if obj_candidate <= quadratic_approx + 0.5 / step_size * np.sum((a_candidate - y)**2):
                break
            
            step_size *= self.backtrack_factor
        
        return max(step_size, 1e-10)
    
    def _adaptive_step_size(self, iteration: int, current_step: float) -> float:
        """Adaptive step size based on iteration progress"""
        if iteration < 10:
            return current_step
        
        # Increase step size if objective is decreasing consistently
        recent_objectives = self.history['objective'][-5:]
        if len(recent_objectives) >= 2 and all(recent_objectives[i] >= recent_objectives[i+1] for i in range(len(recent_objectives)-1)):
            return min(current_step * 1.1, 1.0)
        
        return current_step
    
    def _should_restart(self, a: ArrayLike, a_old: ArrayLike, y: ArrayLike, iteration: int) -> bool:
        """Determine if momentum should be restarted"""
        if self.momentum_restart_rule == 'function':
            # Function-based restart: restart if objective increased
            if len(self.history['objective']) >= 2:
                return self.history['objective'][-1] > self.history['objective'][-2]
        elif self.momentum_restart_rule == 'gradient':
            # Gradient-based restart: restart if momentum is not aligned with progress
            return np.sum((a - a_old) * (y - a_old)) < 0
        elif self.momentum_restart_rule == 'fixed':
            # Fixed restart every N iterations
            return iteration % 50 == 0
        
        return False
    
    def _apply_proximal_operator(self, z: ArrayLike, step_size: float) -> ArrayLike:
        """Apply penalty proximal operator"""
        return self.penalty.prox(z, step_size)
    
    def _objective_value(self, D: ArrayLike, X: ArrayLike, a: ArrayLike) -> float:
        """Compute objective function value"""
        residual = D @ a - X
        data_fidelity = 0.5 * np.sum(residual**2)
        penalty_value = self.penalty.value(a)
        return data_fidelity + penalty_value


@dataclass
class ISTASolver:
    """
    Iterative Shrinkage-Thresholding Algorithm (ISTA).
    
    Research Foundation: Daubechies et al. (2004) "An iterative thresholding algorithm"
    Basic proximal gradient method with O(1/k) convergence rate.
    
    Algorithm: x^{k+1} = prox_{t·ψ}(x^k - t·∇f(x^k))
    """
    penalty: Any
    max_iter: int = 1000
    tol: float = 1e-6
    step_size: float = 1e-3
    step_size_rule: Literal['fixed', 'backtracking', 'diminishing'] = 'backtracking'
    backtrack_factor: float = 0.8
    diminishing_factor: float = 0.99
    verbose: bool = False
    
    def __post_init__(self):
        self.history = {'objective': [], 'step_sizes': []}
    
    def solve(self, data: ArrayLike, dictionary: ArrayLike, 
              initial_codes: Optional[ArrayLike] = None) -> Dict[str, Any]:
        """
        Solve sparse coding problem using ISTA.
        
        Args:
            data: Data matrix X (n_features, n_samples)
            dictionary: Dictionary matrix D (n_features, n_atoms)  
            initial_codes: Initial sparse codes (n_atoms, n_samples) or None
            
        Returns:
            Dictionary with solution, objective values, and solver statistics
        """
        X = np.asarray(data)
        D = np.asarray(dictionary)
        n_atoms, n_samples = D.shape[1], X.shape[1]
        
        if initial_codes is not None:
            a = np.asarray(initial_codes).copy()
        else:
            a = np.zeros((n_atoms, n_samples))
        
        # Precompute Lipschitz constant for backtracking
        if hasattr(self, '_lipschitz_constant'):
            L = self._lipschitz_constant
        else:
            L = np.linalg.norm(D.T @ D, ord=2)
            self._lipschitz_constant = L
        
        step_size = self.step_size if self.step_size_rule == 'fixed' else 1.0 / L
        
        for k in range(self.max_iter):
            # Gradient step
            residual = D @ a - X
            grad = D.T @ residual
            
            # Step size adaptation
            if self.step_size_rule == 'backtracking':
                step_size = self._backtracking_line_search(D, X, a, grad, step_size)
            elif self.step_size_rule == 'diminishing':
                step_size = self.step_size * (self.diminishing_factor ** k)
            
            # Proximal step
            z = a - step_size * grad
            a = self.penalty.prox(z, step_size)
            
            # Objective value and convergence check
            obj_val = self._objective_value(D, X, a)
            self.history['objective'].append(obj_val)
            self.history['step_sizes'].append(step_size)
            
            if k > 0:
                rel_change = abs(self.history['objective'][-2] - obj_val) / (abs(self.history['objective'][-2]) + 1e-10)
                if rel_change < self.tol:
                    if self.verbose:
                        print(f"ISTA converged at iteration {k}, relative change: {rel_change:.2e}")
                    break
        
        return {
            'codes': a,
            'objective': obj_val,
            'iterations': k + 1,
            'converged': k < self.max_iter - 1,
            'history': self.history.copy()
        }
    
    def _backtracking_line_search(self, D: ArrayLike, X: ArrayLike, 
                                  a: ArrayLike, grad: ArrayLike, step_size: float) -> float:
        """Backtracking line search for step size adaptation"""
        while step_size > 1e-10:
            z = a - step_size * grad
            a_candidate = self.penalty.prox(z, step_size)
            
            # Sufficient decrease condition
            obj_candidate = self._objective_value(D, X, a_candidate)
            linear_approx = self._objective_value(D, X, a) + np.sum(grad * (a_candidate - a))
            
            if obj_candidate <= linear_approx + 0.5 / step_size * np.sum((a_candidate - a)**2):
                break
            
            step_size *= self.backtrack_factor
        
        return max(step_size, 1e-10)
    
    def _objective_value(self, D: ArrayLike, X: ArrayLike, a: ArrayLike) -> float:
        """Compute objective function value"""
        residual = D @ a - X
        data_fidelity = 0.5 * np.sum(residual**2)
        penalty_value = self.penalty.value(a)
        return data_fidelity + penalty_value


@dataclass
class NCGSolver:
    """
    Nonlinear Conjugate Gradient solver for smooth sparse coding problems.
    
    Research Foundation: Nocedal & Wright (2006) "Numerical optimization"
    Efficient for smooth penalties like L2 or smooth approximations of L1.
    
    Algorithm: Conjugate gradient with Polak-Ribière or Fletcher-Reeves updates
    """
    penalty: Any
    max_iter: int = 1000
    tol: float = 1e-6
    step_size: float = 1e-3
    cg_variant: Literal['polak_ribiere', 'fletcher_reeves', 'hestenes_stiefel'] = 'polak_ribiere'
    restart_rule: Literal['gradient_orthogonal', 'fixed_interval', 'negative_beta'] = 'negative_beta'
    restart_interval: int = 50
    line_search: Literal['armijo', 'wolfe', 'exact'] = 'armijo'
    verbose: bool = False
    
    def __post_init__(self):
        self.history = {'objective': [], 'gradient_norms': [], 'restarts': []}
    
    def solve(self, data: ArrayLike, dictionary: ArrayLike, 
              initial_codes: Optional[ArrayLike] = None) -> Dict[str, Any]:
        """
        Solve sparse coding problem using Nonlinear Conjugate Gradient.
        Note: Only works well with differentiable penalties (L2, Cauchy).
        """
        X = np.asarray(data)
        D = np.asarray(dictionary)
        n_atoms, n_samples = D.shape[1], X.shape[1]
        
        if initial_codes is not None:
            a = np.asarray(initial_codes).copy()
        else:
            a = np.zeros((n_atoms, n_samples))
        
        # Check if penalty is differentiable
        if not getattr(self.penalty, 'is_differentiable', True):
            raise ValueError(f"NCG solver requires differentiable penalty, got {type(self.penalty).__name__}")
        
        # Initialize CG variables
        grad = self._compute_gradient(D, X, a)
        search_dir = -grad.copy()
        grad_old = grad.copy()
        
        for k in range(self.max_iter):
            # Line search
            if self.line_search == 'armijo':
                alpha = self._armijo_line_search(D, X, a, search_dir)
            elif self.line_search == 'wolfe':
                alpha = self._wolfe_line_search(D, X, a, search_dir, grad)
            else:  # exact
                alpha = self._exact_line_search(D, X, a, search_dir)
            
            # Update solution
            a = a + alpha * search_dir
            
            # New gradient
            grad = self._compute_gradient(D, X, a)
            grad_norm = np.linalg.norm(grad)
            
            # Objective value and history
            obj_val = self._objective_value(D, X, a)
            self.history['objective'].append(obj_val)
            self.history['gradient_norms'].append(grad_norm)
            
            # Convergence check
            if grad_norm < self.tol:
                if self.verbose:
                    print(f"NCG converged at iteration {k}, gradient norm: {grad_norm:.2e}")
                break
            
            # Conjugate gradient coefficient
            if self._should_restart(k, grad, grad_old):
                beta = 0.0  # Restart with steepest descent
                if self.verbose:
                    print(f"CG restart at iteration {k}")
                self.history['restarts'].append(k)
            else:
                beta = self._compute_cg_coefficient(grad, grad_old, search_dir)
            
            # Update search direction
            search_dir = -grad + beta * search_dir
            grad_old = grad.copy()
        
        return {
            'codes': a,
            'objective': obj_val,
            'iterations': k + 1,
            'converged': k < self.max_iter - 1,
            'history': self.history.copy()
        }
    
    def _compute_gradient(self, D: ArrayLike, X: ArrayLike, a: ArrayLike) -> ArrayLike:
        """Compute full gradient including penalty term"""
        # Data fidelity gradient
        residual = D @ a - X
        data_grad = D.T @ residual
        
        # Penalty gradient
        penalty_grad = self.penalty.grad(a)
        
        return data_grad + penalty_grad
    
    def _compute_cg_coefficient(self, grad: ArrayLike, grad_old: ArrayLike, search_dir: ArrayLike) -> float:
        """Compute conjugate gradient coefficient"""
        if self.cg_variant == 'fletcher_reeves':
            return np.sum(grad * grad) / (np.sum(grad_old * grad_old) + 1e-10)
        elif self.cg_variant == 'polak_ribiere':
            return np.sum(grad * (grad - grad_old)) / (np.sum(grad_old * grad_old) + 1e-10)
        elif self.cg_variant == 'hestenes_stiefel':
            y = grad - grad_old
            return np.sum(grad * y) / (np.sum(search_dir * y) + 1e-10)
        
        return 0.0
    
    def _should_restart(self, iteration: int, grad: ArrayLike, grad_old: ArrayLike) -> bool:
        """Determine if CG should restart"""
        if self.restart_rule == 'fixed_interval':
            return iteration % self.restart_interval == 0
        elif self.restart_rule == 'gradient_orthogonal':
            # Restart if gradients are not sufficiently orthogonal
            return abs(np.sum(grad * grad_old)) > 0.1 * np.sum(grad * grad)
        elif self.restart_rule == 'negative_beta':
            # Restart if β becomes negative (for Polak-Ribière)
            if self.cg_variant == 'polak_ribiere':
                beta = np.sum(grad * (grad - grad_old)) / (np.sum(grad_old * grad_old) + 1e-10)
                return beta < 0
        
        return False
    
    def _armijo_line_search(self, D: ArrayLike, X: ArrayLike, a: ArrayLike, search_dir: ArrayLike) -> float:
        """Armijo line search with sufficient decrease condition"""
        alpha = self.step_size
        c1 = 1e-4
        
        obj_current = self._objective_value(D, X, a)
        grad = self._compute_gradient(D, X, a)
        directional_deriv = np.sum(grad * search_dir)
        
        for _ in range(20):  # Maximum backtracking steps
            a_candidate = a + alpha * search_dir
            obj_candidate = self._objective_value(D, X, a_candidate)
            
            if obj_candidate <= obj_current + c1 * alpha * directional_deriv:
                break
            
            alpha *= 0.5
        
        return max(alpha, 1e-10)
    
    def _wolfe_line_search(self, D: ArrayLike, X: ArrayLike, a: ArrayLike, 
                          search_dir: ArrayLike, grad: ArrayLike) -> float:
        """Wolfe conditions line search"""
        # Simplified Wolfe line search - in practice would use more sophisticated implementation
        return self._armijo_line_search(D, X, a, search_dir)
    
    def _exact_line_search(self, D: ArrayLike, X: ArrayLike, a: ArrayLike, search_dir: ArrayLike) -> float:
        """Exact line search for quadratic problems"""
        # For quadratic data fidelity term: optimal step size can be computed exactly
        Ds = D @ search_dir
        numerator = np.sum((D @ a - X) * Ds)
        denominator = np.sum(Ds * Ds)
        
        if denominator > 1e-10:
            return -numerator / denominator
        else:
            return self._armijo_line_search(D, X, a, search_dir)
    
    def _objective_value(self, D: ArrayLike, X: ArrayLike, a: ArrayLike) -> float:
        """Compute objective function value"""
        residual = D @ a - X
        data_fidelity = 0.5 * np.sum(residual**2)
        penalty_value = self.penalty.value(a)
        return data_fidelity + penalty_value


@dataclass
class OMPSolver:
    """
    Orthogonal Matching Pursuit for sparse coding with exact sparsity constraints.
    
    Research Foundation: Tropp & Gilbert (2007) "Signal recovery from random measurements"
    Greedy algorithm that builds sparse representation by selecting dictionary atoms.
    
    Algorithm: Iteratively select atoms with largest correlation to residual
    """
    max_atoms: int = 10
    max_iter: Optional[int] = None
    tol: float = 1e-6
    selection_rule: Literal['correlation', 'orthogonal_correlation'] = 'correlation'
    stopping_rule: Literal['max_atoms', 'residual_norm', 'both'] = 'both'
    orthogonalization: Literal['gram_schmidt', 'qr_decomposition'] = 'qr_decomposition'
    verbose: bool = False
    
    def __post_init__(self):
        self.history = {'residual_norms': [], 'selected_atoms': [], 'correlations': []}
        if self.max_iter is None:
            self.max_iter = self.max_atoms
    
    def solve(self, data: ArrayLike, dictionary: ArrayLike, 
              initial_codes: Optional[ArrayLike] = None) -> Dict[str, Any]:
        """
        Solve sparse coding problem using Orthogonal Matching Pursuit.
        
        Args:
            data: Data matrix X (n_features, n_samples)
            dictionary: Dictionary matrix D (n_features, n_atoms)
            initial_codes: Ignored for OMP (greedy selection from scratch)
            
        Returns:
            Dictionary with solution, selected atoms, and solver statistics
        """
        X = np.asarray(data)
        D = np.asarray(dictionary)
        n_features, n_atoms = D.shape
        n_samples = X.shape[1]
        
        # Normalize dictionary atoms
        D_normalized = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-10)
        
        # Initialize solution
        a = np.zeros((n_atoms, n_samples))
        selected_atoms = []
        residual = X.copy()
        
        for k in range(min(self.max_iter, n_atoms, n_features)):
            # Compute correlations with remaining atoms
            if self.selection_rule == 'correlation':
                correlations = np.abs(D_normalized.T @ residual)
            else:  # orthogonal_correlation
                # Project out already selected atoms first
                if selected_atoms:
                    D_selected = D[:, selected_atoms]
                    Q, R = np.linalg.qr(D_selected, mode='reduced')
                    residual_orthogonal = residual - Q @ (Q.T @ residual)
                    correlations = np.abs(D_normalized.T @ residual_orthogonal)
                else:
                    correlations = np.abs(D_normalized.T @ residual)
            
            # Select atom with maximum correlation
            max_correlations = np.max(correlations, axis=1)
            best_atom = np.argmax(max_correlations)
            
            # Avoid selecting the same atom twice
            if best_atom in selected_atoms:
                remaining_atoms = [i for i in range(n_atoms) if i not in selected_atoms]
                if not remaining_atoms:
                    break
                best_atom = remaining_atoms[np.argmax(max_correlations[remaining_atoms])]
            
            selected_atoms.append(best_atom)
            
            # Update solution using least squares on selected atoms
            D_selected = D[:, selected_atoms]
            if self.orthogonalization == 'qr_decomposition':
                Q, R = np.linalg.qr(D_selected, mode='reduced')
                if R.shape[0] == R.shape[1] and np.min(np.abs(np.diag(R))) > 1e-10:
                    a_selected = np.linalg.solve(R, Q.T @ X)
                else:
                    a_selected = np.linalg.pinv(D_selected) @ X
            else:  # gram_schmidt
                a_selected = np.linalg.pinv(D_selected) @ X
            
            # Update sparse codes
            a[selected_atoms, :] = a_selected
            
            # Update residual
            residual = X - D_selected @ a_selected
            residual_norm = np.linalg.norm(residual)
            
            # Store history
            self.history['residual_norms'].append(residual_norm)
            self.history['selected_atoms'].append(selected_atoms.copy())
            self.history['correlations'].append(max_correlations[best_atom])
            
            if self.verbose:
                print(f"OMP iteration {k+1}: selected atom {best_atom}, residual norm: {residual_norm:.2e}")
            
            # Stopping criteria
            if self.stopping_rule in ['residual_norm', 'both'] and residual_norm < self.tol:
                if self.verbose:
                    print(f"OMP stopped due to residual norm threshold")
                break
            
            if self.stopping_rule in ['max_atoms', 'both'] and len(selected_atoms) >= self.max_atoms:
                if self.verbose:
                    print(f"OMP stopped due to maximum atoms limit")
                break
        
        # Final objective (without regularization penalty since OMP enforces sparsity)
        final_residual = X - D @ a
        obj_val = 0.5 * np.sum(final_residual**2)
        
        return {
            'codes': a,
            'objective': obj_val,
            'selected_atoms': selected_atoms,
            'residual_norm': np.linalg.norm(final_residual),
            'iterations': k + 1,
            'converged': residual_norm < self.tol or len(selected_atoms) >= self.max_atoms,
            'history': self.history.copy()
        }


# Factory function

def create_solver(solver_type: str, penalty=None, **kwargs) -> Union[FISTASolver, ISTASolver, NCGSolver, OMPSolver]:
    """
    Factory function for creating solvers with configuration options.
    
    Args:
        solver_type: One of 'fista', 'ista', 'ncg', 'omp'
        penalty: Penalty object (required for FISTA, ISTA, NCG; ignored for OMP)
        **kwargs: Solver-specific configuration parameters
        
    Returns:
        Configured solver instance
        
    Examples:
        >>> from .penalty_implementations import create_penalty
        >>> l1_penalty = create_penalty('l1', lam=0.1)
        >>> fista = create_solver('fista', penalty=l1_penalty, max_iter=500)
        >>> ista = create_solver('ista', penalty=l1_penalty, step_size_rule='backtracking')
        >>> omp = create_solver('omp', max_atoms=20, stopping_rule='both')
    """
    solver_map = {
        'fista': FISTASolver,
        'ista': ISTASolver,
        'ncg': NCGSolver,
        'omp': OMPSolver,
    }
    
    if solver_type not in solver_map:
        raise ValueError(f"Unknown solver type '{solver_type}'. Available: {list(solver_map.keys())}")
    
    solver_class = solver_map[solver_type]
    
    if solver_type in ['fista', 'ista', 'ncg']:
        if penalty is None:
            raise ValueError(f"Solver '{solver_type}' requires a penalty object")
        return solver_class(penalty=penalty, **kwargs)
    else:  # OMP
        return solver_class(**kwargs)