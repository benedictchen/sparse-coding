"""
Sparse Coding Inference Solvers

Implements research-based algorithms for solving:
argmin_a [1/2||X - D·a||² + penalty(a)]

All solvers follow the InferenceSolver protocol and implement algorithms
from the original research papers with proper citations.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Any, Dict
from .core.interfaces import InferenceSolver, Penalty
from .core.array import ArrayLike
from .fista_batch import power_iter_L


class FISTASolver:
    """Fast Iterative Shrinkage-Thresholding Algorithm.
    
    Reference: Beck & Teboulle (2009). A Fast Iterative Shrinkage-Thresholding Algorithm 
    for Linear Inverse Problems. SIAM Journal on Imaging Sciences.
    
    Accelerated proximal gradient method with optimal O(1/k²) convergence rate.
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6, 
                 backtrack: bool = True, verbose: bool = False):
        self.max_iter = max_iter
        self.tol = tol
        self.backtrack = backtrack
        self.verbose = verbose
    
    @property
    def name(self) -> str:
        return "fista"
    
    @property 
    def supports_batch(self) -> bool:
        return True
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """FISTA algorithm implementation."""
        D = np.asarray(D)
        X = np.asarray(X)
        
        # Ensure X is 2D (features, samples)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_atoms, n_samples = D.shape[1], X.shape[1]
        
        # Initialize variables
        A = np.zeros((n_atoms, n_samples))
        Z = A.copy()
        t = 1.0
        
        # Compute Lipschitz constant for backtracking
        if self.backtrack:
            L = 1.1 * power_iter_L(D)
        else:
            L = kwargs.get('lipschitz', 1.1 * power_iter_L(D))
        
        DtD = D.T @ D
        DtX = D.T @ X
        
        obj_prev = float('inf')
        
        for iteration in range(self.max_iter):
            A_old = A.copy()
            
            # Gradient step: ∇f(z) = D^T(Dz - X)
            grad = DtD @ Z - DtX
            z_grad = Z - grad / L
            
            # Proximal step: apply penalty
            A = penalty.prox(z_grad, 1.0 / L)
            
            # Momentum coefficient update (Nesterov acceleration)
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            
            # Momentum update
            Z = A + (t - 1) / t_new * (A - A_old)
            t = t_new
            
            # Convergence check
            if iteration % 10 == 0:  # Check every 10 iterations for efficiency
                reconstruction = D @ A
                data_fit = 0.5 * np.sum((X - reconstruction)**2)
                penalty_val = sum(penalty.value(A[:, i]) for i in range(n_samples))
                obj = data_fit + penalty_val
                
                if abs(obj - obj_prev) / max(abs(obj), 1e-8) < self.tol:
                    if self.verbose:
                        print(f"FISTA converged after {iteration} iterations")
                    break
                    
                obj_prev = obj
                
                if self.verbose and iteration % 100 == 0:
                    print(f"FISTA iter {iteration}: obj={obj:.6f}")
        
        return A


class ISTASolver:
    """Iterative Shrinkage-Thresholding Algorithm.
    
    Reference: Daubechies et al. (2004). An iterative thresholding algorithm for 
    linear inverse problems with a sparsity constraint.
    
    Basic proximal gradient method with O(1/k) convergence rate.
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6,
                 step_size: Optional[float] = None, verbose: bool = False):
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        self.verbose = verbose
    
    @property
    def name(self) -> str:
        return "ista"
    
    @property
    def supports_batch(self) -> bool:
        return True
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """ISTA algorithm implementation."""
        D = np.asarray(D)
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_atoms, n_samples = D.shape[1], X.shape[1]
        A = np.zeros((n_atoms, n_samples))
        
        # Step size (inverse of Lipschitz constant)
        if self.step_size is None:
            L = 1.1 * power_iter_L(D)
            step_size = 1.0 / L
        else:
            step_size = self.step_size
        
        DtD = D.T @ D
        DtX = D.T @ X
        obj_prev = float('inf')
        
        for iteration in range(self.max_iter):
            # Gradient step
            grad = DtD @ A - DtX
            z = A - step_size * grad
            
            # Proximal step
            A = penalty.prox(z, step_size)
            
            # Convergence check
            if iteration % 10 == 0:
                reconstruction = D @ A
                data_fit = 0.5 * np.sum((X - reconstruction)**2)
                penalty_val = sum(penalty.value(A[:, i]) for i in range(n_samples))
                obj = data_fit + penalty_val
                
                if abs(obj - obj_prev) / max(abs(obj), 1e-8) < self.tol:
                    if self.verbose:
                        print(f"ISTA converged after {iteration} iterations")
                    break
                    
                obj_prev = obj
                
                if self.verbose and iteration % 100 == 0:
                    print(f"ISTA iter {iteration}: obj={obj:.6f}")
        
        return A


class CoordinateDescentSolver:
    """Coordinate Descent for sparse coding.
    
    Reference: Friedman et al. (2007). Pathwise coordinate optimization.
    
    Cyclically optimizes each coefficient while fixing others. 
    Particularly efficient for L1 regularization.
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6,
                 positive: bool = False, verbose: bool = False):
        self.max_iter = max_iter
        self.tol = tol
        self.positive = positive
        self.verbose = verbose
    
    @property
    def name(self) -> str:
        return "coordinate_descent"
    
    @property
    def supports_batch(self) -> bool:
        return True
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """Coordinate descent implementation."""
        D = np.asarray(D)
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_atoms, n_samples = D.shape[1], X.shape[1]
        A = np.zeros((n_atoms, n_samples))
        
        # Precompute D^T @ D diagonal and D^T @ X
        DtD_diag = np.sum(D**2, axis=0)
        DtX = D.T @ X
        
        for iteration in range(self.max_iter):
            A_old = A.copy()
            
            # Cycle through coefficients
            for j in range(n_atoms):
                if DtD_diag[j] == 0:
                    continue
                
                # Compute residual excluding current coefficient
                other_atoms = np.arange(n_atoms) != j
                if np.any(other_atoms):
                    residual = DtX[j] - D[:, j].T @ D[:, other_atoms] @ A[other_atoms]
                else:
                    residual = DtX[j]
                
                # Update coefficient with soft thresholding (assumes L1 penalty)
                if hasattr(penalty, 'lam'):
                    threshold = penalty.lam / DtD_diag[j]
                    A[j] = np.sign(residual) * np.maximum(np.abs(residual) - threshold, 0.0)
                    
                    if self.positive:
                        A[j] = np.maximum(A[j], 0.0)
                else:
                    # General penalty - use single coefficient prox
                    A[j] = penalty.prox(residual / DtD_diag[j], 1.0 / DtD_diag[j])
            
            # Convergence check
            if np.linalg.norm(A - A_old) < self.tol:
                if self.verbose:
                    print(f"Coordinate descent converged after {iteration} iterations")
                break
        
        return A


class OrthogonalMatchingPursuit:
    """Orthogonal Matching Pursuit for sparse coding.
    
    Reference: Pati et al. (1993). Orthogonal matching pursuit: 
    Recursive function approximation with applications to wavelet decomposition.
    
    Greedy algorithm that iteratively selects dictionary atoms with highest correlation.
    """
    
    def __init__(self, n_nonzero_coefs: Optional[int] = None, 
                 tol: Optional[float] = None, verbose: bool = False):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.verbose = verbose
    
    @property
    def name(self) -> str:
        return "omp"
    
    @property
    def supports_batch(self) -> bool:
        return False  # Typically processes samples one by one
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """OMP algorithm implementation."""
        D = np.asarray(D)
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_atoms, n_samples = D.shape[1], X.shape[1]
        A = np.zeros((n_atoms, n_samples))
        
        # Default sparsity level
        if self.n_nonzero_coefs is None:
            sparsity = max(1, min(n_atoms // 4, 50))
        else:
            sparsity = self.n_nonzero_coefs
        
        # Normalize dictionary for correlation computation
        D_norm = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
        
        for sample in range(n_samples):
            x = X[:, sample]
            residual = x.copy()
            selected_atoms = []
            
            for _ in range(sparsity):
                # Compute correlations
                correlations = np.abs(D_norm.T @ residual)
                
                # Avoid already selected atoms
                if selected_atoms:
                    correlations[selected_atoms] = 0
                
                # Select atom with highest correlation
                best_atom = np.argmax(correlations)
                
                if correlations[best_atom] == 0:
                    break
                
                selected_atoms.append(best_atom)
                
                # Least squares solution on selected atoms
                D_selected = D[:, selected_atoms]
                try:
                    coeffs = np.linalg.lstsq(D_selected, x, rcond=None)[0]
                    A[selected_atoms, sample] = coeffs
                    residual = x - D_selected @ coeffs
                except np.linalg.LinAlgError:
                    # Singular matrix - use pseudoinverse
                    coeffs = D_selected.T @ x / (np.sum(D_selected**2, axis=0) + 1e-12)
                    A[selected_atoms, sample] = coeffs
                    residual = x - D_selected @ coeffs
                
                # Check tolerance
                if self.tol is not None and np.linalg.norm(residual) < self.tol:
                    break
        
        return A


class NonlinearConjugateGradient:
    """Nonlinear Conjugate Gradient for smooth penalties.
    
    Reference: Nocedal & Wright (2006). Numerical Optimization, Chapter 5.
    
    Efficient for smooth, differentiable penalty functions (L2, Cauchy, etc.).
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6,
                 beta_method: str = 'polak-ribiere', verbose: bool = False):
        self.max_iter = max_iter
        self.tol = tol
        self.beta_method = beta_method
        self.verbose = verbose
    
    @property
    def name(self) -> str:
        return "ncg"
    
    @property
    def supports_batch(self) -> bool:
        return True
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """NCG algorithm for smooth penalties."""
        if not penalty.is_differentiable:
            raise ValueError("NCG requires differentiable penalty")
        
        D = np.asarray(D)
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_atoms, n_samples = D.shape[1], X.shape[1]
        A = A_flat = np.zeros(n_atoms * n_samples)
        
        # Objective and gradient functions
        def objective(a_flat):
            a_mat = a_flat.reshape(n_atoms, n_samples)
            reconstruction = D @ a_mat
            data_fit = 0.5 * np.sum((X - reconstruction)**2)
            penalty_val = sum(penalty.value(a_mat[:, i]) for i in range(n_samples))
            return data_fit + penalty_val
        
        def gradient(a_flat):
            a_mat = a_flat.reshape(n_atoms, n_samples)
            
            # Data fitting gradient: D^T(Da - X)
            reconstruction = D @ a_mat
            data_grad = D.T @ (reconstruction - X)
            
            # Penalty gradient
            penalty_grad = np.column_stack([penalty.grad(a_mat[:, i]) 
                                          for i in range(n_samples)])
            
            return (data_grad + penalty_grad).flatten()
        
        # Initialize
        grad = gradient(A_flat)
        direction = -grad
        
        for iteration in range(self.max_iter):
            # Armijo line search with backtracking for step size selection
            alpha = 1.0
            obj_current = objective(A_flat)
            
            for _ in range(20):  # Max line search iterations
                A_new = A_flat + alpha * direction
                obj_new = objective(A_new)
                
                # Armijo condition
                if obj_new <= obj_current + 1e-4 * alpha * np.dot(grad, direction):
                    break
                alpha *= 0.5
            
            A_flat = A_new
            grad_new = gradient(A_flat)
            
            # Convergence check
            if np.linalg.norm(grad_new) < self.tol:
                if self.verbose:
                    print(f"NCG converged after {iteration} iterations")
                break
            
            # Compute beta (conjugate gradient coefficient)
            if self.beta_method == 'fletcher-reeves':
                beta = np.dot(grad_new, grad_new) / np.dot(grad, grad)
            elif self.beta_method == 'polak-ribiere':
                beta = np.dot(grad_new, grad_new - grad) / np.dot(grad, grad)
            else:
                beta = 0  # Steepest descent
            
            # Update direction
            direction = -grad_new + beta * direction
            grad = grad_new
            
            if self.verbose and iteration % 100 == 0:
                print(f"NCG iter {iteration}: obj={obj_current:.6f}")
        
        return A_flat.reshape(n_atoms, n_samples)


# Registry of available solvers
SOLVERS = {
    'fista': FISTASolver,
    'ista': ISTASolver,  
    'coordinate_descent': CoordinateDescentSolver,
    'omp': OrthogonalMatchingPursuit,
    'ncg': NonlinearConjugateGradient,
}


def get_solver(name: str, **kwargs):
    """Factory function for solver instantiation."""
    if name not in SOLVERS:
        raise ValueError(f"Unknown solver '{name}'. Available: {list(SOLVERS.keys())}")
    return SOLVERS[name](**kwargs)