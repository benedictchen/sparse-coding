"""
Standard solver implementations for sparse inference.
"""

import numpy as np
from typing import Optional
from ..core.array import ArrayLike, xp
from ..core.interfaces import Penalty
from ..fista_batch import power_iter_L


class FISTASolver:
    """Fast Iterative Shrinkage-Thresholding Algorithm solver."""
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """Solve sparse coding using FISTA."""
        backend = xp(D)
        
        if not penalty.is_prox_friendly:
            raise ValueError("FISTA requires prox-friendly penalty")
        
        n_atoms, n_samples = D.shape[1], X.shape[1]
        
        # Handle single sample
        if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
            return self._solve_single(D, X.ravel(), penalty, backend)
        
        # Batch processing
        A = backend.zeros((n_atoms, n_samples))
        for i in range(n_samples):
            A[:, i] = self._solve_single(D, X[:, i], penalty, backend)
        
        return A
    
    def _solve_single(self, D: ArrayLike, x: ArrayLike, penalty: Penalty, backend) -> ArrayLike:
        """Solve single sparse coding problem."""
        n_atoms = D.shape[1]
        
        # Estimate Lipschitz constant using safe cross-backend method
        D_numpy = np.asarray(D)
        L = float(power_iter_L(D_numpy))
        
        # Initialize
        a = backend.zeros(n_atoms)
        z = backend.zeros(n_atoms)
        t = 1.0
        
        for iteration in range(self.max_iter):
            a_old = a
            
            # Gradient step
            grad = D.T @ (D @ z - x)
            z_grad = z - grad / L
            
            # Proximal step
            a = penalty.prox(z_grad, 1.0 / L)
            
            # Momentum update
            t_new = (1.0 + backend.sqrt(1.0 + 4.0 * t**2)) / 2.0
            beta = (t - 1.0) / t_new
            z = a + beta * (a - a_old)
            t = t_new
            
            # Check convergence
            if backend.linalg.norm(a - a_old) < self.tol:
                break
        
        return a
    
    @property
    def name(self) -> str:
        return "fista"
    
    @property
    def supports_batch(self) -> bool:
        return True


class ISTASolver:
    """Iterative Shrinkage-Thresholding Algorithm solver."""
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """Solve sparse coding using ISTA."""
        backend = xp(D)
        
        if not penalty.is_prox_friendly:
            raise ValueError("ISTA requires prox-friendly penalty")
        
        n_atoms, n_samples = D.shape[1], X.shape[1]
        
        # Handle single sample
        if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
            return self._solve_single(D, X.ravel(), penalty, backend)
        
        # Batch processing
        A = backend.zeros((n_atoms, n_samples))
        for i in range(n_samples):
            A[:, i] = self._solve_single(D, X[:, i], penalty, backend)
        
        return A
    
    def _solve_single(self, D: ArrayLike, x: ArrayLike, penalty: Penalty, backend) -> ArrayLike:
        """Solve single sparse coding problem."""
        n_atoms = D.shape[1]
        
        # Estimate Lipschitz constant using safe cross-backend method
        D_numpy = np.asarray(D)
        L = float(power_iter_L(D_numpy))
        
        # Initialize
        a = backend.zeros(n_atoms)
        
        for iteration in range(self.max_iter):
            a_old = a
            
            # Gradient step
            grad = D.T @ (D @ a - x)
            z = a - grad / L
            
            # Proximal step
            a = penalty.prox(z, 1.0 / L)
            
            # Check convergence
            if backend.linalg.norm(a - a_old) < self.tol:
                break
        
        return a
    
    @property
    def name(self) -> str:
        return "ista"
    
    @property
    def supports_batch(self) -> bool:
        return True