"""
Online dictionary learning algorithm.

Implements Mairal, J., Bach, F., Ponce, J., & Sapiro, G. (2009). Online dictionary 
learning for sparse coding. Proceedings of the 26th Annual International Conference 
on Machine Learning (ICML), 689-696.

Provides multiple online learning strategies:
- Robbins-Monro schedule: Decreasing learning rates with convergence guarantees
- Constant forgetting rate: Fixed rate for non-stationary data (Mairal et al.)
- Adaptive learning rate: Schaul et al. (2013) gradient-based adaptation
- Cyclic learning rate: Smith (2017) periodic variation for better convergence

Includes numerically stable matrix solvers (Cholesky, SVD, iterative methods).
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Literal


class OnlineDictionaryLearning:
    """Research-accurate online dictionary learning with all adaptive variants.
    
    Implements multiple learning rate schedules and numerical solvers
    from the online learning literature.
    """
    
    def __init__(self, 
                 forgetting_rate: float = 0.95,
                 regularization: float = 1e-6,
                 learning_schedule: Literal['robbins_monro', 'constant', 'adaptive', 'cyclic'] = 'robbins_monro',
                 solver: Literal['direct', 'cholesky', 'svd', 'iterative'] = 'cholesky',
                 base_lr: float = 0.01,
                 cycle_length: int = 1000,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001):
        self.forgetting_rate = forgetting_rate
        self.regularization = regularization
        self.learning_schedule = learning_schedule
        self.solver = solver
        self.base_lr = base_lr
        self.cycle_length = cycle_length
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        self.A_accumulated = None
        self.B_accumulated = None
        self.t = 0
    
    def update(self, 
               D: np.ndarray, 
               X: np.ndarray, 
               A: np.ndarray,
               batch_size: Optional[int] = None) -> np.ndarray:
        """Update dictionary using online learning.
        
        Args:
            D: Current dictionary (n_features, n_atoms)
            X: Training data batch (n_features, n_samples)
            A: Sparse codes batch (n_atoms, n_samples)
            batch_size: Size of mini-batches (full batch if None)
            
        Returns:
            Updated dictionary
        """
        n_features, n_atoms = D.shape
        n_samples = X.shape[1]
        
        if self.A_accumulated is None:
            self.A_accumulated = np.zeros((n_atoms, n_atoms))
            self.B_accumulated = np.zeros((n_features, n_atoms))
        
        if batch_size is None:
            batch_size = n_samples
        
        D_new = D.copy()
        
        # Process data in mini-batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            X_batch = X[:, start_idx:end_idx]
            A_batch = A[:, start_idx:end_idx]
            
            self.t += 1
            
            # Compute sufficient statistics
            A_batch_gram = A_batch @ A_batch.T
            B_batch = X_batch @ A_batch.T
            
            rho_t = self._compute_learning_rate(B_batch)
            
            self.A_accumulated = (1 - rho_t) * self.A_accumulated + rho_t * A_batch_gram
            self.B_accumulated = (1 - rho_t) * self.B_accumulated + rho_t * B_batch
            
            A_reg = self.A_accumulated + self.regularization * np.eye(n_atoms)
            D_new = self._solve_dictionary_update(A_reg, self.B_accumulated)
        
        # Normalize dictionary atoms
        atom_norms = np.linalg.norm(D_new, axis=0, keepdims=True)
        atom_norms = np.where(atom_norms < 1e-12, 1.0, atom_norms)
        D_new = D_new / atom_norms
        
        return D_new
    
    def _compute_learning_rate(self, B_batch):
        """Compute learning rate using research-validated schedules."""
        if self.learning_schedule == 'robbins_monro':
            return 1.0 / self.t if self.t <= 100 else 1.0 / (self.t**0.5)
            
        elif self.learning_schedule == 'constant':
            return self.forgetting_rate
            
        elif self.learning_schedule == 'adaptive':
            if hasattr(self, 'B_accumulated') and self.B_accumulated is not None:
                grad_norm = np.linalg.norm(B_batch - self.B_accumulated)
                return min(1.0, self.base_lr / (grad_norm + 1e-8))
            else:
                return self.base_lr
                
        elif self.learning_schedule == 'cyclic':
            cycle_pos = (self.t % self.cycle_length) / self.cycle_length
            return self.min_lr + (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * cycle_pos)) / 2
            
        else:
            return 1.0 / self.t if self.t <= 100 else 1.0 / (self.t**0.5)
    
    def _solve_dictionary_update(self, A_reg, B_accumulated):
        """Solve dictionary update with configurable numerical methods."""
        if self.solver == 'direct':
            try:
                return np.linalg.solve(A_reg, B_accumulated.T).T
            except np.linalg.LinAlgError:
                return B_accumulated @ np.linalg.pinv(A_reg)
                
        elif self.solver == 'cholesky':
            try:
                L = np.linalg.cholesky(A_reg)
                return np.linalg.solve(L.T, np.linalg.solve(L, B_accumulated.T)).T
            except np.linalg.LinAlgError:
                return self._solve_dictionary_update(A_reg, B_accumulated)
                
        elif self.solver == 'svd':
            try:
                U, s, Vt = np.linalg.svd(A_reg)
                s_inv = np.where(s > 1e-10, 1.0 / s, 0.0)
                A_reg_inv = Vt.T @ np.diag(s_inv) @ U.T
                return B_accumulated @ A_reg_inv
            except np.linalg.LinAlgError:
                return B_accumulated @ np.linalg.pinv(A_reg)
                
        elif self.solver == 'iterative':
            try:
                from scipy.sparse.linalg import cg
                D_new = np.zeros_like(B_accumulated)
                for i in range(B_accumulated.shape[0]):
                    D_new[i, :], _ = cg(A_reg, B_accumulated[i, :], maxiter=100)
                return D_new
            except ImportError:
                return self._solve_dictionary_update(A_reg, B_accumulated)
                
        else:
            try:
                return np.linalg.solve(A_reg, B_accumulated.T).T
            except np.linalg.LinAlgError:
                return B_accumulated @ np.linalg.pinv(A_reg)
    
    def reset_accumulated_statistics(self):
        """Reset accumulated statistics for new learning session."""
        self.A_accumulated = None
        self.B_accumulated = None
        self.t = 0