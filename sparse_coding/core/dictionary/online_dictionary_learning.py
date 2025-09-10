"""
Online dictionary learning algorithm.

Implements Mairal et al. (2010) stochastic approximation approach for 
large-scale dictionary learning with streaming data.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


class OnlineDictionaryLearning:
    """Online Dictionary Learning Algorithm.
    
    Reference: Mairal et al. (2010). Online dictionary learning for sparse 
    coding.
    """
    
    def __init__(self, 
                 forgetting_rate: float = 0.95,
                 regularization: float = 1e-6):
        self.forgetting_rate = forgetting_rate
        self.regularization = regularization
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
            
            # FIXME: Multiple research-accurate online learning rate schedules needed
            #
            # ISSUE: Current learning rate schedule may not be optimal for all scenarios
            #
            # SOLUTION 1: Robbins-Monro schedule (current approach)
            # Exponential forgetting update with decreasing learning rate
            rho_t = 1.0 / self.t if self.t <= 100 else 1.0 / (self.t**0.5)
            
            # SOLUTION 2: Constant forgetting rate (Mairal et al. 2010)
            # Use fixed forgetting rate for non-stationary data:
            # rho_t = self.forgetting_rate
            
            # SOLUTION 3: Adaptive learning rate (Schaul et al. 2013)
            # Adjust learning rate based on gradient magnitude:
            # grad_norm = np.linalg.norm(B_batch - self.B_accumulated)
            # rho_t = min(1.0, base_lr / (grad_norm + 1e-8))
            
            # SOLUTION 4: Cyclic learning rate (Smith 2017)
            # Periodic variation for better convergence:
            # cycle_length = 1000; max_lr = 0.1; min_lr = 0.001
            # cycle_pos = (self.t % cycle_length) / cycle_length
            # rho_t = min_lr + (max_lr - min_lr) * (1 + np.cos(np.pi * cycle_pos)) / 2
            
            self.A_accumulated = (1 - rho_t) * self.A_accumulated + rho_t * A_batch_gram
            self.B_accumulated = (1 - rho_t) * self.B_accumulated + rho_t * B_batch
            
            # FIXME: Dictionary update solver lacks research-accurate alternatives
            #
            # ISSUE: Current solver may be numerically unstable for ill-conditioned matrices
            #
            # SOLUTION 1: Regularized linear system solver (current approach)
            # Dictionary update: solve D = B_accumulated @ (A_accumulated + λI)^(-1)
            A_reg = self.A_accumulated + self.regularization * np.eye(n_atoms)
            
            try:
                D_new = np.linalg.solve(A_reg, self.B_accumulated.T).T
            except np.linalg.LinAlgError:
                D_new = self.B_accumulated @ np.linalg.pinv(A_reg)
            
            # SOLUTION 2: Cholesky decomposition for symmetric positive definite
            # More efficient for well-conditioned Gram matrices:
            # try:
            #     L = np.linalg.cholesky(A_reg)
            #     D_new = np.linalg.solve(L.T, np.linalg.solve(L, self.B_accumulated.T)).T
            # except np.linalg.LinAlgError:
            #     fallback to SVD
            
            # SOLUTION 3: SVD-based solver for maximum numerical stability
            # Handles rank-deficient and ill-conditioned cases:
            # U, s, Vt = np.linalg.svd(A_reg)
            # s_inv = 1.0 / (s + regularization)
            # A_reg_inv = Vt.T @ np.diag(s_inv) @ U.T
            # D_new = self.B_accumulated @ A_reg_inv
            
            # SOLUTION 4: Iterative solver (CG) for large-scale problems
            # from scipy.sparse.linalg import cg
            # D_new = np.zeros_like(self.B_accumulated)
            # for i in range(n_features):
            #     D_new[i, :], _ = cg(A_reg, self.B_accumulated[i, :], maxiter=100)
        
        # Normalize dictionary atoms
        atom_norms = np.linalg.norm(D_new, axis=0, keepdims=True)
        atom_norms = np.where(atom_norms < 1e-12, 1.0, atom_norms)
        D_new = D_new / atom_norms
        
        return D_new