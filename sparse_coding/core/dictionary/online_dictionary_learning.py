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
            
            # Exponential forgetting update
            rho_t = 1.0 / self.t if self.t <= 100 else 1.0 / (self.t**0.5)
            
            self.A_accumulated = (1 - rho_t) * self.A_accumulated + rho_t * A_batch_gram
            self.B_accumulated = (1 - rho_t) * self.B_accumulated + rho_t * B_batch
            
            # Dictionary update: solve D = B_accumulated @ (A_accumulated + λI)^(-1)
            A_reg = self.A_accumulated + self.regularization * np.eye(n_atoms)
            
            try:
                D_new = np.linalg.solve(A_reg, self.B_accumulated.T).T
            except np.linalg.LinAlgError:
                D_new = self.B_accumulated @ np.linalg.pinv(A_reg)
        
        # Normalize dictionary atoms
        atom_norms = np.linalg.norm(D_new, axis=0, keepdims=True)
        atom_norms = np.where(atom_norms < 1e-12, 1.0, atom_norms)
        D_new = D_new / atom_norms
        
        return D_new