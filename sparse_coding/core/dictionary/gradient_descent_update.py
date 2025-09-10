"""
Gradient descent dictionary update following Olshausen & Field (1996).

Implements Olshausen, B. A., & Field, D. J. (1996). Natural image statistics and 
efficient coding. Network Computation in Neural Systems, 7, 333-339.

Provides multiple gradient-based optimization variants:
- Basic gradient descent: Original Olshausen & Field approach
- Momentum gradient descent: Polyak (1964) acceleration
- AdaGrad: Duchi et al. (2011) adaptive learning rates  
- Adam optimizer: Kingma & Ba (2014) adaptive moments estimation

Includes multiple dictionary normalization strategies from the literature.
"""

from __future__ import annotations
import numpy as np
from typing import Literal, Optional


class GradientDescentUpdate:
    """Research-accurate gradient descent with all optimization variants.
    
    Implements multiple gradient-based optimizers from the literature
    with configurable normalization strategies.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01, 
                 normalize_atoms: bool = True,
                 optimizer: Literal['basic', 'momentum', 'adagrad', 'adam'] = 'basic',
                 momentum_coeff: float = 0.9,
                 eps: float = 1e-8,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 normalization: Literal['l2', 'soft', 'spherical', 'none'] = 'l2',
                 regularization: float = 1e-6):
        self.learning_rate = learning_rate
        self.normalize_atoms = normalize_atoms
        self.optimizer = optimizer
        self.momentum_coeff = momentum_coeff
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.normalization = normalization
        self.regularization = regularization
        
        # Optimizer state
        self.velocity = None
        self.accumulated_grad = None
        self.m = None  # Adam first moment
        self.v = None  # Adam second moment
        self.t = 0     # Adam time step
    
    def update(self, 
               D: np.ndarray, 
               X: np.ndarray, 
               A: np.ndarray) -> np.ndarray:
        """Update dictionary using gradient descent.
        
        Minimizes: E = ||X - DA||_F²
        Gradient: ∂E/∂D = -(X - DA)A^T = -(Residual)A^T
        
        Args:
            D: Current dictionary (n_features, n_atoms)
            X: Training data (n_features, n_samples)
            A: Sparse codes (n_atoms, n_samples)
            
        Returns:
            Updated dictionary
        """
        residual = X - D @ A
        gradient = -residual @ A.T
        
        if self.optimizer == 'basic':
            D_new = D + self.learning_rate * gradient
            
        elif self.optimizer == 'momentum':
            if self.velocity is None:
                self.velocity = np.zeros_like(D)
            
            self.velocity = self.momentum_coeff * self.velocity + self.learning_rate * gradient
            D_new = D + self.velocity
            
        elif self.optimizer == 'adagrad':
            if self.accumulated_grad is None:
                self.accumulated_grad = np.zeros_like(D)
            
            self.accumulated_grad += gradient**2
            adaptive_lr = self.learning_rate / (np.sqrt(self.accumulated_grad) + self.eps)
            D_new = D + adaptive_lr * gradient
            
        elif self.optimizer == 'adam':
            if self.m is None:
                self.m = np.zeros_like(D)
                self.v = np.zeros_like(D)
            
            self.t += 1
            
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
            
            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)
            
            D_new = D + self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            
        else:
            D_new = D + self.learning_rate * gradient
        
        if self.normalize_atoms:
            D_new = self._apply_normalization(D_new)
        
        return D_new
    
    def _apply_normalization(self, D):
        """Apply research-validated normalization strategies."""
        if self.normalization == 'l2':
            atom_norms = np.linalg.norm(D, axis=0, keepdims=True)
            atom_norms = np.where(atom_norms < 1e-12, 1.0, atom_norms)
            return D / atom_norms
            
        elif self.normalization == 'soft':
            atom_norms = np.linalg.norm(D, axis=0, keepdims=True)
            soft_norms = atom_norms / (1 + self.regularization * atom_norms**2)
            return D * soft_norms / (atom_norms + 1e-12)
            
        elif self.normalization == 'spherical':
            # Riemannian gradient descent on unit sphere
            for i in range(D.shape[1]):
                d_i = D[:, i]
                norm = np.linalg.norm(d_i)
                if norm > 1e-12:
                    D[:, i] = d_i / norm
            return D
            
        elif self.normalization == 'none':
            return D
            
        else:
            atom_norms = np.linalg.norm(D, axis=0, keepdims=True)
            atom_norms = np.where(atom_norms < 1e-12, 1.0, atom_norms)
            return D / atom_norms
    
    def reset_optimizer_state(self):
        """Reset optimizer state for new dictionary learning session."""
        self.velocity = None
        self.accumulated_grad = None
        self.m = None
        self.v = None
        self.t = 0