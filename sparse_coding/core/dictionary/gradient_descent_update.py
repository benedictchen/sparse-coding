"""
Gradient descent dictionary update following Olshausen & Field (1996).

Implements the original dictionary learning approach from natural image 
statistics using gradient descent optimization.
"""

from __future__ import annotations
import numpy as np


class GradientDescentUpdate:
    """Gradient descent dictionary update.
    
    Reference: Olshausen & Field (1996). Natural image statistics and efficient 
    coding.
    """
    
    def __init__(self, learning_rate: float = 0.01, normalize_atoms: bool = True):
        self.learning_rate = learning_rate
        self.normalize_atoms = normalize_atoms
    
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
        # Compute reconstruction residual
        residual = X - D @ A
        
        # Gradient of dictionary: -residual @ A^T
        gradient = -residual @ A.T
        
        # Gradient descent update
        D_new = D + self.learning_rate * gradient
        
        if self.normalize_atoms:
            # Normalize each dictionary atom to unit norm
            atom_norms = np.linalg.norm(D_new, axis=0, keepdims=True)
            atom_norms = np.where(atom_norms < 1e-12, 1.0, atom_norms)
            D_new = D_new / atom_norms
        
        return D_new