"""
Method of Optimal Directions (MOD) for dictionary update.

Implements Engan et al. (1999) closed-form dictionary update using 
least squares optimization.
"""

from __future__ import annotations
import numpy as np


class MethodOptimalDirections:
    """Method of Optimal Directions dictionary update.
    
    Reference: Engan et al. (1999). Method of optimal directions for frame 
    design.
    """
    
    def __init__(self, regularization: float = 1e-6):
        self.regularization = regularization
    
    def update(self, 
               D: np.ndarray, 
               X: np.ndarray, 
               A: np.ndarray) -> np.ndarray:
        """Update dictionary using MOD algorithm.
        
        Solves: D* = argmin_D ||X - DA||_F² 
        Solution: D = XA^T(AA^T + εI)^(-1)
        
        Args:
            D: Current dictionary (n_features, n_atoms)
            X: Training data (n_features, n_samples)
            A: Sparse codes (n_atoms, n_samples)
            
        Returns:
            Updated dictionary
        """
        # Compute Gram matrix with regularization
        AAT = A @ A.T
        np.fill_diagonal(AAT, AAT.diagonal() + self.regularization)
        
        # Solve linear system: D = XA^T(AA^T + εI)^(-1)
        try:
            XAT = X @ A.T
            D_new = np.linalg.solve(AAT, XAT.T).T
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            D_new = X @ A.T @ np.linalg.pinv(AAT)
        
        # Normalize dictionary atoms
        atom_norms = np.linalg.norm(D_new, axis=0, keepdims=True)
        atom_norms = np.where(atom_norms < 1e-12, 1.0, atom_norms)
        D_new = D_new / atom_norms
        
        return D_new