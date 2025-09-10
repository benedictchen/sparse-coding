"""
Orthogonal Matching Pursuit (OMP) greedy solver.

Implements Pati et al. (1993) greedy algorithm for sparse approximation with
exact sparsity constraints.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


class OrthogonalMatchingPursuit:
    """Orthogonal Matching Pursuit greedy sparse solver.
    
    Reference: Pati et al. (1993). Orthogonal matching pursuit: Recursive 
    function approximation with applications to wavelet decomposition.
    """
    
    def __init__(self, sparsity: Optional[int] = None, tol: float = 1e-6):
        self.sparsity = sparsity
        self.tol = tol
    
    def solve(self, 
              D: np.ndarray, 
              x: np.ndarray,
              max_sparsity: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Solve sparse coding problem using OMP.
        
        Args:
            D: Dictionary matrix (n_features, n_atoms)
            x: Signal vector (n_features,)
            max_sparsity: Maximum number of non-zero coefficients
            
        Returns:
            Sparse codes and iteration count
        """
        n_features, n_atoms = D.shape
        
        if max_sparsity is None:
            max_sparsity = self.sparsity or min(n_features, n_atoms // 2)
        
        # Normalize dictionary atoms for correlation computation
        D_normalized = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
        
        a = np.zeros(n_atoms)
        residual = x.copy()
        selected_indices = []
        
        for k in range(max_sparsity):
            if np.linalg.norm(residual) < self.tol:
                return a, k
            
            # Find atom with maximum correlation to residual
            correlations = np.abs(D_normalized.T @ residual)
            
            # Avoid selecting already chosen atoms
            for idx in selected_indices:
                correlations[idx] = 0.0
            
            best_idx = int(np.argmax(correlations))
            selected_indices.append(best_idx)
            
            # Solve least squares on selected atoms
            D_selected = D[:, selected_indices]
            try:
                # Use pseudo-inverse for stability
                coeffs = np.linalg.pinv(D_selected) @ x
                
                # Update sparse vector
                a.fill(0.0)
                a[selected_indices] = coeffs
                
                # Update residual
                residual = x - D_selected @ coeffs
                
            except np.linalg.LinAlgError:
                # If singular, return current solution
                return a, k
        
        return a, max_sparsity