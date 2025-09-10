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
            
            # FIXME: Multiple research-accurate OMP implementation variants needed
            # 
            # ISSUE: Current implementation lacks key OMP variants from literature
            # 
            # SOLUTION 1: Standard OMP (Pati et al. 1993) - Current approach
            # Find atom with maximum correlation to residual
            correlations = np.abs(D_normalized.T @ residual)
            
            # SOLUTION 2: Weak OMP (Temlyakov 2000)
            # Select atoms with correlation above threshold rather than maximum:
            # threshold = self.weak_threshold * np.max(correlations)
            # weak_candidates = np.where(correlations >= threshold)[0]
            # best_idx = weak_candidates[np.random.randint(len(weak_candidates))]
            
            # SOLUTION 3: Regularized OMP (Needell & Vershynin 2010)
            # Add small regularization to correlation computation:
            # regularized_corr = correlations + reg_param * np.random.randn(n_atoms)
            
            # SOLUTION 4: Stagewise OMP (Donoho et al. 2012)
            # Multiple weak selections per iteration for better approximation
            
            # Avoid selecting already chosen atoms
            for idx in selected_indices:
                correlations[idx] = 0.0
            
            best_idx = int(np.argmax(correlations))
            selected_indices.append(best_idx)
            
            # FIXME: Least squares solver lacks research-accurate variants
            #
            # ISSUE: Only pseudo-inverse used, missing numerically stable alternatives
            #
            # SOLUTION 1: Current pseudo-inverse approach (numerically unstable)
            D_selected = D[:, selected_indices]
            try:
                coeffs = np.linalg.pinv(D_selected) @ x
                
                # SOLUTION 2: QR decomposition for numerical stability (Golub & Van Loan)
                # Q, R = np.linalg.qr(D_selected)
                # coeffs = np.linalg.solve(R, Q.T @ x)
                
                # SOLUTION 3: Cholesky decomposition for symmetric positive definite
                # G = D_selected.T @ D_selected + regularization * np.eye(len(selected_indices))
                # coeffs = np.linalg.solve(G, D_selected.T @ x)
                
                # SOLUTION 4: SVD for maximum numerical stability
                # U, s, Vt = np.linalg.svd(D_selected, full_matrices=False)
                # coeffs = Vt.T @ (np.diag(1/s) @ (U.T @ x))
                
                # Update sparse vector
                a.fill(0.0)
                a[selected_indices] = coeffs
                
                # Update residual
                residual = x - D_selected @ coeffs
                
            except np.linalg.LinAlgError:
                # If singular, return current solution
                return a, k
        
        return a, max_sparsity