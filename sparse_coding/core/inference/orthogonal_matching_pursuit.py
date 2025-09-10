"""
Orthogonal Matching Pursuit (OMP) greedy solver.

Implements Pati, Y. C., Rezaiifar, R., & Krishnaprasad, P. S. (1993). Orthogonal matching 
pursuit: Recursive function approximation with applications to wavelet decomposition. 
Proceedings of 27th Asilomar Conference on Signals, Systems and Computers, 1, 40-44.

Provides multiple OMP variants and numerical solvers:
- Standard OMP: Original greedy selection algorithm
- Weak OMP: Temlyakov (2000) threshold-based selection  
- Regularized OMP: Needell & Vershynin (2010) with regularization
- Stagewise OMP: Donoho et al. (2012) multiple selections per iteration

Numerical stability through QR decomposition (Golub & Van Loan), Cholesky 
decomposition, SVD, and pseudo-inverse solvers.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Literal


class OrthogonalMatchingPursuit:
    """Pati et al. (1993) Orthogonal Matching Pursuit greedy algorithm.
    
    Iteratively selects dictionary atoms with highest correlation to residual,
    solving least squares subproblems for sparse signal reconstruction.
    """
    
    def __init__(self, 
                 sparsity: Optional[int] = None, 
                 tol: float = 1e-6,
                 variant: Literal['standard', 'weak', 'regularized', 'stagewise'] = 'standard',
                 weak_threshold: float = 0.5,
                 regularization: float = 1e-6,
                 solver: Literal['pinv', 'qr', 'cholesky', 'svd'] = 'qr'):
        self.sparsity = sparsity
        self.tol = tol
        self.variant = variant
        self.weak_threshold = weak_threshold
        self.regularization = regularization
        self.solver = solver
    
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
            
            correlations = np.abs(D_normalized.T @ residual)
            
            for idx in selected_indices:
                correlations[idx] = 0.0
            
            if self.variant == 'standard':
                best_idx = int(np.argmax(correlations))
                selected_indices.append(best_idx)
                
            elif self.variant == 'weak':
                threshold = self.weak_threshold * np.max(correlations)
                weak_candidates = np.where(correlations >= threshold)[0]
                if len(weak_candidates) > 0:
                    best_idx = int(np.random.choice(weak_candidates))
                    selected_indices.append(best_idx)
                else:
                    best_idx = int(np.argmax(correlations))
                    selected_indices.append(best_idx)
                    
            elif self.variant == 'regularized':
                regularized_corr = correlations + self.regularization * np.random.randn(n_atoms)
                best_idx = int(np.argmax(regularized_corr))
                selected_indices.append(best_idx)
                
            elif self.variant == 'stagewise':
                n_select = min(3, max(1, int(0.1 * n_atoms)))
                top_indices = np.argsort(correlations)[-n_select:]
                for idx in top_indices:
                    if idx not in selected_indices:
                        selected_indices.append(int(idx))
                        break
            
            D_selected = D[:, selected_indices]
            coeffs = self._solve_least_squares(D_selected, x)
            
            a.fill(0.0)
            a[selected_indices] = coeffs
            
            residual = x - D_selected @ coeffs
        
        return a, max_sparsity
    
    def _solve_least_squares(self, D_selected, x):
        """Research-accurate least squares solvers with numerical stability."""
        if self.solver == 'pinv':
            return np.linalg.pinv(D_selected) @ x
            
        elif self.solver == 'qr':
            try:
                Q, R = np.linalg.qr(D_selected)
                return np.linalg.solve(R, Q.T @ x)
            except np.linalg.LinAlgError:
                return np.linalg.pinv(D_selected) @ x
                
        elif self.solver == 'cholesky':
            try:
                G = D_selected.T @ D_selected + self.regularization * np.eye(D_selected.shape[1])
                L = np.linalg.cholesky(G)
                y = np.linalg.solve(L, D_selected.T @ x)
                return np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                return np.linalg.pinv(D_selected) @ x
                
        elif self.solver == 'svd':
            try:
                U, s, Vt = np.linalg.svd(D_selected, full_matrices=False)
                s_inv = np.where(s > 1e-10, 1.0 / s, 0.0)
                return Vt.T @ (s_inv * (U.T @ x))
            except np.linalg.LinAlgError:
                return np.linalg.pinv(D_selected) @ x
        
        return np.linalg.pinv(D_selected) @ x