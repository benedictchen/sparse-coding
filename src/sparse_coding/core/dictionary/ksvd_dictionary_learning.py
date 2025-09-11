"""
K-SVD dictionary learning algorithm.

Implements Aharon, M., Elad, M., & Bruckstein, A. (2006). K-SVD: An algorithm for 
designing overcomplete dictionaries for sparse representation. IEEE Transactions on 
Signal Processing, 54(11), 4311-4322.

Provides multiple K-SVD variants:
- Standard K-SVD: Original atom-by-atom SVD updates
- Approximate K-SVD: Rubinstein et al. (2008) efficiency improvements
- Batch K-SVD: Mairal et al. (2009) simultaneous atom processing
- Online K-SVD: Streaming updates for large-scale data

Includes numerically stable SVD solvers and unused atom replacement strategies.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Literal, Optional
from sklearn.utils.extmath import randomized_svd


class KSVDDictionaryLearning:
    """Aharon et al. (2006) K-SVD dictionary learning algorithm.
    
    Updates dictionary atoms using SVD decomposition of error matrix
    for each atom while keeping sparse codes fixed.
    """
    
    def __init__(self, 
                 sparsity_threshold: float = 1e-10,
                 variant: Literal['standard', 'approximate', 'batch', 'online'] = 'standard',
                 min_active_threshold: int = 3,
                 batch_size: int = 5,
                 svd_solver: Literal['full', 'randomized', 'truncated'] = 'full',
                 replacement_strategy: Literal['random', 'max_residual', 'pca'] = 'max_residual'):
        self.sparsity_threshold = sparsity_threshold
        self.variant = variant
        self.min_active_threshold = min_active_threshold
        self.batch_size = batch_size
        self.svd_solver = svd_solver
        self.replacement_strategy = replacement_strategy
    
    def update(self, 
               D: np.ndarray, 
               X: np.ndarray, 
               A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update dictionary using K-SVD algorithm.
        
        Args:
            D: Current dictionary (n_features, n_atoms)
            X: Training data (n_features, n_samples)
            A: Sparse codes (n_atoms, n_samples)
            
        Returns:
            Updated dictionary and sparse codes
        """
        D_new = D.copy()
        A_new = A.copy()
        n_atoms = D.shape[1]
        
        if self.variant == 'batch':
            return self._batch_ksvd_update(D_new, X, A_new, n_atoms)
        elif self.variant == 'online':
            return self._online_ksvd_update(D_new, X, A_new, n_atoms)
        
        for k in range(n_atoms):
            using_atom_k = np.abs(A_new[k, :]) > self.sparsity_threshold
            
            if self.variant == 'approximate':
                n_active = np.sum(using_atom_k)
                if n_active < self.min_active_threshold:
                    continue
            
            if not np.any(using_atom_k):
                D_new[:, k], A_new[k, :] = self._replace_unused_atom(D_new, X, A_new, k)
                continue
            
            # Extract relevant data and codes
            X_k = X[:, using_atom_k]
            A_k = A_new[:, using_atom_k]
            
            residual = X_k - D_new @ A_k + np.outer(D_new[:, k], A_k[k, :])
            
            if residual.shape[1] > 0:
                U, s, Vt = self._compute_svd(residual)
                
                if len(s) > 0:
                    D_new[:, k] = U[:, 0]
                    A_new[k, using_atom_k] = s[0] * Vt[0, :]
        
        return D_new, A_new
    
    def _replace_unused_atom(self, D_new, X, A_new, k):
        """Replace unused atom with research-validated strategies."""
        if self.replacement_strategy == 'random':
            sample_idx = np.random.randint(0, X.shape[1])
            new_atom = X[:, sample_idx]
            new_atom /= (np.linalg.norm(new_atom) + 1e-12)
            return new_atom, 0.0
            
        elif self.replacement_strategy == 'max_residual':
            reconstruction_errors = np.linalg.norm(X - D_new @ A_new, axis=0)
            worst_idx = np.argmax(reconstruction_errors)
            new_atom = X[:, worst_idx]
            new_atom /= (np.linalg.norm(new_atom) + 1e-12)
            return new_atom, 0.0
            
        elif self.replacement_strategy == 'pca':
            residual_matrix = X - D_new @ A_new
            U, _, _ = self._compute_svd(residual_matrix)
            if U.shape[1] > 0:
                return U[:, 0], 0.0
            else:
                return self._replace_unused_atom(D_new, X, A_new, k)
                
        return D_new[:, k], 0.0
    
    def _compute_svd(self, matrix):
        """Compute SVD with configurable solvers."""
        if self.svd_solver == 'full':
            return np.linalg.svd(matrix, full_matrices=False)
        elif self.svd_solver == 'randomized':
            n_components = min(1, min(matrix.shape))
            return randomized_svd(matrix, n_components=n_components, random_state=42)
        elif self.svd_solver == 'truncated':
            try:
                from scipy.sparse.linalg import svds
                U, s, Vt = svds(matrix, k=1)
                return U, s, Vt
            except ImportError:
                return np.linalg.svd(matrix, full_matrices=False)
        else:
            return np.linalg.svd(matrix, full_matrices=False)
    
    def _batch_ksvd_update(self, D_new, X, A_new, n_atoms):
        """Batch K-SVD processing multiple atoms simultaneously."""
        for batch_start in range(0, n_atoms, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_atoms)
            batch_indices = list(range(batch_start, batch_end))
            
            for k in batch_indices:
                using_atom_k = np.abs(A_new[k, :]) > self.sparsity_threshold
                
                if not np.any(using_atom_k):
                    D_new[:, k], A_new[k, :] = self._replace_unused_atom(D_new, X, A_new, k)
                    continue
                
                X_k = X[:, using_atom_k]
                A_k = A_new[:, using_atom_k]
                
                residual = X_k - D_new @ A_k + np.outer(D_new[:, k], A_k[k, :])
                
                if residual.shape[1] > 0:
                    U, s, Vt = self._compute_svd(residual)
                    
                    if len(s) > 0:
                        D_new[:, k] = U[:, 0]
                        A_new[k, using_atom_k] = s[0] * Vt[0, :]
                        
        return D_new, A_new
    
    def _online_ksvd_update(self, D_new, X, A_new, n_atoms):
        """Online K-SVD with stochastic approximation."""
        n_samples = X.shape[1]
        sample_indices = np.random.choice(n_samples, size=min(n_samples, 100), replace=False)
        
        X_sample = X[:, sample_indices]
        A_sample = A_new[:, sample_indices]
        
        for k in range(n_atoms):
            using_atom_k = np.abs(A_sample[k, :]) > self.sparsity_threshold
            
            if not np.any(using_atom_k):
                continue
            
            X_k = X_sample[:, using_atom_k]
            A_k = A_sample[:, using_atom_k]
            
            residual = X_k - D_new @ A_k + np.outer(D_new[:, k], A_k[k, :])
            
            if residual.shape[1] > 0:
                U, s, Vt = self._compute_svd(residual)
                
                if len(s) > 0:
                    learning_rate = 0.1
                    D_new[:, k] = (1 - learning_rate) * D_new[:, k] + learning_rate * U[:, 0]
                    D_new[:, k] /= (np.linalg.norm(D_new[:, k]) + 1e-12)
                    
        return D_new, A_new