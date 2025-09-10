"""
K-SVD dictionary learning algorithm.

Implements Aharon et al. (2006) K-SVD algorithm for dictionary learning
with atom-by-atom updates using SVD decomposition.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


class KSVDDictionaryLearning:
    """K-SVD Dictionary Learning Algorithm.
    
    Reference: Aharon et al. (2006). K-SVD: An algorithm for designing 
    overcomplete dictionaries for sparse representation.
    """
    
    def __init__(self, sparsity_threshold: float = 1e-10):
        self.sparsity_threshold = sparsity_threshold
    
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
        
        for k in range(n_atoms):
            # Find samples using atom k
            using_atom_k = np.abs(A_new[k, :]) > self.sparsity_threshold
            
            if not np.any(using_atom_k):
                # Replace unused atom with random data sample
                sample_idx = np.random.randint(0, X.shape[1])
                D_new[:, k] = X[:, sample_idx]
                D_new[:, k] /= (np.linalg.norm(D_new[:, k]) + 1e-12)
                continue
            
            # Extract relevant data and codes
            X_k = X[:, using_atom_k]
            A_k = A_new[:, using_atom_k]
            
            # Compute residual without atom k
            residual = X_k - D_new @ A_k + np.outer(D_new[:, k], A_k[k, :])
            
            # SVD decomposition of residual
            if residual.shape[1] > 0:
                U, s, Vt = np.linalg.svd(residual, full_matrices=False)
                
                if len(s) > 0:
                    # Update atom k with first left singular vector
                    D_new[:, k] = U[:, 0]
                    
                    # Update corresponding coefficients
                    A_new[k, using_atom_k] = s[0] * Vt[0, :]
        
        return D_new, A_new