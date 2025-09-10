"""
Dictionary update implementations for sparse coding dictionary learning.

Based on:
- Engan et al. (1999) "Method of optimal directions for frame design"
- Aharon et al. (2006) "K-SVD: An Algorithm for Designing Overcomplete Dictionaries"
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
from enum import Enum
from ..core.array import ArrayLike, xp
from ..core.interfaces import DictUpdater


class UpdaterType(Enum):
    """Available dictionary update methods with research references."""
    MOD = "mod"                  # Method of Optimal Directions - Engan et al. (1999)
    GRADIENT_DESCENT = "grad_d"  # Gradient descent on dictionary
    KSVD = "ksvd"               # Aharon et al. (2006) K-SVD algorithm


@dataclass 
class UpdaterConfig:
    """
    Configuration parameters for dictionary update algorithms.
    
    Controls the method and parameters for dictionary learning including
    closed-form solutions, gradient-based methods, and SVD-based approaches.
    """
    updater_type: UpdaterType = UpdaterType.MOD
    learning_rate: float = 0.01
    regularization: float = 1e-6  # For numerical stability
    normalize_atoms: bool = True
    max_atom_updates: int = 1     # For K-SVD


class MODUpdater:
    """
    Method of Optimal Directions (MOD) - Engan et al. (1999)
    
    Research Foundation: Engan et al. (1999) "Method of optimal directions for frame design"
    
    Algorithm: Solve closed-form least squares problem
        D_new = argmin_D ||X - DA||_F^2 = XA^T(AA^T + εI)^(-1)
        
    Properties:
    - Closed-form solution
    - Fast computation 
    - May require column normalization after update
    - Can be numerically unstable without regularization
    """
    
    def __init__(self, config: UpdaterConfig):
        self.config = config
        self.regularization = config.regularization
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """
        MOD dictionary update step
        
        Formula: D = XA^T(AA^T + εI)^(-1)
        
        Computes the dictionary that minimizes reconstruction error via least squares.
        """
        backend = xp(D)
        
        # Compute AA^T with regularization
        AAT = A @ A.T
        AAT_reg = AAT + self.regularization * backend.eye(AAT.shape[0])
        
        # Solve: D = XA^T(AA^T + εI)^(-1) 
        try:
            # Use Cholesky if possible (faster for positive definite)
            L = backend.linalg.cholesky(AAT_reg)
            # Solve L @ L.T @ D.T = A @ X.T via forward/backward substitution
            temp = backend.linalg.solve_triangular(L, A @ X.T, lower=True)
            D_new = backend.linalg.solve_triangular(L.T, temp, lower=False).T
        except:
            # Fallback to general solver
            D_new = X @ A.T @ backend.linalg.inv(AAT_reg)
        
        # Normalize columns if requested
        if self.config.normalize_atoms:
            atom_norms = backend.linalg.norm(D_new, axis=0, keepdims=True)
            atom_norms = backend.maximum(atom_norms, 1e-12)  # Avoid division by zero
            D_new = D_new / atom_norms
        
        return D_new
    
    @property
    def name(self) -> str:
        return "mod"
    
    @property
    def requires_normalization(self) -> bool:
        return True  # MOD can change column norms significantly


class GradientDictUpdater:
    """
    Gradient descent dictionary update
    
    Algorithm: D_new = D - η∇_D(||X - DA||_F^2)
    where ∇_D(||X - DA||_F^2) = -(X - DA)A^T
    
    Properties:
    - Simple and stable
    - Slower convergence than MOD
    - Less sensitive to numerical issues
    """
    
    def __init__(self, config: UpdaterConfig):
        self.config = config
        self.learning_rate = config.learning_rate
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """Gradient descent dictionary update"""
        backend = xp(D)
        
        # Compute gradient: ∇_D(||X - DA||_F^2) = -(X - DA)A^T
        residual = X - D @ A
        gradient = -residual @ A.T
        
        # Gradient descent step
        D_new = D - self.learning_rate * gradient
        
        # Normalize columns if requested
        if self.config.normalize_atoms:
            atom_norms = backend.linalg.norm(D_new, axis=0, keepdims=True)
            atom_norms = backend.maximum(atom_norms, 1e-12)
            D_new = D_new / atom_norms
        
        return D_new
    
    @property
    def name(self) -> str:
        return "grad_d"
    
    @property
    def requires_normalization(self) -> bool:
        return True


class KSVDUpdater:
    """
    K-SVD Dictionary Update - Aharon et al. (2006)
    
    Research Foundation: Aharon et al. (2006) "K-SVD: An Algorithm for Designing Overcomplete Dictionaries"
    
    Algorithm:
    1. For each atom k:
        a. Find samples using this atom: Ωₖ = {i : A[k,i] ≠ 0}
        b. Compute error without atom k: E_k = X - Σⱼ≠ₖ D[j]A[j,:]
        c. Restrict to relevant samples: E_k^R = E_k[:,Ωₖ]
        d. SVD: E_k^R = UΣV^T, update D[k] = U[:,0], A[k,Ωₖ] = Σ[0,0]V[0,:]
        
    Properties:
    - Optimal single-atom updates
    - Maintains sparsity pattern of A
    - More expensive than MOD but higher quality
    """
    
    def __init__(self, config: UpdaterConfig):
        self.config = config
        self.max_atom_updates = config.max_atom_updates
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """K-SVD dictionary update"""
        backend = xp(D)
        n_features, n_atoms = D.shape
        n_samples = A.shape[1]
        
        D_new = D.copy()
        A_new = A.copy()
        
        # Update a subset of atoms (for efficiency)
        atoms_to_update = min(self.max_atom_updates, n_atoms)
        atom_indices = np.random.choice(n_atoms, atoms_to_update, replace=False)
        
        for k in atom_indices:
            # Find samples that use this atom (sparsity pattern)
            usage_mask = backend.abs(A_new[k, :]) > 1e-12
            
            if backend.sum(usage_mask) == 0:
                # No samples use this atom, skip or reinitialize randomly
                continue
            
            # Compute error matrix without atom k
            E_k = X - (D_new @ A_new - backend.outer(D_new[:, k], A_new[k, :]))
            
            # Restrict to samples using atom k
            E_k_restricted = E_k[:, usage_mask]
            
            if E_k_restricted.shape[1] == 0:
                continue
            
            # SVD of restricted error matrix
            try:
                U, s, Vt = backend.linalg.svd(E_k_restricted, full_matrices=False)
                
                if len(s) > 0:
                    # Update dictionary atom (first left singular vector)
                    D_new[:, k] = U[:, 0]
                    
                    # Update corresponding coefficients
                    A_new[k, usage_mask] = s[0] * Vt[0, :]
                    
            except np.linalg.LinAlgError:
                # Handle numerical issues
                continue
        
        # Normalize atoms if requested
        if self.config.normalize_atoms:
            atom_norms = backend.linalg.norm(D_new, axis=0, keepdims=True)
            atom_norms = backend.maximum(atom_norms, 1e-12)
            D_new = D_new / atom_norms
        
        return D_new
    
    @property
    def name(self) -> str:
        return "ksvd"
    
    @property
    def requires_normalization(self) -> bool:
        return False  # K-SVD maintains normalization naturally