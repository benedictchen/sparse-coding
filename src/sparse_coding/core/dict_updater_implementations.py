"""
Dictionary learning algorithms for sparse coding.

Implements multiple dictionary update methods from the sparse coding literature:
- Method of Optimal Directions (MOD)
- K-SVD algorithm  
- Gradient descent dictionary updates
- Online dictionary learning
- Block coordinate descent
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Dict, Any, Optional, Tuple
from .array import ArrayLike, ensure_array
from .interfaces import DictUpdater


# Consolidated Dictionary Updater Implementations
# Single architecture pattern with concrete classes and simple factory

@dataclass  
class ModUpdater:
    """
    MOD: Method of Optimal Directions (Engan et al., 1999).
    
    Provides closed-form dictionary update by solving least squares problem:
    min_D ||X - DA||_F^2 subject to normalized columns.
    
    Solution: D = XA^T(AA^T + εI)^(-1) followed by column normalization.
    
    Reference:
    Engan, K., Aase, S. O., & Husøy, J. H. (1999). Method of optimal 
    directions for frame design. ICASSP, Vol. 5, pp. 2443-2446.
    """
    eps: float = 1e-7  # Regularization for numerical stability
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """
        MOD dictionary update (Engan et al., 1999).
        
        Solves: min_D ||X - DA||_F^2
        Closed-form solution: D = XA^T(AA^T + εI)^(-1)
        """
        D, X, A = ensure_array(D), ensure_array(X), ensure_array(A)
        
        # MOD closed-form update: D = XA^T(AA^T + εI)^(-1)
        AAt = A @ A.T
        # Add regularization for numerical stability  
        AAt.flat[::AAt.shape[0]+1] += self.eps
        
        try:
            # Solve linear system: (AA^T + εI) * D^T = A * X^T
            D_new = np.linalg.solve(AAt, A @ X.T).T
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            AAt_inv = np.linalg.pinv(AAt)
            D_new = X @ A.T @ AAt_inv
        
        # Normalize dictionary atoms to unit norm
        return self._normalize_columns(D_new)
    
    def _normalize_columns(self, D: ArrayLike) -> ArrayLike:
        """Normalize dictionary columns to unit L2 norm."""
        D = ensure_array(D)
        norms = np.linalg.norm(D, axis=0, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)  # Avoid division by zero
        return D / norms
    
    @property
    def name(self) -> str:
        return "mod"
    
    @property  
    def requires_normalization(self) -> bool:
        return False  # Already normalized in step()


@dataclass
class KsvdUpdater:
    """
    K-SVD: Algorithm for designing overcomplete dictionaries (Aharon et al., 2006).
    
    Updates dictionary atoms sequentially using SVD while adjusting sparse codes.
    Each atom is updated by SVD of the error matrix excluding that atom.
    
    Reference:
    Aharon, M., Elad, M., & Bruckstein, A. (2006). K-SVD: An algorithm 
    for designing overcomplete dictionaries for sparse representation.
    IEEE Transactions on Signal Processing, 54(11), 4311-4322.
    """
    n_iterations: int = 1  # Number of K-SVD sweeps per update
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """
        K-SVD dictionary update (Aharon et al., 2006).
        
        For each atom k:
        1. Find samples using atom k
        2. Compute error without atom k  
        3. SVD update: d_k = u_1, a_k = σ_1 * v_1^T
        """
        D, X, A = ensure_array(D), ensure_array(X), ensure_array(A)
        D_new = D.copy()
        A_new = A.copy()
        n_atoms = D.shape[1]
        
        for iteration in range(self.n_iterations):
            for k in range(n_atoms):
                # Find samples that use atom k (non-zero coefficients)
                omega_k = np.abs(A_new[k, :]) > 1e-12
                
                if not np.any(omega_k):
                    # Reinitialize unused atom with random sample
                    random_idx = np.random.randint(0, X.shape[1])
                    D_new[:, k] = X[:, random_idx]
                    D_new[:, k] /= np.linalg.norm(D_new[:, k]) + 1e-12
                    continue
                
                # Error matrix excluding atom k: E_k = X_ω_k - Σ_{j≠k} d_j a_j^T
                # Equivalently: E_k = X_ω_k - D_ω_k * A_ω_k + d_k * a_k^T
                X_omega = X[:, omega_k]
                A_omega = A_new[:, omega_k]
                E_k = X_omega - D_new @ A_omega + np.outer(D_new[:, k], A_new[k, omega_k])
                
                # SVD of error matrix
                try:
                    U, s, Vt = np.linalg.svd(E_k, full_matrices=False)
                    if len(s) > 0:
                        # Update dictionary atom: d_k = u_1 (first left singular vector)
                        D_new[:, k] = U[:, 0]
                        # Update sparse codes: a_k[ω_k] = σ_1 * v_1^T  
                        A_new[k, omega_k] = s[0] * Vt[0, :]
                except np.linalg.LinAlgError:
                    # Skip update if SVD fails
                    continue
        
        return D_new
    
    @property
    def name(self) -> str:
        return "ksvd"
    
    @property
    def requires_normalization(self) -> bool:
        return False  # K-SVD maintains unit norms through SVD


@dataclass
class GradientUpdater:
    """
    Gradient descent dictionary updater (Olshausen & Field, 1996).
    
    Updates dictionary using gradient descent on Frobenius norm objective:
    min_D 0.5||X - DA||_F^2
    
    Gradient: ∇_D = -(X - DA)A^T
    Update: D ← D - η∇_D
    
    Reference:  
    Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell 
    receptive field properties by learning a sparse code for natural images.
    Nature, 381(6583), 607-609.
    """
    learning_rate: float = 0.01
    momentum: float = 0.0
    _momentum_buffer: Optional[ArrayLike] = None
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """
        Gradient descent dictionary update.
        
        Gradient of 0.5||X - DA||_F^2 w.r.t. D is: ∇_D = -(X - DA)A^T
        """
        D, X, A = ensure_array(D), ensure_array(X), ensure_array(A)
        
        # Compute reconstruction residual
        residual = X - D @ A
        
        # Gradient: ∇_D = -residual * A^T 
        gradient = -residual @ A.T
        
        # Apply momentum if requested
        if self.momentum > 0:
            if self._momentum_buffer is None:
                self._momentum_buffer = np.zeros_like(gradient)
            
            self._momentum_buffer = self.momentum * self._momentum_buffer + (1 - self.momentum) * gradient
            update = self._momentum_buffer
        else:
            update = gradient
        
        # Gradient descent update
        D_new = D - self.learning_rate * update
        
        return D_new
    
    @property
    def name(self) -> str:
        return "gradient"
    
    @property
    def requires_normalization(self) -> bool:
        return True  # Gradient updates can change column norms


@dataclass
class OnlineUpdater:
    """
    Online/stochastic dictionary learning (Mairal et al., 2010).
    
    Processes data in mini-batches with stochastic gradient updates.
    Maintains running averages for stable online learning.
    
    Reference:
    Mairal, J., Bach, F., Ponce, J., & Sapiro, G. (2010). Online dictionary 
    learning for sparse coding. ICML, pp. 689-696.
    """
    learning_rate: float = 0.01
    momentum: float = 0.9
    forgetting_factor: float = 0.95
    _momentum_buffer: Optional[ArrayLike] = None
    _n_updates: int = 0
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """
        Online dictionary update with forgetting factor.
        
        Uses exponential forgetting of past gradients for online adaptation.
        """
        D, X, A = ensure_array(D), ensure_array(X), ensure_array(A)
        
        # Compute current gradient  
        residual = X - D @ A
        current_grad = -residual @ A.T
        
        # Initialize momentum buffer
        if self._momentum_buffer is None:
            self._momentum_buffer = np.zeros_like(current_grad)
        
        # Update with forgetting factor (exponential moving average)
        self._n_updates += 1
        decay = self.forgetting_factor ** self._n_updates
        
        self._momentum_buffer = (self.momentum * self._momentum_buffer + 
                               (1 - self.momentum) * current_grad)
        
        # Adaptive learning rate (decreases with more updates)
        adaptive_lr = self.learning_rate / (1 + 0.01 * self._n_updates)
        
        # Online update
        D_new = D - adaptive_lr * self._momentum_buffer
        
        return D_new
    
    def reset(self):
        """Reset online learner state."""
        self._momentum_buffer = None
        self._n_updates = 0
    
    @property
    def name(self) -> str:
        return "online"
    
    @property
    def requires_normalization(self) -> bool:
        return True  # Online updates typically need normalization


@dataclass
class BlockCoordinateUpdater:
    """
    Block coordinate descent for dictionary learning.
    
    Updates dictionary one column (atom) at a time while keeping others fixed.
    Can be more stable than full gradient updates.
    """
    max_inner_iter: int = 5
    tol: float = 1e-8
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """
        Block coordinate descent over dictionary atoms.
        
        For each atom d_k, solve: min_{d_k} ||X - Σ_j d_j a_j^T||_F^2
        """
        D, X, A = ensure_array(D), ensure_array(X), ensure_array(A)
        D_new = D.copy()
        n_atoms = D.shape[1]
        
        for k in range(n_atoms):
            # Find samples using atom k
            active_samples = np.abs(A[k, :]) > 1e-12
            
            if not np.any(active_samples):
                continue
            
            # Compute residual without atom k
            residual_k = X[:, active_samples] - D_new @ A[:, active_samples] + np.outer(D_new[:, k], A[k, active_samples])
            
            # Optimal update for atom k
            a_k = A[k, active_samples]
            if np.sum(a_k * a_k) > self.tol:
                d_k_new = residual_k @ a_k / np.sum(a_k * a_k)
                
                # Normalize
                norm_k = np.linalg.norm(d_k_new)
                if norm_k > 1e-12:
                    D_new[:, k] = d_k_new / norm_k
        
        return D_new
    
    @property
    def name(self) -> str:
        return "block_coordinate"
    
    @property  
    def requires_normalization(self) -> bool:
        return False  # Normalized within update


# Simple Factory for Dictionary Updaters
class DictUpdaterFactory:
    """Clean, single-responsibility factory for creating dictionary updaters."""
    
    _updaters = {
        'mod': ModUpdater,
        'ksvd': KsvdUpdater,
        'gradient': GradientUpdater,
        'online': OnlineUpdater,
        'block_coordinate': BlockCoordinateUpdater
    }
    
    @staticmethod
    def create(method: str, **kwargs) -> DictUpdater:
        """Create updater instance by name."""
        if method not in DictUpdaterFactory._updaters:
            available = list(DictUpdaterFactory._updaters.keys())
            raise ValueError(f"Unknown updater '{method}'. Available: {available}")
        
        updater_cls = DictUpdaterFactory._updaters[method]
        return updater_cls(**kwargs)
    
    @staticmethod
    def list_available() -> list:
        """List available updater methods."""
        return list(DictUpdaterFactory._updaters.keys())


# Legacy compatibility layer
class DictUpdaterRegistry:
    """Simplified registry for backward compatibility - delegates to factory."""
    
    @staticmethod
    def get_updater(method: str) -> DictUpdater:
        """Get updater (delegates to factory)."""
        return DictUpdaterFactory.create(method)
    
    @staticmethod
    def recommend_updater(scenario: str) -> DictUpdater:
        """Recommend updater for scenario."""
        recommendations = {
            'batch': 'ksvd',      
            'online': 'online',   
            'fast': 'mod',        
            'stable': 'block_coordinate'
        }
        
        method = recommendations.get(scenario, 'ksvd')
        return DictUpdaterFactory.create(method)


# Legacy compatibility
DICT_UPDATER_REGISTRY = DictUpdaterRegistry()


# Simplified Configuration System
@dataclass
class DictUpdaterConfig:
    """Clean configuration for dictionary updater creation."""
    
    # Core updater parameters
    method: str = 'ksvd'  # 'mod', 'ksvd', 'gradient', 'online', 'block_coordinate'
    
    # MOD specific
    mod_eps: float = 1e-7
    
    # K-SVD specific
    ksvd_iterations: int = 1
    
    # Gradient specific  
    learning_rate: float = 0.01
    momentum: float = 0.0
    
    # Online specific
    forgetting_factor: float = 0.95
    
    # Block coordinate specific
    max_inner_iter: int = 5


def create_dict_updater(config: DictUpdaterConfig) -> DictUpdater:
    """Create dictionary updater from configuration."""
    # Extract method-specific parameters
    updater_kwargs = {}
    
    if config.method == 'mod':
        updater_kwargs['eps'] = config.mod_eps
    elif config.method == 'ksvd':
        updater_kwargs['n_iterations'] = config.ksvd_iterations  
    elif config.method == 'gradient':
        updater_kwargs.update({
            'learning_rate': config.learning_rate,
            'momentum': config.momentum
        })
    elif config.method == 'online':
        updater_kwargs.update({
            'learning_rate': config.learning_rate,
            'momentum': config.momentum,
            'forgetting_factor': config.forgetting_factor
        })
    elif config.method == 'block_coordinate':
        updater_kwargs['max_inner_iter'] = config.max_inner_iter
    
    # Single creation path - use factory
    return DictUpdaterFactory.create(config.method, **updater_kwargs)