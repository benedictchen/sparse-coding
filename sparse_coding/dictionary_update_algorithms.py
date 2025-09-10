"""
Dictionary Update Methods for Sparse Coding

Implements research-based dictionary update algorithms from original papers.
All updaters follow the DictUpdater protocol.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Any, Dict
from .core.interfaces import DictUpdater
from .core.array import ArrayLike


class MODUpdater:
    """Method of Optimal Directions dictionary update.
    
    Reference: Engan et al. (1999). Method of optimal directions for frame design.
    
    Closed-form least squares solution: D = X A^T (A A^T + εI)^{-1}
    Optimal when sparse codes are fixed.
    """
    
    def __init__(self, regularization: float = 1e-7, normalize: bool = True):
        self.regularization = regularization
        self.normalize = normalize
    
    @property
    def name(self) -> str:
        return "mod"
    
    @property
    def requires_normalization(self) -> bool:
        return self.normalize
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """MOD dictionary update step."""
        X = np.asarray(X)
        A = np.asarray(A)
        
        # Compute A A^T with regularization
        AtA = A @ A.T
        AtA_reg = AtA + self.regularization * np.eye(AtA.shape[0])
        
        try:
            # Solve: D = X A^T (A A^T + εI)^{-1}
            D_new = X @ A.T @ np.linalg.inv(AtA_reg)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            D_new = X @ np.linalg.pinv(A)
        
        # Normalize columns if required
        if self.normalize:
            norms = np.linalg.norm(D_new, axis=0, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            D_new = D_new / norms
        
        return D_new


class GradientDescentUpdater:
    """Gradient descent dictionary update.
    
    Reference: Olshausen & Field (1996). Emergence of simple-cell receptive field properties.
    
    Updates dictionary using gradient of reconstruction error:
    D ← D - η ∇_D [1/2 ||X - DA||_F^2]
    """
    
    def __init__(self, learning_rate: float = 0.01, normalize: bool = True,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        self.learning_rate = learning_rate
        self.normalize = normalize
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._momentum_buffer = None
    
    @property
    def name(self) -> str:
        return "grad_d"
    
    @property
    def requires_normalization(self) -> bool:
        return self.normalize
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """Gradient descent dictionary update."""
        D = np.asarray(D)
        X = np.asarray(X)
        A = np.asarray(A)
        
        # Compute gradient: ∇_D [1/2 ||X - DA||_F^2] = -(X - DA)A^T
        reconstruction = D @ A
        residual = X - reconstruction
        gradient = -residual @ A.T
        
        # Add weight decay
        if self.weight_decay > 0:
            gradient += self.weight_decay * D
        
        # Apply momentum
        if self.momentum > 0:
            if self._momentum_buffer is None:
                self._momentum_buffer = np.zeros_like(gradient)
            
            self._momentum_buffer = self.momentum * self._momentum_buffer + gradient
            gradient = self._momentum_buffer
        
        # Update dictionary
        D_new = D - self.learning_rate * gradient
        
        # Normalize columns if required
        if self.normalize:
            norms = np.linalg.norm(D_new, axis=0, keepdims=True)
            norms[norms == 0] = 1
            D_new = D_new / norms
        
        return D_new


class KSVDUpdater:
    """K-SVD dictionary update algorithm.
    
    Reference: Aharon et al. (2006). K-SVD: An Algorithm for Designing Overcomplete 
    Dictionaries for Sparse Representation.
    
    Updates one dictionary atom at a time using SVD while maintaining sparsity.
    """
    
    def __init__(self, preserve_dc: bool = False, verbose: bool = False):
        self.preserve_dc = preserve_dc  # Keep DC component fixed
        self.verbose = verbose
    
    @property
    def name(self) -> str:
        return "ksvd"
    
    @property
    def requires_normalization(self) -> bool:
        return False  # K-SVD inherently maintains unit norms
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """K-SVD dictionary update."""
        D = np.asarray(D).copy()
        X = np.asarray(X)
        A = np.asarray(A).copy()
        
        n_features, n_atoms = D.shape
        n_samples = X.shape[1]
        
        # Update each atom sequentially
        start_idx = 1 if self.preserve_dc else 0
        
        for k in range(start_idx, n_atoms):
            # Find samples that use atom k
            using_atom_k = np.nonzero(A[k, :])[0]
            
            if len(using_atom_k) == 0:
                # No samples use this atom - reinitialize randomly
                D[:, k] = np.random.randn(n_features)
                D[:, k] /= np.linalg.norm(D[:, k])
                continue
            
            # Compute error without atom k: E_k = X - ∑_{j≠k} d_j a_j
            D[:, k] = 0  # Remove current atom
            E_k = X[:, using_atom_k] - D @ A[:, using_atom_k]
            
            # Extract coefficients for atom k
            a_k = A[k, using_atom_k]
            
            # SVD of error matrix: E_k = U Σ V^T
            try:
                U, sigma, Vt = np.linalg.svd(E_k, full_matrices=False)
                
                # Update dictionary atom: d_k = u_1 (first left singular vector)
                D[:, k] = U[:, 0]
                
                # Update coefficients: a_k = σ_1 v_1^T
                A[k, using_atom_k] = sigma[0] * Vt[0, :]
                
            except np.linalg.LinAlgError:
                # SVD decomposition failed - fallback to column normalization
                if np.linalg.norm(E_k) > 1e-12:
                    D[:, k] = E_k[:, 0] / np.linalg.norm(E_k[:, 0])
                else:
                    D[:, k] = np.random.randn(n_features)
                    D[:, k] /= np.linalg.norm(D[:, k])
        
        return D


class OnlineDictionaryUpdater:
    """Online/stochastic dictionary learning update.
    
    Reference: Mairal et al. (2010). Online dictionary learning for sparse coding.
    
    Stochastic gradient descent with running averages for efficient streaming updates.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9,
                 batch_size: Optional[int] = None, normalize: bool = True):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.normalize = normalize
        self._running_avg = None
        self._step_count = 0
    
    @property
    def name(self) -> str:
        return "online_sgd"
    
    @property
    def requires_normalization(self) -> bool:
        return self.normalize
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """Online dictionary update step."""
        D = np.asarray(D)
        X = np.asarray(X)
        A = np.asarray(A)
        
        # Sample mini-batch if specified
        if self.batch_size is not None and X.shape[1] > self.batch_size:
            indices = np.random.choice(X.shape[1], self.batch_size, replace=False)
            X_batch = X[:, indices]
            A_batch = A[:, indices]
        else:
            X_batch = X
            A_batch = A
        
        # Compute gradient on mini-batch
        reconstruction = D @ A_batch
        residual = X_batch - reconstruction
        gradient = -residual @ A_batch.T / A_batch.shape[1]  # Average over batch
        
        # Initialize running average
        if self._running_avg is None:
            self._running_avg = np.zeros_like(gradient)
        
        # Update running average with momentum
        self._running_avg = self.momentum * self._running_avg + (1 - self.momentum) * gradient
        
        # Adaptive learning rate (decreases with steps)
        self._step_count += 1
        adaptive_lr = self.learning_rate / (1 + 0.001 * self._step_count)
        
        # Update dictionary
        D_new = D - adaptive_lr * self._running_avg
        
        # Normalize columns if required
        if self.normalize:
            norms = np.linalg.norm(D_new, axis=0, keepdims=True)
            norms[norms == 0] = 1
            D_new = D_new / norms
        
        return D_new


class AdamDictionaryUpdater:
    """Adam optimizer for dictionary updates.
    
    Reference: Kingma & Ba (2014). Adam: A Method for Stochastic Optimization.
    
    Adaptive learning rate with momentum and second-moment estimation.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, normalize: bool = True):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.normalize = normalize
        self._m = None  # First moment
        self._v = None  # Second moment
        self._step_count = 0
    
    @property
    def name(self) -> str:
        return "adam"
    
    @property
    def requires_normalization(self) -> bool:
        return self.normalize
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """Adam dictionary update step."""
        D = np.asarray(D)
        X = np.asarray(X)
        A = np.asarray(A)
        
        # Compute gradient
        reconstruction = D @ A
        residual = X - reconstruction
        gradient = -residual @ A.T / A.shape[1]
        
        # Initialize moments
        if self._m is None:
            self._m = np.zeros_like(gradient)
            self._v = np.zeros_like(gradient)
        
        self._step_count += 1
        
        # Update biased first moment estimate
        self._m = self.beta1 * self._m + (1 - self.beta1) * gradient
        
        # Update biased second moment estimate
        self._v = self.beta2 * self._v + (1 - self.beta2) * gradient**2
        
        # Compute bias-corrected moments
        m_hat = self._m / (1 - self.beta1**self._step_count)
        v_hat = self._v / (1 - self.beta2**self._step_count)
        
        # Update dictionary
        D_new = D - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Normalize columns if required
        if self.normalize:
            norms = np.linalg.norm(D_new, axis=0, keepdims=True)
            norms[norms == 0] = 1
            D_new = D_new / norms
        
        return D_new


class ProjectedGradientUpdater:
    """Projected gradient descent with constraints.
    
    Reference: Bertsekas (1999). Nonlinear Programming, Chapter 2.
    
    Enforces constraints on dictionary atoms (e.g., non-negativity, bounded norm).
    """
    
    def __init__(self, learning_rate: float = 0.01, projection: str = 'unit_norm',
                 clip_range: Optional[tuple] = None):
        self.learning_rate = learning_rate
        self.projection = projection
        self.clip_range = clip_range
    
    @property
    def name(self) -> str:
        return "projected_gradient"
    
    @property
    def requires_normalization(self) -> bool:
        return False  # Projection handles constraints
    
    def _project(self, D: np.ndarray) -> np.ndarray:
        """Apply projection to satisfy constraints."""
        if self.projection == 'unit_norm':
            # Project to unit sphere
            norms = np.linalg.norm(D, axis=0, keepdims=True)
            norms[norms == 0] = 1
            return D / norms
        
        elif self.projection == 'non_negative':
            # Project to non-negative orthant
            D_proj = np.maximum(D, 0)
            norms = np.linalg.norm(D_proj, axis=0, keepdims=True)
            norms[norms == 0] = 1
            return D_proj / norms
        
        elif self.projection == 'box' and self.clip_range is not None:
            # Project to box constraints
            D_proj = np.clip(D, self.clip_range[0], self.clip_range[1])
            norms = np.linalg.norm(D_proj, axis=0, keepdims=True)
            norms[norms == 0] = 1
            return D_proj / norms
        
        else:
            return D
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """Projected gradient descent step."""
        D = np.asarray(D)
        X = np.asarray(X)
        A = np.asarray(A)
        
        # Compute gradient
        reconstruction = D @ A
        residual = X - reconstruction
        gradient = -residual @ A.T
        
        # Gradient step
        D_new = D - self.learning_rate * gradient
        
        # Apply projection
        D_new = self._project(D_new)
        
        return D_new


# Registry of available dictionary updaters
DICT_UPDATERS = {
    'mod': MODUpdater,
    'grad_d': GradientDescentUpdater,
    'ksvd': KSVDUpdater,
    'online_sgd': OnlineDictionaryUpdater,
    'adam': AdamDictionaryUpdater,
    'projected_gradient': ProjectedGradientUpdater,
}


def get_dict_updater(name: str, **kwargs):
    """Factory function for dictionary updater instantiation."""
    if name not in DICT_UPDATERS:
        raise ValueError(f"Unknown updater '{name}'. Available: {list(DICT_UPDATERS.keys())}")
    return DICT_UPDATERS[name](**kwargs)