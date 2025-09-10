"""
Standard dictionary updater implementations.
"""

import numpy as np
from ..core.array import ArrayLike, xp


class MODUpdater:
    """Method of Optimal Directions (MOD) dictionary updater."""
    
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """MOD dictionary update step."""
        backend = xp(D)
        
        # MOD: D = X A^T (A A^T + eps I)^-1
        At = A.T
        G = A @ At
        G_reg = G + self.eps * backend.eye(G.shape[0])
        
        # Use solve instead of inv for stability
        D_new = backend.linalg.solve(G_reg, (X @ At).T).T
        
        # Normalize columns
        norms = backend.linalg.norm(D_new, axis=0, keepdims=True)
        norms = backend.where(norms < 1e-12, 1.0, norms)
        
        return D_new / norms
    
    @property
    def name(self) -> str:
        return "mod"
    
    @property
    def requires_normalization(self) -> bool:
        return False  # Already normalized internally


class GradDUpdater:
    """Gradient-based dictionary updater."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """Gradient dictionary update step."""
        backend = xp(D)
        
        # Gradient: dD/dL = (X - D*A) * A^T
        residual = X - D @ A
        gradient = residual @ A.T
        
        # Update
        D_new = D + self.learning_rate * gradient
        
        # Normalize columns
        norms = backend.linalg.norm(D_new, axis=0, keepdims=True)
        norms = backend.where(norms < 1e-12, 1.0, norms)
        
        return D_new / norms
    
    @property
    def name(self) -> str:
        return "grad_d"
    
    @property
    def requires_normalization(self) -> bool:
        return False  # Already normalized internally