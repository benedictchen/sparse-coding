"""
Standard penalty implementations for sparse coding.
"""

import numpy as np
from typing import Any
from ..core.array import ArrayLike, xp


class L1Penalty:
    """L1 penalty (Lasso regularization)."""
    
    def __init__(self, lam: float = 0.1):
        self.lam = lam
    
    def value(self, a: ArrayLike) -> float:
        """Compute L1 penalty value."""
        backend = xp(a)
        return float(self.lam * backend.sum(backend.abs(a)))
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """L1 proximal operator (soft thresholding)."""
        backend = xp(z)
        threshold = self.lam * t
        return backend.sign(z) * backend.maximum(backend.abs(z) - threshold, 0.0)
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """L1 subgradient (sign function)."""
        backend = xp(a)
        return self.lam * backend.sign(a)
    
    @property
    def is_prox_friendly(self) -> bool:
        return True
    
    @property 
    def is_differentiable(self) -> bool:
        return False


class L2Penalty:
    """L2 penalty (Ridge regularization)."""
    
    def __init__(self, lam: float = 0.1):
        self.lam = lam
    
    def value(self, a: ArrayLike) -> float:
        """Compute L2 penalty value."""
        backend = xp(a)
        return float(0.5 * self.lam * backend.sum(a * a))
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """L2 proximal operator (shrinkage)."""
        shrinkage = 1.0 / (1.0 + self.lam * t)
        return shrinkage * z
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """L2 gradient."""
        return self.lam * a
    
    @property
    def is_prox_friendly(self) -> bool:
        return True
    
    @property
    def is_differentiable(self) -> bool:
        return True


class ElasticNetPenalty:
    """Elastic Net penalty (L1 + L2 combination)."""
    
    def __init__(self, lam: float = 0.1, l1_ratio: float = 0.5):
        self.lam = lam
        self.l1_ratio = l1_ratio
        self.l2_ratio = 1.0 - l1_ratio
    
    def value(self, a: ArrayLike) -> float:
        """Compute Elastic Net penalty value."""
        backend = xp(a)
        l1_term = backend.sum(backend.abs(a))
        l2_term = 0.5 * backend.sum(a * a)
        return float(self.lam * (self.l1_ratio * l1_term + self.l2_ratio * l2_term))
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """Elastic Net proximal operator."""
        backend = xp(z)
        
        # L2 shrinkage factor
        shrinkage = 1.0 / (1.0 + self.lam * self.l2_ratio * t)
        
        # L1 threshold
        threshold = self.lam * self.l1_ratio * t * shrinkage
        
        # Apply shrinkage then soft thresholding
        z_shrunk = shrinkage * z
        return backend.sign(z_shrunk) * backend.maximum(backend.abs(z_shrunk) - threshold, 0.0)
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """Elastic Net subgradient."""
        backend = xp(a)
        l1_subgrad = backend.sign(a)
        l2_grad = a
        return self.lam * (self.l1_ratio * l1_subgrad + self.l2_ratio * l2_grad)
    
    @property
    def is_prox_friendly(self) -> bool:
        return True
    
    @property
    def is_differentiable(self) -> bool:
        return False  # Due to L1 component