"""
Penalty function implementations for sparse coding optimization.

Based on:
- Tibshirani (1996) "Regression shrinkage and selection via the lasso"
- Zou & Hastie (2005) "Regularization and variable selection via elastic net"
- Zhang (1997) "Parameter estimation techniques: a tutorial with application"
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
from enum import Enum
from ..core.array import ArrayLike, xp
from ..core.interfaces import Penalty


class PenaltyType(Enum):
    """Available penalty function types with research references."""
    L1 = "l1"                    # Tibshirani (1996) - Lasso
    L2 = "l2"                    # Ridge regression
    ELASTIC_NET = "elastic_net"   # Zou & Hastie (2005)
    CAUCHY = "cauchy"            # Zhang (1997) - Robust penalty
    TOP_K = "top_k"              # Hard sparsity constraint


@dataclass
class PenaltyConfig:
    """
    Configuration for penalty functions in sparse coding optimization.
    
    Provides comprehensive parameter control for different regularization approaches
    including L1/L2 penalties, elastic net combinations, and robust penalty functions.
    """
    penalty_type: PenaltyType = PenaltyType.L1
    lam: float = 0.1                    # Main penalty strength
    l1_ratio: float = 0.5              # For Elastic Net: 0=pure L2, 1=pure L1
    sigma: float = 1.0                 # For Cauchy penalties
    k: int = 10                        # For Top-K constraint
    use_soft_thresholding: bool = True  # Soft vs hard thresholding
    numerical_stability_eps: float = 1e-10


class L1Penalty:
    """
    L1 penalty (Lasso) implementation
    
    Research Foundation: Tibshirani (1996) "Regression shrinkage and selection via the lasso"
    
    Mathematical Definition:
        ψ(a) = λ * ||a||₁ = λ * Σᵢ|aᵢ|
        
    Proximal Operator (soft thresholding):
        prox_{tψ}(z) = sign(z) * max(|z| - tλ, 0)
        
    Properties:
        - Non-differentiable at zero (uses subgradient)
        - Induces exact sparsity (sets coefficients to zero)
        - Convex and prox-friendly
    """
    
    def __init__(self, config: PenaltyConfig):
        self.config = config
        self.lam = config.lam
        
    def value(self, a: ArrayLike) -> float:
        """L1 penalty value: λ * ||a||₁"""
        backend = xp(a)
        return float(self.lam * backend.sum(backend.abs(a)))
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """
        L1 proximal operator: soft thresholding
        
        Formula: prox_{tλ}(z) = sign(z) * max(|z| - tλ, 0)
        
        Implements the soft-thresholding operator for L1 regularization.
        """
        backend = xp(z)
        threshold = self.lam * t
        
        if self.config.use_soft_thresholding:
            return backend.sign(z) * backend.maximum(backend.abs(z) - threshold, 0.0)
        else:
            return z * (backend.abs(z) > threshold).astype(z.dtype)
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """L1 subgradient: λ * sign(a)"""
        backend = xp(a)
        return self.lam * backend.sign(a)
    
    @property
    def is_prox_friendly(self) -> bool:
        return True
    
    @property
    def is_differentiable(self) -> bool:
        return False


class L2Penalty:
    """
    L2 penalty (Ridge) implementation
    
    Research Foundation: Ridge regression - Hoerl & Kennard (1970)
    
    Mathematical Definition:
        ψ(a) = (λ/2) * ||a||₂² = (λ/2) * Σᵢaᵢ²
        
    Proximal Operator (shrinkage):
        prox_{tψ}(z) = z / (1 + tλ)
        
    Properties:
        - Differentiable everywhere
        - Shrinks coefficients but doesn't zero them
        - Good for multicollinearity
    """
    
    def __init__(self, config: PenaltyConfig):
        self.config = config
        self.lam = config.lam
        
    def value(self, a: ArrayLike) -> float:
        """L2 penalty value: (λ/2) * ||a||₂²"""
        backend = xp(a)
        return float(0.5 * self.lam * backend.sum(a * a))
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """L2 proximal operator: shrinkage"""
        return z / (1.0 + t * self.lam)
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """L2 gradient: λ * a"""
        return self.lam * a
    
    @property
    def is_prox_friendly(self) -> bool:
        return True
    
    @property
    def is_differentiable(self) -> bool:
        return True


class ElasticNetPenalty:
    """
    Elastic Net penalty implementation
    
    Research Foundation: Zou & Hastie (2005) "Regularization and variable selection via elastic net"
    
    Mathematical Definition:
        ψ(a) = λ * [α||a||₁ + (1-α)/2 * ||a||₂²]
        
    where α is the L1 ratio parameter (l1_ratio in config)
    
    Properties:
        - Combines L1 (sparsity) and L2 (stability) penalties
        - α=1: Pure L1 (Lasso), α=0: Pure L2 (Ridge)
        - Good for grouped variable selection
    """
    
    def __init__(self, config: PenaltyConfig):
        self.config = config
        self.lam = config.lam
        self.l1_ratio = config.l1_ratio
        
    def value(self, a: ArrayLike) -> float:
        """Elastic net penalty value"""
        backend = xp(a)
        l1_term = backend.sum(backend.abs(a))
        l2_term = 0.5 * backend.sum(a * a)
        return float(self.lam * (self.l1_ratio * l1_term + (1 - self.l1_ratio) * l2_term))
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """
        Elastic net proximal operator
        
        Two-step process:
        1. Apply L1 soft thresholding: S(z, tλα)
        2. Apply L2 shrinkage: result / (1 + tλ(1-α))
        """
        backend = xp(z)
        threshold = t * self.lam * self.l1_ratio
        
        # Step 1: L1 soft thresholding
        l1_prox = backend.sign(z) * backend.maximum(backend.abs(z) - threshold, 0.0)
        
        # Step 2: L2 shrinkage
        shrinkage_factor = 1.0 + t * self.lam * (1 - self.l1_ratio)
        return l1_prox / shrinkage_factor
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """Elastic net gradient/subgradient"""
        backend = xp(a)
        l1_grad = backend.sign(a)  # Subgradient of L1
        l2_grad = a               # Gradient of L2
        return self.lam * (self.l1_ratio * l1_grad + (1 - self.l1_ratio) * l2_grad)
    
    @property
    def is_prox_friendly(self) -> bool:
        return True
    
    @property
    def is_differentiable(self) -> bool:
        return False  # Due to L1 component


class CauchyPenalty:
    """
    Cauchy penalty implementation for robust sparse coding
    
    Research Foundation: Zhang (1997) "Parameter estimation techniques"
    
    Mathematical Definition:
        ψ(a) = λ * Σᵢ log(1 + (aᵢ/σ)²)
        
    Properties:
        - Robust to outliers
        - Non-convex but differentiable
        - Requires iterative reweighted algorithms
    """
    
    def __init__(self, config: PenaltyConfig):
        self.config = config
        self.lam = config.lam
        self.sigma = config.sigma
        
    def value(self, a: ArrayLike) -> float:
        """Cauchy penalty value"""
        backend = xp(a)
        normalized = a / self.sigma
        return float(self.lam * backend.sum(backend.log(1.0 + normalized * normalized)))
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """Cauchy proximal operator (approximate using Newton-Raphson)"""
        backend = xp(z)
        
        # Approximate solution using iterative method
        x = z.copy()
        for _ in range(5):  # Newton iterations
            sigma2 = self.sigma ** 2
            factor = 2 * t * self.lam / sigma2
            denominator = 1.0 + (x / self.sigma) ** 2
            gradient = x - z + factor * x / denominator
            hessian = 1.0 + factor * (sigma2 - x ** 2) / (denominator ** 2)
            x = x - gradient / hessian
        
        return x
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """Cauchy gradient"""
        backend = xp(a)
        normalized = a / self.sigma
        denominator = 1.0 + normalized * normalized
        return self.lam * 2 * normalized / (self.sigma * denominator)
    
    @property
    def is_prox_friendly(self) -> bool:
        return False  # Requires iterative solution
    
    @property
    def is_differentiable(self) -> bool:
        return True


class TopKPenalty:
    """
    Top-K sparsity constraint implementation
    
    Enforces exact sparsity by keeping only the K largest coefficients.
    
    Properties:
        - Non-convex projection operator
        - Guarantees exact sparsity level
        - Used with greedy algorithms like OMP
    """
    
    def __init__(self, config: PenaltyConfig):
        self.config = config
        self.k = config.k
        
    def value(self, a: ArrayLike) -> float:
        """Top-K constraint value (0 if feasible, inf otherwise)"""
        backend = xp(a)
        sparsity = backend.sum(backend.abs(a) > 1e-12)
        return 0.0 if sparsity <= self.k else float('inf')
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """Top-K projection: keep only K largest coefficients"""
        backend = xp(z)
        
        # Find indices of K largest absolute values
        abs_z = backend.abs(z)
        if hasattr(backend, 'argpartition'):
            indices = backend.argpartition(-abs_z, self.k)[:self.k]
        else:
            # Fallback for backends without argpartition
            sorted_indices = backend.argsort(-abs_z)
            indices = sorted_indices[:self.k]
        
        # Create result with only top K coefficients
        result = backend.zeros_like(z)
        result[indices] = z[indices]
        return result
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """Top-K gradient (subdifferential is complex, return zero)"""
        backend = xp(a)
        return backend.zeros_like(a)
    
    @property
    def is_prox_friendly(self) -> bool:
        return True
    
    @property
    def is_differentiable(self) -> bool:
        return False