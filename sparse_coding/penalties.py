from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Penalty:
    """
    Abstract base class for penalty functions in sparse coding.
    
    All penalty functions must implement the proximal operator (prox)
    and value function for optimization algorithms.
    """
    
    def prox(self, z: np.ndarray, t: float) -> np.ndarray: 
        """
        Proximal operator: prox_t(z) = argmin_x (0.5 * ||x - z||^2 + t * penalty(x))
        
        Parameters
        ----------
        z : np.ndarray
            Input point
        t : float
            Step size parameter
            
        Returns
        -------
        x : np.ndarray
            Solution of proximal operator
            
        Examples
        --------
        L1 soft thresholding: np.sign(z) * np.maximum(np.abs(z) - t*lam, 0.0)
        """
        raise NotImplementedError("Subclasses must implement prox()")
    
    def value(self, a: np.ndarray) -> float: 
        """
        Evaluate penalty function at point a.
        
        Parameters
        ----------
        a : np.ndarray
            Point to evaluate
            
        Returns
        -------
        penalty_value : float
            Value of penalty function
            
        Examples
        --------
        L1 penalty: lam * np.sum(np.abs(a))
        """
        raise NotImplementedError("Subclasses must implement value()")

@dataclass
class L1(Penalty):
    lam: float
    def prox(self, z, t): return np.sign(z) * np.maximum(np.abs(z) - t*self.lam, 0.0)
    def value(self, a): return self.lam * float(np.sum(np.abs(a)))

@dataclass
class ElasticNet(Penalty):
    l1: float; l2: float
    def prox(self, z, t): return (np.sign(z) * np.maximum(np.abs(z) - t*self.l1, 0.0)) / (1.0 + t*self.l2)
    def value(self, a): return self.l1 * float(np.sum(np.abs(a))) + 0.5*self.l2 * float(np.sum(a*a))

@dataclass
class NonNegL1(Penalty):
    lam: float
    def prox(self, z, t): return np.maximum(z - t*self.lam, 0.0)
    def value(self, a): return self.lam * float(np.sum(a)) if np.all(a>=0) else np.inf


@dataclass
class L2(Penalty):
    """L2 (Ridge) regularization penalty.
    
    Reference: Hoerl & Kennard (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems.
    Provides smooth penalty that shrinks coefficients but doesn't enforce sparsity.
    """
    lam: float
    
    def prox(self, z: np.ndarray, t: float) -> np.ndarray:
        """Proximal operator for L2 penalty: shrinkage operator."""
        return z / (1.0 + t * self.lam)
    
    def value(self, a: np.ndarray) -> float:
        """L2 penalty value: λ * 0.5 * ||a||²"""
        return self.lam * 0.5 * float(np.sum(a * a))
    
    def grad(self, a: np.ndarray) -> np.ndarray:
        """Gradient of L2 penalty: λ * a"""
        return self.lam * a
    
    @property
    def is_differentiable(self) -> bool:
        return True


@dataclass  
class CauchyPenalty(Penalty):
    """Cauchy penalty for robust sparse coding.
    
    Reference: Dennis & Welsch (1978). Techniques for nonlinear least squares and robust regression.
    More robust to outliers than L1, promotes sparsity while handling large coefficients gracefully.
    """
    lam: float
    sigma: float = 1.0
    
    def prox(self, z: np.ndarray, t: float) -> np.ndarray:
        """Proximal operator for Cauchy penalty (no closed form - iterative solution)."""
        # Iterative proximal solution for Cauchy penalty
        a = z.copy()
        for _ in range(10):  # Fixed iterations for efficiency
            grad_penalty = self.grad(a)
            a = z - t * grad_penalty
        return a
    
    def value(self, a: np.ndarray) -> float:
        """Cauchy penalty: λ * Σ log(1 + (a/σ)²)"""
        return self.lam * float(np.sum(np.log(1 + (a / self.sigma) ** 2)))
    
    def grad(self, a: np.ndarray) -> np.ndarray:
        """Gradient of Cauchy penalty: λ * (2a/σ²) / (1 + (a/σ)²)"""
        return self.lam * (2 * a / self.sigma**2) / (1 + (a / self.sigma)**2)
    
    @property
    def is_differentiable(self) -> bool:
        return True
    
    @property
    def is_prox_friendly(self) -> bool:
        return False  # Requires iterative prox computation


@dataclass
class TopKConstraint(Penalty):
    """Top-K sparsity constraint penalty.
    
    Reference: Blumensath & Davies (2009). Iterative hard thresholding for compressed sensing.
    Enforces exact sparsity by keeping only K largest coefficients.
    """
    k: int
    
    def prox(self, z: np.ndarray, t: float) -> np.ndarray:
        """Top-K proximal operator: hard thresholding to K largest elements."""
        if self.k >= len(z):
            return z
        
        # Find indices of K largest absolute values
        indices = np.argpartition(np.abs(z), -self.k)[-self.k:]
        result = np.zeros_like(z)
        result[indices] = z[indices]
        return result
    
    def value(self, a: np.ndarray) -> float:
        """Constraint indicator: 0 if ||a||₀ ≤ K, ∞ otherwise."""
        return 0.0 if np.sum(np.abs(a) > 1e-10) <= self.k else np.inf
    
    def grad(self, a: np.ndarray) -> np.ndarray:
        """Constraint has no meaningful gradient (non-convex)."""
        raise NotImplementedError("TopK constraint is non-differentiable")
    
    @property
    def is_differentiable(self) -> bool:
        return False
    
    @property
    def is_prox_friendly(self) -> bool:
        return True  # Efficient sorting-based prox


@dataclass
class HuberPenalty(Penalty):
    """Huber penalty - hybrid L1/L2 for robustness.
    
    Reference: Huber (1964). Robust Estimation of a Location Parameter.
    Quadratic for small values (like L2), linear for large values (like L1).
    """
    lam: float
    delta: float = 1.0
    
    def prox(self, z: np.ndarray, t: float) -> np.ndarray:
        """Huber proximal operator."""
        threshold = t * self.lam
        # Soft thresholding for |z| > δ + threshold
        # Shrinkage for |z| ≤ δ + threshold
        abs_z = np.abs(z)
        result = np.zeros_like(z)
        
        # Linear region: soft thresholding
        linear_mask = abs_z > self.delta + threshold
        result[linear_mask] = np.sign(z[linear_mask]) * np.maximum(
            abs_z[linear_mask] - threshold, 0.0)
        
        # Quadratic region: shrinkage
        quad_mask = ~linear_mask
        result[quad_mask] = z[quad_mask] / (1.0 + t * self.lam / self.delta)
        
        return result
    
    def value(self, a: np.ndarray) -> float:
        """Huber penalty value."""
        abs_a = np.abs(a)
        quad_part = np.sum(0.5 * abs_a**2 / self.delta * (abs_a <= self.delta))
        linear_part = np.sum((abs_a - 0.5 * self.delta) * (abs_a > self.delta))
        return self.lam * (quad_part + linear_part)
    
    def grad(self, a: np.ndarray) -> np.ndarray:
        """Huber penalty gradient."""
        abs_a = np.abs(a)
        grad = np.zeros_like(a)
        
        # Quadratic region
        quad_mask = abs_a <= self.delta
        grad[quad_mask] = self.lam * a[quad_mask] / self.delta
        
        # Linear region  
        linear_mask = abs_a > self.delta
        grad[linear_mask] = self.lam * np.sign(a[linear_mask])
        
        return grad
    
    @property
    def is_differentiable(self) -> bool:
        return False  # Not differentiable at boundaries


@dataclass
class GroupLasso(Penalty):
    """Group Lasso penalty for structured sparsity.
    
    Reference: Yuan & Lin (2006). Model selection and estimation in regression with grouped variables.
    Promotes sparsity at the group level while allowing within-group density.
    """
    lam: float
    groups: List[List[int]]  # List of index groups
    
    def __post_init__(self):
        # Precompute group assignments for efficiency
        self.group_map = {}
        for group_id, indices in enumerate(self.groups):
            for idx in indices:
                self.group_map[idx] = group_id
    
    def prox(self, z: np.ndarray, t: float) -> np.ndarray:
        """Group Lasso proximal operator: group-wise soft thresholding."""
        result = z.copy()
        threshold = t * self.lam
        
        for group_indices in self.groups:
            group_z = z[group_indices]
            group_norm = np.linalg.norm(group_z)
            
            if group_norm > threshold:
                # Scale entire group
                scale = (group_norm - threshold) / group_norm
                result[group_indices] = scale * group_z
            else:
                # Zero entire group
                result[group_indices] = 0.0
        
        return result
    
    def value(self, a: np.ndarray) -> float:
        """Group Lasso penalty: λ * Σ ||a_g||₂"""
        penalty = 0.0
        for group_indices in self.groups:
            penalty += np.linalg.norm(a[group_indices])
        return self.lam * penalty
    
    def grad(self, a: np.ndarray) -> np.ndarray:
        """Group Lasso subgradient."""
        grad = np.zeros_like(a)
        
        for group_indices in self.groups:
            group_a = a[group_indices]
            group_norm = np.linalg.norm(group_a)
            
            if group_norm > 1e-10:  # Avoid division by zero
                grad[group_indices] = self.lam * group_a / group_norm
            # else: subgradient is any vector with norm ≤ λ (set to 0)
        
        return grad
    
    @property
    def is_differentiable(self) -> bool:
        return False  # Not differentiable when group norm = 0
