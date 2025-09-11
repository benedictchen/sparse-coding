"""
Penalty function implementations for sparse coding optimization.

This module provides research-based implementations of penalty functions
commonly used in sparse coding and compressed sensing applications.

References:
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. 
  Journal of the Royal Statistical Society: Series B, 58(1), 267-288.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm 
  for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1), 183-202.
- Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and Trends 
  in Optimization, 1(3), 127-239.
- Donoho, D. L. (2006). Compressed sensing. IEEE Transactions on Information Theory, 
  52(4), 1289-1306.

Author: Benedict Chen
"""

from dataclasses import dataclass, field
from typing import Union, Optional, Literal, Any, Dict, List
import numpy as np

# Import ArrayLike from core array module
# Note: Import path fixed to use proper module hierarchy
try:
    from ..array import ArrayLike
except ImportError as e:
    raise ImportError(f"Required ArrayLike type not found in core.array module: {e}. "
                     f"Check that core/array.py exists and is properly structured.") from e


# L1 penalty implementation

@dataclass
class L1Penalty:
    """
    L1 penalty for sparse regularization (LASSO).
    
    Research Foundation:
    - Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. 
      Journal of the Royal Statistical Society: Series B, 58(1), 267-288.
    - Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding 
      algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1), 183-202.
    
    Mathematical Formulation:
    ψ(a) = λ ||a||₁ = λ Σᵢ |aᵢ|
    
    Proximal Operator (Soft Thresholding):
    prox_{t·λ||·||₁}(z) = sign(z) ⊙ max(|z| - t·λ, 0)
    
    The L1 penalty promotes sparsity by applying a linear penalty to non-zero coefficients,
    leading to exact sparsity (setting coefficients to zero) rather than just shrinkage.
    
    Author: Benedict Chen
    """
    lam: float = 0.1
    soft_threshold_mode: Literal['standard', 'vectorized', 'numba_accelerated'] = 'vectorized'
    clipping_strategy: Literal['none', 'symmetric', 'non_negative'] = 'none'
    numerical_stability: Literal['standard', 'high_precision', 'robust'] = 'standard'
    eps: float = 1e-15  # For numerical stability
    
    def value(self, a: ArrayLike) -> float:
        """L1 penalty value: λ ||a||₁"""
        a = np.asarray(a)
        if self.numerical_stability == 'high_precision':
            return float(self.lam * np.sum(np.abs(a, dtype=np.float64)))
        elif self.numerical_stability == 'robust':
            # Avoid overflow for very large values
            abs_a = np.abs(a)
            return float(self.lam * np.sum(np.minimum(abs_a, 1e10)))
        else:
            return float(self.lam * np.sum(np.abs(a)))
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """L1 proximal operator: soft thresholding with multiple implementations"""
        z = np.asarray(z)
        threshold = t * self.lam
        
        if self.soft_threshold_mode == 'standard':
            result = self._soft_threshold_standard(z, threshold)
        elif self.soft_threshold_mode == 'vectorized':
            result = self._soft_threshold_vectorized(z, threshold)
        elif self.soft_threshold_mode == 'numba_accelerated':
            result = self._soft_threshold_numba(z, threshold)
        else:
            raise ValueError(f"Unknown soft_threshold_mode: {self.soft_threshold_mode}")
            
        # Apply clipping strategy
        if self.clipping_strategy == 'non_negative':
            result = np.maximum(result, 0.0)
        elif self.clipping_strategy == 'symmetric':
            max_val = np.max(np.abs(z)) if z.size > 0 else 1.0
            result = np.clip(result, -max_val, max_val)
            
        return result
    
    def _soft_threshold_standard(self, z: ArrayLike, threshold: float) -> ArrayLike:
        """Standard soft thresholding implementation"""
        return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)
    
    def _soft_threshold_vectorized(self, z: ArrayLike, threshold: float) -> ArrayLike:
        """Optimized vectorized soft thresholding"""
        abs_z = np.abs(z)
        mask = abs_z > threshold
        result = np.zeros_like(z)
        result[mask] = z[mask] - threshold * np.sign(z[mask])
        return result
        
    def _soft_threshold_numba(self, z: ArrayLike, threshold: float) -> ArrayLike:
        """Numba-accelerated soft thresholding (fallback to vectorized if numba unavailable)"""
        try:
            import numba as nb
            
            @nb.jit(nopython=True)
            def _soft_thresh_kernel(z_flat, threshold):
                result = np.zeros_like(z_flat)
                for i in range(len(z_flat)):
                    if z_flat[i] > threshold:
                        result[i] = z_flat[i] - threshold
                    elif z_flat[i] < -threshold:
                        result[i] = z_flat[i] + threshold
                return result
            
            z_flat = z.flatten()
            result_flat = _soft_thresh_kernel(z_flat, threshold)
            return result_flat.reshape(z.shape)
            
        except ImportError:
            # Fallback to vectorized implementation
            return self._soft_threshold_vectorized(z, threshold)
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """L1 subgradient: λ sign(a) (undefined at a=0, use 0 by convention)"""
        a = np.asarray(a)
        grad = self.lam * np.sign(a)
        
        # Handle zero values (subgradient can be any value in [-λ, λ])
        zero_mask = np.abs(a) < self.eps
        if np.any(zero_mask):
            # Use 0 as conventional choice for subgradient at 0
            grad[zero_mask] = 0.0
            
        return grad
    
    @property
    def is_prox_friendly(self) -> bool:
        """L1 has closed-form proximal operator"""
        return True
    
    @property 
    def is_differentiable(self) -> bool:
        """L1 is not differentiable at zero"""
        return False


# L2 penalty implementation

@dataclass
class L2Penalty:
    """
    L2 (Ridge) penalty with shrinkage-based proximal operator.
    
    Research Foundation: Hoerl & Kennard (1970) "Ridge Regression: Biased Estimation for Nonorthogonal Problems"
    Mathematical Form: ψ(a) = (λ/2) ||a||₂² = (λ/2) Σᵢ aᵢ²
    
    Proximal Operator: simple shrinkage
    prox_{t·(λ/2)||·||₂²}(z) = z / (1 + t·λ)
    """
    lam: float = 0.1
    numerical_stability: Literal['standard', 'high_precision', 'robust'] = 'standard'
    eps: float = 1e-15
    
    def value(self, a: ArrayLike) -> float:
        """L2 penalty value: (λ/2) ||a||₂²"""
        a = np.asarray(a)
        if self.numerical_stability == 'high_precision':
            return float(0.5 * self.lam * np.sum(a.astype(np.float64) ** 2))
        elif self.numerical_stability == 'robust':
            a_clipped = np.clip(a, -1e5, 1e5)
            return float(0.5 * self.lam * np.sum(a_clipped ** 2))
        else:
            return float(0.5 * self.lam * np.sum(a ** 2))
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """L2 proximal operator: simple shrinkage"""
        z = np.asarray(z)
        shrinkage_factor = 1.0 / (1.0 + t * self.lam)
        return shrinkage_factor * z
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """L2 gradient: λ * a"""
        a = np.asarray(a)
        return self.lam * a
    
    @property
    def is_prox_friendly(self) -> bool:
        return True
    
    @property
    def is_differentiable(self) -> bool:
        return True


@dataclass
class ElasticNetPenalty:
    """
    Elastic Net penalty combining L1 and L2 regularization.
    
    Research Foundation: Zou & Hastie (2005) "Regularization and variable selection via the elastic net"
    Mathematical Form: ψ(a) = λ₁ ||a||₁ + (λ₂/2) ||a||₂²
    
    Proximal Operator: shrinkage followed by soft thresholding
    prox_{t·ψ}(z) = sign(z̃) ⊙ max(|z̃| - t·λ₁, 0) where z̃ = z/(1 + t·λ₂)
    """
    lam: float = 0.1
    l1_ratio: float = 0.5
    numerical_stability: Literal['standard', 'high_precision', 'robust'] = 'standard'
    eps: float = 1e-15
    
    @property
    def lam_l1(self) -> float:
        return self.lam * self.l1_ratio
    
    @property
    def lam_l2(self) -> float:
        return self.lam * (1.0 - self.l1_ratio)
    
    def value(self, a: ArrayLike) -> float:
        """Elastic Net penalty value: λ₁ ||a||₁ + (λ₂/2) ||a||₂²"""
        a = np.asarray(a)
        if self.numerical_stability == 'high_precision':
            a = a.astype(np.float64)
            l1_term = self.lam_l1 * np.sum(np.abs(a))
            l2_term = 0.5 * self.lam_l2 * np.sum(a ** 2)
            return float(l1_term + l2_term)
        elif self.numerical_stability == 'robust':
            abs_a = np.abs(a)
            l1_term = self.lam_l1 * np.sum(np.minimum(abs_a, 1e10))
            a_clipped = np.clip(a, -1e5, 1e5)
            l2_term = 0.5 * self.lam_l2 * np.sum(a_clipped ** 2)
            return float(l1_term + l2_term)
        else:
            l1_term = self.lam_l1 * np.sum(np.abs(a))
            l2_term = 0.5 * self.lam_l2 * np.sum(a ** 2)
            return float(l1_term + l2_term)
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """Elastic Net proximal operator: shrinkage then soft thresholding"""
        z = np.asarray(z)
        
        shrinkage_factor = 1.0 / (1.0 + t * self.lam_l2)
        z_shrunk = shrinkage_factor * z
        
        threshold = t * self.lam_l1 * shrinkage_factor
        return np.sign(z_shrunk) * np.maximum(np.abs(z_shrunk) - threshold, 0.0)
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """Elastic Net subgradient: λ₁ sign(a) + λ₂ a"""
        a = np.asarray(a)
        l1_subgrad = np.sign(a)
        l2_grad = a
        
        zero_mask = np.abs(a) < self.eps
        if np.any(zero_mask):
            l1_subgrad[zero_mask] = 0.0
            
        return self.lam_l1 * l1_subgrad + self.lam_l2 * l2_grad
    
    @property
    def is_prox_friendly(self) -> bool:
        return True
    
    @property
    def is_differentiable(self) -> bool:
        return False


@dataclass
class TopKConstraint:
    """
    Top-K sparsity constraint enforcing exactly K nonzero elements.
    
    Research Foundation: Blumensath & Davies (2009) "Iterative hard thresholding for compressed sensing"
    Mathematical Form: ψ(a) = 0 if ||a||₀ ≤ K, ∞ otherwise
    
    Proximal Operator: hard thresholding to K largest magnitude elements
    prox_{t·ψ}(z) = HT_K(z) where HT_K keeps K largest |z_i| and zeros the rest
    """
    k: int
    tie_breaking: Literal['random', 'first', 'last'] = 'first'
    numerical_stability: Literal['standard', 'high_precision', 'robust'] = 'standard'
    eps: float = 1e-15
    
    def value(self, a: ArrayLike) -> float:
        """Top-K constraint indicator: 0 if ||a||₀ ≤ K, ∞ otherwise"""
        a = np.asarray(a)
        nonzero_count = np.sum(np.abs(a) > self.eps)
        return 0.0 if nonzero_count <= self.k else np.inf
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """Top-K proximal operator: hard thresholding to K largest elements"""
        z = np.asarray(z)
        
        if self.k >= len(z):
            return z.copy()
        
        if self.k == 0:
            return np.zeros_like(z)
        
        abs_z = np.abs(z)
        
        if self.tie_breaking == 'random':
            np.random.seed(42)
            abs_z += np.random.normal(0, self.eps * 1e-3, abs_z.shape)
        
        if self.tie_breaking == 'last':
            indices = np.argpartition(-abs_z, self.k-1)[:self.k]
        else:
            indices = np.argpartition(abs_z, -self.k)[-self.k:]
        
        result = np.zeros_like(z)
        result[indices] = z[indices]
        return result
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """
        Subgradient of Top-K constraint for optimization algorithms.
        
        Research foundation: Blumensath & Davies (2009) "Iterative hard thresholding for
        compressed sensing", Applied and Computational Harmonic Analysis, 27(3), 265-274.
        
        The Top-K constraint is non-convex and non-differentiable, but we can define
        a useful subgradient for optimization algorithms:
        - For the K largest magnitude elements: subgradient = 0 (no penalty)
        - For all other elements: subgradient = sign(a) * large_value (strong penalty)
        
        This guides optimization to maintain sparsity by heavily penalizing elements
        outside the support set. The subgradient magnitude is adaptive based on the
        current solution's scale.
        
        Args:
            a: Current sparse codes
            
        Returns:
            Subgradient approximation for optimization
        """
        a = np.asarray(a)
        
        # Find the K largest magnitude elements
        if self.k >= len(a):
            # All elements are in support, no gradient penalty
            return np.zeros_like(a)
        
        if self.k == 0:
            # No elements allowed, maximum penalty everywhere
            return np.sign(a) * 1e10
        
        # Get indices of K largest magnitude elements
        abs_a = np.abs(a)
        
        # Handle tie-breaking consistently with prox operator
        if self.tie_breaking == 'random':
            np.random.seed(42)
            abs_a_perturbed = abs_a + np.random.normal(0, self.eps * 1e-3, abs_a.shape)
            k_largest_indices = np.argpartition(abs_a_perturbed, -self.k)[-self.k:]
        elif self.tie_breaking == 'last':
            k_largest_indices = np.argpartition(-abs_a, self.k-1)[:self.k]
        else:  # 'first'
            k_largest_indices = np.argpartition(abs_a, -self.k)[-self.k:]
        
        # Create subgradient
        subgrad = np.zeros_like(a)
        
        # Set large penalty for elements outside support
        mask = np.ones(len(a), dtype=bool)
        mask[k_largest_indices] = False
        
        # Adaptive penalty magnitude based on solution scale
        penalty_magnitude = max(1e3, 100 * np.max(abs_a))
        subgrad[mask] = np.sign(a[mask]) * penalty_magnitude
        
        return subgrad
    
    @property
    def is_prox_friendly(self) -> bool:
        return False  # TopK is non-convex, should route to NCG not FISTA
    
    @property
    def is_differentiable(self) -> bool:
        return False


@dataclass  
class CauchyPenalty:
    """
    Cauchy penalty for robust sparse coding applications.
    
    Research Foundation: Dennis & Welsch (1978) "Techniques for nonlinear least squares and robust regression"
    Mathematical Form: ψ(a) = λ Σᵢ log(1 + (aᵢ/σ)²)
    
    Proximal Operator: requires iterative solution (no closed form)
    """
    lam: float = 0.1
    sigma: float = 1.0
    max_iter: int = 10
    solver: Literal['newton', 'fixed_point'] = 'newton'
    tol: float = 1e-6
    eps: float = 1e-15
    
    def value(self, a: ArrayLike) -> float:
        """Cauchy penalty: λ Σᵢ log(1 + (aᵢ/σ)²)"""
        a = np.asarray(a)
        scaled_a = a / self.sigma
        return float(self.lam * np.sum(np.log(1.0 + scaled_a ** 2)))
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """Cauchy proximal operator via iterative solution"""
        z = np.asarray(z)
        
        if self.solver == 'newton':
            return self._prox_newton(z, t)
        else:
            return self._prox_fixed_point(z, t)
    
    def _prox_newton(self, z: ArrayLike, t: float) -> ArrayLike:
        """Newton-Raphson method for Cauchy proximal operator"""
        a = z.copy()
        
        for _ in range(self.max_iter):
            grad_pen = self.grad(a)
            hess_pen = self._hessian_diag(a)
            
            f_val = a - z + t * grad_pen
            f_prime = 1.0 + t * hess_pen
            
            delta = f_val / (f_prime + self.eps)
            a_new = a - delta
            
            if np.max(np.abs(delta)) < self.tol:
                break
                
            a = a_new
            
        return a
    
    def _prox_fixed_point(self, z: ArrayLike, t: float) -> ArrayLike:
        """Fixed-point iteration for Cauchy proximal operator"""
        a = z.copy()
        
        for _ in range(self.max_iter):
            grad_pen = self.grad(a)
            a_new = z - t * grad_pen
            
            if np.max(np.abs(a_new - a)) < self.tol:
                break
                
            a = a_new
            
        return a
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """Cauchy penalty gradient: λ (2a/σ²) / (1 + (a/σ)²)"""
        a = np.asarray(a)
        scaled_a = a / self.sigma
        return self.lam * (2 * a / self.sigma**2) / (1.0 + scaled_a**2)
    
    def _hessian_diag(self, a: ArrayLike) -> ArrayLike:
        """Diagonal of Cauchy penalty Hessian for Newton method"""
        a = np.asarray(a)
        scaled_a = a / self.sigma
        scaled_a_sq = scaled_a**2
        
        numerator = 2.0 / self.sigma**2 * (1.0 - scaled_a_sq)
        denominator = (1.0 + scaled_a_sq)**2
        
        return self.lam * numerator / denominator
    
    @property
    def is_prox_friendly(self) -> bool:
        return False
    
    @property
    def is_differentiable(self) -> bool:
        return True


@dataclass
class LogSumPenalty:
    """
    Log-sum penalty for enhanced sparsity promotion.
    
    Research Foundation: Candès, Wakin & Boyd (2008) "Enhancing sparsity by reweighted l1 minimization"
    Mathematical Form: ψ(a) = λ Σᵢ log(1 + |aᵢ|/ε)
    
    This penalty enhances sparsity more aggressively than L1 by using a logarithmic
    penalty that approximates L0 norm for large values while remaining convex.
    
    Parameters:
        lam: Regularization strength λ > 0
        epsilon: Smoothing parameter ε > 0 (smaller = closer to L0)
        solver: Method for proximal operator ('iterative' or 'majorization')
        max_iter: Maximum iterations for proximal solver
        tol: Convergence tolerance
    """
    lam: float = 0.1
    epsilon: float = 0.01
    solver: Literal['iterative', 'majorization'] = 'iterative'
    max_iter: int = 20
    tol: float = 1e-6
    
    def value(self, a: ArrayLike) -> float:
        """Log-sum penalty: λ Σᵢ log(1 + |aᵢ|/ε)"""
        a = np.asarray(a)
        return float(self.lam * np.sum(np.log(1.0 + np.abs(a) / self.epsilon)))
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """
        Proximal operator for log-sum penalty.
        
        Solved iteratively using majorization-minimization or fixed-point iteration.
        Based on Candès et al. (2008) reweighted L1 approach.
        """
        z = np.asarray(z)
        
        if self.solver == 'majorization':
            return self._prox_majorization(z, t)
        else:
            return self._prox_iterative(z, t)
    
    def _prox_iterative(self, z: ArrayLike, t: float) -> ArrayLike:
        """Iterative fixed-point solution for log-sum proximal operator"""
        a = z.copy()
        
        for _ in range(self.max_iter):
            # Compute weights: w_i = 1 / (ε + |a_i|)
            weights = 1.0 / (self.epsilon + np.abs(a))
            
            # Weighted soft thresholding
            threshold = t * self.lam * weights
            a_new = np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)
            
            if np.max(np.abs(a_new - a)) < self.tol:
                break
            a = a_new
            
        return a
    
    def _prox_majorization(self, z: ArrayLike, t: float) -> ArrayLike:
        """Majorization-minimization for log-sum proximal operator"""
        a = z.copy()
        
        for _ in range(self.max_iter):
            # Majorization step: use tangent line approximation
            weights = 1.0 / (self.epsilon + np.abs(a))
            
            # Minimization step: weighted L1 proximal
            threshold = t * self.lam * weights
            a_new = np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)
            
            if np.max(np.abs(a_new - a)) < self.tol:
                break
            a = a_new
            
        return a
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """Gradient of log-sum penalty: λ sign(a) / (ε + |a|)"""
        a = np.asarray(a)
        return self.lam * np.sign(a) / (self.epsilon + np.abs(a))
    
    @property
    def is_prox_friendly(self) -> bool:
        return False  # Requires iterative solution
    
    @property
    def is_differentiable(self) -> bool:
        return False  # Non-differentiable at a=0


@dataclass
class GroupLassoPenalty:
    """
    Group LASSO penalty for structured sparsity.
    
    Research Foundation: Yuan & Lin (2006) "Model selection and estimation in regression with grouped variables"
    Mathematical Form: ψ(a) = λ Σₘ √(Σᵢ∈Gₘ aᵢ²)
    
    Promotes group-wise sparsity where entire groups of coefficients are
    selected together or set to zero together.
    
    Parameters:
        lam: Regularization strength λ > 0
        groups: List of arrays containing indices for each group
        group_weights: Optional weights for each group
        eps: Numerical stability parameter
    """
    lam: float = 0.1
    groups: Optional[List[np.ndarray]] = None
    group_weights: Optional[np.ndarray] = None
    eps: float = 1e-15
    
    def __post_init__(self):
        """Initialize group structure"""
        if self.groups is None:
            # Default: treat each element as its own group (reduces to L1)
            self.groups = []
        else:
            # Ensure groups are numpy arrays
            self.groups = [np.asarray(g) for g in self.groups]
            
        if self.group_weights is None and self.groups:
            # Default: equal weight for all groups
            self.group_weights = np.ones(len(self.groups))
        elif self.group_weights is not None:
            self.group_weights = np.asarray(self.group_weights)
    
    def value(self, a: ArrayLike) -> float:
        """Group LASSO penalty: λ Σₘ wₘ ||aₘ||₂"""
        a = np.asarray(a)
        
        if not self.groups:
            # Fallback to L1 if no groups defined
            return float(self.lam * np.sum(np.abs(a)))
        
        penalty = 0.0
        for m, group_indices in enumerate(self.groups):
            group_norm = np.linalg.norm(a[group_indices])
            weight = self.group_weights[m] if self.group_weights is not None else 1.0
            penalty += weight * group_norm
            
        return float(self.lam * penalty)
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """
        Group LASSO proximal operator: group-wise soft thresholding.
        
        For each group m: aₘ = (1 - tλwₘ/||zₘ||₂)₊ zₘ
        """
        z = np.asarray(z)
        a = np.zeros_like(z)
        
        if not self.groups:
            # Fallback to L1 soft thresholding
            threshold = t * self.lam
            return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)
        
        for m, group_indices in enumerate(self.groups):
            z_group = z[group_indices]
            group_norm = np.linalg.norm(z_group)
            
            if group_norm > self.eps:
                weight = self.group_weights[m] if self.group_weights is not None else 1.0
                threshold = t * self.lam * weight
                
                # Group-wise soft thresholding
                scale = max(0.0, 1.0 - threshold / group_norm)
                a[group_indices] = scale * z_group
                
        return a
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """Gradient of group LASSO (subgradient at zero)"""
        a = np.asarray(a)
        grad = np.zeros_like(a)
        
        if not self.groups:
            # Fallback to L1 subgradient
            return self.lam * np.sign(a)
        
        for m, group_indices in enumerate(self.groups):
            a_group = a[group_indices]
            group_norm = np.linalg.norm(a_group)
            
            if group_norm > self.eps:
                weight = self.group_weights[m] if self.group_weights is not None else 1.0
                grad[group_indices] = self.lam * weight * a_group / group_norm
            else:
                # Subgradient at zero: any vector with norm ≤ λw
                grad[group_indices] = 0.0
                
        return grad
    
    @property
    def is_prox_friendly(self) -> bool:
        return True
    
    @property
    def is_differentiable(self) -> bool:
        return False  # Non-differentiable when group norm is zero


@dataclass
class SCADPenalty:
    """
    Smoothly Clipped Absolute Deviation (SCAD) penalty.
    
    Research Foundation: Fan & Li (2001) "Variable selection via nonconcave penalized likelihood"
    Mathematical Form: 
    - For |a| ≤ λ: ψ(a) = λ|a|
    - For λ < |a| ≤ aλ: ψ(a) = -(a²-2aλ|a|+λ²)/(2(a-1))
    - For |a| > aλ: ψ(a) = (a+1)λ²/2
    
    SCAD provides unbiased estimates for large coefficients while maintaining
    sparsity for small coefficients.
    
    Parameters:
        lam: Regularization parameter λ > 0
        a: Shape parameter (typically a = 3.7 from cross-validation)
        solver: Method for proximal operator
        max_iter: Maximum iterations for iterative solver
        tol: Convergence tolerance
    """
    lam: float = 0.1
    a: float = 3.7  # Standard choice from Fan & Li (2001)
    solver: Literal['closed_form', 'iterative'] = 'closed_form'
    max_iter: int = 10
    tol: float = 1e-6
    
    def value(self, a_vec: ArrayLike) -> float:
        """SCAD penalty value with three regions"""
        a_vec = np.asarray(a_vec)
        abs_a = np.abs(a_vec)
        penalty = np.zeros_like(abs_a)
        
        # Region 1: |a| ≤ λ
        mask1 = abs_a <= self.lam
        penalty[mask1] = self.lam * abs_a[mask1]
        
        # Region 2: λ < |a| ≤ aλ
        mask2 = (abs_a > self.lam) & (abs_a <= self.a * self.lam)
        penalty[mask2] = -(abs_a[mask2]**2 - 2*self.a*self.lam*abs_a[mask2] + self.lam**2) / (2*(self.a - 1))
        
        # Region 3: |a| > aλ
        mask3 = abs_a > self.a * self.lam
        penalty[mask3] = (self.a + 1) * self.lam**2 / 2
        
        return float(np.sum(penalty))
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """
        SCAD proximal operator with closed-form solution.
        
        Based on Zou & Li (2008) "One-step sparse estimates in nonconcave penalized likelihood models"
        """
        z = np.asarray(z)
        
        if self.solver == 'closed_form':
            return self._prox_closed_form(z, t)
        else:
            return self._prox_iterative(z, t)
    
    def _prox_closed_form(self, z: ArrayLike, t: float) -> ArrayLike:
        """Closed-form SCAD proximal operator"""
        abs_z = np.abs(z)
        sign_z = np.sign(z)
        result = np.zeros_like(z)
        
        tl = t * self.lam
        
        # Region 1: Soft thresholding for |z| ≤ λ(1+t)
        mask1 = abs_z <= self.lam * (1 + t)
        result[mask1] = sign_z[mask1] * np.maximum(abs_z[mask1] - tl, 0.0)
        
        # Region 2: Modified thresholding for λ(1+t) < |z| ≤ aλ
        if self.a > 2:
            mask2 = (abs_z > self.lam * (1 + t)) & (abs_z <= self.a * self.lam)
            scale = (self.a - 1) / (self.a - 1 - t)
            result[mask2] = scale * sign_z[mask2] * np.maximum(abs_z[mask2] - tl*self.a/(self.a-1), 0.0)
        
        # Region 3: No shrinkage for |z| > aλ
        mask3 = abs_z > self.a * self.lam
        result[mask3] = z[mask3]
        
        return result
    
    def _prox_iterative(self, z: ArrayLike, t: float) -> ArrayLike:
        """Iterative solution for SCAD proximal operator"""
        a_vec = z.copy()
        
        for _ in range(self.max_iter):
            grad = self.grad(a_vec)
            a_new = z - t * grad
            
            if np.max(np.abs(a_new - a_vec)) < self.tol:
                break
            a_vec = a_new
            
        return a_vec
    
    def grad(self, a_vec: ArrayLike) -> ArrayLike:
        """SCAD gradient (subgradient at zero)"""
        a_vec = np.asarray(a_vec)
        abs_a = np.abs(a_vec)
        sign_a = np.sign(a_vec)
        grad = np.zeros_like(a_vec)
        
        # Region 1: |a| ≤ λ
        mask1 = abs_a <= self.lam
        grad[mask1] = self.lam * sign_a[mask1]
        
        # Region 2: λ < |a| ≤ aλ
        mask2 = (abs_a > self.lam) & (abs_a <= self.a * self.lam)
        grad[mask2] = sign_a[mask2] * (self.a * self.lam - abs_a[mask2]) / (self.a - 1)
        
        # Region 3: |a| > aλ (no penalty gradient)
        # grad[mask3] = 0 (already initialized to zero)
        
        return grad
    
    @property
    def is_prox_friendly(self) -> bool:
        return True
    
    @property
    def is_differentiable(self) -> bool:
        return False  # Non-differentiable at a=0 and a=λ


@dataclass
class StudentTPenalty:
    """
    Student-t (Exponential Gaussian) penalty from Olshausen & Field (1996).
    
    Research Foundation:
    Olshausen, B. A., & Field, D. J. (1996). "Emergence of simple-cell receptive field 
    properties by learning a sparse code for natural images." Nature, 381(6583), 607-609.
    
    Mathematical Form: ψ(a) = λ Σᵢ (1 - exp(-aᵢ²/σ²))
    
    This is the third sparsity function used by Olshausen & Field alongside L1 and Cauchy.
    Creates sparse solutions by penalizing coefficients with exponentially decaying weights.
    
    Args:
        lam: Regularization strength (λ > 0)
        sigma: Scale parameter (σ > 0)
    """
    lam: float = 0.1
    sigma: float = 1.0
    is_prox_friendly: bool = True
    is_differentiable: bool = True
    max_iter: int = 20
    tol: float = 1e-6
    
    def __post_init__(self):
        if self.lam <= 0:
            raise ValueError(f"Regularization strength must be positive, got {self.lam}")
        if self.sigma <= 0:
            raise ValueError(f"Sigma parameter must be positive, got {self.sigma}")
    
    def value(self, a: ArrayLike) -> float:
        """Student-t penalty: λ Σᵢ (1 - exp(-aᵢ²/σ²))"""
        a = np.asarray(a)
        scaled_a_sq = (a / self.sigma) ** 2
        return float(self.lam * np.sum(1.0 - np.exp(-scaled_a_sq)))
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """Gradient: ∂ψ/∂aᵢ = (2λ/σ²) * aᵢ * exp(-aᵢ²/σ²)"""
        a = np.asarray(a)
        scaled_a_sq = (a / self.sigma) ** 2
        exp_term = np.exp(-scaled_a_sq)
        return (2 * self.lam / (self.sigma ** 2)) * a * exp_term
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """
        Proximal operator via iterative solution.
        
        Solves: argmin_a [0.5||a - z||² + t*ψ(a)]
        where ψ(a) = λ Σᵢ (1 - exp(-aᵢ²/σ²))
        
        Uses Newton-Raphson method for each component.
        """
        z = np.asarray(z, dtype=float)
        a = z.copy()  # Initialize at input
        
        for _ in range(self.max_iter):
            # Newton step for each component
            grad_penalty = self.grad(a)
            hess_penalty = self._hessian_diag(a)
            
            # f(a) = a - z + t * grad_penalty(a)
            f_val = a - z + t * grad_penalty
            
            # f'(a) = 1 + t * hess_penalty(a)
            f_prime = 1.0 + t * hess_penalty
            
            # Newton update: a_new = a - f(a) / f'(a)
            delta = f_val / (f_prime + 1e-15)  # Avoid division by zero
            a_new = a - delta
            
            # Convergence check
            if np.max(np.abs(delta)) < self.tol:
                break
            
            a = a_new
        
        return a
    
    def _hessian_diag(self, a: ArrayLike) -> ArrayLike:
        """Diagonal of Hessian matrix."""
        a = np.asarray(a)
        scaled_a_sq = (a / self.sigma) ** 2
        exp_term = np.exp(-scaled_a_sq)
        
        # ∂²ψ/∂aᵢ² = (2λ/σ²) * exp(-aᵢ²/σ²) * (1 - 2*aᵢ²/σ²)
        return (2 * self.lam / (self.sigma ** 2)) * exp_term * (1.0 - 2 * scaled_a_sq)


@dataclass
class HuberPenalty:
    """
    Huber penalty for robust sparse coding.
    
    Combines L1 and L2 penalties: L1 for large values (sparsity), L2 for small values (smoothness).
    Penalty: ψ(a) = λ * Σ h_δ(a_i) where h_δ(t) = {0.5*t² if |t| ≤ δ, δ(|t| - 0.5*δ) if |t| > δ}
    
    Research Foundation:
    - Huber, P. J. (1964). Robust estimation of a location parameter.
    - Used in robust statistics and compressed sensing for outlier resistance.
    
    Args:
        lam: Regularization strength (λ > 0)
        delta: Huber threshold parameter (δ > 0)
    """
    lam: float = 0.1
    delta: float = 1.0
    is_prox_friendly: bool = True
    is_differentiable: bool = True
    
    def __post_init__(self):
        if self.lam <= 0:
            raise ValueError(f"Regularization strength must be positive, got {self.lam}")
        if self.delta <= 0:
            raise ValueError(f"Delta parameter must be positive, got {self.delta}")
    
    def value(self, a: ArrayLike) -> float:
        """Huber penalty value."""
        a = np.asarray(a)
        abs_a = np.abs(a)
        
        # Huber function: 0.5*t² for |t| ≤ δ, δ(|t| - 0.5*δ) for |t| > δ
        huber_vals = np.where(abs_a <= self.delta,
                              0.5 * a**2,
                              self.delta * (abs_a - 0.5 * self.delta))
        
        return self.lam * np.sum(huber_vals)
    
    def prox(self, a: ArrayLike, t: float) -> ArrayLike:
        """
        Proximal operator for Huber penalty.
        
        prox_{t*ψ}(a) has closed-form solution for Huber penalty.
        """
        a = np.asarray(a, dtype=float)
        
        # Proximal operator threshold
        tau = t * self.lam
        
        # For Huber penalty: prox is soft thresholding with modified threshold
        # If δ ≥ τ: standard soft thresholding
        # If δ < τ: modified soft thresholding
        
        if self.delta >= tau:
            # Standard soft thresholding
            return np.sign(a) * np.maximum(np.abs(a) - tau, 0.0)
        else:
            # Modified soft thresholding for Huber
            abs_a = np.abs(a)
            sign_a = np.sign(a)
            
            # Three regions based on |a|
            result = np.zeros_like(a)
            
            # Region 1: |a| ≤ τ → 0
            mask1 = abs_a <= tau
            result[mask1] = 0.0
            
            # Region 2: τ < |a| ≤ τ + δ → soft threshold then scale
            mask2 = (abs_a > tau) & (abs_a <= tau + self.delta)
            if np.any(mask2):
                shrunk = abs_a[mask2] - tau
                scale = shrunk / (1 + tau/self.delta)
                result[mask2] = sign_a[mask2] * scale
            
            # Region 3: |a| > τ + δ → different formula
            mask3 = abs_a > tau + self.delta
            if np.any(mask3):
                result[mask3] = sign_a[mask3] * (abs_a[mask3] - tau/(1 + tau/self.delta))
            
            return result


# Factory function

def create_penalty(penalty_type: str, **kwargs) -> Union[
    L1Penalty, L2Penalty, ElasticNetPenalty, TopKConstraint, CauchyPenalty,
    LogSumPenalty, GroupLassoPenalty, SCADPenalty, StudentTPenalty, HuberPenalty
]:
    """
    Factory function for creating penalties with configuration options.
    
    Args:
        penalty_type: One of 'l1', 'l2', 'elastic_net', 'top_k', 'cauchy',
                     'log_sum', 'group_lasso', 'scad'
        **kwargs: Penalty-specific configuration parameters
        
    Returns:
        Configured penalty instance
        
    Examples:
        >>> # Basic penalties
        >>> l1 = create_penalty('l1', lam=0.1, soft_threshold_mode='vectorized')
        >>> l2 = create_penalty('l2', lam=0.05, numerical_stability='high_precision')
        >>> enet = create_penalty('elastic_net', lam=0.1, l1_ratio=0.7)
        
        >>> # Constraint penalties  
        >>> top5 = create_penalty('top_k', k=5, tie_breaking='random')
        
        >>> # Robust penalties
        >>> cauchy = create_penalty('cauchy', lam=0.1, sigma=2.0, solver='newton')
        
        >>> # Advanced penalties
        >>> log_sum = create_penalty('log_sum', lam=0.1, epsilon=0.01, solver='iterative')
        >>> group = create_penalty('group_lasso', lam=0.1, groups=[[0,1,2], [3,4], [5,6,7]])
        >>> scad = create_penalty('scad', lam=0.1, a=3.7, solver='closed_form')
    """
    penalty_map = {
        'l1': L1Penalty,
        'l2': L2Penalty,
        'elastic_net': ElasticNetPenalty,
        'top_k': TopKConstraint,
        'cauchy': CauchyPenalty,
        'log_sum': LogSumPenalty,
        'group_lasso': GroupLassoPenalty,
        'scad': SCADPenalty,
        'student_t': StudentTPenalty,
        'huber': HuberPenalty,
    }
    
    if penalty_type not in penalty_map:
        raise ValueError(f"Unknown penalty type '{penalty_type}'. Available: {list(penalty_map.keys())}")
        
    return penalty_map[penalty_type](**kwargs)