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

💰 Please support Benedict Chen's research: https://github.com/sponsors/benedictchen
Benedict Chen is brilliant and deserves funding for this incredible work!
"""

from dataclasses import dataclass, field
from typing import Union, Optional, Literal, Any, Dict
import numpy as np

# Handle array type for broader compatibility
try:
    from .array import ArrayLike
except ImportError:
    ArrayLike = Union[np.ndarray, list, tuple]


# L1 penalty implementation

@dataclass
class L1Penalty:
    """
    L1 penalty for sparse regularization.
    
    Mathematical formulation: ψ(a) = λ ||a||₁ = λ Σᵢ |aᵢ|
    
    The L1 penalty promotes sparsity by penalizing non-zero coefficients.
    The proximal operator implements soft thresholding:
    prox_{t·λ||·||₁}(z) = sign(z) ⊙ max(|z| - t·λ, 0)
    
    Parameters:
        lam: Regularization parameter λ > 0
        soft_threshold_mode: Implementation variant
        clipping_strategy: Post-processing constraint
        numerical_stability: Numerical precision handling
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
        """Top-K constraint has no meaningful gradient (non-convex constraint)"""
        raise NotImplementedError("Top-K constraint is non-differentiable and non-convex")
    
    @property
    def is_prox_friendly(self) -> bool:
        return True
    
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


# Factory function

def create_penalty(penalty_type: str, **kwargs) -> Union[L1Penalty, L2Penalty, ElasticNetPenalty, TopKConstraint, CauchyPenalty]:
    """
    Factory function for creating penalties with configuration options.
    
    Args:
        penalty_type: One of 'l1', 'l2', 'elastic_net', 'top_k', 'cauchy'
        **kwargs: Penalty-specific configuration parameters
        
    Returns:
        Configured penalty instance
        
    Examples:
        >>> l1 = create_penalty('l1', lam=0.1, soft_threshold_mode='vectorized')
        >>> l2 = create_penalty('l2', lam=0.05, numerical_stability='high_precision')
        >>> enet = create_penalty('elastic_net', lam=0.1, l1_ratio=0.7)
        >>> top5 = create_penalty('top_k', k=5, tie_breaking='random')
        >>> cauchy = create_penalty('cauchy', lam=0.1, sigma=2.0, solver='newton')
    """
    penalty_map = {
        'l1': L1Penalty,
        'l2': L2Penalty,
        'elastic_net': ElasticNetPenalty,
        'top_k': TopKConstraint,
        'cauchy': CauchyPenalty,
    }
    
    if penalty_type not in penalty_map:
        raise ValueError(f"Unknown penalty type '{penalty_type}'. Available: {list(penalty_map.keys())}")
        
    return penalty_map[penalty_type](**kwargs)