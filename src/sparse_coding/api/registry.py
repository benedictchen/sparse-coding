"""
Plugin registry system for extensible sparse coding components.

Enables registration and discovery of penalties, solvers, and dictionary updaters
without import dependencies. Supports both decorator and direct registration.
"""

from typing import Dict, Any, Type, Callable, Union, Optional, List
import inspect
import warnings
from functools import wraps

# Global registry storage
_REGISTRY: Dict[str, Dict[str, Any]] = {
    "penalty": {},
    "solver": {}, 
    "dict_updater": {},
    "learner": {}
}

# Component type validation
VALID_KINDS = {"penalty", "solver", "dict_updater", "learner"}


def register(kind: str, name: str, *, override: bool = False):
    """
    Register a component in the plugin system.
    
    Can be used as decorator or called directly.
    
    Args:
        kind: Component type ('penalty', 'solver', 'dict_updater', 'learner')
        name: Unique name within the kind
        override: Whether to allow overriding existing registrations
        
    Returns:
        Decorator function or original class/function
        
    Examples:
        # Using as decorator
        @register("penalty", "l1")
        class L1Penalty:
            def __init__(self, lam: float = 0.1):
                self.lam = lam
            def prox(self, z, t):
                return np.sign(z) * np.maximum(np.abs(z) - t*self.lam, 0.0)
            def value(self, a):
                return self.lam * np.sum(np.abs(a))
        
        # Direct registration
        register("penalty", "custom", MyCustomPenalty)
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"Invalid kind '{kind}'. Must be one of: {VALID_KINDS}")
    
    def decorator(cls_or_fn: Union[Type, Callable]) -> Union[Type, Callable]:
        # Check for conflicts
        if name in _REGISTRY[kind] and not override:
            existing = _REGISTRY[kind][name]
            if existing is not cls_or_fn:  # Allow re-registration of same object
                warnings.warn(
                    f"Overriding existing {kind} '{name}': {existing} -> {cls_or_fn}. "
                    f"Use override=True to suppress this warning."
                )
        
        # Validate component interface
        _validate_component(cls_or_fn, kind)
        
        # Add metadata
        if hasattr(cls_or_fn, '__dict__'):
            cls_or_fn._sparse_coding_registry = {
                'kind': kind,
                'name': name,
                'module': cls_or_fn.__module__,
                'qualname': getattr(cls_or_fn, '__qualname__', str(cls_or_fn))
            }
        
        # Register
        _REGISTRY[kind][name] = cls_or_fn
        return cls_or_fn
    
    return decorator


def get_registry(kind: str, name: str) -> Any:
    """
    Get registered component by kind and name.
    
    Args:
        kind: Component type
        name: Component name
        
    Returns:
        Registered component class or function
        
    Raises:
        KeyError: If component not found
        ValueError: If kind invalid
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"Invalid kind '{kind}'. Must be one of: {VALID_KINDS}")
    
    if name not in _REGISTRY[kind]:
        available = list(_REGISTRY[kind].keys())
        raise KeyError(
            f"No {kind} named '{name}' found. Available: {available}"
        )
    
    return _REGISTRY[kind][name]


def list_registered(kind: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
    """
    List all registered components.
    
    Args:
        kind: Specific kind to list, or None for all
        
    Returns:
        Dict mapping kinds to component names, or list of names for specific kind
    """
    if kind is None:
        return {k: list(v.keys()) for k, v in _REGISTRY.items()}
    
    if kind not in VALID_KINDS:
        raise ValueError(f"Invalid kind '{kind}'. Must be one of: {VALID_KINDS}")
    
    return list(_REGISTRY[kind].keys())


def create_from_config(config: Dict[str, Any]) -> Any:
    """
    Create component instance from configuration.
    
    Args:
        config: Configuration dict with 'kind', 'name', and optional 'params'
        
    Returns:
        Instantiated component
        
    Example:
        config = {
            'kind': 'penalty', 
            'name': 'l1',
            'params': {'lam': 0.1}
        }
        penalty = create_from_config(config)
    """
    required_keys = {'kind', 'name'}
    if not required_keys.issubset(config.keys()):
        missing = required_keys - config.keys()
        raise ValueError(f"Config missing required keys: {missing}")
    
    kind = config['kind']
    name = config['name']
    params = config.get('params', {})
    
    # Get component class
    component_cls = get_registry(kind, name)
    
    # Instantiate with parameters
    try:
        if inspect.isclass(component_cls):
            return component_cls(**params)
        else:
            # Function - call with params
            return component_cls(**params) if params else component_cls
    except TypeError as e:
        raise TypeError(
            f"Failed to instantiate {kind} '{name}' with params {params}: {e}"
        ) from e


def unregister(kind: str, name: str) -> bool:
    """
    Remove component from registry.
    
    Args:
        kind: Component type
        name: Component name
        
    Returns:
        True if removed, False if not found
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"Invalid kind '{kind}'. Must be one of: {VALID_KINDS}")
    
    if name in _REGISTRY[kind]:
        del _REGISTRY[kind][name]
        return True
    return False


def clear_registry(kind: Optional[str] = None) -> None:
    """
    Clear registry entries.
    
    Args:
        kind: Specific kind to clear, or None for all
    """
    if kind is None:
        for k in _REGISTRY:
            _REGISTRY[k].clear()
    else:
        if kind not in VALID_KINDS:
            raise ValueError(f"Invalid kind '{kind}'. Must be one of: {VALID_KINDS}")
        _REGISTRY[kind].clear()


def get_registry_info(kind: str, name: str) -> Dict[str, Any]:
    """
    Get metadata about registered component.
    
    Args:
        kind: Component type
        name: Component name
        
    Returns:
        Dict with component metadata
    """
    component = get_registry(kind, name)
    
    info = {
        'kind': kind,
        'name': name,
        'type': type(component).__name__,
        'module': getattr(component, '__module__', 'unknown'),
        'qualname': getattr(component, '__qualname__', str(component)),
        'is_class': inspect.isclass(component),
        'is_function': inspect.isfunction(component),
    }
    
    # Add custom registry metadata if available
    if hasattr(component, '_sparse_coding_registry'):
        info.update(component._sparse_coding_registry)
    
    # Add signature info
    try:
        if inspect.isclass(component):
            sig = inspect.signature(component.__init__)
            # Remove 'self' parameter
            params = {k: v for k, v in sig.parameters.items() if k != 'self'}
        else:
            sig = inspect.signature(component)
            params = dict(sig.parameters)
        
        info['parameters'] = {
            name: {
                'kind': str(param.kind),
                'default': param.default if param.default != param.empty else None,
                'annotation': str(param.annotation) if param.annotation != param.empty else None
            }
            for name, param in params.items()
        }
    except (ValueError, TypeError):
        info['parameters'] = {}
    
    return info


def _validate_component(component: Any, kind: str) -> None:
    """
    Validate component implements expected interface.
    
    Args:
        component: Component to validate
        kind: Expected component type
    """
    # Import protocols here to avoid circular imports
    from ..core.interfaces import Penalty, InferenceSolver, DictUpdater, Learner
    
    protocol_map = {
        'penalty': Penalty,
        'solver': InferenceSolver, 
        'dict_updater': DictUpdater,
        'learner': Learner
    }
    
    if kind not in protocol_map:
        return  # Skip validation for unknown kinds
    
    protocol = protocol_map[kind]
    
    # Check required methods exist (basic duck typing validation)
    required_methods = [
        name for name, obj in inspect.getmembers(protocol)
        if not name.startswith('_') and callable(obj)
    ]
    
    missing_methods = []
    for method_name in required_methods:
        if not hasattr(component, method_name):
            # Check if it's a class that might implement the method
            if inspect.isclass(component):
                # Look for method in class definition
                if not any(hasattr(cls, method_name) for cls in component.__mro__):
                    missing_methods.append(method_name)
            else:
                missing_methods.append(method_name)
    
    if missing_methods:
        warnings.warn(
            f"{kind.title()} '{component}' may not implement required methods: {missing_methods}. "
            f"This may cause runtime errors."
        )


# Convenience aliases
get = get_registry
create = create_from_config
list_all = list_registered


# Example implementations that work with the registry
import numpy as np

class ExampleL1Penalty:
    """
    L1 penalty implementation for sparse coding (Tibshirani, 1996).
    
    Based on: Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
    Journal of the Royal Statistical Society, 58(1), 267-288.
    
    Mathematical formulation: ψ(a) = λ * ||a||₁ = λ * Σᵢ |aᵢ|
    """
    
    def __init__(self, lam: float = 0.1):
        self.lam = lam
    
    def prox(self, z, t):
        """
        L1 proximal operator (soft thresholding).
        
        Mathematical form: prox_{t·λ·||·||₁}(z) = sign(z) ⊙ max(|z| - tλ, 0)
        """
        return np.sign(z) * np.maximum(np.abs(z) - t * self.lam, 0.0)
    
    def value(self, a):
        """L1 penalty value: λ * ||a||₁"""
        return self.lam * np.sum(np.abs(a))
    
    def grad(self, a):
        """
        L1 subgradient: ∂ψ/∂a = λ * sign(a)
        
        Note: Not differentiable at zero, returns subgradient.
        """
        return self.lam * np.sign(a)
    
    def __repr__(self):
        return f"ExampleL1Penalty(lam={self.lam})"


class ExampleL2Penalty:
    """
    L2 penalty implementation for ridge regression (Hoerl & Kennard, 1970).
    
    Based on: Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: biased estimation 
    for nonorthogonal problems. Technometrics, 12(1), 55-67.
    
    Mathematical formulation: ψ(a) = (λ/2) * ||a||₂² = (λ/2) * Σᵢ aᵢ²
    """
    
    def __init__(self, lam: float = 0.1):
        self.lam = lam
    
    def prox(self, z, t):
        """
        L2 proximal operator (shrinkage).
        
        Mathematical form: prox_{t·λ·||·||₂²/2}(z) = z / (1 + tλ)
        """
        return z / (1.0 + t * self.lam)
    
    def value(self, a):
        """L2 penalty value: (λ/2) * ||a||₂²"""
        return 0.5 * self.lam * np.sum(a * a)
    
    def grad(self, a):
        """L2 gradient: ∂ψ/∂a = λ * a"""
        return self.lam * a
    
    def __repr__(self):
        return f"ExampleL2Penalty(lam={self.lam})"


class ExampleElasticNetPenalty:
    """
    Elastic Net penalty implementation (Zou & Hastie, 2005).
    
    Based on: Zou, H., & Hastie, T. (2005). Regularization and variable selection 
    via the elastic net. Journal of the Royal Statistical Society, 67(2), 301-320.
    
    Mathematical formulation: ψ(a) = λ * (α*||a||₁ + (1-α)/2*||a||₂²)
    """
    
    def __init__(self, lam: float = 0.1, l1_ratio: float = 0.5):
        self.lam = lam
        self.l1_ratio = l1_ratio
    
    def prox(self, z, t):
        """
        Elastic Net proximal operator.
        
        Two-step process:
        1. L1 soft thresholding: z' = sign(z) ⊙ max(|z| - t*λ*α, 0)
        2. L2 shrinkage: result = z' / (1 + t*λ*(1-α))
        """
        l1_prox = np.sign(z) * np.maximum(np.abs(z) - t * self.lam * self.l1_ratio, 0.0)
        return l1_prox / (1.0 + t * self.lam * (1 - self.l1_ratio))
    
    def value(self, a):
        """Elastic Net penalty value: λ * (α*||a||₁ + (1-α)/2*||a||₂²)"""
        l1_term = self.l1_ratio * np.sum(np.abs(a))
        l2_term = (1 - self.l1_ratio) * 0.5 * np.sum(a * a)
        return self.lam * (l1_term + l2_term)
    
    def grad(self, a):
        """Elastic Net gradient: ∂ψ/∂a = λ * (α*sign(a) + (1-α)*a)"""
        l1_grad = self.l1_ratio * np.sign(a)
        l2_grad = (1 - self.l1_ratio) * a
        return self.lam * (l1_grad + l2_grad)
    
    def __repr__(self):
        return f"ExampleElasticNetPenalty(lam={self.lam}, l1_ratio={self.l1_ratio})"


class ExampleCauchyPenalty:
    """
    Cauchy penalty for robust sparse coding (Nikolova, 2013).
    
    Based on: Nikolova, M. (2013). Description of the minimizers of least squares 
    regularized with ℓ0-norm. SIAM Journal on Matrix Analysis and Applications, 34(4), 1464-1484.
    
    Mathematical formulation: ψ(a) = λ * Σᵢ log(1 + (aᵢ/σ)²)
    """
    
    def __init__(self, lam: float = 0.1, sigma: float = 1.0):
        self.lam = lam
        self.sigma = sigma
    
    def prox(self, z, t):
        """
        Cauchy proximal operator (iterative solution required).
        
        Approximated using Newton's method for the proximal equation:
        x + t*λ*(2x/σ²)/(1 + (x/σ)²) = z
        """
        result = z.copy()
        for _ in range(5):  # Newton iterations
            ratio = result / self.sigma
            ratio_sq = ratio * ratio
            f = result + t * self.lam * (2 * ratio / self.sigma) / (1 + ratio_sq) - z
            df = 1 + t * self.lam * 2 / (self.sigma * self.sigma) * (1 - ratio_sq) / ((1 + ratio_sq) ** 2)
            result = result - f / (df + 1e-8)
        return result
    
    def value(self, a):
        """Cauchy penalty value: λ * Σᵢ log(1 + (aᵢ/σ)²)"""
        return self.lam * np.sum(np.log(1 + (a / self.sigma) ** 2))
    
    def grad(self, a):
        """Cauchy gradient: ∂ψ/∂a = λ * (2a/σ²) / (1 + (a/σ)²)"""
        ratio = a / self.sigma
        return self.lam * (2 * ratio / self.sigma) / (1 + ratio * ratio)
    
    def __repr__(self):
        return f"ExampleCauchyPenalty(lam={self.lam}, sigma={self.sigma})"


class ExampleFISTASolver:
    """
    FISTA solver implementation (Beck & Teboulle, 2009).
    
    Based on: Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding 
    algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1), 183-202.
    
    Implements accelerated proximal gradient method with optimal O(1/k²) convergence rate.
    """
    
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
    
    def solve(self, D, X, penalty, **kwargs):
        """
        Solve sparse coding using FISTA algorithm.
        
        Implements accelerated proximal gradient method (Beck & Teboulle, 2009):
        min_A [0.5||X - DA||_F^2 + penalty(A)]
        
        Args:
            D: Dictionary matrix (n_features, n_atoms)
            X: Data matrix (n_features, n_samples) 
            penalty: Penalty function with prox() method
            
        Returns:
            A: Sparse codes (n_atoms, n_samples)
        """
        n_atoms, n_samples = D.shape[1], X.shape[1]
        
        # Compute Lipschitz constant L = σ_max(D)^2
        L = np.linalg.norm(D, ord=2) ** 2
        step_size = 1.0 / L
        
        # Initialize variables
        A = np.zeros((n_atoms, n_samples))
        Z = A.copy()
        t = 1.0
        
        # Track convergence
        prev_objective = float('inf')
        
        for iteration in range(self.max_iter):
            A_old = A.copy()
            
            # Gradient of 0.5||X - DZ||^2 w.r.t. Z
            residual = D @ Z - X
            gradient = D.T @ residual
            
            # Gradient step
            Z_grad = Z - step_size * gradient
            
            # Proximal step
            A = penalty.prox(Z_grad, step_size)
            
            # Momentum coefficient update (FISTA)
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            
            # Momentum update
            Z = A + ((t - 1.0) / t_new) * (A - A_old)
            t = t_new
            
            # Check convergence
            if iteration % 10 == 0:  # Check every 10 iterations for efficiency
                objective = 0.5 * np.sum(residual ** 2) + penalty.value(A)
                rel_change = abs(prev_objective - objective) / (abs(prev_objective) + 1e-8)
                
                if rel_change < self.tol:
                    break
                    
                prev_objective = objective
        
        return A
    
    @property
    def name(self):
        return "fista"
    
    @property
    def supports_batch(self):
        return True
    
    def __repr__(self):
        return f"ExampleFISTASolver(max_iter={self.max_iter}, tol={self.tol})"


class ExampleMODUpdater:
    """
    Method of Optimal Directions (MOD) dictionary updater (Engan et al., 1999).
    
    Based on: Engan, K., Aase, S. O., & Husøy, J. H. (1999). Method of optimal 
    directions for frame design. In Proceedings of ICASSP (Vol. 5, pp. 2443-2446).
    
    Provides closed-form dictionary update by solving: min_D ||X - DA||_F^2
    Solution: D = XA^T(AA^T + εI)^(-1)
    """
    
    def __init__(self, eps: float = 1e-7):
        self.eps = eps  # Regularization for numerical stability
    
    def step(self, D, X, A, **kwargs):
        """
        Dictionary update using Method of Optimal Directions.
        
        Solves: min_D ||X - DA||_F^2 subject to unit norm columns
        
        Args:
            D: Current dictionary (n_features, n_atoms)
            X: Data matrix (n_features, n_samples)
            A: Current sparse codes (n_atoms, n_samples)
            
        Returns:
            Updated dictionary with normalized columns
        """
        # MOD update: D = XA^T(AA^T + εI)^(-1)
        AAt = A @ A.T + self.eps * np.eye(A.shape[0])
        D_new = X @ A.T @ np.linalg.inv(AAt)
        
        # Normalize dictionary atoms to unit norm
        norms = np.linalg.norm(D_new, axis=0)
        norms[norms < 1e-12] = 1.0  # Avoid division by zero
        D_new = D_new / norms
        
        return D_new
    
    @property
    def name(self):
        return "mod"
    
    @property
    def requires_normalization(self):
        return False  # Already normalized in step()
    
    def __repr__(self):
        return f"ExampleMODUpdater(eps={self.eps})"


class ExampleGradientDictUpdater:
    """
    Gradient descent dictionary updater (Olshausen & Field, 1996).
    
    Based on: Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell 
    receptive field properties by learning a sparse code for natural images.
    
    Updates dictionary using gradient: D ← D - η∇_D[0.5||X - DA||_F^2]
    Gradient: ∇_D = -(X - DA)A^T
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    def step(self, D, X, A, **kwargs):
        """
        Dictionary update using gradient descent.
        
        Args:
            D: Current dictionary (n_features, n_atoms)
            X: Data matrix (n_features, n_samples)
            A: Current sparse codes (n_atoms, n_samples)
            
        Returns:
            Updated dictionary
        """
        # Compute reconstruction error
        residual = X - D @ A
        
        # Gradient: ∇_D = -(X - DA)A^T
        gradient = -residual @ A.T
        
        # Gradient descent update
        D_new = D - self.learning_rate * gradient
        
        return D_new
    
    @property
    def name(self):
        return "grad_d"
    
    @property
    def requires_normalization(self):
        return True  # Need to normalize after gradient update
    
    def __repr__(self):
        return f"ExampleGradientDictUpdater(lr={self.learning_rate})"


class ExampleKSVDUpdater:
    """
    K-SVD dictionary updater (Aharon et al., 2006).
    
    Based on: Aharon, M., Elad, M., & Bruckstein, A. (2006). K-SVD: An algorithm 
    for designing overcomplete dictionaries for sparse representation.
    
    Updates dictionary atoms sequentially using SVD while adjusting sparse codes.
    """
    
    def __init__(self):
        pass
    
    def step(self, D, X, A, **kwargs):
        """
        Dictionary update using K-SVD algorithm.
        
        Args:
            D: Current dictionary (n_features, n_atoms)
            X: Data matrix (n_features, n_samples)
            A: Current sparse codes (n_atoms, n_samples)
            
        Returns:
            Updated dictionary and sparse codes
        """
        D_new = D.copy()
        A_new = A.copy()
        
        for k in range(D.shape[1]):
            # Find samples that use atom k
            active_samples = np.abs(A[k, :]) > 1e-12
            
            if not np.any(active_samples):
                continue  # Skip unused atoms
            
            # Compute error without atom k
            E_k = X[:, active_samples] - D_new @ A_new[:, active_samples] + np.outer(D_new[:, k], A_new[k, active_samples])
            
            # SVD of error matrix
            U, s, Vt = np.linalg.svd(E_k, full_matrices=False)
            
            # Update atom k and corresponding codes
            if len(s) > 0:
                D_new[:, k] = U[:, 0]  # First left singular vector
                A_new[k, active_samples] = s[0] * Vt[0, :]  # Scaled first right singular vector
        
        return D_new
    
    @property
    def name(self):
        return "ksvd"
    
    @property
    def requires_normalization(self):
        return False  # K-SVD maintains unit norms through SVD
    
    def __repr__(self):
        return "ExampleKSVDUpdater()"


# Auto-register the examples
# Register penalty implementations with research-based configurations
register("penalty", "l1")(ExampleL1Penalty)
register("penalty", "l2")(ExampleL2Penalty) 
register("penalty", "ridge")(ExampleL2Penalty)  # Alias for ridge regression
register("penalty", "elastic_net")(ExampleElasticNetPenalty)
register("penalty", "cauchy")(ExampleCauchyPenalty)

# Register solver implementations
register("solver", "fista")(ExampleFISTASolver)

# Register dictionary updater implementations  
register("dict_updater", "mod")(ExampleMODUpdater)
register("dict_updater", "grad_d")(ExampleGradientDictUpdater)
register("dict_updater", "ksvd")(ExampleKSVDUpdater)