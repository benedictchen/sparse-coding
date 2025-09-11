"""
Sparse coding optimization algorithms.

Implements multiple algorithms for solving sparse optimization problems:
- Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
- Iterative Shrinkage-Thresholding Algorithm (ISTA)
- Orthogonal Matching Pursuit (OMP)
- Nonlinear Conjugate Gradient (NCG)
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, Dict, Any, Optional, Callable, Tuple
from .array import ArrayLike, ensure_array
from .interfaces import Penalty, InferenceSolver


# SOLUTION 2: Separate Concrete Solver Classes

@dataclass
class FistaSolver:
    """
    FISTA: Fast Iterative Shrinkage-Thresholding Algorithm (Beck & Teboulle, 2009).
    
    Accelerated proximal gradient method with optimal O(1/k²) convergence rate.
    Uses Nesterov momentum for acceleration beyond standard ISTA.
    
    Reference:
    Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding 
    algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1), 183-202.
    """
    max_iter: int = 1000
    tol: float = 1e-6
    backtrack: bool = True
    backtrack_factor: float = 0.8
    adaptive_restart: bool = True
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """
        Solve: min_A [0.5||X - DA||_F^2 + penalty(A)]
        
        FISTA algorithm (Beck & Teboulle, 2009):
        1. Gradient step: z_grad = z - (1/L)∇f(z)  
        2. Proximal step: a = prox_{t·ψ}(z_grad)
        3. Momentum update: z = a + β(a - a_old) where β = (t-1)/t_new
        """
        D, X = ensure_array(D), ensure_array(X)
        n_atoms, n_samples = D.shape[1], X.shape[1]
        
        # Compute Lipschitz constant L = σ_max(D)^2
        L = np.linalg.norm(D, ord=2) ** 2
        step_size = 1.0 / L
        
        # Initialize FISTA variables
        A = np.zeros((n_atoms, n_samples))
        Z = A.copy()
        t = 1.0  # Momentum parameter
        
        for iteration in range(self.max_iter):
            A_old = A.copy()
            
            # Gradient of data fidelity term: ∇f(Z) = D^T(DZ - X)
            residual = D @ Z - X
            gradient = D.T @ residual
            
            # Gradient step with optional backtracking
            current_step = self._backtracking_line_search(D, X, Z, gradient, step_size, penalty) if self.backtrack else step_size
            Z_grad = Z - current_step * gradient
            
            # Proximal step: A = prox_{t·ψ}(Z_grad)
            A = penalty.prox(Z_grad, current_step)
            
            # FISTA momentum coefficient: t_new = (1 + √(1 + 4t²))/2
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            beta = (t - 1.0) / t_new
            
            # Adaptive restart (O'Donoghue & Candès, 2015)
            if self.adaptive_restart and np.sum((A - A_old) * (Z - A_old)) < 0:
                t = t_new = 1.0
                beta = 0.0
            
            # Momentum update: Z = A + β(A - A_old)
            Z = A + beta * (A - A_old)
            t = t_new
            
            # Convergence check
            if np.linalg.norm(A - A_old) < self.tol * max(1.0, np.linalg.norm(A)):
                break
        
        return A
    
    def _backtracking_line_search(self, D: ArrayLike, X: ArrayLike, z: ArrayLike, 
                                  grad: ArrayLike, step_size: float, penalty: Penalty) -> float:
        """Backtracking line search for step size selection."""
        while step_size > 1e-12:
            z_grad = z - step_size * grad
            a_candidate = penalty.prox(z_grad, step_size)
            
            # Check sufficient decrease condition
            residual_candidate = D @ a_candidate - X
            obj_candidate = 0.5 * np.sum(residual_candidate**2) + penalty.value(a_candidate)
            
            residual_z = D @ z - X  
            quadratic_approx = 0.5 * np.sum(residual_z**2) + np.sum(grad * (a_candidate - z))
            
            if obj_candidate <= quadratic_approx + 0.5 / step_size * np.sum((a_candidate - z)**2):
                break
            
            step_size *= self.backtrack_factor
        
        return max(step_size, 1e-12)
    
    @property
    def name(self) -> str:
        return "fista"
    
    @property 
    def supports_batch(self) -> bool:
        return True


@dataclass  
class IstaSolver:
    """
    ISTA: Basic Iterative Shrinkage-Thresholding Algorithm.
    
    Standard proximal gradient method without acceleration.
    Simpler than FISTA but with O(1/k) convergence rate.
    
    Reference:
    Daubechies, I., et al. (2004). An iterative thresholding algorithm 
    for linear inverse problems. Communications on Pure and Applied Mathematics, 57(11), 1413-1457.
    """
    max_iter: int = 1000
    tol: float = 1e-6
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """
        Solve: min_A [0.5||X - DA||_F^2 + penalty(A)]
        
        ISTA algorithm:
        1. Gradient step: a_grad = a - (1/L)∇f(a)
        2. Proximal step: a = prox_{t·ψ}(a_grad)
        """
        D, X = ensure_array(D), ensure_array(X)
        n_atoms, n_samples = D.shape[1], X.shape[1]
        
        # Compute Lipschitz constant
        L = np.linalg.norm(D, ord=2) ** 2
        step_size = 1.0 / L
        
        A = np.zeros((n_atoms, n_samples))
        
        for iteration in range(self.max_iter):
            A_old = A.copy()
            
            # Gradient step
            residual = D @ A - X
            gradient = D.T @ residual
            A_grad = A - step_size * gradient
            
            # Proximal step
            A = penalty.prox(A_grad, step_size)
            
            # Convergence check
            if np.linalg.norm(A - A_old) < self.tol * max(1.0, np.linalg.norm(A)):
                break
        
        return A
    
    @property
    def name(self) -> str:
        return "ista"
    
    @property
    def supports_batch(self) -> bool:
        return True


@dataclass
class OmpSolver:
    """
    OMP: Orthogonal Matching Pursuit (Tropp & Gilbert, 2007).
    
    Greedy algorithm for sparse approximation. Selects dictionary atoms
    one by one based on correlation with residual.
    
    Reference:
    Tropp, J. A., & Gilbert, A. C. (2007). Signal recovery from random 
    measurements via orthogonal matching pursuit. IEEE Transactions on Information Theory, 53(12), 4655-4666.
    """
    sparsity_level: int = 10
    tol: float = 1e-10
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """
        OMP algorithm (Tropp & Gilbert, 2007):
        1. Find atom with highest correlation to residual
        2. Add to active set and solve least squares
        3. Update residual and repeat
        """
        D, X = ensure_array(D), ensure_array(X)
        n_atoms, n_samples = D.shape[1], X.shape[1]
        A = np.zeros((n_atoms, n_samples))
        
        for sample_idx in range(n_samples):
            x = X[:, sample_idx]
            residual = x.copy()
            active_set = []
            
            for _ in range(min(self.sparsity_level, n_atoms)):
                # Find atom with maximum correlation
                correlations = np.abs(D.T @ residual)
                best_atom = np.argmax(correlations)
                
                if best_atom in active_set or correlations[best_atom] < self.tol:
                    break
                
                active_set.append(best_atom)
                
                # Least squares solution on active set
                D_active = D[:, active_set]
                try:
                    coeffs = np.linalg.lstsq(D_active, x, rcond=None)[0]
                    A[active_set, sample_idx] = coeffs
                    residual = x - D_active @ coeffs
                except np.linalg.LinAlgError:
                    break
                
                # Check residual norm
                if np.linalg.norm(residual) < self.tol:
                    break
        
        return A
    
    @property
    def name(self) -> str:
        return "omp"
    
    @property
    def supports_batch(self) -> bool:
        return False  # OMP typically processes samples individually


@dataclass
class NcgSolver:
    """
    NCG: Nonlinear Conjugate Gradient for smooth penalties.
    
    Uses conjugate gradient for penalties that are differentiable.
    More efficient than proximal methods for smooth objectives.
    
    Reference:
    Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer.
    """
    max_iter: int = 1000
    tol: float = 1e-6
    line_search_max_iter: int = 20
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """
        NCG for smooth penalties using Polak-Ribière formula.
        
        Only works when penalty.is_differentiable is True.
        """
        if not penalty.is_differentiable:
            raise ValueError("NCG solver requires differentiable penalty")
        
        D, X = ensure_array(D), ensure_array(X)
        n_atoms, n_samples = D.shape[1], X.shape[1]
        A = np.zeros((n_atoms, n_samples))
        
        # Process each sample separately for nonlinear CG
        for sample_idx in range(n_samples):
            x = X[:, sample_idx]
            a = np.zeros(n_atoms)
            
            # Initial gradient
            residual = D @ a - x
            grad_data = D.T @ residual
            grad_penalty = penalty.grad(a)
            grad = grad_data + grad_penalty
            direction = -grad
            
            for iteration in range(self.max_iter):
                # Line search
                alpha = self._armijo_line_search(D, x, a, direction, penalty)
                
                a_old = a.copy()
                grad_old = grad.copy()
                a = a + alpha * direction
                
                # New gradient
                residual = D @ a - x
                grad_data = D.T @ residual  
                grad_penalty = penalty.grad(a)
                grad = grad_data + grad_penalty
                
                # Convergence check
                if np.linalg.norm(a - a_old) < self.tol * max(1.0, np.linalg.norm(a)):
                    break
                
                # Polak-Ribière beta
                y = grad - grad_old
                beta = max(0.0, np.dot(grad, y) / (np.dot(grad_old, grad_old) + 1e-12))
                direction = -grad + beta * direction
            
            A[:, sample_idx] = a
        
        return A
    
    def _armijo_line_search(self, D: ArrayLike, x: ArrayLike, a: ArrayLike, 
                           direction: ArrayLike, penalty: Penalty) -> float:
        """Armijo line search for step size."""
        alpha = 1.0
        c1 = 1e-4  # Armijo parameter
        
        # Current objective
        residual = D @ a - x
        obj_current = 0.5 * np.dot(residual, residual) + penalty.value(a)
        
        # Directional derivative
        grad_data = D.T @ residual
        grad_penalty = penalty.grad(a)
        grad_total = grad_data + grad_penalty
        directional_deriv = np.dot(grad_total, direction)
        
        for _ in range(self.line_search_max_iter):
            a_new = a + alpha * direction
            residual_new = D @ a_new - x
            obj_new = 0.5 * np.dot(residual_new, residual_new) + penalty.value(a_new)
            
            # Armijo condition
            if obj_new <= obj_current + c1 * alpha * directional_deriv:
                break
            
            alpha *= 0.5
        
        return max(alpha, 1e-8)
    
    @property
    def name(self) -> str:
        return "ncg"
    
    @property
    def supports_batch(self) -> bool:
        return False  # NCG processes samples individually for nonlinear problems


# SOLUTION 3: Solver Factory Pattern
class SolverFactory:
    """Factory for creating solver instances with configuration."""
    
    _solvers = {
        'fista': FistaSolver,
        'ista': IstaSolver, 
        'omp': OmpSolver,
        'ncg': NcgSolver
    }
    
    @staticmethod
    def create_solver(algorithm: str, **kwargs) -> InferenceSolver:
        """Create solver instance by name."""
        if algorithm not in SolverFactory._solvers:
            available = list(SolverFactory._solvers.keys())
            raise ValueError(f"Unknown solver '{algorithm}'. Available: {available}")
        
        solver_cls = SolverFactory._solvers[algorithm]
        return solver_cls(**kwargs)
    
    @staticmethod
    def list_available() -> list:
        """List available solver algorithms."""
        return list(SolverFactory._solvers.keys())
    
    @staticmethod
    def register_solver(name: str, solver_cls: type):
        """Register new solver type."""
        SolverFactory._solvers[name] = solver_cls


# SOLUTION 4: Registry-Based Solver System  
class SolverRegistry:
    """Registry-based solver management system."""
    
    def __init__(self):
        self._registry = {}
        self._auto_selection_rules = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default solvers."""
        self.register('fista', FistaSolver())
        self.register('ista', IstaSolver())
        self.register('omp', OmpSolver())  
        self.register('ncg', NcgSolver())
        
        # Auto-selection rules based on penalty properties
        self._auto_selection_rules = {
            (True, False): 'fista',   # prox_friendly=True, differentiable=False (L1, ElasticNet)
            (True, True): 'fista',    # prox_friendly=True, differentiable=True (L2) 
            (False, True): 'ncg',     # prox_friendly=False, differentiable=True (Cauchy)
            (False, False): 'fista'   # fallback
        }
    
    def register(self, name: str, solver: InferenceSolver):
        """Register solver instance."""
        self._registry[name] = solver
    
    def get_solver(self, algorithm: str) -> InferenceSolver:
        """Get solver by name."""
        if algorithm not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(f"Solver '{algorithm}' not found. Available: {available}")
        return self._registry[algorithm]
    
    def auto_select_solver(self, penalty: Penalty) -> InferenceSolver:
        """Automatically select best solver for penalty type."""
        key = (penalty.is_prox_friendly, penalty.is_differentiable)
        algorithm = self._auto_selection_rules.get(key, 'fista')
        return self.get_solver(algorithm)
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, 
              algorithm: str = 'auto', **kwargs) -> ArrayLike:
        """Solve using registry-managed solver."""
        if algorithm == 'auto':
            solver = self.auto_select_solver(penalty)
        else:
            solver = self.get_solver(algorithm)
        
        return solver.solve(D, X, penalty, **kwargs)


# Global registry instance
SOLVER_REGISTRY = SolverRegistry()


# Configuration system for overlapping solutions  
@dataclass
class SolverConfig:
    """Configuration for solver creation with solution selection."""
    
    # Core solver parameters
    algorithm: str = 'fista'  # 'fista', 'ista', 'omp', 'ncg', 'auto'
    max_iter: int = 1000
    tol: float = 1e-6
    
    # FISTA specific
    backtrack: bool = True
    adaptive_restart: bool = True
    
    # OMP specific  
    sparsity_level: int = 10
    
    # Solution pattern selection
    use_factory: bool = False      # Solution 3: Factory pattern
    use_registry: bool = True      # Solution 4: Registry system (default)
    enable_auto_selection: bool = True  # Auto-select based on penalty
    
    
def create_solver(config: SolverConfig) -> InferenceSolver:
    """
    Create solver with configurable solution patterns.
    
    Allows users to choose between factory pattern and registry system.
    """
    # Extract algorithm-specific parameters
    solver_kwargs = {}
    if config.algorithm in ['fista', 'ista']:
        solver_kwargs.update({
            'max_iter': config.max_iter,
            'tol': config.tol
        })
        if config.algorithm == 'fista':
            solver_kwargs.update({
                'backtrack': config.backtrack,
                'adaptive_restart': config.adaptive_restart
            })
    elif config.algorithm == 'omp':
        solver_kwargs.update({
            'sparsity_level': config.sparsity_level,
            'tol': config.tol
        })
    elif config.algorithm == 'ncg':
        solver_kwargs.update({
            'max_iter': config.max_iter,
            'tol': config.tol
        })
    
    # Create solver using selected pattern
    if config.use_factory:
        # Solution 3: Factory pattern
        return SolverFactory.create_solver(config.algorithm, **solver_kwargs)
    elif config.use_registry:
        # Solution 4: Registry system  
        if config.algorithm == 'auto' and not config.enable_auto_selection:
            config.algorithm = 'fista'  # fallback
        
        # Create new solver instance with parameters
        if config.algorithm != 'auto':
            solver = SolverFactory.create_solver(config.algorithm, **solver_kwargs)
            SOLVER_REGISTRY.register(f'{config.algorithm}_configured', solver)
            return solver
        else:
            # Auto-selection will happen at solve time
            return SOLVER_REGISTRY
    else:
        # Direct instantiation (Solution 2)
        return SolverFactory.create_solver(config.algorithm, **solver_kwargs)