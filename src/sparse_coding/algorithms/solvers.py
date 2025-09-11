"""
Sparse coding solver implementations.

Based on:
- Beck & Teboulle (2009) "A Fast Iterative Shrinkage-Thresholding Algorithm"
- Pati et al. (1993) "Orthogonal matching pursuit: recursive function approximation"
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
from enum import Enum
from ..core.array import ArrayLike, xp
from ..core.interfaces import InferenceSolver, Penalty


class SolverType(Enum):
    """Available solver types with research references."""
    FISTA = "fista"      # Beck & Teboulle (2009) - Fast ISTA
    ISTA = "ista"        # Basic proximal gradient
    OMP = "omp"          # Pati et al. (1993) - Orthogonal matching pursuit


@dataclass
class SolverConfig:
    """
    Configuration parameters for sparse coding inference algorithms.
    
    Controls algorithm selection and convergence behavior for different
    optimization approaches including proximal methods and greedy algorithms.
    """
    solver_type: SolverType = SolverType.FISTA
    max_iter: int = 100
    tol: float = 1e-6
    line_search: bool = False
    backtrack_factor: float = 0.8
    initial_step_size: float = 1.0
    adaptive_restart: bool = True  # For FISTA
    verbose: bool = False


class FISTASolver:
    """
    Fast Iterative Shrinkage-Thresholding Algorithm
    
    Research Foundation: Beck & Teboulle (2009) "A Fast Iterative Shrinkage-Thresholding Algorithm"
    
    Algorithm:
    ```
    1. Initialize: a₀ = 0, y₀ = a₀, t₀ = 1
    2. For k = 0, 1, 2, ...:
        a. Compute gradient: ∇f(yₖ) = Dᵀ(Dyₖ - x)
        b. Proximal step: aₖ₊₁ = proxψ(yₖ - (1/L)∇f(yₖ))
        c. Update momentum: tₖ₊₁ = (1 + √(1 + 4tₖ²))/2
        d. Update iterate: yₖ₊₁ = aₖ₊₁ + ((tₖ-1)/tₖ₊₁)(aₖ₊₁ - aₖ)
    ```
    
    Properties:
    - O(1/k²) convergence rate (vs O(1/k) for ISTA)
    - Requires prox-friendly penalties
    - Uses Nesterov acceleration
    """
    
    def __init__(self, config: SolverConfig):
        self.config = config
        self.max_iter = config.max_iter
        self.tol = config.tol
        
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """
        Solve sparse coding using FISTA
        
        Args:
            D: Dictionary matrix (n_features, n_atoms)
            X: Data matrix (n_features, n_samples)
            penalty: Penalty function (must be prox-friendly)
            
        Returns:
            Sparse codes A (n_atoms, n_samples)
        """
        if not penalty.is_prox_friendly:
            raise ValueError("FISTA requires prox-friendly penalty")
        
        backend = xp(D)
        n_atoms, n_samples = D.shape[1], X.shape[1]
        
        # Handle single sample
        if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
            return self._solve_single(D, X.ravel(), penalty, backend)
        
        # Batch processing
        A = backend.zeros((n_atoms, n_samples))
        for i in range(n_samples):
            A[:, i] = self._solve_single(D, X[:, i], penalty, backend)
        
        return A
    
    def _solve_single(self, D: ArrayLike, x: ArrayLike, penalty: Penalty, backend) -> ArrayLike:
        """Solve single FISTA problem with all optimizations"""
        n_atoms = D.shape[1]
        
        # Estimate Lipschitz constant (spectral norm squared)
        L = float(backend.linalg.norm(D, ord=2) ** 2)
        
        # Initialize FISTA variables
        a = backend.zeros(n_atoms)  # Current iterate
        y = a.copy()               # Momentum point
        t = 1.0                    # Momentum parameter
        
        prev_objective = float('inf')
        
        for iteration in range(self.max_iter):
            # Compute gradient at momentum point: ∇f(y) = D^T(Dy - x)
            residual = D @ y - x
            grad = D.T @ residual
            
            # Proximal gradient step: a_{k+1} = prox_{t*penalty}(y - (1/L)*grad)
            a_new = penalty.prox(y - grad / L, 1.0 / L)
            
            # Compute objective for convergence check
            data_fit = 0.5 * backend.sum(residual * residual)
            penalty_val = penalty.value(a_new)
            objective = float(data_fit + penalty_val)
            
            # Check convergence
            if abs(objective - prev_objective) < self.tol:
                if self.config.verbose:
                    print(f"FISTA converged at iteration {iteration}")
                break
            
            # FISTA momentum update
            t_new = 0.5 * (1.0 + backend.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_new
            y = a_new + beta * (a_new - a)
            
            # Adaptive restart (Beck & Teboulle 2009)
            if self.config.adaptive_restart:
                if backend.dot(a_new - a, y - a_new) > 0:
                    y = a_new
                    t_new = 1.0
            
            # Update for next iteration
            a = a_new
            t = t_new
            prev_objective = objective
        
        return a
    
    @property
    def name(self) -> str:
        return "fista"
    
    @property
    def supports_batch(self) -> bool:
        return True


class ISTASolver:
    """
    Iterative Shrinkage-Thresholding Algorithm
    
    Research Foundation: Basic proximal gradient method
    
    Algorithm:
    ```
    1. Initialize: a₀ = 0
    2. For k = 0, 1, 2, ...:
        a. Compute gradient: ∇f(aₖ) = Dᵀ(Daₖ - x)
        b. Proximal step: aₖ₊₁ = proxψ(aₖ - (1/L)∇f(aₖ))
    ```
    
    Properties:
    - O(1/k) convergence rate
    - Simpler than FISTA, no momentum
    - More stable for ill-conditioned problems
    """
    
    def __init__(self, config: SolverConfig):
        self.config = config
        self.max_iter = config.max_iter
        self.tol = config.tol
        
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """
        Solve sparse coding using ISTA
        
        Args:
            D: Dictionary matrix (n_features, n_atoms)
            X: Data matrix (n_features, n_samples)
            penalty: Penalty function (must be prox-friendly)
            
        Returns:
            Sparse codes A (n_atoms, n_samples)
        """
        if not penalty.is_prox_friendly:
            raise ValueError("ISTA requires prox-friendly penalty")
        
        backend = xp(D)
        n_atoms, n_samples = D.shape[1], X.shape[1]
        
        # Handle single sample
        if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
            return self._solve_single(D, X.ravel(), penalty, backend)
        
        # Batch processing
        A = backend.zeros((n_atoms, n_samples))
        for i in range(n_samples):
            A[:, i] = self._solve_single(D, X[:, i], penalty, backend)
        
        return A
    
    def _solve_single(self, D: ArrayLike, x: ArrayLike, penalty: Penalty, backend) -> ArrayLike:
        """Solve single ISTA problem"""
        n_atoms = D.shape[1]
        
        # Estimate Lipschitz constant
        L = float(backend.linalg.norm(D, ord=2) ** 2)
        
        # Initialize
        a = backend.zeros(n_atoms)
        prev_objective = float('inf')
        
        for iteration in range(self.max_iter):
            # Compute gradient: ∇f(a) = D^T(Da - x)
            residual = D @ a - x
            grad = D.T @ residual
            
            # Proximal gradient step
            a = penalty.prox(a - grad / L, 1.0 / L)
            
            # Check convergence
            data_fit = 0.5 * backend.sum(residual * residual)
            penalty_val = penalty.value(a)
            objective = float(data_fit + penalty_val)
            
            if abs(objective - prev_objective) < self.tol:
                if self.config.verbose:
                    print(f"ISTA converged at iteration {iteration}")
                break
                
            prev_objective = objective
        
        return a
    
    @property
    def name(self) -> str:
        return "ista"
    
    @property
    def supports_batch(self) -> bool:
        return True


class OMPSolver:
    """
    Orthogonal Matching Pursuit
    
    Research Foundation: Pati et al. (1993) "Orthogonal matching pursuit: recursive function approximation"
    
    Algorithm:
    ```
    1. Initialize: residual r = x, active set S = ∅
    2. While ||r|| > tolerance and |S| < max_atoms:
        a. Find best atom: j = argmax |⟨D_j, r⟩|
        b. Update active set: S = S ∪ {j}
        c. Solve least squares: a_S = (D_S^T D_S)^(-1) D_S^T x
        d. Update residual: r = x - D_S a_S
    ```
    
    Properties:
    - Greedy algorithm
    - Exact sparsity control
    - Good for very sparse solutions
    - Fast for small sparsity levels
    """
    
    def __init__(self, config: SolverConfig):
        self.config = config
        self.max_iter = config.max_iter  # Max number of atoms to select
        self.tol = config.tol
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, max_atoms: Optional[int] = None, **kwargs) -> ArrayLike:
        """
        Solve sparse coding using OMP
        
        Note: OMP doesn't directly use penalty functions - sparsity controlled by max_atoms
        """
        backend = xp(D)
        n_atoms, n_samples = D.shape[1], X.shape[1]
        
        if max_atoms is None:
            max_atoms = min(self.max_iter, n_atoms // 4)  # Default: 25% sparsity
        
        # Handle single sample
        if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
            return self._solve_single(D, X.ravel(), max_atoms, backend)
        
        # Process samples sequentially (OMP is inherently sequential)
        A = backend.zeros((n_atoms, n_samples))
        for i in range(n_samples):
            A[:, i] = self._solve_single(D, X[:, i], max_atoms, backend)
        
        return A
    
    def _solve_single(self, D: ArrayLike, x: ArrayLike, max_atoms: int, backend) -> ArrayLike:
        """Solve single OMP problem"""
        n_atoms = D.shape[1]
        
        # Initialize
        residual = x.copy()
        active_set = []
        a = backend.zeros(n_atoms)
        
        # Normalize dictionary columns for correlation computation
        D_normalized = D / (backend.linalg.norm(D, axis=0, keepdims=True) + 1e-10)
        
        for iteration in range(max_atoms):
            # Find atom with highest correlation to residual
            correlations = backend.abs(D_normalized.T @ residual)
            
            # Exclude already selected atoms
            for idx in active_set:
                correlations[idx] = 0
            
            best_atom = int(backend.argmax(correlations))
            max_corr = correlations[best_atom]
            
            # Check stopping criterion
            if max_corr < self.tol:
                break
            
            # Add to active set
            active_set.append(best_atom)
            
            # Solve least squares on active set
            D_active = D[:, active_set]
            
            try:
                # Use pseudoinverse for numerical stability
                a_active = backend.linalg.pinv(D_active) @ x
                a[active_set] = a_active
                
                # Update residual
                residual = x - D_active @ a_active
                
                # Check residual stopping criterion
                if backend.linalg.norm(residual) < self.tol:
                    break
                    
            except np.linalg.LinAlgError:
                # Handle singular matrix
                break
        
        return a
    
    @property
    def name(self) -> str:
        return "omp"
    
    @property
    def supports_batch(self) -> bool:
        return False  # OMP processes samples sequentially