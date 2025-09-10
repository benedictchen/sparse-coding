"""
Protocol interfaces for clean separation of concerns.

Defines contracts for: Penalty, InferenceSolver, DictUpdater, Learner
Enables composition and plugin-style extensibility.
"""

from typing import Protocol, Any, Dict, Optional, Tuple
from .array import ArrayLike


class Penalty(Protocol):
    """
    Penalty/regularization term interface.
    
    Supports both proximal (FISTA-style) and differentiable (gradient-based) penalties.
    """
    
    def value(self, a: ArrayLike) -> float:
        """
        Evaluate penalty function ψ(a).
        
        Args:
            a: Sparse codes
            
        Returns:
            Penalty value (scalar)
        """
        # FIXME: Subclasses must implement this method
        # Example implementations:
        # L1: return self.lam * np.sum(np.abs(a))
        # L2: return self.lam * 0.5 * np.sum(a**2)  
        # Elastic Net: return self.lam * (self.l1_ratio * np.sum(np.abs(a)) + 
        #                                 (1-self.l1_ratio) * 0.5 * np.sum(a**2))
        # Cauchy: return self.lam * np.sum(np.log(1 + (a/self.sigma)**2))
        raise NotImplementedError("Subclasses must implement value()")
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """
        Proximal operator: prox_{t·ψ}(z) = argmin_a [ψ(a) + 1/(2t)||a - z||²].
        
        Args:
            z: Input point
            t: Proximal parameter
            
        Returns:
            Proximal point
        """
        # FIXME: Subclasses must implement this method  
        # Example implementations:
        # L1 (soft thresholding): return np.sign(z) * np.maximum(np.abs(z) - t * self.lam, 0)
        # L2 (shrinkage): return z / (1 + t * self.lam)
        # Elastic Net: l1_prox = soft_thresh(z, t*lam*l1_ratio); return l1_prox / (1 + t*lam*(1-l1_ratio))
        # TopK: indices = np.argpartition(np.abs(z), -k)[-k:]; result = np.zeros_like(z); result[indices] = z[indices]; return result
        raise NotImplementedError("Subclasses must implement prox()")
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """
        Gradient of penalty: ∇ψ(a).
        
        Args:
            a: Sparse codes
            
        Returns:
            Gradient w.r.t. a
        """
        # FIXME: Subclasses must implement this method for gradient-based solvers
        # Example implementations:
        # L1: return self.lam * np.sign(a)  # (subgradient, not differentiable at 0)
        # L2: return self.lam * a
        # Elastic Net: return self.lam * (self.l1_ratio * np.sign(a) + (1-self.l1_ratio) * a)
        # Cauchy: return self.lam * (2 * a / self.sigma**2) / (1 + (a/self.sigma)**2)
        raise NotImplementedError("Subclasses must implement grad() for gradient-based optimization")
    
    @property
    def is_prox_friendly(self) -> bool:
        """Whether penalty supports efficient proximal operator."""
        # FIXME: Subclasses should implement this property
        # Examples:
        # L1, L2, Elastic Net: return True (have closed-form prox operators)
        # General functions: return False (need iterative prox computation)
        # TopK constraint: return True (can compute prox via sorting)
        return True  # Default assumption for most sparse coding penalties
    
    @property
    def is_differentiable(self) -> bool:
        """Whether penalty is differentiable everywhere."""
        # FIXME: Subclasses should implement this property
        # Examples:
        # L1: return False (not differentiable at 0)
        # L2, Cauchy: return True (smooth everywhere)
        # Elastic Net: return False (L1 component makes it non-differentiable)
        return False  # Conservative default for sparsity-inducing penalties


class InferenceSolver(Protocol):
    """
    Sparse inference solver interface.
    
    Solves: argmin_a [1/2||X - D·a||² + penalty(a)]
    """
    
    def solve(self, D: ArrayLike, X: ArrayLike, penalty: Penalty, **kwargs) -> ArrayLike:
        """
        Solve sparse coding inference problem.
        
        Args:
            D: Dictionary matrix (n_features, n_atoms)
            X: Data matrix (n_features, n_samples) 
            penalty: Penalty function
            **kwargs: Solver-specific parameters
            
        Returns:
            Sparse codes A (n_atoms, n_samples)
        """
        # FIXME: Subclasses must implement the core solving logic
        # Example implementations:
        # FISTA: Use accelerated proximal gradient method with momentum
        # ISTA: Use basic proximal gradient method  
        # NCG: Use nonlinear conjugate gradient for smooth penalties
        # OMP: Use orthogonal matching pursuit for very sparse solutions
        # Example FISTA structure:
        #   1. Initialize: a = zeros, z = zeros, t = 1
        #   2. For iteration in range(max_iter):
        #        a_old = a
        #        grad = D.T @ (D @ z - X)  # Gradient of 0.5||X - Dz||^2
        #        z_grad = z - grad / L  # Gradient step (L = Lipschitz constant)
        #        a = penalty.prox(z_grad, 1/L)  # Proximal step
        #        t_new = (1 + sqrt(1 + 4*t^2)) / 2  # Momentum coefficient
        #        z = a + (t-1)/t_new * (a - a_old)  # Momentum update
        #        t = t_new
        #        if converged: break
        #   3. Return a
        raise NotImplementedError("Subclasses must implement solve()")
    
    @property
    def name(self) -> str:
        """Solver name for registry/logging."""
        # FIXME: Subclasses should return a descriptive name
        # Examples: "fista", "ista", "ncg", "omp", "lasso_cd"
        return self.__class__.__name__.lower().replace('solver', '')
    
    @property
    def supports_batch(self) -> bool:
        """Whether solver can handle multiple samples efficiently."""
        # FIXME: Subclasses should specify batch processing capability
        # Examples:
        # FISTA/ISTA: return True (can vectorize across samples)
        # OMP: return False (typically processes samples sequentially)
        # Coordinate Descent: return True (can batch efficiently)
        return True  # Default assumption for proximal methods


class DictUpdater(Protocol):
    """
    Dictionary update interface.
    
    Updates dictionary given current sparse codes.
    """
    
    def step(self, D: ArrayLike, X: ArrayLike, A: ArrayLike, **kwargs) -> ArrayLike:
        """
        Dictionary update step.
        
        Args:
            D: Current dictionary (n_features, n_atoms)
            X: Data matrix (n_features, n_samples)
            A: Current sparse codes (n_atoms, n_samples)  
            **kwargs: Updater-specific parameters
            
        Returns:
            Updated dictionary
        """
        # FIXME: Subclasses must implement dictionary update logic
        # Example implementations:
        # MOD (Method of Optimal Directions): 
        #   return solve(A @ A.T + eps*I, A @ X.T).T  # Closed-form solution
        # Gradient Descent:
        #   grad = -(X - D @ A) @ A.T  # Gradient of 0.5||X - DA||_F^2
        #   return D - learning_rate * grad
        # K-SVD: 
        #   For each atom: isolate error, SVD update, sparse code adjustment
        # Online/Stochastic:
        #   running_avg = momentum * running_avg + (1-momentum) * current_gradient
        #   return D - learning_rate * running_avg
        raise NotImplementedError("Subclasses must implement step()")
    
    @property
    def name(self) -> str:
        """Updater name for registry/logging."""
        # FIXME: Subclasses should return a descriptive name
        # Examples: "mod", "grad_d", "ksvd", "online_sgd" 
        return self.__class__.__name__.lower().replace('updater', '')
    
    @property
    def requires_normalization(self) -> bool:
        """Whether dictionary columns need post-normalization."""
        # FIXME: Subclasses should specify normalization requirement
        # Examples:
        # MOD: return True (closed-form solution may change column norms)
        # Gradient descent: return True (gradients can change norms)
        # K-SVD: return False (SVD inherently maintains unit norms)
        return True  # Conservative default - most methods need normalization


class Learner(Protocol):
    """
    High-level dictionary learning orchestrator.
    
    Coordinates inference solver + dictionary updater.
    """
    
    def fit(self, X: ArrayLike, **kwargs) -> 'Learner':
        """
        Learn dictionary from data.
        
        Args:
            X: Training data
            **kwargs: Training parameters
            
        Returns:
            Self (for chaining)
        """
        # FIXME: Subclasses must implement dictionary learning logic
        # Example implementations:
        # - K-SVD: alternating sparse coding and dictionary update steps
        # - Method of Optimal Directions (MOD): least squares dictionary update
        # - Online dictionary learning: stochastic gradient descent on dictionary atoms
        # 
        # Basic pattern:
        # for step in range(n_steps):
        #     codes = self.encode(X)  # Sparse coding step
        #     self.dictionary = update_dictionary(X, codes)  # Dictionary update
        ...
    
    def encode(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """
        Encode data using learned dictionary.
        
        Args:
            X: Data to encode
            **kwargs: Encoding parameters
            
        Returns:
            Sparse codes
        """
        # FIXME: Subclasses must implement encoding logic
        # Example implementations:
        # - Use sparse coding solver with learned dictionary
        # - Apply penalty function for sparsity constraint
        # - Return sparse coefficient matrix
        # 
        # Basic pattern:
        # solver = self._get_solver()
        # penalty = self._get_penalty()
        # codes = solver.solve(self.dictionary, X, penalty)
        # return codes
        ...
    
    def decode(self, A: ArrayLike) -> ArrayLike:
        """
        Decode sparse codes back to data space.
        
        Args:
            A: Sparse codes
            
        Returns:
            Reconstructed data
        """
        # FIXME: Subclasses must implement decoding logic
        # Example implementations:
        # - Simple matrix multiplication: X_reconstructed = Dictionary @ codes
        # - Apply normalization or post-processing if needed
        # 
        # Basic pattern:
        # return self.dictionary @ A
        ...
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        """Learned dictionary matrix."""
        # FIXME: Subclasses must implement dictionary property
        # Example implementations:
        # - Return learned dictionary matrix (features x atoms)
        # - Ensure proper normalization (unit norm columns)
        # - Handle case when not yet fitted (return None)
        # 
        # Basic pattern:
        # if hasattr(self, '_dictionary'):
        #     return self._dictionary
        # return None
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get learner configuration for serialization."""
        # FIXME: Subclasses must implement configuration serialization
        # Example implementations:
        # - Return dict of all learnable parameters and hyperparameters
        # - Include model architecture, training settings, etc.
        # - Exclude large arrays (dictionary should be saved separately)
        # 
        # Basic pattern:
        # return {
        #     'n_atoms': self.n_atoms,
        #     'penalty_type': self.penalty_type,
        #     'solver_params': self.solver_params,
        #     'hyperparameters': self.hyperparameters
        # }
        ...
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set learner configuration from serialization."""
        # FIXME: Subclasses must implement configuration deserialization
        # Example implementations:
        # - Restore all parameters from config dict
        # - Validate configuration integrity
        # - Initialize internal state properly
        # 
        # Basic pattern:
        # self.n_atoms = config['n_atoms']
        # self.penalty_type = config['penalty_type']
        # self.solver_params = config['solver_params']
        # self._reinitialize_from_config()
        ...


class StreamingLearner(Protocol):
    """
    Streaming/online learning interface.
    
    Extends Learner with incremental updates.
    """
    
    def partial_fit(self, X: ArrayLike, **kwargs) -> 'StreamingLearner':
        """
        Incremental learning step.
        
        Args:
            X: Mini-batch of data
            **kwargs: Update parameters
            
        Returns:
            Self (for chaining)
        """
        ...
    
    def reset(self) -> None:
        """Reset learner state for fresh training."""
        ...
    
    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed."""
        ...


# Type aliases for common patterns
PenaltyConfig = Dict[str, Any]
SolverConfig = Dict[str, Any] 
UpdaterConfig = Dict[str, Any]
LearnerConfig = Dict[str, Any]

# Configuration schema for validation
CONFIG_SCHEMA = {
    "penalty": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "params": {"type": "object"}
        },
        "required": ["name"]
    },
    "solver": {
        "type": "object", 
        "properties": {
            "name": {"type": "string"},
            "params": {"type": "object"}
        },
        "required": ["name"]
    },
    "dict_updater": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "params": {"type": "object"}
        },
        "required": ["name"]
    }
}