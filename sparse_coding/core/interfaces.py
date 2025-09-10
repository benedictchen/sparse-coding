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
        ...
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """
        Proximal operator: prox_{t·ψ}(z) = argmin_a [ψ(a) + 1/(2t)||a - z||²].
        
        Args:
            z: Input point
            t: Proximal parameter
            
        Returns:
            Proximal point
        """
        ...
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """
        Gradient of penalty: ∇ψ(a).
        
        Args:
            a: Sparse codes
            
        Returns:
            Gradient w.r.t. a
        """
        ...
    
    @property
    def is_prox_friendly(self) -> bool:
        """Whether penalty supports efficient proximal operator."""
        ...
    
    @property
    def is_differentiable(self) -> bool:
        """Whether penalty is differentiable everywhere."""
        ...


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
        ...
    
    @property
    def name(self) -> str:
        """Solver name for registry/logging."""
        ...
    
    @property
    def supports_batch(self) -> bool:
        """Whether solver can handle multiple samples efficiently."""
        ...


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
        ...
    
    @property
    def name(self) -> str:
        """Updater name for registry/logging."""
        ...
    
    @property
    def requires_normalization(self) -> bool:
        """Whether dictionary columns need post-normalization."""
        ...


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
        ...
    
    def decode(self, A: ArrayLike) -> ArrayLike:
        """
        Decode sparse codes back to data space.
        
        Args:
            A: Sparse codes
            
        Returns:
            Reconstructed data
        """
        ...
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        """Learned dictionary matrix."""
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get learner configuration for serialization."""
        ...
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set learner configuration from serialization."""
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