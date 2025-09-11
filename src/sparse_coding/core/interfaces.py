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
        raise NotImplementedError("Concrete penalty classes must implement value() method")
    
    def prox(self, z: ArrayLike, t: float) -> ArrayLike:
        """
        Proximal operator: prox_{t·ψ}(z) = argmin_a [ψ(a) + 1/(2t)||a - z||²].
        
        Args:
            z: Input point
            t: Proximal parameter
            
        Returns:
            Proximal point
        """
        raise NotImplementedError("Concrete penalty classes must implement prox() method")
    
    def grad(self, a: ArrayLike) -> ArrayLike:
        """
        Gradient of penalty: ∇ψ(a).
        
        Args:
            a: Sparse codes
            
        Returns:
            Gradient w.r.t. a
        """
        # Default implementation for non-differentiable penalties
        # Subclasses should override for differentiable penalties
        raise NotImplementedError(f"{self.__class__.__name__} must implement grad() for gradient-based optimization")
    
    @property
    def is_prox_friendly(self) -> bool:
        """Whether penalty supports efficient proximal operator."""
        # Default: assume proximal operator is available if prox() is implemented
        try:
            # Check if prox method exists and is not the base NotImplementedError version
            import numpy as np
            test_array = np.array([1.0])
            self.prox(test_array, 1.0)
            return True
        except (NotImplementedError, AttributeError):
            return False
    
    @property
    def is_differentiable(self) -> bool:
        """Whether penalty is differentiable everywhere."""
        # Default: check if grad() is implemented
        try:
            import numpy as np
            test_array = np.array([1.0])
            self.grad(test_array)
            return True
        except (NotImplementedError, AttributeError):
            return False


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
        raise NotImplementedError(f"{self.__class__.__name__} must implement solve() method")
    
    @property
    def name(self) -> str:
        """Solver name for registry/logging."""
        # Default implementation returns class name
        return self.__class__.__name__.replace('Solver', '').lower()
    
    @property
    def supports_batch(self) -> bool:
        """Whether solver can handle multiple samples efficiently."""
        # Default: assume batch processing is supported
        # Subclasses can override if they only support single samples
        return True


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
        raise NotImplementedError(f"{self.__class__.__name__} must implement step() method")
    
    @property
    def name(self) -> str:
        """Updater name for registry/logging."""
        # Default implementation returns class name
        return self.__class__.__name__.replace('Updater', '').lower()
    
    @property
    def requires_normalization(self) -> bool:
        """Whether dictionary columns need post-normalization."""
        # Default: most methods require normalization to unit norm
        # Subclasses can override if they handle normalization internally
        return True


class Learner(Protocol):
    """
    High-level dictionary learning orchestrator.
    
    Coordinates inference solver + dictionary updater.
    """
    
    def fit(self, X: ArrayLike, **kwargs) -> 'Learner':
        """
        Learn dictionary from data.
        
        Args:
            X: Training data (n_features, n_samples)
            **kwargs: Training parameters
            
        Returns:
            Self (for chaining)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement fit() method")
    
    def encode(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """
        Encode data using learned dictionary.
        
        Args:
            X: Data to encode
            **kwargs: Encoding parameters
            
        Returns:
            Sparse codes
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement encode() method")
    
    def decode(self, A: ArrayLike) -> ArrayLike:
        """
        Decode sparse codes back to data space.
        
        Args:
            A: Sparse codes (n_atoms, n_samples)
            
        Returns:
            Reconstructed data (n_features, n_samples)
        """
        # Default implementation using dictionary matrix
        if hasattr(self, 'dictionary') and self.dictionary is not None:
            import numpy as np
            return np.dot(self.dictionary, A)
        else:
            raise NotImplementedError(f"{self.__class__.__name__} must implement decode() or have a dictionary attribute")
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        """Learned dictionary matrix."""
        # Default: check for _dictionary attribute
        if hasattr(self, '_dictionary'):
            return self._dictionary
        return None
    
    def get_config(self) -> Dict[str, Any]:
        """Get learner configuration for serialization."""
        # Default implementation returns basic configuration
        config = {
            'class': self.__class__.__name__,
            'module': self.__class__.__module__
        }
        # Add any configuration attributes
        if hasattr(self, 'config'):
            config['params'] = self.config
        return config
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set learner configuration from serialization."""
        # Default implementation sets config attribute
        if 'params' in config:
            self.config = config['params']


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
        raise NotImplementedError(f"{self.__class__.__name__} must implement partial_fit() for streaming learning")
    
    def reset(self) -> None:
        """Reset learner state for fresh training."""
        # Default implementation resets common attributes
        if hasattr(self, '_dictionary'):
            self._dictionary = None
        if hasattr(self, '_n_samples_seen'):
            self._n_samples_seen = 0
        if hasattr(self, '_training_history'):
            self._training_history = []
    
    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed."""
        # Default implementation
        if hasattr(self, '_n_samples_seen'):
            return self._n_samples_seen
        return 0


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