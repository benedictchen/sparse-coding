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
        # FIXME: FAKE CODE - Protocol implementing method mapping logic
        #
        # PROBLEM: Protocol contains method mapping dictionaries and string manipulation.
        # This couples interface to implementation details and creates brittle magic behavior.
        #
        # SOLUTION: Make this an abstract property for concrete classes to implement.
        raise NotImplementedError("Concrete updater classes must define name property")
    
    @property
    def requires_normalization(self) -> bool:
        """Whether dictionary columns need post-normalization."""
        # FIXME: FAKE CODE - Protocol implementing normalization decision logic
        #
        # PROBLEM: Protocol contains algorithm-specific knowledge about whether
        # dictionary normalization is required. This violates separation of concerns.
        #
        # RESEARCH CONTEXT: Dictionary normalization requirements vary by algorithm:
        # - MOD: Closed-form solution preserves meaningful scale
        # - K-SVD: SVD intrinsically produces unit-norm atoms  
        # - Gradient descent: Can drift from unit norm during updates
        # - Online SGD: Typically requires periodic normalization
        #
        # SOLUTION: Let each concrete updater class define its normalization needs
        # based on its specific algorithmic properties.
        raise NotImplementedError("Concrete updater classes must define requires_normalization property")


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
        # FIXME: MASSIVE FAKE CODE VIOLATION - Protocol implementing dictionary learning algorithms
        #
        # PROBLEM: This Protocol contains a COMPLETE dictionary learning system with
        # K-SVD, MOD, and online learning implementations. This is 50+ lines of fake code
        # that should be in concrete learner classes, not a Protocol interface.
        #
        # RESEARCH CONTEXT: Dictionary learning is the central problem in sparse coding:
        # - Two-stage alternating minimization: sparse coding + dictionary update
        # - K-SVD algorithm alternates between OMP/FISTA and SVD-based updates
        # - MOD uses closed-form solutions for dictionary updates
        # - Online learning processes mini-batches with streaming updates
        #
        # SOLUTION 1: Pure Protocol Interface (Recommended)
        # Remove ALL algorithmic logic:
        #
        # def fit(self, X: ArrayLike, **kwargs) -> 'Learner':
        #     """Learn dictionary from data."""
        #     ...  # No implementation - force concrete learners
        #
        # SOLUTION 2: Concrete Learner Implementations
        # Create separate classes for each learning approach:
        #
        # class KsvdLearner:
        #     def fit(self, X, **kwargs):
        #         # K-SVD alternating minimization
        #         for step in range(n_steps):
        #             codes = self.solver.solve(self.dictionary, X, self.penalty)
        #             self.dictionary = self.updater.step(self.dictionary, X, codes)
        #
        # class ModLearner:
        #     def fit(self, X, **kwargs):  
        #         # MOD alternating optimization
        #
        # class OnlineLearner:
        #     def fit(self, X, **kwargs):
        #         # Streaming dictionary learning
        #
        # SOLUTION 3: Composite Learner Pattern
        # Use composition with solver + updater components:
        #
        # class CompositeLearner:
        #     def __init__(self, solver, updater, penalty):
        #         self.solver = solver
        #         self.updater = updater  
        #         self.penalty = penalty
        #     
        #     def fit(self, X, **kwargs):
        #         # Generic alternating minimization
        #         for step in range(kwargs.get('n_steps', 100)):
        #             codes = self.solver.solve(self.dictionary, X, self.penalty)
        #             self.dictionary = self.updater.step(self.dictionary, X, codes)
        #
        # References:
        # - Aharon et al. (2006). K-SVD dictionary learning algorithm
        # - Engan et al. (1999). MOD method of optimal directions  
        # - Mairal et al. (2010). Online dictionary learning for sparse coding
        # - Olshausen & Field (1996). Sparse coding with overcomplete bases
        
        raise NotImplementedError("Concrete learner classes must implement fit() method")
    
    def encode(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """
        Encode data using learned dictionary.
        
        Args:
            X: Data to encode
            **kwargs: Encoding parameters
            
        Returns:
            Sparse codes
        """
        # FIXME: FAKE CODE - Protocol implementing encoding logic
        #
        # PROBLEM: Protocol contains implementation logic for encoding including
        # error checking, solver selection, and algorithm dispatch.
        #
        # SOLUTION: Pure interface - let concrete learners implement encoding.
        raise NotImplementedError("Concrete learner classes must implement encode() method")
    
    def decode(self, A: ArrayLike) -> ArrayLike:
        """
        Decode sparse codes back to data space.
        
        Args:
            A: Sparse codes (n_atoms, n_samples)
            
        Returns:
            Reconstructed data (n_features, n_samples)
        """
        # FIXME: FAKE CODE - Protocol implementing decoding (dictionary multiplication)
        #
        # PROBLEM: Protocol assumes all learners store dictionary in `_dictionary` 
        # attribute and use simple matrix multiplication for reconstruction.
        #
        # RESEARCH CONTEXT: Reconstruction in sparse coding is X̂ = D @ A where
        # D is the learned dictionary and A are the sparse codes.
        #
        # SOLUTION: Let concrete learners implement their specific decoding logic.
        raise NotImplementedError("Concrete learner classes must implement decode() method")
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        """Learned dictionary matrix."""
        # FIXME: FAKE CODE - Protocol implementing dictionary access with hasattr() magic
        #
        # PROBLEM: Protocol makes assumptions about internal storage attribute naming.
        # 
        # SOLUTION: Make this an abstract property for concrete learners to implement.
        raise NotImplementedError("Concrete learner classes must define dictionary property")
    
    def get_config(self) -> Dict[str, Any]:
        """Get learner configuration for serialization."""
        # FIXME: EXTENSIVE FAKE CODE - Protocol implementing serialization logic
        #
        # PROBLEM: Protocol contains ~25 lines of serialization logic including:
        # - Class name introspection
        # - Dictionary shape extraction with hasattr() checks
        # - Penalty type inspection and lambda extraction
        # - Generic attribute iteration and extraction
        #
        # This couples Protocol to specific implementation details and storage patterns.
        #
        # SOLUTION: Let concrete learners implement their own configuration serialization
        # based on their specific state and parameters.
        raise NotImplementedError("Concrete learner classes must implement get_config() method")
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set learner configuration from serialization."""
        # FIXME: EXTENSIVE FAKE CODE - Protocol implementing deserialization with penalty factories
        #
        # PROBLEM: Protocol contains complex deserialization logic including:
        # - Hardcoded penalty class imports and factory mapping
        # - Attribute setting via setattr() with magic attribute names  
        # - State initialization assumptions (_dictionary = None)
        #
        # This creates tight coupling between Protocol and concrete penalty implementations.
        #
        # SOLUTION: Let concrete learners implement their own configuration deserialization.
        raise NotImplementedError("Concrete learner classes must implement set_config() method")


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
        # FIXME: FAKE CODE - Protocol implementing streaming learning algorithm
        #
        # PROBLEM: Protocol contains complete online learning implementation with
        # dictionary initialization, solver/updater dispatch, and state management.
        #
        # RESEARCH CONTEXT: Online dictionary learning (Mairal et al. 2010) processes
        # data in mini-batches with stochastic gradient updates. This is a complex
        # algorithm that belongs in concrete implementations.
        #
        # SOLUTION: Pure protocol interface for streaming learners.
        raise NotImplementedError("Concrete streaming learner classes must implement partial_fit() method")
    
    def reset(self) -> None:
        """Reset learner state for fresh training."""
        # FIXME: FAKE CODE - Protocol implementing state management with hardcoded attributes
        #
        # PROBLEM: Protocol assumes specific internal state attributes like
        # _dictionary, _n_samples_seen, _momentum_buffer and implements reset logic.
        #
        # SOLUTION: Let concrete streaming learners manage their own state.
        raise NotImplementedError("Concrete streaming learner classes must implement reset() method")
    
    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed."""
        # FIXME: FAKE CODE - Protocol implementing sample counting with getattr() defaults
        #
        # PROBLEM: Protocol assumes internal _n_samples_seen attribute with default fallback.
        #
        # SOLUTION: Let concrete streaming learners track their own sample counts.
        raise NotImplementedError("Concrete streaming learner classes must define n_samples_seen property")


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