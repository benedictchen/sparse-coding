"""
Protocol interfaces for clean separation of concerns.

Defines contracts for: Penalty, InferenceSolver, DictUpdater, Learner
Enables composition and plugin-style extensibility.

This file now imports from modularized implementations for maintainability.
"""

from typing import Protocol, Any, Dict, Optional, Tuple
from .array import ArrayLike

# Import modularized implementations
from .penalties.penalty_protocol import PenaltyProtocol as Penalty
from .inference.fista_accelerated_solver import FISTASolver
from .inference.ista_basic_solver import ISTASolver
from .inference.nonlinear_conjugate_gradient import NonlinearConjugateGradient
from .inference.orthogonal_matching_pursuit import OrthogonalMatchingPursuit
from .dictionary.method_optimal_directions import MethodOptimalDirections
from .dictionary.ksvd_dictionary_learning import KSVDDictionaryLearning
from .dictionary.gradient_descent_update import GradientDescentUpdate
from .dictionary.online_dictionary_learning import OnlineDictionaryLearning
from .orchestration.olshausen_field_learner import OlshausenFieldLearner, SparseCodingConfig


# Protocol interfaces for type checking and extensibility
class InferenceSolver(Protocol):
    """Protocol for sparse inference algorithms."""
    
    def solve(self, D: ArrayLike, x: ArrayLike, *args, **kwargs) -> Tuple[ArrayLike, int]:
        """Solve sparse coding inference problem."""
        ...


class DictUpdater(Protocol):
    """Protocol for dictionary update algorithms."""
    
    def update(self, D: ArrayLike, X: ArrayLike, A: ArrayLike) -> ArrayLike:
        """Update dictionary given data and sparse codes."""
        ...


class Learner(Protocol):
    """Protocol for complete sparse coding learners."""
    
    def fit(self, X: ArrayLike, penalty: Penalty) -> 'Learner':
        """Learn dictionary from training data."""
        ...
    
    def encode(self, X: ArrayLike, penalty: Penalty) -> ArrayLike:
        """Encode signals using learned dictionary."""
        ...
    
    def decode(self, sparse_codes: ArrayLike) -> ArrayLike:
        """Reconstruct signals from sparse codes."""
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        ...
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set new configuration."""
        ...


class StreamingLearner(Protocol):
    """Protocol for online/streaming sparse coding learners."""
    
    def partial_fit(self, X: ArrayLike, penalty: Penalty) -> 'StreamingLearner':
        """Update model with new batch of data."""
        ...
    
    def encode(self, X: ArrayLike, penalty: Penalty) -> ArrayLike:
        """Encode signals using current dictionary."""
        ...
    
    def decode(self, sparse_codes: ArrayLike) -> ArrayLike:
        """Reconstruct signals from sparse codes."""
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        ...
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set new configuration."""
        ...


# Concrete implementations registry
INFERENCE_ALGORITHMS = {
    'fista': FISTASolver,
    'ista': ISTASolver,
    'ncg': NonlinearConjugateGradient,
    'omp': OrthogonalMatchingPursuit
}

DICTIONARY_ALGORITHMS = {
    'mod': MethodOptimalDirections,
    'ksvd': KSVDDictionaryLearning,
    'gradient': GradientDescentUpdate,
    'online': OnlineDictionaryLearning
}

LEARNER_IMPLEMENTATIONS = {
    'olshausen_field': OlshausenFieldLearner
}


__all__ = [
    # Protocols
    'Penalty', 'InferenceSolver', 'DictUpdater', 'Learner', 'StreamingLearner',
    # Concrete implementations
    'FISTASolver', 'ISTASolver', 'NonlinearConjugateGradient', 'OrthogonalMatchingPursuit',
    'MethodOptimalDirections', 'KSVDDictionaryLearning', 'GradientDescentUpdate', 'OnlineDictionaryLearning',
    'OlshausenFieldLearner', 'SparseCodingConfig',
    # Registries
    'INFERENCE_ALGORITHMS', 'DICTIONARY_ALGORITHMS', 'LEARNER_IMPLEMENTATIONS'
]