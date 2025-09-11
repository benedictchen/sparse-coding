"""
Penalty function protocol for sparse coding optimization.

Defines the mathematical interface required by all penalty functions used in
sparse coding algorithms following Tibshirani (1996) LASSO formulation.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class PenaltyProtocol(Protocol):
    """Protocol for penalty functions in sparse coding optimization.
    
    Implements the mathematical interface required for penalty terms R(a) in the
    sparse coding objective: minimize ||x - Da||² + λR(a)
    
    Reference: Tibshirani (1996). Regression shrinkage and selection via the lasso.
    """
    
    def value(self, a: np.ndarray) -> float:
        """Evaluate penalty function R(a)."""
        ...
    
    def prox(self, z: np.ndarray, t: float) -> np.ndarray:
        """Proximal operator: argmin_a (½||a-z||² + t·R(a))."""
        ...
    
    def grad(self, a: np.ndarray) -> np.ndarray:
        """Gradient/subgradient of penalty function."""
        ...
    
    @property
    def is_prox_friendly(self) -> bool:
        """Whether proximal operator has closed-form solution."""
        ...
    
    @property
    def is_differentiable(self) -> bool:
        """Whether penalty function is everywhere differentiable."""
        ...