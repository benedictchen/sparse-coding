"""
Penalty function protocols for sparse coding optimization.

This module defines the mathematical interface for penalty functions used in
sparse coding formulations, following Tibshirani (1996) LASSO framework.
"""

from .penalty_protocol import PenaltyProtocol

__all__ = ['PenaltyProtocol']