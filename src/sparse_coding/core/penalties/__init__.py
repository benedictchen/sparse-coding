"""
Penalty function protocols and implementations for sparse coding optimization.

This module defines the mathematical interface for penalty functions used in
sparse coding formulations, following Tibshirani (1996) LASSO framework.
"""

from .penalty_protocol import PenaltyProtocol
from .implementations import L1Penalty, L2Penalty, ElasticNetPenalty, CauchyPenalty

__all__ = ['PenaltyProtocol', 'L1Penalty', 'L2Penalty', 'ElasticNetPenalty', 'CauchyPenalty']