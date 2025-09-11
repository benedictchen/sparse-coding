"""
Sparse coding inference algorithms.

This module implements sparse inference algorithms for solving:
minimize ||x - Da||² + λR(a)

Following Olshausen & Field (1996) sparse coding framework.
"""

from .fista_accelerated_solver import FISTASolver
from .ista_basic_solver import ISTASolver  
from .nonlinear_conjugate_gradient import NonlinearConjugateGradient
from .orthogonal_matching_pursuit import OrthogonalMatchingPursuit

__all__ = [
    'FISTASolver',
    'ISTASolver', 
    'NonlinearConjugateGradient',
    'OrthogonalMatchingPursuit'
]