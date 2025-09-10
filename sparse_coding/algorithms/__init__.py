"""
Algorithm implementations for sparse coding.

This package contains concrete implementations of penalty functions, 
solvers, and dictionary update methods used in sparse coding.

Modules:
    penalties: L1, L2, Elastic Net, Cauchy, and Top-K penalty implementations
    solvers: FISTA, ISTA, and OMP sparse coding solvers
    dict_updaters: MOD, Gradient Descent, and K-SVD dictionary updaters
"""

from .penalties import L1Penalty, L2Penalty, ElasticNetPenalty, CauchyPenalty, TopKPenalty
from .solvers import FISTASolver, ISTASolver, OMPSolver  
from .dictionary_update_algorithms import MODUpdater, GradientDictUpdater, KSVDUpdater

__all__ = [
    # Penalty functions
    'L1Penalty', 'L2Penalty', 'ElasticNetPenalty', 'CauchyPenalty', 'TopKPenalty',
    
    # Sparse coding solvers
    'FISTASolver', 'ISTASolver', 'OMPSolver',
    
    # Dictionary updaters  
    'MODUpdater', 'GradientDictUpdater', 'KSVDUpdater'
]