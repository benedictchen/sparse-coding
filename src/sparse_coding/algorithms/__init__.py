"""
Algorithm implementations for sparse coding.

This package contains concrete implementations of penalty functions, 
solvers, and dictionary update methods used in sparse coding.

Modules:
    penalties: L1, L2, Elastic Net, Cauchy, and Top-K penalty implementations
    solvers: FISTA, ISTA, and OMP sparse coding solvers
    dict_updaters: MOD, Gradient Descent, and K-SVD dictionary updaters
"""

from ..core.penalties import L1Penalty, L2Penalty, ElasticNetPenalty, CauchyPenalty
from .solvers import FISTASolver, ISTASolver, OMPSolver  
from ..core.dict_updater_implementations import ModUpdater as MODUpdater, GradientUpdater as GradientDictUpdater, KsvdUpdater as KSVDUpdater

__all__ = [
    # Penalty functions
    'L1Penalty', 'L2Penalty', 'ElasticNetPenalty', 'CauchyPenalty',
    
    # Sparse coding solvers
    'FISTASolver', 'ISTASolver', 'OMPSolver',
    
    # Dictionary updaters  
    'MODUpdater', 'GradientDictUpdater', 'KSVDUpdater'
]