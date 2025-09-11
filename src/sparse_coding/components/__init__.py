"""
Default component implementations for the plugin system.

Registers standard penalties, solvers, and dictionary updaters.
"""

from ..core.penalties import L1Penalty, L2Penalty, ElasticNetPenalty
from .solvers import FISTASolver, ISTASolver
from ..core.dict_updater_implementations import ModUpdater as MODUpdater, GradientUpdater as GradDUpdater
from .register_defaults import register_default_components

# Auto-register default components on import
register_default_components()

__all__ = [
    'L1Penalty', 'L2Penalty', 'ElasticNetPenalty',
    'FISTASolver', 'ISTASolver', 
    'MODUpdater', 'GradDUpdater',
    'register_default_components'
]