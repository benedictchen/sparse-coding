"""
Algorithm factory functions for configurable sparse coding implementations.

This package provides factory functions that allow users to instantiate
penalty functions, solvers, and dictionary updaters with various configuration
options, enabling flexible algorithm combinations.
"""

from .algorithm_factory import (
    create_penalty, create_solver, create_dict_updater, create_complete_learner,
    PenaltyType, SolverType, UpdaterType,
    PenaltyConfig, SolverConfig, UpdaterConfig, LearnerConfig
)

__all__ = [
    # Factory functions
    'create_penalty', 'create_solver', 'create_dict_updater', 'create_complete_learner',
    
    # Configuration types
    'PenaltyType', 'SolverType', 'UpdaterType', 
    'PenaltyConfig', 'SolverConfig', 'UpdaterConfig', 'LearnerConfig'
]