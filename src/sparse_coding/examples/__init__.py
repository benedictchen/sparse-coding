"""
Validation and demonstration examples for sparse coding algorithms.

This package contains validation demos that test all
penalty functions, solvers, and dictionary update methods with
research-based implementations.
"""

from .algorithm_validation_demo import (
    demo_all_penalties, demo_all_solvers, 
    demo_all_updaters, demo_dictionary_learning_pipeline,
    run_demo
)

__all__ = [
    'demo_all_penalties', 'demo_all_solvers',
    'demo_all_updaters', 'demo_dictionary_learning_pipeline', 
    'run_demo'
]