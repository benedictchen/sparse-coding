"""
Validation and demonstration examples for sparse coding algorithms.

This package contains comprehensive validation demos that test all
penalty functions, solvers, and dictionary update methods with
research-accurate implementations.
"""

from .algorithm_validation_demo import (
    demo_penalty_functions, demo_sparse_coding_solvers, 
    demo_dictionary_updaters, demo_complete_pipeline,
    run_validation_suite
)

__all__ = [
    'demo_penalty_functions', 'demo_sparse_coding_solvers',
    'demo_dictionary_updaters', 'demo_complete_pipeline', 
    'run_validation_suite'
]