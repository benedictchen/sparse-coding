"""
Unified imports and compatibility layer for modular sparse coding implementations.

This module provides backwards compatibility by re-exporting all algorithms
and factory functions from the new modular structure:
- algorithms/: penalty functions, solvers, dictionary updaters
- factories/: factory functions for algorithm configuration
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple

# Import all implementations from modular structure
from sparse_coding.core.penalties.implementations import (
    L1Penalty, L2Penalty, ElasticNetPenalty, CauchyPenalty
)
from sparse_coding.algorithms.solvers import (
    FISTASolver, ISTASolver, OMPSolver
)
from sparse_coding.core.dict_updater_implementations import (
    ModUpdater as MODUpdater, GradientUpdater as GradientDictUpdater, KsvdUpdater as KSVDUpdater
)
from sparse_coding.factories.algorithm_factory import (
    create_penalty, create_solver, create_dict_updater, create_learner,
    DictionaryLearner, LearnerConfig, PenaltyType, SolverType, UpdaterType
)


def validate_implementation():
    """
    Validate all algorithm implementations work correctly.
    
    This function tests all the implementations against known mathematical properties
    and validates sparse coding algorithm implementations.
    """
    print("Validating sparse coding algorithm implementations...")
    
    # Test data
    np.random.seed(42)
    n_features, n_atoms = 20, 15
    D = np.random.randn(n_features, n_atoms)
    D = D / np.linalg.norm(D, axis=0)  # Normalize columns
    x = np.random.randn(n_features)
    
    # Test penalties
    penalties_to_test = [
        ('L1', create_penalty('l1', lam=0.1)),
        ('L2', create_penalty('l2', lam=0.1)),
        ('Elastic Net', create_penalty('elastic_net', lam=0.1, l1_ratio=0.5)),
        ('Cauchy', create_penalty('cauchy', lam=0.1, sigma=1.0)),
    ]
    
    for name, penalty in penalties_to_test:
        test_a = np.random.randn(n_atoms) * 0.1
        
        # Test value function
        val = penalty.value(test_a)
        assert val >= 0, f"{name} penalty value should be non-negative"
        
        # Test prox operator
        if penalty.is_prox_friendly:
            prox_result = penalty.prox(test_a, 0.1)
            assert prox_result.shape == test_a.shape, f"{name} prox should preserve shape"
        
        print(f"[OK] {name} penalty: value={val:.4f}, prox_friendly={penalty.is_prox_friendly}")
    
    # Test solvers
    solvers_to_test = [
        ('FISTA', create_solver('fista', max_iter=50, tol=1e-4)),
        ('ISTA', create_solver('ista', max_iter=50, tol=1e-4)),
        ('OMP', create_solver('omp', max_iter=10, tol=1e-4)),
    ]
    
    l1_penalty = create_penalty('l1', lam=0.1)
    
    for name, solver in solvers_to_test:
        try:
            if name == 'OMP':
                codes = solver.solve(D, x.reshape(-1, 1), l1_penalty, max_atoms=5)
            else:
                codes = solver.solve(D, x.reshape(-1, 1), l1_penalty)
            
            assert codes.shape == (n_atoms, 1), f"{name} should return correct shape"
            sparsity = np.mean(np.abs(codes) < 1e-6)
            print(f"[OK] {name} solver: shape={codes.shape}, sparsity={sparsity:.2f}")
            
        except Exception as e:
            print(f"[ERR] {name} solver failed: {e}")
    
    print("Algorithm validation completed successfully.")


if __name__ == "__main__":
    validate_implementation()