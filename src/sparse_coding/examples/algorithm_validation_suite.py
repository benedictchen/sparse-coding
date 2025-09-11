#!/usr/bin/env python3
"""
Simple validation script for modular sparse coding implementation.
Tests that all algorithm components work correctly.
"""

import numpy as np
import sys
import os

# Add current directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    from algorithms.penalties import L1Penalty, L2Penalty, PenaltyConfig, PenaltyType
    from algorithms.solvers import FISTASolver, ISTASolver, SolverConfig, SolverType
    from algorithms.dict_updaters import MODUpdater, UpdaterConfig, UpdaterType
    from factories.algorithm_factory import create_penalty, create_solver, create_learner
    print("[OK] All modular imports successful")
except ImportError as e:
    print(f"[ERR] Import failed: {e}")
    sys.exit(1)

def test_penalties():
    """Test penalty function implementations"""
    print("Testing penalty functions...")
    
    test_vector = np.array([0.8, -0.3, 0.05, 0.0, -0.7, 0.02])
    
    penalties = [
        ("L1", create_penalty('l1', lam=0.1)),
        ("L2", create_penalty('l2', lam=0.1)),
        ("Elastic Net", create_penalty('elastic_net', lam=0.1, l1_ratio=0.7))
    ]
    
    for name, penalty in penalties:
        value = penalty.value(test_vector)
        if penalty.is_prox_friendly:
            prox_result = penalty.prox(test_vector, 0.1)
            sparsity = np.mean(np.abs(prox_result) < 0.01)
            print(f"  [OK] {name}: value={value:.3f}, sparsity={sparsity:.2f}")
        else:
            print(f"  [OK] {name}: value={value:.3f}, not prox-friendly")

def test_solvers():
    """Test solver implementations"""
    print("Testing sparse coding solvers...")
    
    # Create test problem
    np.random.seed(42)
    n_features, n_atoms = 15, 10
    D = np.random.randn(n_features, n_atoms)
    D = D / np.linalg.norm(D, axis=0)
    x = np.random.randn(n_features)
    
    penalty = create_penalty('l1', lam=0.1)
    
    solvers = [
        ("FISTA", create_solver('fista', max_iter=50, tol=1e-4)),
        ("ISTA", create_solver('ista', max_iter=50, tol=1e-4)),
        ("OMP", create_solver('omp', max_iter=5, tol=1e-4))
    ]
    
    for name, solver in solvers:
        try:
            if name == 'OMP':
                codes = solver.solve(D, x.reshape(-1, 1), penalty, max_atoms=3)
            else:
                codes = solver.solve(D, x.reshape(-1, 1), penalty)
            
            error = np.linalg.norm(D @ codes.ravel() - x)
            sparsity = np.mean(np.abs(codes.ravel()) < 1e-6)
            print(f"  [OK] {name}: error={error:.3f}, sparsity={sparsity:.2f}")
        except Exception as e:
            print(f"  [ERR] {name}: {str(e)}")

def test_dictionary_learner():
    """Test dictionary learning system"""
    print("Testing dictionary learner...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_features, n_samples = 20, 50
    X = np.random.randn(n_features, n_samples) * 0.1
    
    try:
        learner = create_learner(
            penalty_type='l1',
            solver_type='fista',
            updater_type='mod',
            n_atoms=15,
            lam=0.1,
            n_iterations=5,
            verbose=False
        )
        
        # Fit and encode
        learner.fit(X)
        codes = learner.encode(X[:, :10])
        reconstruction = learner.decode(codes)
        
        error = np.linalg.norm(X[:, :10] - reconstruction) / np.linalg.norm(X[:, :10])
        sparsity = np.mean(np.abs(codes) < 1e-6)
        
        print(f"  [OK] Dictionary learning: error={error:.3f}, sparsity={sparsity:.2f}")
        print(f"  [OK] Dictionary shape: {learner.dictionary.shape}")
        
    except Exception as e:
        print(f"  [ERR] Dictionary learning: {str(e)}")

if __name__ == "__main__":
    print("Validating modular sparse coding implementation")
    print("=" * 50)
    
    test_penalties()
    test_solvers()
    test_dictionary_learner()
    
    print("=" * 50)
    print("Modular implementation validation completed")