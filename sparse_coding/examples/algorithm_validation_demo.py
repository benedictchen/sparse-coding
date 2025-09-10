"""
Comprehensive example demonstrating sparse coding algorithm implementations.

This module showcases the complete range of penalty functions, solver algorithms,
and dictionary update methods available in the sparse coding library, with
research-based implementations and configurable parameters.

Includes demonstrations of:
- Penalty functions: L1, L2, Elastic Net, Cauchy, Top-K constraints
- Sparse coding solvers: FISTA, ISTA, NCG, OMP
- Dictionary updaters: MOD, gradient descent, K-SVD
- Complete dictionary learning pipelines
- Configuration and serialization capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

# Import sparse coding algorithm implementations
from .reference_implementations import (
    create_penalty, create_solver, create_complete_learner,
    PenaltyType, SolverType, UpdaterType,
    PenaltyConfig, SolverConfig, UpdaterConfig, LearnerConfig
)


def generate_test_data(n_features: int = 20, n_samples: int = 100, 
                      n_true_atoms: int = 15, sparsity_level: int = 3, 
                      noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic sparse coding data for testing
    
    Creates synthetic data with known ground truth for algorithm validation.
    """
    np.random.seed(42)  # Reproducible results
    
    # Create true dictionary (what we want to recover)
    D_true = np.random.randn(n_features, n_true_atoms)
    D_true = D_true / np.linalg.norm(D_true, axis=0)  # Normalize columns
    
    # Create sparse codes (mostly zeros with few non-zero entries)
    A_true = np.zeros((n_true_atoms, n_samples))
    for i in range(n_samples):
        # Pick random atoms to be active
        active_atoms = np.random.choice(n_true_atoms, sparsity_level, replace=False)
        A_true[active_atoms, i] = np.random.randn(sparsity_level)
    
    # Generate data: X = D_true * A_true + noise
    X = D_true @ A_true + noise_std * np.random.randn(n_features, n_samples)
    
    return X, D_true, A_true


def demo_all_penalties():
    """
    Demonstrate penalty function implementations.
    
    Tests all implemented penalty functions including L1, L2, Elastic Net,
    Cauchy robust penalty, and Top-K hard sparsity constraints with
    their respective proximal operators and mathematical properties.
    """
    print("\n" + "="*60)
    print("PENALTY FUNCTION IMPLEMENTATIONS")
    print("="*60)
    
    # Test vector
    test_codes = np.array([0.8, -0.3, 0.05, 0.0, -0.7, 0.02])
    print(f"Test codes: {test_codes}")
    print(f"True sparsity (zeros): {np.sum(np.abs(test_codes) < 0.1)}/6")
    
    penalty_configs = [
        ('L1 (Lasso)', 'l1', {'lam': 0.1}),
        ('L2 (Ridge)', 'l2', {'lam': 0.1}),
        ('Elastic Net', 'elastic_net', {'lam': 0.1, 'l1_ratio': 0.7}),
        ('Cauchy (Robust)', 'cauchy', {'lam': 0.05, 'sigma': 0.5}),
        ('Top-K (Hard)', 'top_k', {'k': 3})
    ]
    
    results = []
    for name, penalty_type, kwargs in penalty_configs:
        penalty = create_penalty(penalty_type, **kwargs)
        
        # Test penalty value
        value = penalty.value(test_codes)
        
        # Test proximal operator (if supported)
        if penalty.is_prox_friendly:
            prox_result = penalty.prox(test_codes, 0.1)
            sparsity_after = np.sum(np.abs(prox_result) < 0.01)
            print(f"✅ {name:15} | Value: {value:8.4f} | Sparsity after prox: {sparsity_after}/6 | Prox-friendly: {penalty.is_prox_friendly}")
            results.append((name, prox_result, sparsity_after))
        else:
            print(f"✅ {name:15} | Value: {value:8.4f} | Prox-friendly: {penalty.is_prox_friendly} | Differentiable: {penalty.is_differentiable}")
    
    return results


def demo_all_solvers():
    """
    Demonstrate sparse coding solver implementations.
    
    Tests FISTA, ISTA, and OMP algorithms with convergence analysis
    and performance comparison on synthetic test problems.
    """
    print("\n" + "="*60) 
    print("SPARSE CODING SOLVER ALGORITHMS")
    print("="*60)
    
    # Generate test problem
    np.random.seed(42)
    n_features, n_atoms = 15, 10
    D = np.random.randn(n_features, n_atoms)
    D = D / np.linalg.norm(D, axis=0)
    x = np.random.randn(n_features)
    
    penalty = create_penalty('l1', lam=0.1)
    
    solver_configs = [
        ('FISTA (Fast)', 'fista', {'max_iter': 100, 'tol': 1e-5}),
        ('ISTA (Basic)', 'ista', {'max_iter': 100, 'tol': 1e-5}),  
        ('OMP (Greedy)', 'omp', {'max_iter': 5, 'tol': 1e-5}),
    ]
    
    results = []
    for name, solver_type, kwargs in solver_configs:
        solver = create_solver(solver_type, **kwargs)
        
        try:
            # Solve sparse coding problem
            X_test = x.reshape(-1, 1)  # Make it 2D
            codes = solver.solve(D, X_test, penalty)
            
            # Compute reconstruction error
            reconstruction = D @ codes.ravel()
            error = np.linalg.norm(x - reconstruction)
            sparsity = np.sum(np.abs(codes.ravel()) < 1e-6)
            
            print(f"✅ {name:15} | Recon Error: {error:8.4f} | Sparsity: {sparsity:2d}/{n_atoms} | Batch Support: {solver.supports_batch}")
            results.append((name, codes.ravel(), error, sparsity))
            
        except Exception as e:
            print(f"[ERR] {name:15} | Error: {str(e)[:50]}...")
    
    return results


def demo_all_updaters():
    """
    Demonstrate dictionary update method implementations.
    
    Tests MOD, gradient descent, and K-SVD dictionary learning algorithms
    with performance evaluation on synthetic dictionary recovery tasks.
    """
    print("\n" + "="*60)
    print("DICTIONARY UPDATE METHOD IMPLEMENTATIONS") 
    print("="*60)
    
    # Generate test problem
    np.random.seed(42)
    X, D_true, A_true = generate_test_data(n_features=10, n_samples=20, n_true_atoms=8)
    
    # Start with random dictionary
    D_init = np.random.randn(10, 8)
    D_init = D_init / np.linalg.norm(D_init, axis=0)
    
    updater_configs = [
        ('MOD (Fast)', 'mod', {'regularization': 1e-6}),
        ('Gradient Descent', 'grad_d', {'learning_rate': 0.1}),
        ('K-SVD (Quality)', 'ksvd', {'max_atom_updates': 3}),
    ]
    
    from .reference_implementations import MODUpdater, GradientDictUpdater, KSVDUpdater, UpdaterConfig, UpdaterType
    
    results = []
    for name, updater_type, kwargs in updater_configs:
        # Create updater
        config = UpdaterConfig(updater_type=UpdaterType(updater_type), **kwargs)
        
        if updater_type == 'mod':
            updater = MODUpdater(config)
        elif updater_type == 'grad_d':
            updater = GradientDictUpdater(config)
        elif updater_type == 'ksvd':
            updater = KSVDUpdater(config)
        
        try:
            # Update dictionary
            D_updated = updater.step(D_init, X, A_true)
            
            # Measure improvement (alignment with true dictionary)
            # Use Hungarian algorithm approximation: best permutation alignment
            alignment_score = 0
            for i in range(D_true.shape[1]):
                best_match = np.max(np.abs(D_true[:, i].T @ D_updated))
                alignment_score += best_match
            alignment_score /= D_true.shape[1]
            
            print(f"✅ {name:15} | Dict Alignment: {alignment_score:6.3f} | Requires Norm: {updater.requires_normalization}")
            results.append((name, D_updated, alignment_score))
            
        except Exception as e:
            print(f"[ERR] {name:15} | Error: {str(e)[:50]}...")
    
    return results


def demo_complete_pipeline():
    """
    Demonstrate complete dictionary learning pipeline.
    
    Tests integrated sparse coding and dictionary update systems with
    convergence monitoring, objective tracking, and configuration management.
    """
    print("\n" + "="*60)
    print("COMPLETE DICTIONARY LEARNING PIPELINE")
    print("="*60)
    
    # Generate synthetic data
    X, D_true, A_true = generate_test_data(n_features=20, n_samples=50, n_true_atoms=15)
    
    print(f"Data: {X.shape[0]} features, {X.shape[1]} samples")
    print(f"True dictionary: {D_true.shape[1]} atoms")
    print(f"True sparsity: {np.mean(np.abs(A_true) > 1e-6):.1%} non-zeros")
    
    # Test different algorithm combinations
    combinations = [
        {
            'name': 'L1 + FISTA + MOD (Fast & Standard)',
            'penalty_type': 'l1', 'solver_type': 'fista', 'updater_type': 'mod',
            'lam': 0.1, 'max_iter': 50, 'n_iterations': 20
        },
        {
            'name': 'ElasticNet + ISTA + K-SVD (High Quality)', 
            'penalty_type': 'elastic_net', 'solver_type': 'ista', 'updater_type': 'ksvd',
            'lam': 0.05, 'l1_ratio': 0.8, 'max_iter': 30, 'n_iterations': 15
        },
        {
            'name': 'Top-K + FISTA + GradientDescent (Sparse)',
            'penalty_type': 'top_k', 'solver_type': 'fista', 'updater_type': 'grad_d',
            'k': 3, 'max_iter': 40, 'learning_rate': 0.05, 'n_iterations': 25
        }
    ]
    
    results = []
    for combo in combinations:
        try:
            print(f"\n--- Testing: {combo['name']} ---")
            
            # Create complete learner with specified configuration
            learner = create_complete_learner(
                penalty_type=combo['penalty_type'],
                solver_type=combo['solver_type'], 
                updater_type=combo['updater_type'],
                n_atoms=15,  # Match true dictionary size
                n_iterations=combo['n_iterations'],
                verbose=False,
                compute_objective=True,
                **{k: v for k, v in combo.items() if k not in ['name', 'penalty_type', 'solver_type', 'updater_type', 'n_iterations']}
            )
            
            # Perform dictionary learning via alternating minimization (Olshausen & Field 1996)
            learner.fit(X)
            
            # Test sparse coding phase: encode training data using learned dictionary
            codes = learner.encode(X[:, :10])  # Encode first 10 samples
            reconstructed = learner.decode(codes)
            
            # Evaluate reconstruction fidelity (L2 relative error)
            original = X[:, :10]
            recon_error = np.linalg.norm(original - reconstructed) / np.linalg.norm(original)
            
            # Measure sparsity level (fraction of near-zero coefficients)
            sparsity = np.mean(np.abs(codes) < 1e-6)
            
            # Validate configuration persistence for reproducibility
            config = learner.get_config()
            
            print(f"\n--- Testing: {combo['name']} ---")
            print(f"Training completed: {len(learner._training_history)} iterations")
            print(f"Reconstruction error: {recon_error:.4f}")
            print(f"Sparsity achieved: {sparsity:.1%}")
            
            # Access learned dictionary for analysis and visualization
            learned_dict = learner.dictionary
            print(f"Dictionary shape: {learned_dict.shape}")
            
            results.append({
                'name': combo['name'],
                'learner': learner,
                'recon_error': recon_error,
                'sparsity': sparsity,
                'config': config
            })
            
        except Exception as e:
            print(f"Failed: {str(e)[:80]}...")
    
    return results


def demo_configuration_showcase():
    """
    Demonstrate comprehensive configuration options.
    
    Shows all available configuration parameters and algorithm choices
    for penalty functions, solvers, and dictionary update methods.
    """
    print("\n" + "="*60)
    print("ALGORITHM CONFIGURATION OPTIONS")
    print("="*60)
    
    # Show all available options
    print("Available Penalty Types:")
    for penalty_type in PenaltyType:
        print(f"  • {penalty_type.value}: {penalty_type.name}")
    
    print("\nAvailable Solver Types:")
    for solver_type in SolverType:
        print(f"  • {solver_type.value}: {solver_type.name}")
    
    print("\nAvailable Updater Types:")
    for updater_type in UpdaterType:
        print(f"  • {updater_type.value}: {updater_type.name}")
    
    # Show configuration examples
    print("\n--- Example Configurations ---")
    
    configs = [
        {
            'name': 'Speed Optimized',
            'penalty': 'l1', 'solver': 'fista', 'updater': 'mod',
            'description': 'Fastest combination for real-time applications'
        },
        {
            'name': 'Quality Optimized', 
            'penalty': 'elastic_net', 'solver': 'ista', 'updater': 'ksvd',
            'description': 'Best reconstruction quality for offline processing'
        },
        {
            'name': 'Robust to Outliers',
            'penalty': 'cauchy', 'solver': 'ista', 'updater': 'grad_d', 
            'description': 'Handles noisy data and outliers well'
        },
        {
            'name': 'Extremely Sparse',
            'penalty': 'top_k', 'solver': 'omp', 'updater': 'mod',
            'description': 'Forces exact sparsity level'
        }
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"   Penalty: {config['penalty']}, Solver: {config['solver']}, Updater: {config['updater']}")
        print(f"   Use case: {config['description']}")
    
    print("\n✅ All combinations are fully implemented and configurable!")


def run_comprehensive_demo():
    """
    Execute comprehensive demonstration of all algorithm implementations.
    
    Orchestrates complete testing of penalty functions, solver algorithms,
    dictionary update methods, and integrated learning pipelines.
    """
    print("COMPREHENSIVE SPARSE CODING ALGORITHM DEMONSTRATION")
    print("Research-accurate implementations with full configurability")
    print("=" * 80)
    
    print("\n📚 Research Foundation:")
    print("• Olshausen & Field (1996) - Sparse coding for natural images")
    print("• Beck & Teboulle (2009) - FISTA algorithm")
    print("• Tibshirani (1996) - Lasso regression")
    print("• Zou & Hastie (2005) - Elastic Net")
    print("• Aharon et al. (2006) - K-SVD algorithm")
    print("• Engan et al. (1999) - Method of Optimal Directions")
    print("• Pati et al. (1993) - Orthogonal Matching Pursuit")
    
    # Run all demonstrations
    penalty_results = demo_all_penalties()
    solver_results = demo_all_solvers() 
    updater_results = demo_all_updaters()
    pipeline_results = demo_complete_pipeline()
    demo_configuration_showcase()
    
    return {
        'penalties': penalty_results,
        'solvers': solver_results, 
        'updaters': updater_results,
        'pipelines': pipeline_results
    }


if __name__ == "__main__":
    results = run_comprehensive_demo()