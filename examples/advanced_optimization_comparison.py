#!/usr/bin/env python3
"""
Advanced Optimization Methods Comparison

Compares different optimization algorithms for sparse coding:
- ISTA (Iterative Soft Thresholding)
- FISTA (Fast ISTA)
- Coordinate Descent
- Adaptive FISTA

Shows convergence properties and computational efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys

# Add sparse_coding to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sparse_coding.advanced_optimization import (
    AdvancedOptimizer, L1Proximal, ElasticNetProximal,
    create_advanced_sparse_coder
)


def create_test_problem(n_features: int = 100, n_atoms: int = 50, sparsity: float = 0.1):
    """Create a test sparse coding problem with known ground truth"""
    
    # Generate random dictionary
    np.random.seed(42)
    dictionary = np.random.randn(n_features, n_atoms)
    dictionary = dictionary / np.linalg.norm(dictionary, axis=0)
    
    # Generate sparse ground truth
    true_coeffs = np.zeros(n_atoms)
    n_active = int(sparsity * n_atoms)
    active_indices = np.random.choice(n_atoms, n_active, replace=False)
    true_coeffs[active_indices] = np.random.randn(n_active)
    
    # Generate noisy observation
    signal = dictionary @ true_coeffs + 0.01 * np.random.randn(n_features)
    
    return dictionary, signal, true_coeffs


def run_optimization_comparison():
    """Compare different optimization algorithms"""
    
    print("🔬 Advanced Optimization Methods Comparison")
    print("==========================================")
    
    # Create test problem
    dictionary, signal, true_coeffs = create_test_problem(n_features=50, n_atoms=100, sparsity=0.15)
    sparsity_penalty = 0.1
    
    print(f"Problem size: {dictionary.shape[0]} features, {dictionary.shape[1]} atoms")
    print(f"Ground truth sparsity: {np.mean(np.abs(true_coeffs) > 1e-6):.3f}")
    print(f"Sparsity penalty: {sparsity_penalty}")
    
    # Initialize optimizers
    methods = {
        'ISTA': 'ista',
        'FISTA': 'fista', 
        'Coordinate Descent': 'coordinate_descent',
        'Adaptive FISTA': 'adaptive_fista'
    }
    
    results = {}
    
    print("\n🚀 Running optimization methods...")
    
    for method_name, method_func in methods.items():
        print(f"\nRunning {method_name}...")
        
        # Create optimizer
        optimizer = create_advanced_sparse_coder(
            dictionary, 
            penalty_type='l1',
            penalty_params={'lam': sparsity_penalty},
            max_iter=1000,
            tolerance=1e-8
        )
        
        # Run optimization
        start_time = time.time()
        result = getattr(optimizer, method_func)(signal)
        end_time = time.time()
        
        # Store results
        results[method_name] = {
            'solution': result['solution'],
            'iterations': result['iterations'],
            'converged': result['converged'],
            'final_objective': result['final_objective'],
            'runtime': end_time - start_time,
            'history': result['history']
        }
        
        # Compute recovery error
        recovery_error = np.linalg.norm(result['solution'] - true_coeffs)
        
        print(f"  ✓ Converged: {result['converged']}")
        print(f"  ✓ Iterations: {result['iterations']}")
        print(f"  ✓ Runtime: {end_time - start_time:.3f}s")
        print(f"  ✓ Recovery error: {recovery_error:.6f}")
        print(f"  ✓ Final objective: {result['final_objective']:.6f}")
    
    # Create comparison plots
    print("\n📊 Creating comparison plots...")
    
    # Convergence plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Optimization Methods Comparison', fontsize=16)
    
    # Objective convergence
    ax = axes[0, 0]
    for method_name, result in results.items():
        history = result['history']
        if 'objectives' in history:
            ax.semilogy(history['objectives'], label=method_name)
    ax.set_title('Objective Function Convergence')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residual convergence
    ax = axes[0, 1]
    for method_name, result in results.items():
        history = result['history']
        if 'residuals' in history:
            ax.semilogy(history['residuals'], label=method_name)
    ax.set_title('Residual Convergence')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||x_k - x_{k-1}||')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Runtime comparison
    ax = axes[1, 0]
    methods_list = list(results.keys())
    runtimes = [results[method]['runtime'] for method in methods_list]
    ax.bar(methods_list, runtimes, alpha=0.7)
    ax.set_title('Runtime Comparison')
    ax.set_ylabel('Time (seconds)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Recovery error comparison
    ax = axes[1, 1]
    recovery_errors = [np.linalg.norm(results[method]['solution'] - true_coeffs) 
                      for method in methods_list]
    ax.bar(methods_list, recovery_errors, alpha=0.7)
    ax.set_title('Recovery Error')
    ax.set_ylabel('||x* - x_true||')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Solution comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle('Solution Comparison', fontsize=16)
    
    axes = axes.flatten()
    
    # Ground truth
    axes[0].stem(true_coeffs, linefmt='k-', markerfmt='ko', basefmt='k-')
    axes[0].set_title('Ground Truth')
    axes[0].grid(True, alpha=0.3)
    
    # Show solutions from different methods
    for i, (method_name, result) in enumerate(list(results.items())[:3]):
        axes[i+1].stem(result['solution'], alpha=0.7)
        axes[i+1].set_title(f'{method_name} Solution')
        axes[i+1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\n📋 Summary Table:")
    print("="*80)
    print(f"{'Method':<20} {'Iterations':<12} {'Runtime (s)':<12} {'Recovery Error':<15} {'Converged':<10}")
    print("="*80)
    
    for method_name, result in results.items():
        recovery_error = np.linalg.norm(result['solution'] - true_coeffs)
        print(f"{method_name:<20} {result['iterations']:<12} {result['runtime']:<12.3f} "
              f"{recovery_error:<15.6f} {str(result['converged']):<10}")
    
    print("="*80)
    
    # Save results
    output_dir = Path("optimization_comparison_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "optimization_comparison.png", dpi=150, bbox_inches='tight')
    
    print(f"\n💾 Results saved to: {output_dir.absolute()}")


def demonstrate_different_penalties():
    """Demonstrate different penalty functions"""
    
    print("\n\n🎯 Different Penalty Functions Demo")
    print("==================================")
    
    # Create test problem
    dictionary, signal, true_coeffs = create_test_problem(n_features=30, n_atoms=50, sparsity=0.2)
    
    penalties = {
        'L1': {'type': 'l1', 'params': {'lam': 0.1}},
        'Elastic Net': {'type': 'elastic_net', 'params': {'l1': 0.08, 'l2': 0.02}},
        'Non-negative L1': {'type': 'non_negative_l1', 'params': {'lam': 0.1}}
    }
    
    penalty_results = {}
    
    for penalty_name, penalty_config in penalties.items():
        print(f"\nTesting {penalty_name} penalty...")
        
        optimizer = create_advanced_sparse_coder(
            dictionary,
            penalty_type=penalty_config['type'],
            penalty_params=penalty_config['params'],
            max_iter=500
        )
        
        result = optimizer.fista(signal)
        penalty_results[penalty_name] = result
        
        sparsity = np.mean(np.abs(result['solution']) > 1e-6)
        recovery_error = np.linalg.norm(result['solution'] - true_coeffs)
        
        print(f"  ✓ Iterations: {result['iterations']}")
        print(f"  ✓ Sparsity: {sparsity:.3f}")
        print(f"  ✓ Recovery error: {recovery_error:.6f}")
    
    # Plot penalty comparison
    fig, axes = plt.subplots(1, len(penalties), figsize=(15, 4))
    fig.suptitle('Different Penalty Functions', fontsize=16)
    
    for i, (penalty_name, result) in enumerate(penalty_results.items()):
        axes[i].stem(result['solution'], alpha=0.7)
        sparsity = np.mean(np.abs(result['solution']) > 1e-6)
        axes[i].set_title(f'{penalty_name}\n(Sparsity: {sparsity:.3f})')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Run all demonstrations"""
    run_optimization_comparison()
    demonstrate_different_penalties()
    
    print("\n✅ Advanced optimization comparison completed!")


if __name__ == "__main__":
    main()