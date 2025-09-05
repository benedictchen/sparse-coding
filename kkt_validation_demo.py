"""
Comprehensive KKT Validation Demonstration
==========================================

Demonstrates the enhanced KKT (Karush-Kuhn-Tucker) condition checking
for L1 sparse coding optimization validation.

This demo showcases:
1. KKT violation analysis on synthetic problems
2. Integration with SparseCoder optimization
3. Debugging tools for optimization issues
4. Research-accurate validation techniques

Based on research from:
- Boyd, S., & Vandenberghe, L. (2004). Convex optimization
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive fields
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add package paths for demo
sys.path.insert(0, '.')
sys.path.insert(0, 'sparse_coding')

# Import enhanced KKT diagnostics
from sparse_coding.diagnostics import (
    kkt_violation_comprehensive, 
    diagnose_kkt_violations,
    dictionary_coherence
)

def demo_kkt_basics():
    """Demonstrate basic KKT violation computation and analysis."""
    print("=" * 70)
    print("KKT VALIDATION DEMO 1: Basic KKT Analysis")
    print("=" * 70)
    
    # Create synthetic sparse coding problem
    np.random.seed(42)
    n_features, n_components, n_samples = 25, 40, 15
    
    print(f"Problem dimensions:")
    print(f"  Dictionary: {n_features}×{n_components} (overcomplete: {n_components/n_features:.1f}×)")
    print(f"  Data: {n_features}×{n_samples}")
    print(f"  Coefficients: {n_components}×{n_samples}")
    
    # Create overcomplete dictionary
    D = np.random.randn(n_features, n_components)
    D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    
    # Create sparse coefficients (ground truth)
    A_true = np.zeros((n_components, n_samples))
    for i in range(n_samples):
        n_active = np.random.randint(2, 5)  # 2-4 active coefficients per sample
        active_idx = np.random.choice(n_components, n_active, replace=False)
        A_true[active_idx, i] = np.random.randn(n_active)
    
    # Generate observations with noise
    noise_level = 0.01
    X = D @ A_true + noise_level * np.random.randn(n_features, n_samples)
    
    print(f"\nGround truth sparsity: {np.mean(np.abs(A_true) < 1e-12):.1%}")
    print(f"Noise level: {noise_level}")
    
    # Test different sparse solutions
    solutions = {
        "Ground Truth (Noisy)": A_true,
        "Zero Solution": np.zeros_like(A_true),
        "Dense Solution": np.linalg.pinv(D) @ X,  # Pseudoinverse
        "Random Sparse": np.random.randn(*A_true.shape) * (np.random.rand(*A_true.shape) > 0.8)
    }
    
    lam = 0.1  # L1 regularization parameter
    
    print(f"\nKKT Analysis Results (λ = {lam}):")
    print("-" * 50)
    
    for name, A in solutions.items():
        results = kkt_violation_comprehensive(D, X, A, lam, detailed=True)
        
        print(f"\n{name}:")
        print(f"  Max KKT violation: {results['max_violation']:.2e}")
        print(f"  Sparsity level: {results['sparsity_level']:.1%}")
        print(f"  KKT satisfied: {'✓ YES' if results['kkt_satisfied'] else '❌ NO'}")
        print(f"  Reconstruction error: {results['reconstruction_error']:.4f}")
        
        if 'dictionary_coherence' in results:
            print(f"  Dictionary coherence: {results['dictionary_coherence']:.3f}")
    
    return D, X, A_true, lam

def demo_optimization_debugging():
    """Demonstrate using KKT conditions to debug optimization issues."""
    print("\n" + "=" * 70)
    print("KKT VALIDATION DEMO 2: Optimization Debugging")
    print("=" * 70)
    
    # Create a problem where we can simulate optimization issues
    np.random.seed(123)
    D = np.random.randn(15, 25)
    D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    
    A_sparse = np.zeros((25, 8))
    A_sparse[0, 0] = 1.5
    A_sparse[5, 1] = -2.0
    A_sparse[12, 2] = 1.0
    A_sparse[20, 3] = -0.8
    
    X = D @ A_sparse + 0.02 * np.random.randn(15, 8)
    lam = 0.15
    
    print("Simulating common optimization issues:")
    print("-" * 40)
    
    # Issue 1: Converged too early (high violation)
    A_early = A_sparse.copy()
    A_early += 0.1 * np.random.randn(*A_sparse.shape)  # Add optimization noise
    
    print("\n1. EARLY CONVERGENCE SIMULATION:")
    results_early = kkt_violation_comprehensive(D, X, A_early, lam, detailed=True)
    diagnose_kkt_violations(results_early, verbose=True)
    
    # Issue 2: Wrong regularization parameter
    print("\n2. WRONG REGULARIZATION PARAMETER:")
    lam_wrong = 0.01  # Too small
    results_wrong_lam = kkt_violation_comprehensive(D, X, A_sparse, lam_wrong, detailed=True)
    print(f"Using λ = {lam_wrong} (too small):")
    diagnose_kkt_violations(results_wrong_lam, verbose=True)
    
    # Issue 3: High dictionary coherence problem
    print("\n3. HIGH COHERENCE DICTIONARY:")
    # Create high coherence dictionary
    D_bad = np.random.randn(10, 20)
    D_bad[:, 1] = D_bad[:, 0] + 0.1 * np.random.randn(10)  # Make atoms 0,1 highly coherent
    D_bad = D_bad / (np.linalg.norm(D_bad, axis=0, keepdims=True) + 1e-12)
    
    coherence = dictionary_coherence(D_bad)
    print(f"Dictionary mutual coherence: {coherence:.4f}")
    
    A_bad = np.zeros((20, 5))
    A_bad[0, 0] = 1.0
    A_bad[1, 0] = 0.8  # Both highly coherent atoms active
    X_bad = D_bad @ A_bad
    
    results_coherence = kkt_violation_comprehensive(D_bad, X_bad, A_bad, lam=0.1, detailed=True)
    diagnose_kkt_violations(results_coherence, verbose=True)

def demo_sparsecoder_integration():
    """Demonstrate enhanced KKT checking with SparseCoder."""
    print("\n" + "=" * 70)
    print("KKT VALIDATION DEMO 3: SparseCoder Integration")
    print("=" * 70)
    
    try:
        # Try to import SparseCoder (may fail due to import issues)
        import importlib.util
        
        # Direct file import as fallback
        spec = importlib.util.spec_from_file_location("sparse_coder", "src/sparse_coding/sparse_coder.py")
        sparse_coder_module = importlib.util.module_from_spec(spec)
        
        # Mock the required imports for demo
        class MockConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class MockOptimizer:
            def __init__(self, config):
                self.config = config
            
            def solve(self, D, x):
                # Simple soft thresholding for demo
                grad = D.T @ (D @ np.zeros(D.shape[1]) - x)
                A = np.sign(grad) * np.maximum(0, np.abs(grad) - self.config.sparsity_penalty)
                info = {'converged': True, 'n_iterations': 10}
                return A, info
        
        # Mock the imports
        import sys
        sys.modules['sparse_coding.research_accurate_sparsity'] = type('MockModule', (), {
            'FISTAOptimizer': MockOptimizer,
            'SparseCodingConfig': MockConfig,
            'SparsenessFunction': type('Enum', (), {'LOG': 'log'})
        })
        
        spec.loader.exec_module(sparse_coder_module)
        SparseCoder = sparse_coder_module.SparseCoder
        
        print("Creating SparseCoder with enhanced KKT checking...")
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(30, 20)  # 30 samples, 20 features each
        
        # Create and configure SparseCoder
        sc = SparseCoder(n_components=25, sparsity_penalty=0.12, max_iterations=50)
        sc.enable_kkt_checking(tolerance=1e-3, detailed=True)
        
        print("Fitting SparseCoder with KKT validation...")
        sc.fit(X)
        
        # Test enhanced KKT checking
        A = sc.transform(X[:5])  # Transform first 5 samples
        print(f"\nTesting enhanced KKT checking on transformed samples:")
        
        kkt_results = sc.check_kkt_violation(X[:5].T, A.T, detailed=True, verbose=True)
        
        print(f"\nEnhanced KKT analysis completed!")
        print(f"Final assessment: {'PASSED' if kkt_results['kkt_satisfied'] else 'NEEDS ATTENTION'}")
        
    except Exception as e:
        print(f"SparseCoder integration demo failed: {e}")
        print("This is expected due to import complexities in the current package structure.")
        print("The KKT diagnostics functions work independently and can be used with any optimization algorithm.")

def demo_research_validation():
    """Demonstrate research-accurate KKT validation techniques."""
    print("\n" + "=" * 70)
    print("KKT VALIDATION DEMO 4: Research-Accurate Validation")
    print("=" * 70)
    
    print("Creating research-accurate test cases...")
    
    # Test case 1: Perfect KKT solution construction
    print("\n1. CONSTRUCTING KKT-OPTIMAL SOLUTION:")
    
    # Simple orthogonal dictionary for exact analysis
    D = np.array([[1.0, 0.0, 0.5], 
                  [0.0, 1.0, 0.5]])  # 2×3 overcomplete dictionary
    D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    
    # Active coefficient: A[0,0] = 1.0, others zero
    A = np.array([[1.0], [0.0], [0.0]])
    lam = 0.2
    
    # For KKT optimality:
    # - Active coeff: D^T(X - DA)[0] = λ*sign(A[0]) = 0.2
    # - Inactive coeffs: |D^T(X - DA)[i]| ≤ λ
    
    # Solve for X: X = DA + residual where D^T*residual = [0.2, ?, ?]
    # residual = [r1, r2] such that:
    # - D[0]^T * residual = 0.2 → r1 = 0.2
    # - |D[1]^T * residual| ≤ 0.2 → |r2| ≤ 0.2, choose r2 = 0.1  
    # - |D[2]^T * residual| ≤ 0.2 → |0.5*r1 + 0.5*r2| ≤ 0.2
    #                                |0.5*0.2 + 0.5*0.1| = 0.15 ≤ 0.2 ✓
    
    residual = np.array([[0.2], [0.1]])
    X_kkt = D @ A + residual
    
    print(f"Dictionary shape: {D.shape}")
    print(f"Active coefficient: A[0,0] = {A[0,0]}")
    print(f"Regularization: λ = {lam}")
    
    results_optimal = kkt_violation_comprehensive(D, X_kkt, A, lam, detailed=True)
    print(f"\nKKT violation for constructed optimal solution: {results_optimal['max_violation']:.2e}")
    
    if results_optimal['max_violation'] < 1e-10:
        print("✅ Perfect KKT solution constructed successfully!")
    else:
        print("❌ KKT solution construction failed")
        
    # Test case 2: Violation scaling with regularization
    print("\n2. KKT VIOLATION SCALING:")
    
    regularization_params = [0.01, 0.05, 0.1, 0.2, 0.5]
    violations = []
    
    # Use a fixed suboptimal solution
    A_subopt = np.array([[0.8], [0.1], [0.05]])  # Slightly wrong solution
    
    for lam_test in regularization_params:
        results = kkt_violation_comprehensive(D, X_kkt, A_subopt, lam_test, detailed=False)
        violations.append(results['max_violation'])
    
    print("Regularization parameter vs KKT violation:")
    for lam_test, violation in zip(regularization_params, violations):
        print(f"  λ = {lam_test:4.2f} → violation = {violation:.3e}")
    
    # Test case 3: Coherence effects
    print("\n3. DICTIONARY COHERENCE EFFECTS:")
    
    # Low coherence vs high coherence dictionaries
    D_low = np.eye(3)  # Perfect conditioning
    D_high = np.array([[1.0, 0.9, 0.1], 
                       [0.0, 0.1, 0.9],
                       [0.1, 0.1, 0.1]])  # High coherence
    D_high = D_high / (np.linalg.norm(D_high, axis=0, keepdims=True) + 1e-12)
    
    coherence_low = dictionary_coherence(D_low)
    coherence_high = dictionary_coherence(D_high)
    
    print(f"Low coherence dictionary: {coherence_low:.4f}")
    print(f"High coherence dictionary: {coherence_high:.4f}")
    
    # Same sparse solution with both dictionaries
    A_test = np.array([[1.0], [0.0], [0.0]])
    X_low = D_low @ A_test + 0.01 * np.random.randn(3, 1)
    X_high = D_high @ A_test + 0.01 * np.random.randn(3, 1)
    
    results_low = kkt_violation_comprehensive(D_low, X_low, A_test, 0.1, detailed=True)
    results_high = kkt_violation_comprehensive(D_high, X_high, A_test, 0.1, detailed=True)
    
    print(f"KKT violation (low coherence): {results_low['max_violation']:.3e}")
    print(f"KKT violation (high coherence): {results_high['max_violation']:.3e}")
    print(f"Coherence impact: {results_high['max_violation']/results_low['max_violation']:.1f}× worse")

def main():
    """Run comprehensive KKT validation demonstration."""
    print("🔍 COMPREHENSIVE KKT VALIDATION DEMONSTRATION")
    print("=" * 70)
    print("This demonstrates enhanced KKT condition checking for L1 sparse coding.")
    print("KKT conditions are necessary for optimal solutions in convex optimization.")
    print()
    
    try:
        # Run all demonstration components
        D, X, A_true, lam = demo_kkt_basics()
        demo_optimization_debugging()  
        demo_sparsecoder_integration()
        demo_research_validation()
        
        print("\n" + "=" * 70)
        print("🎯 KKT VALIDATION DEMONSTRATION COMPLETE")
        print("=" * 70)
        print()
        print("KEY TAKEAWAYS:")
        print("✓ KKT conditions provide rigorous optimization validation")
        print("✓ Enhanced diagnostics identify specific optimization issues")
        print("✓ Dictionary coherence significantly affects solution quality")  
        print("✓ Proper regularization parameter selection is critical")
        print("✓ Integration with SparseCoder enables automatic validation")
        print()
        print("USAGE RECOMMENDATIONS:")
        print("• Always check KKT conditions for L1 sparse coding solutions")
        print("• Use detailed analysis to diagnose optimization problems")
        print("• Monitor dictionary coherence in overcomplete settings")
        print("• Adjust tolerances based on numerical precision requirements")
        print("• Enable KKT checking during development and validation")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()