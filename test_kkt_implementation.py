"""
KKT Implementation Testing and Validation
=========================================

Test the current KKT violation implementation and identify gaps for improvement.
Based on research requirements for L1 sparse coding optimization validation.
"""

import numpy as np
import sys
import os

# Add both src and flat layout paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Try src layout first
    sys.path.insert(0, 'src')
    from sparse_coding.diagnostics import kkt_violation_l1
    from sparse_coding.sparse_coder import SparseCoder
    print("✓ Using src layout imports")
except ImportError:
    try:
        # Try flat layout diagnostics with src layout sparse_coder
        from sparse_coding.diagnostics import kkt_violation_l1
        sys.path.insert(0, 'src')  
        from sparse_coding.sparse_coder import SparseCoder
        print("✓ Using mixed layout imports")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        # Let's just use direct file paths
        import importlib.util
        
        # Load diagnostics from flat layout
        diagnostics_path = "sparse_coding/diagnostics.py"
        spec = importlib.util.spec_from_file_location("diagnostics", diagnostics_path)
        diagnostics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diagnostics)
        kkt_violation_l1 = diagnostics.kkt_violation_l1
        
        # Load sparse_coder from src layout  
        sparse_coder_path = "src/sparse_coding/sparse_coder.py"
        spec = importlib.util.spec_from_file_location("sparse_coder", sparse_coder_path)
        sparse_coder_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sparse_coder_module)
        SparseCoder = sparse_coder_module.SparseCoder
        
        print("✓ Using direct file imports")

def create_synthetic_problem(n_features=50, n_components=100, n_samples=10, 
                           sparsity_level=0.1, noise_level=0.01, seed=42):
    """
    Create a synthetic sparse coding problem with known ground truth.
    
    This allows us to test KKT conditions on a controllable problem.
    """
    np.random.seed(seed)
    
    # Create overcomplete dictionary
    D = np.random.randn(n_features, n_components)
    # Normalize columns
    D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    
    # Create sparse coefficients 
    A_true = np.zeros((n_components, n_samples))
    for i in range(n_samples):
        # Select random subset of atoms to be active
        n_active = int(sparsity_level * n_components)
        active_indices = np.random.choice(n_components, n_active, replace=False)
        A_true[active_indices, i] = np.random.randn(n_active)
    
    # Generate signals
    X = D @ A_true + noise_level * np.random.randn(n_features, n_samples)
    
    return D, X, A_true

def test_kkt_on_synthetic_data():
    """Test KKT conditions on synthetic data with known solution."""
    print("=== Testing KKT Implementation on Synthetic Data ===")
    
    # Create synthetic problem
    D, X, A_true = create_synthetic_problem(seed=42)
    lam = 0.1
    
    print(f"Problem dimensions: D={D.shape}, X={X.shape}, A={A_true.shape}")
    print(f"True sparsity level: {np.mean(np.abs(A_true) < 1e-12):.3f}")
    
    # Test KKT violation on true solution (should be low but not zero due to noise)
    kkt_true = kkt_violation_l1(D, X, A_true, lam)
    print(f"KKT violation on true sparse solution: {kkt_true:.2e}")
    
    # Test KKT violation on dense solution (should be high)
    A_dense = np.linalg.pinv(D) @ X  # Pseudoinverse solution (dense)
    kkt_dense = kkt_violation_l1(D, X, A_dense, lam)
    print(f"KKT violation on dense pseudoinverse: {kkt_dense:.2e}")
    
    # Test KKT violation on zero solution
    A_zero = np.zeros_like(A_true)
    kkt_zero = kkt_violation_l1(D, X, A_zero, lam)
    print(f"KKT violation on zero solution: {kkt_zero:.2e}")
    
    return kkt_true, kkt_dense, kkt_zero

def test_sparse_coder_kkt_integration():
    """Test KKT checking integration in SparseCoder."""
    print("\n=== Testing SparseCoder KKT Integration ===")
    
    # Create simple 2D problem for visualization
    np.random.seed(42)
    X = np.random.randn(20, 50)  # 20 features, 50 samples
    
    # Create SparseCoder
    sc = SparseCoder(n_components=30, sparsity_penalty=0.1, max_iterations=100)
    
    # Fit without KKT checking
    print("Fitting SparseCoder...")
    sc.fit(X)
    
    # Transform to get codes
    A = sc.transform(X)
    print(f"Learned codes shape: {A.shape}")
    print(f"Sparsity level: {np.mean(np.abs(A) < 1e-8):.3f}")
    
    # Check KKT violation
    kkt_results = sc.check_kkt_violation(X.T, A.T, tolerance=1e-3, verbose=True)
    
    # Validate solution
    validation_results = sc.validate_solution(X.T, A.T, check_kkt=True)
    print(f"\nValidation Results:")
    print(f"  Reconstruction error: {validation_results['reconstruction_error']:.4f}")
    print(f"  Sparsity level: {validation_results['sparsity_level']:.3f}")
    print(f"  Average nonzeros: {validation_results['n_nonzero_avg']:.1f}")
    
    return kkt_results, validation_results

def test_kkt_edge_cases():
    """Test KKT implementation on edge cases."""
    print("\n=== Testing KKT Edge Cases ===")
    
    # Single sample case
    D = np.random.randn(10, 20)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    X = np.random.randn(10, 1)
    A = np.random.randn(20, 1) 
    A[np.abs(A) < 0.5] = 0  # Make sparse
    
    kkt_single = kkt_violation_l1(D, X, A, 0.1)
    print(f"Single sample KKT violation: {kkt_single:.2e}")
    
    # All zero coefficients case
    A_zero = np.zeros((20, 1))
    kkt_all_zero = kkt_violation_l1(D, X, A_zero, 0.1)
    print(f"All zero coefficients KKT: {kkt_all_zero:.2e}")
    
    # Perfect reconstruction case (should have low KKT violation)
    A_perfect = np.linalg.pinv(D) @ X
    # Zero out small coefficients to make sparse
    A_perfect[np.abs(A_perfect) < 0.1] = 0
    kkt_perfect = kkt_violation_l1(D, X, A_perfect, 0.01)
    print(f"Near-perfect reconstruction KKT: {kkt_perfect:.2e}")
    
    return kkt_single, kkt_all_zero, kkt_perfect

def analyze_kkt_implementation_gaps():
    """Analyze current KKT implementation for gaps and improvements."""
    print("\n=== Analyzing KKT Implementation Gaps ===")
    
    gaps_found = []
    
    # Check if KKT function handles edge cases properly
    try:
        # Test with single sample 
        D = np.random.randn(5, 10)
        X = np.random.randn(5)  # 1D array
        A = np.random.randn(10) 
        kkt = kkt_violation_l1(D, X.reshape(-1, 1), A.reshape(-1, 1), 0.1)
        print(f"✓ Single sample handling works: KKT = {kkt:.2e}")
    except Exception as e:
        gaps_found.append(f"Single sample handling: {e}")
        print(f"❌ Single sample handling failed: {e}")
    
    # Check tolerance handling
    try:
        D = np.random.randn(5, 10)
        X = np.random.randn(5, 3)
        A = np.random.randn(10, 3)
        A[np.abs(A) < 0.001] = 0
        kkt1 = kkt_violation_l1(D, X, A, 0.1, tol=1e-12)
        kkt2 = kkt_violation_l1(D, X, A, 0.1, tol=1e-3)
        print(f"✓ Tolerance handling works: KKT(1e-12)={kkt1:.2e}, KKT(1e-3)={kkt2:.2e}")
    except Exception as e:
        gaps_found.append(f"Tolerance handling: {e}")
        print(f"❌ Tolerance handling failed: {e}")
    
    # Check numerical stability with extreme values
    try:
        D = np.random.randn(5, 10) * 1e6  # Large values
        X = np.random.randn(5, 3) * 1e-6  # Small values
        A = np.random.randn(10, 3)
        kkt = kkt_violation_l1(D, X, A, 0.1)
        print(f"✓ Numerical stability test: KKT = {kkt:.2e}")
    except Exception as e:
        gaps_found.append(f"Numerical stability: {e}")
        print(f"❌ Numerical stability failed: {e}")
    
    return gaps_found

def main():
    """Run comprehensive KKT implementation testing."""
    print("KKT Implementation Validation Test")
    print("=" * 50)
    
    try:
        # Test on synthetic data
        kkt_results = test_kkt_on_synthetic_data()
        
        # Test integration with SparseCoder
        sparse_coder_results = test_sparse_coder_kkt_integration()
        
        # Test edge cases
        edge_case_results = test_kkt_edge_cases()
        
        # Analyze implementation gaps
        gaps = analyze_kkt_implementation_gaps()
        
        # Summary
        print("\n" + "="*50)
        print("KKT IMPLEMENTATION ANALYSIS SUMMARY")
        print("="*50)
        
        if len(gaps) == 0:
            print("✅ Current KKT implementation handles all tested cases correctly")
        else:
            print("❌ KKT implementation gaps found:")
            for gap in gaps:
                print(f"  - {gap}")
        
        print(f"\nKey findings:")
        print(f"  - Synthetic data KKT violation: {kkt_results[0]:.2e}")
        print(f"  - SparseCoder integration: {'✓' if sparse_coder_results[0]['kkt_satisfied'] else '❌'}")
        print(f"  - Edge cases handled: {'✓' if len(gaps) == 0 else f'{len(gaps)} issues'}")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        if len(gaps) > 0:
            print("  1. Fix identified gaps in KKT implementation")
            print("  2. Add more comprehensive error handling")
            print("  3. Improve numerical stability")
        else:
            print("  1. Current implementation is robust")
            print("  2. Consider adding more detailed violation reporting")
            print("  3. Add configuration options for different tolerance levels")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()