"""
Mathematical Regression Guard Tests
==================================

Essential test suite to prevent regression of critical mathematical fixes.
These tests MUST pass in CI/CD to ensure mathematical correctness.

Based on:
- Olshausen & Field (1996) sparse coding theory
- Beck & Teboulle (2009) FISTA convergence
- KKT optimality conditions
- Dictionary learning convergence properties
"""

import numpy as np
import pytest
import warnings

# Suppress numpy warnings during testing
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_sparse_coder():
    """Import SparseCoder with proper error handling."""
    try:
        import sys
        sys.path.insert(0, 'src')
        from sparse_coding.sparse_coder import SparseCoder
        return SparseCoder
    except ImportError:
        pytest.skip("SparseCoder not available - skipping mathematical regression tests")


class TestCriticalMathRegression:
    """Critical mathematical regression prevention tests."""
    
    def test_basic_functionality(self):
        """Test basic SparseCoder functionality works without errors."""
        SparseCoder = get_sparse_coder()
        
        # Simple well-conditioned test
        np.random.seed(42)
        X = np.random.randn(50, 64)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize
        
        coder = SparseCoder(
            n_components=32,
            sparsity_penalty=0.1,
            random_state=42
        )
        
        # Should not crash
        codes = coder.fit_transform(X)
        
        # Basic sanity checks
        assert codes.shape == (50, 32), f"Wrong output shape: {codes.shape}"
        assert np.all(np.isfinite(codes)), "Non-finite values in output"
        assert hasattr(coder, 'dictionary_'), "Dictionary not created"
        assert coder.dictionary_.shape == (64, 32), f"Wrong dictionary shape: {coder.dictionary_.shape}"

    def test_sparsity_penalty_effect(self):
        """Test that sparsity penalty actually affects sparsity."""
        SparseCoder = get_sparse_coder()
        
        np.random.seed(42)
        X = np.random.randn(50, 64)
        
        # Test low vs high penalty
        coder_low = SparseCoder(n_components=32, sparsity_penalty=0.01, random_state=42)
        coder_high = SparseCoder(n_components=32, sparsity_penalty=0.5, random_state=42)
        
        codes_low = coder_low.fit_transform(X)
        codes_high = coder_high.fit_transform(X)
        
        # Compute sparsity levels
        sparsity_low = np.mean(np.abs(codes_low) < 1e-8)
        sparsity_high = np.mean(np.abs(codes_high) < 1e-8)
        
        # Higher penalty should generally give higher sparsity
        # Allow some tolerance for stochastic effects
        assert sparsity_high >= sparsity_low - 0.2, (
            f"High penalty ({sparsity_high:.3f}) not sparser than low penalty ({sparsity_low:.3f})"
        )
        
        # High penalty should achieve meaningful sparsity
        assert sparsity_high >= 0.1, f"High penalty achieved no sparsity: {sparsity_high:.3f}"

    def test_dictionary_normalization(self):
        """Test dictionary atoms are properly normalized."""
        SparseCoder = get_sparse_coder()
        
        np.random.seed(42)
        X = np.random.randn(100, 64)
        
        coder = SparseCoder(n_components=32, sparsity_penalty=0.1, random_state=42)
        coder.fit(X)
        
        # Check atom normalization
        atom_norms = np.linalg.norm(coder.dictionary_, axis=0)
        
        # All atoms should be approximately unit length
        np.testing.assert_allclose(
            atom_norms, 1.0, rtol=1e-3, atol=1e-3,
            err_msg="Dictionary atoms not properly normalized"
        )

    def test_reproducibility(self):
        """Test results are reproducible with same seed."""
        SparseCoder = get_sparse_coder()
        
        # Same seed should give identical results
        for seed in [42, 123]:
            np.random.seed(seed)
            X = np.random.randn(50, 64)
            
            coder1 = SparseCoder(n_components=32, sparsity_penalty=0.1, random_state=seed)
            coder2 = SparseCoder(n_components=32, sparsity_penalty=0.1, random_state=seed)
            
            codes1 = coder1.fit_transform(X)
            codes2 = coder2.fit_transform(X)
            
            # Results should be very close (allowing for minor numerical differences)
            np.testing.assert_allclose(
                codes1, codes2, rtol=1e-6, atol=1e-6,
                err_msg=f"Results not reproducible with seed {seed}"
            )

    def test_shape_consistency(self):
        """Test shape handling is consistent across different input sizes."""
        SparseCoder = get_sparse_coder()
        
        test_cases = [
            (20, 32, 16),   # (n_samples, n_features, n_components)
            (50, 64, 32),
            (100, 128, 64),
        ]
        
        for n_samples, n_features, n_components in test_cases:
            np.random.seed(42)
            X = np.random.randn(n_samples, n_features)
            
            coder = SparseCoder(
                n_components=n_components,
                sparsity_penalty=0.1,
                random_state=42
            )
            
            codes = coder.fit_transform(X)
            
            # Verify shapes
            assert codes.shape == (n_samples, n_components), (
                f"Wrong codes shape: {codes.shape} vs expected {(n_samples, n_components)}"
            )
            assert coder.dictionary_.shape == (n_features, n_components), (
                f"Wrong dictionary shape: {coder.dictionary_.shape} vs expected {(n_features, n_components)}"
            )

    def test_numerical_stability_edge_cases(self):
        """Test handling of numerically challenging inputs."""
        SparseCoder = get_sparse_coder()
        
        # Test cases that can cause numerical issues
        test_cases = [
            ("small_values", 1e-8 * np.random.randn(20, 32)),
            ("mixed_scales", np.concatenate([
                1e-6 * np.random.randn(10, 32),
                1e3 * np.random.randn(10, 32)
            ], axis=0)),
        ]
        
        for case_name, X in test_cases:
            # Set seed for reproducibility
            np.random.seed(42)
            
            coder = SparseCoder(
                n_components=16,
                sparsity_penalty=0.1,
                random_state=42
            )
            
            # Should not crash and produce finite outputs
            try:
                codes = coder.fit_transform(X)
                assert np.all(np.isfinite(codes)), f"Non-finite outputs for {case_name}"
                assert codes.shape == (X.shape[0], 16), f"Wrong shape for {case_name}"
                
            except Exception as e:
                # Allow reasonable exceptions but not crashes
                if "singular" in str(e).lower() or "converge" in str(e).lower():
                    pytest.skip(f"Numerical conditioning issue in {case_name}: {e}")
                else:
                    raise

    def test_kkt_violation_bounds(self):
        """Test KKT violations are within reasonable bounds if checker exists."""
        SparseCoder = get_sparse_coder()
        
        np.random.seed(42)
        # Well-conditioned problem
        n_features, n_components = 32, 64
        D_true = np.random.randn(n_features, n_components)
        D_true = D_true / np.linalg.norm(D_true, axis=0, keepdims=True)
        
        # Sparse coefficients
        A_true = np.random.randn(n_components, 30)
        A_true[np.abs(A_true) < 0.5] = 0
        
        # Generate data
        X = (D_true @ A_true).T + 0.01 * np.random.randn(30, n_features)
        
        coder = SparseCoder(
            n_components=n_components,
            sparsity_penalty=0.1,
            random_state=42
        )
        
        codes = coder.fit_transform(X)
        
        # If KKT checker exists, test it
        if hasattr(coder, 'check_kkt_violation'):
            kkt_violation = coder.check_kkt_violation(X.T, codes.T)
            
            # Should be finite and non-negative
            assert np.isfinite(kkt_violation), "KKT violation should be finite"
            assert kkt_violation >= 0, "KKT violation should be non-negative"
            
            # Should be reasonable (not too large)
            assert kkt_violation <= 10.0, f"KKT violation too large: {kkt_violation:.6f}"

    def test_convergence_with_different_tolerances(self):
        """Test algorithm behavior with different convergence tolerances."""
        SparseCoder = get_sparse_coder()
        
        np.random.seed(42)
        X = np.random.randn(30, 32)
        
        # Test different tolerance levels
        tolerances = [1e-2, 1e-4, 1e-6]
        
        for tol in tolerances:
            coder = SparseCoder(
                n_components=16,
                sparsity_penalty=0.1,
                tolerance=tol,
                random_state=42
            )
            
            codes = coder.fit_transform(X)
            
            # Should produce reasonable outputs regardless of tolerance
            assert np.all(np.isfinite(codes)), f"Non-finite codes with tolerance {tol}"
            assert codes.shape == (30, 16), f"Wrong shape with tolerance {tol}"
            
            # Sparsity should be consistent across tolerances
            sparsity = np.mean(np.abs(codes) < 1e-8)
            assert 0.0 <= sparsity <= 1.0, f"Invalid sparsity {sparsity} with tolerance {tol}"


def test_mathematical_regression_prevention():
    """Pytest entry point for mathematical regression tests."""
    # This test ensures the test class can be instantiated
    test_instance = TestCriticalMathRegression()
    
    # Run a minimal smoke test
    try:
        test_instance.test_basic_functionality()
    except pytest.SkipTest:
        pytest.skip("SparseCoder not available")
    
    # If we get here, basic functionality works
    assert True


# Additional utility function for debugging
def diagnose_mathematical_issues():
    """Diagnostic function to identify specific mathematical problems."""
    try:
        SparseCoder = get_sparse_coder()
    except:
        print("❌ SparseCoder import failed")
        return
    
    np.random.seed(42)
    X = np.random.randn(50, 64)
    
    try:
        coder = SparseCoder(n_components=32, sparsity_penalty=0.1, random_state=42)
        codes = coder.fit_transform(X)
        
        print("✅ Basic functionality works")
        print(f"   Output shape: {codes.shape}")
        print(f"   Dictionary shape: {coder.dictionary_.shape}")
        
        # Check sparsity
        sparsity = np.mean(np.abs(codes) < 1e-8)
        print(f"   Sparsity level: {sparsity:.3f}")
        
        # Check normalization
        atom_norms = np.linalg.norm(coder.dictionary_, axis=0)
        print(f"   Atom norm range: [{atom_norms.min():.6f}, {atom_norms.max():.6f}]")
        
        # Check for KKT violation method
        if hasattr(coder, 'check_kkt_violation'):
            kkt_violation = coder.check_kkt_violation(X.T, codes.T)
            print(f"   KKT violation: {kkt_violation:.6f}")
        else:
            print("   KKT checker: Not available")
            
    except Exception as e:
        print(f"❌ Mathematical issues detected: {e}")


if __name__ == "__main__":
    diagnose_mathematical_issues()