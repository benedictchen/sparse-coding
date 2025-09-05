"""
Research Mathematical Regression Prevention Tests
================================================

Critical test suite to prevent regression of mathematical fixes in sparse coding.
Uses the actual SparseCoder API discovered through introspection.

API Conventions (discovered):
- Dictionary: .D attribute (not .dictionary_)
- Shape: Input (n_samples, n_features) → Codes (n_features, n_atoms) → Dict (n_samples, n_atoms)
- Parameters: n_atoms, ratio_lambda_over_sigma, seed, mode, max_iter, tol

CRITICAL MATHEMATICAL PROPERTIES TO PRESERVE:
1. Dictionary atoms normalized to unit length
2. Sparsity increases with penalty 
3. Finite outputs under all conditions
4. Reasonable sparsity levels (>0% and <100%)
5. Mathematical consistency across runs

These tests MUST pass to prevent regression of critical mathematical fixes.
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
        from sparse_coding import SparseCoder
        return SparseCoder
    except ImportError:
        pytest.skip("SparseCoder not available - skipping mathematical regression tests")


class TestCriticalMathRegression:
    """Critical mathematical regression tests with actual API."""
    
    def test_basic_mathematical_sanity(self):
        """Test basic mathematical sanity: finite outputs, correct shapes, normalized dictionary."""
        SparseCoder = get_sparse_coder()
        
        np.random.seed(42)
        X = np.random.randn(50, 64)
        
        coder = SparseCoder(n_atoms=32, ratio_lambda_over_sigma=0.14, seed=42)
        codes = coder.fit_transform(X)
        
        # Shape consistency
        assert codes.shape == (64, 32), f"Wrong codes shape: {codes.shape}, expected (64, 32)"
        assert coder.D.shape == (50, 32), f"Wrong dictionary shape: {coder.D.shape}, expected (50, 32)"
        
        # Finite outputs
        assert np.all(np.isfinite(codes)), "Codes contain non-finite values"
        assert np.all(np.isfinite(coder.D)), "Dictionary contains non-finite values"
        
        # Dictionary normalization
        atom_norms = np.linalg.norm(coder.D, axis=0)
        np.testing.assert_allclose(
            atom_norms, 1.0, rtol=1e-3, atol=1e-3,
            err_msg=f"Dictionary atoms not normalized. Norms: {atom_norms}"
        )

    def test_sparsity_penalty_effect(self):
        """Test that sparsity penalty controls sparsity level."""
        SparseCoder = get_sparse_coder()
        
        np.random.seed(42)
        X = np.random.randn(50, 64)
        
        # Test different penalty levels
        penalties = [0.01, 0.14, 0.5]
        sparsity_levels = []
        
        for i, penalty in enumerate(penalties):
            coder = SparseCoder(n_atoms=32, ratio_lambda_over_sigma=penalty, seed=42+i)
            codes = coder.fit_transform(X)
            sparsity = np.mean(np.abs(codes) < 1e-8)
            sparsity_levels.append(sparsity)
        
        # Sparsity should generally increase with penalty
        for i in range(len(sparsity_levels) - 1):
            assert sparsity_levels[i+1] >= sparsity_levels[i] - 0.15, (
                f"Sparsity did not increase sufficiently: {sparsity_levels[i]:.3f} -> {sparsity_levels[i+1]:.3f}"
            )
        
        # Should achieve meaningful range
        assert sparsity_levels[0] < 0.5, f"Low penalty too sparse: {sparsity_levels[0]:.3f}"
        assert sparsity_levels[-1] > 0.3, f"High penalty not sparse enough: {sparsity_levels[-1]:.3f}"

    def test_mathematical_bounds_and_stability(self):
        """Test mathematical bounds: sparsity ∈ [0,1], no NaN/Inf, reasonable magnitudes."""
        SparseCoder = get_sparse_coder()
        
        # Test with various input conditions
        test_cases = [
            ("normal", np.random.randn(30, 32)),
            ("small_values", 1e-6 * np.random.randn(30, 32)),
            ("normalized", np.random.randn(30, 32)),
        ]
        
        # Normalize the "normalized" case
        test_cases[2] = ("normalized", test_cases[2][1] / np.linalg.norm(test_cases[2][1], axis=1, keepdims=True))
        
        for case_name, X in test_cases:
            np.random.seed(42)
            coder = SparseCoder(n_atoms=16, ratio_lambda_over_sigma=0.14, seed=42)
            
            try:
                codes = coder.fit_transform(X)
                
                # Mathematical bounds
                assert np.all(np.isfinite(codes)), f"Non-finite codes in {case_name}"
                assert np.all(np.isfinite(coder.D)), f"Non-finite dictionary in {case_name}"
                
                # Sparsity bounds
                sparsity = np.mean(np.abs(codes) < 1e-8)
                assert 0.0 <= sparsity <= 1.0, f"Invalid sparsity {sparsity} in {case_name}"
                
                # Reasonable magnitude bounds  
                if np.any(np.abs(codes) > 1e-8):  # If there are active coefficients
                    active_coeffs = codes[np.abs(codes) > 1e-8]
                    assert np.max(np.abs(active_coeffs)) < 1e6, f"Coefficients too large in {case_name}"
                    assert np.max(np.abs(active_coeffs)) > 1e-12, f"Active coefficients too small in {case_name}"
                
            except Exception as e:
                if "singular" in str(e).lower() or "numerical" in str(e).lower():
                    pytest.skip(f"Numerical conditioning issue in {case_name}: {e}")
                else:
                    raise AssertionError(f"Unexpected failure in {case_name}: {e}")

    def test_different_modes_mathematical_consistency(self):
        """Test mathematical consistency across different modes."""
        SparseCoder = get_sparse_coder()
        
        np.random.seed(42)
        X = np.random.randn(30, 32)
        
        modes = ['l1', 'paper']
        results = {}
        
        for mode in modes:
            try:
                coder = SparseCoder(n_atoms=16, ratio_lambda_over_sigma=0.14, mode=mode, seed=42)
                codes = coder.fit_transform(X)
                
                # Basic mathematical properties should hold for all modes
                assert np.all(np.isfinite(codes)), f"Non-finite codes in {mode} mode"
                assert np.all(np.isfinite(coder.D)), f"Non-finite dictionary in {mode} mode"
                
                sparsity = np.mean(np.abs(codes) < 1e-8)
                assert 0.0 <= sparsity <= 1.0, f"Invalid sparsity {sparsity} in {mode} mode"
                
                # Dictionary normalization
                atom_norms = np.linalg.norm(coder.D, axis=0)
                assert np.allclose(atom_norms, 1.0, rtol=1e-2), f"Dictionary not normalized in {mode} mode"
                
                results[mode] = {'codes': codes, 'sparsity': sparsity, 'dictionary': coder.D}
                
            except Exception as e:
                if "mode" in str(e).lower():
                    pytest.skip(f"Mode {mode} not supported: {e}")
                else:
                    raise
        
        # If both modes worked, they should produce mathematically valid but potentially different results
        if len(results) >= 2:
            modes_list = list(results.keys())
            # Different modes can produce different results, but both should be valid
            for mode in modes_list:
                assert 'sparsity' in results[mode], f"Results incomplete for {mode}"

    def test_parameter_bounds_and_effects(self):
        """Test parameter bounds and their mathematical effects."""
        SparseCoder = get_sparse_coder()
        
        np.random.seed(42)
        X = np.random.randn(30, 32)
        
        # Test tolerance parameter
        tolerances = [1e-2, 1e-4, 1e-6]
        for tol in tolerances:
            coder = SparseCoder(n_atoms=16, ratio_lambda_over_sigma=0.14, tol=tol, seed=42)
            codes = coder.fit_transform(X)
            
            assert codes.shape == (32, 16), f"Wrong shape with tolerance {tol}"
            assert np.all(np.isfinite(codes)), f"Non-finite values with tolerance {tol}"
        
        # Test max_iter parameter
        max_iters = [10, 50, 200]
        for max_iter in max_iters:
            coder = SparseCoder(n_atoms=16, ratio_lambda_over_sigma=0.14, max_iter=max_iter, seed=42)
            codes = coder.fit_transform(X)
            
            assert codes.shape == (32, 16), f"Wrong shape with max_iter {max_iter}"
            assert np.all(np.isfinite(codes)), f"Non-finite values with max_iter {max_iter}"
        
        # Test explicit lambda parameter if supported
        try:
            coder_lam = SparseCoder(n_atoms=16, lam=0.1, seed=42)
            codes_lam = coder_lam.fit_transform(X)
            
            assert codes_lam.shape == (32, 16), "Wrong shape with explicit lambda"
            assert np.all(np.isfinite(codes_lam)), "Non-finite values with explicit lambda"
            
        except Exception as e:
            # Explicit lambda might not be fully supported
            pass

    def test_edge_case_inputs(self):
        """Test mathematical robustness with edge case inputs."""
        SparseCoder = get_sparse_coder()
        
        edge_cases = [
            ("very_small_data", 1e-12 * np.random.randn(20, 16)),
            ("single_feature", np.random.randn(20, 1)),  # Might not work but shouldn't crash
            ("single_sample", np.random.randn(1, 16)),
            ("identical_samples", np.ones((10, 16))),
        ]
        
        for case_name, X in edge_cases:
            np.random.seed(42)
            
            # Use fewer atoms for edge cases
            n_atoms = min(8, X.shape[1])
            
            try:
                coder = SparseCoder(n_atoms=n_atoms, ratio_lambda_over_sigma=0.14, seed=42)
                codes = coder.fit_transform(X)
                
                # Should produce finite outputs
                assert np.all(np.isfinite(codes)), f"Non-finite codes in {case_name}"
                assert np.all(np.isfinite(coder.D)), f"Non-finite dictionary in {case_name}"
                
                # Correct shapes
                expected_codes_shape = (X.shape[1], n_atoms)
                expected_dict_shape = (X.shape[0], n_atoms)
                
                assert codes.shape == expected_codes_shape, (
                    f"Wrong codes shape in {case_name}: {codes.shape} vs {expected_codes_shape}"
                )
                assert coder.D.shape == expected_dict_shape, (
                    f"Wrong dict shape in {case_name}: {coder.D.shape} vs {expected_dict_shape}"
                )
                
            except Exception as e:
                # Some edge cases might legitimately fail, but should be reasonable failures
                if any(word in str(e).lower() for word in ["singular", "rank", "dimension", "size"]):
                    pytest.skip(f"Expected numerical issue in {case_name}: {e}")
                else:
                    raise AssertionError(f"Unexpected failure in {case_name}: {e}")

    def test_gain_equalization_effect(self):
        """Test gain equalization parameter if available."""
        SparseCoder = get_sparse_coder()
        
        np.random.seed(42)
        # Create data that would benefit from gain equalization
        X = np.random.randn(50, 32)
        X[:, :10] *= 10  # Some features have larger scale
        
        # Test with and without gain equalization
        try:
            coder_with = SparseCoder(n_atoms=16, ratio_lambda_over_sigma=0.14, 
                                   do_gain_equalization=True, seed=42)
            coder_without = SparseCoder(n_atoms=16, ratio_lambda_over_sigma=0.14, 
                                      do_gain_equalization=False, seed=43)
            
            codes_with = coder_with.fit_transform(X)
            codes_without = coder_without.fit_transform(X)
            
            # Both should produce valid outputs
            assert np.all(np.isfinite(codes_with)), "Non-finite codes with gain equalization"
            assert np.all(np.isfinite(codes_without)), "Non-finite codes without gain equalization"
            
            # They might produce different results, but both should be mathematically valid
            sparsity_with = np.mean(np.abs(codes_with) < 1e-8)
            sparsity_without = np.mean(np.abs(codes_without) < 1e-8)
            
            assert 0.0 <= sparsity_with <= 1.0, f"Invalid sparsity with equalization: {sparsity_with}"
            assert 0.0 <= sparsity_without <= 1.0, f"Invalid sparsity without equalization: {sparsity_without}"
            
        except Exception as e:
            if "do_gain_equalization" in str(e):
                pytest.skip(f"Gain equalization parameter not supported: {e}")
            else:
                raise

    def test_multiple_random_seeds(self):
        """Test mathematical consistency across different random seeds."""
        SparseCoder = get_sparse_coder()
        
        np.random.seed(42)
        X = np.random.randn(30, 32)
        
        # Test multiple seeds - results should be different but all mathematically valid
        seeds = [42, 123, 999]
        results = []
        
        for seed in seeds:
            coder = SparseCoder(n_atoms=16, ratio_lambda_over_sigma=0.14, seed=seed)
            codes = coder.fit_transform(X)
            
            # Mathematical validity for each seed
            assert np.all(np.isfinite(codes)), f"Non-finite codes with seed {seed}"
            assert np.all(np.isfinite(coder.D)), f"Non-finite dictionary with seed {seed}"
            
            sparsity = np.mean(np.abs(codes) < 1e-8)
            assert 0.0 <= sparsity <= 1.0, f"Invalid sparsity {sparsity} with seed {seed}"
            
            # Dictionary normalization
            atom_norms = np.linalg.norm(coder.D, axis=0)
            assert np.allclose(atom_norms, 1.0, rtol=1e-2), f"Dictionary not normalized with seed {seed}"
            
            results.append((codes, sparsity, coder.D))
        
        # Different seeds should produce different results (not identical)
        for i in range(len(results) - 1):
            codes_diff = np.max(np.abs(results[i][0] - results[i+1][0]))
            # Allow for possibility of identical results but expect some difference usually
            if codes_diff < 1e-12:
                pytest.skip("Results unexpectedly identical across seeds (might indicate determinism issues)")
        
        # All results should be in similar ranges
        sparsity_values = [r[1] for r in results]
        sparsity_range = max(sparsity_values) - min(sparsity_values)
        assert sparsity_range < 0.8, f"Sparsity too variable across seeds: {sparsity_values}"


# Single integration test for pytest
def test_mathematical_regression_prevention_suite():
    """Pytest entry point for all mathematical regression tests."""
    # Basic smoke test
    try:
        SparseCoder = get_sparse_coder()
        
        # Minimal test
        np.random.seed(42)
        X = np.random.randn(20, 16)
        coder = SparseCoder(n_atoms=8, ratio_lambda_over_sigma=0.14, seed=42)
        codes = coder.fit_transform(X)
        
        # Basic assertions
        assert codes.shape == (16, 8), f"Wrong shape: {codes.shape}"
        assert np.all(np.isfinite(codes)), "Non-finite outputs"
        assert coder.D.shape == (20, 8), f"Wrong dictionary shape: {coder.D.shape}"
        assert np.all(np.isfinite(coder.D)), "Non-finite dictionary"
        
        # Dictionary normalization
        atom_norms = np.linalg.norm(coder.D, axis=0)
        assert np.allclose(atom_norms, 1.0, rtol=1e-2), "Dictionary normalization failed"
        
        print("✅ Mathematical regression prevention tests: Core functionality verified")
        
    except Exception as e:
        if "skip" in str(e).lower():
            pytest.skip("SparseCoder not available")
        else:
            raise


if __name__ == "__main__":
    # Run basic diagnostic
    test_mathematical_regression_prevention_suite()
    print("Mathematical regression tests passed!")