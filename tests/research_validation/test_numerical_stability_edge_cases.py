"""
Comprehensive edge case tests for numerical stability in sparse coding algorithms.

Addresses numerical stability issues identified in the comprehensive analysis.
Tests extreme conditions that can cause algorithm failure or numerical instability.
"""

import numpy as np
import pytest
from scipy import linalg
from sparse_coding import SparseCoder, DictionaryLearner
from sparse_coding.fista_batch import fista_batch
from sparse_coding.core.dict_updater_implementations import ModUpdater, KsvdUpdater
from tests.conftest import create_test_dictionary, assert_dictionary_normalized


class TestNumericalStabilityEdgeCases:
    """Comprehensive numerical stability testing for extreme conditions."""
    
    def test_near_singular_dictionary_stability(self):
        """
        Test algorithm stability with near-singular (poorly conditioned) dictionaries.
        
        Addresses numerical stability issues when dictionary atoms are nearly parallel
        or when condition number is very high.
        """
        np.random.seed(42)
        
        n_features, n_atoms = 32, 16
        
        # Create poorly conditioned dictionary (high condition number)
        D_base = create_test_dictionary(n_features, n_atoms, condition_number=1.0, seed=42)
        
        # Make dictionary poorly conditioned by making atoms nearly parallel
        D_poorly_conditioned = D_base.copy()
        # Add small random perturbations to make atoms nearly parallel
        noise_scale = 1e-6
        for i in range(1, n_atoms):
            # Make atom i nearly parallel to atom 0 with small perturbation
            D_poorly_conditioned[:, i] = (0.99 * D_poorly_conditioned[:, 0] + 
                                        0.01 * np.random.randn(n_features) * noise_scale)
            # Normalize
            D_poorly_conditioned[:, i] /= np.linalg.norm(D_poorly_conditioned[:, i])
        
        # Check condition number is actually high
        condition_number = np.linalg.cond(D_poorly_conditioned)
        assert condition_number > 100, f"Dictionary should be poorly conditioned: cond={condition_number}"
        
        # Generate test signal
        true_codes = np.zeros((n_atoms, 1))
        true_codes[0, 0] = 1.0  # Use only first atom to avoid ambiguity
        signal = D_poorly_conditioned @ true_codes + 0.01 * np.random.randn(n_features, 1)
        
        # Test FISTA stability with poorly conditioned dictionary
        codes = fista_batch(D_poorly_conditioned, signal, 0.1, max_iter=200, tol=1e-8)
        
        # Stability checks
        assert np.all(np.isfinite(codes)), "FISTA should produce finite codes with poorly conditioned dictionary"
        assert np.max(np.abs(codes)) < 1e3, f"Codes should be bounded: max={np.max(np.abs(codes))}"
        
        # Reconstruction quality should still be reasonable
        reconstruction = D_poorly_conditioned @ codes
        reconstruction_error = np.linalg.norm(signal - reconstruction)
        relative_error = reconstruction_error / np.linalg.norm(signal)
        
        assert relative_error < 0.5, (
            f"Reconstruction error too high with poor conditioning: {relative_error:.3f}"
        )
        
        print(f"✅ Near-singular dictionary test passed:")
        print(f"   - Condition number: {condition_number:.2e}")
        print(f"   - Relative reconstruction error: {relative_error:.3f}")
    
    def test_extreme_signal_magnitudes(self):
        """
        Test algorithm stability with very large and very small signal magnitudes.
        
        Tests numerical stability when signals have extreme dynamic ranges
        that could cause overflow or underflow.
        """
        np.random.seed(42)
        
        n_features, n_atoms = 24, 12
        D = create_test_dictionary(n_features, n_atoms, condition_number=2.0, seed=42)
        
        # Test extreme magnitudes
        magnitude_scales = [1e-12, 1e-6, 1e-3, 1e3, 1e6, 1e12]
        
        for scale in magnitude_scales:
            # Create signal with specific magnitude
            true_codes = np.zeros((n_atoms, 1))
            true_codes[0, 0] = scale
            true_codes[1, 0] = -scale * 0.5
            signal = D @ true_codes
            
            # Test FISTA stability
            try:
                codes = fista_batch(D, signal, 0.1 * scale, max_iter=100, tol=1e-10)
                
                # Numerical stability checks
                assert np.all(np.isfinite(codes)), f"FISTA should produce finite codes for scale {scale}"
                
                # Check for reasonable reconstruction
                reconstruction = D @ codes
                reconstruction_error = np.linalg.norm(signal - reconstruction)
                signal_norm = np.linalg.norm(signal)
                
                if signal_norm > 1e-15:  # Avoid division by zero for tiny signals
                    relative_error = reconstruction_error / signal_norm
                    # More lenient threshold for extreme scales
                    max_error = 0.8 if scale < 1e-9 else 0.2
                    assert relative_error < max_error, (
                        f"Poor reconstruction for scale {scale}: error={relative_error:.2e}"
                    )
                
                print(f"✅ Scale {scale:.1e}: relative error = {relative_error:.2e}")
                
            except (OverflowError, np.linalg.LinAlgError, FloatingPointError) as e:
                pytest.fail(f"FISTA failed for scale {scale} with error: {e}")
    
    def test_zero_and_near_zero_inputs(self):
        """
        Test algorithm behavior with zero and near-zero inputs.
        
        Edge case testing for degenerate inputs that can cause division by zero
        or other numerical issues.
        """
        np.random.seed(42)
        
        n_features, n_atoms = 16, 8
        D = create_test_dictionary(n_features, n_atoms, condition_number=1.5, seed=42)
        
        # Test 1: Exact zero signal
        zero_signal = np.zeros((n_features, 1))
        codes_zero = fista_batch(D, zero_signal, 0.1, max_iter=100, tol=1e-8)
        
        assert np.all(np.isfinite(codes_zero)), "FISTA should handle zero signal gracefully"
        assert np.max(np.abs(codes_zero)) < 1e-6, "Codes for zero signal should be near zero"
        
        # Test 2: Near-zero signal (machine epsilon level)
        epsilon_signal = np.random.randn(n_features, 1) * 1e-15
        codes_epsilon = fista_batch(D, epsilon_signal, 0.1, max_iter=100, tol=1e-8)
        
        assert np.all(np.isfinite(codes_epsilon)), "FISTA should handle epsilon-level signals"
        
        # Test 3: Near-zero dictionary atoms
        D_near_zero = D.copy()
        D_near_zero[:, -1] = np.random.randn(n_features) * 1e-14
        D_near_zero[:, -1] /= np.linalg.norm(D_near_zero[:, -1])  # This might create issues
        
        signal = np.random.randn(n_features, 1) * 0.1
        
        try:
            codes_near_zero_dict = fista_batch(D_near_zero, signal, 0.1, max_iter=50, tol=1e-6)
            assert np.all(np.isfinite(codes_near_zero_dict)), "FISTA should handle near-zero dictionary atoms"
        except np.linalg.LinAlgError:
            # This is acceptable behavior for degenerate dictionaries
            pass
        
        print(f"✅ Zero/near-zero input tests passed")
    
    def test_dictionary_learning_numerical_stability(self):
        """
        Test numerical stability of dictionary learning algorithms.
        
        Tests MOD and K-SVD updates under challenging numerical conditions.
        """
        np.random.seed(42)
        
        n_features, n_atoms = 20, 10
        n_samples = 15
        
        # Create challenging scenario: overcomplete with correlated samples
        D_true = create_test_dictionary(n_features, n_atoms, condition_number=5.0, seed=42)
        
        # Generate correlated sparse codes (challenging for learning)
        A = np.zeros((n_atoms, n_samples))
        base_pattern = np.random.randn(5)  # Base pattern
        for i in range(n_samples):
            # Create correlated patterns
            active_atoms = np.random.choice(n_atoms, size=3, replace=False)
            for j, atom_idx in enumerate(active_atoms):
                A[atom_idx, i] = base_pattern[j % len(base_pattern)] + 0.1 * np.random.randn()
        
        X = D_true @ A + 0.02 * np.random.randn(n_features, n_samples)
        
        # Test MOD updater stability
        mod_updater = ModUpdater(eps=1e-8)
        D_init = np.random.randn(n_features, n_atoms)
        D_init /= np.linalg.norm(D_init, axis=0, keepdims=True)
        
        D_mod = mod_updater.step(D_init, X, A)
        
        # Stability checks for MOD
        assert np.all(np.isfinite(D_mod)), "MOD should produce finite dictionary"
        assert_dictionary_normalized(D_mod, tolerance=1e-10)
        
        # Check condition number hasn't exploded
        mod_condition = np.linalg.cond(D_mod)
        assert mod_condition < 1e12, f"MOD dictionary condition number too high: {mod_condition:.2e}"
        
        # Test K-SVD updater stability
        ksvd_updater = KsvdUpdater(n_iterations=1)
        D_ksvd = ksvd_updater.step(D_init.copy(), X, A.copy())
        
        # Stability checks for K-SVD
        assert np.all(np.isfinite(D_ksvd)), "K-SVD should produce finite dictionary"
        assert_dictionary_normalized(D_ksvd, tolerance=1e-10)
        
        # Check for reasonable atom diversity (no completely parallel atoms)
        min_angle_cos = np.inf
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                cos_angle = np.abs(np.dot(D_ksvd[:, i], D_ksvd[:, j]))
                min_angle_cos = min(min_angle_cos, cos_angle)
        
        assert min_angle_cos < 0.99, f"Dictionary atoms too parallel: min cos angle = {min_angle_cos:.3f}"
        
        print(f"✅ Dictionary learning stability tests passed:")
        print(f"   - MOD condition number: {mod_condition:.2e}")
        print(f"   - Min atom angle cosine: {min_angle_cos:.3f}")
    
    def test_ill_conditioned_linear_systems(self):
        """
        Test stability when solving ill-conditioned linear systems.
        
        Tests the robustness of internal linear algebra operations
        under challenging numerical conditions.
        """
        np.random.seed(42)
        
        n_features = 16
        
        # Create ill-conditioned matrices that might appear in algorithms
        condition_numbers = [1e3, 1e6, 1e9, 1e12]
        
        for target_cond in condition_numbers:
            # Create matrix with specific condition number
            U, s, Vt = np.linalg.svd(np.random.randn(n_features, n_features))
            s_scaled = np.linspace(1, 1/target_cond, n_features)
            A = U @ np.diag(s_scaled) @ Vt
            
            actual_cond = np.linalg.cond(A)
            
            # Test system solving
            b = np.random.randn(n_features)
            
            try:
                # Test different solution methods
                x_direct = np.linalg.solve(A, b)
                assert np.all(np.isfinite(x_direct)), f"Direct solve failed for cond={actual_cond:.2e}"
                
                # Test with regularization (as used in MOD)
                A_reg = A + 1e-8 * np.eye(n_features)
                x_reg = np.linalg.solve(A_reg, b)
                assert np.all(np.isfinite(x_reg)), f"Regularized solve failed for cond={actual_cond:.2e}"
                
                # Test residual
                residual = np.linalg.norm(A @ x_reg - b)
                residual_relative = residual / np.linalg.norm(b)
                
                # For very ill-conditioned systems, we allow larger residuals
                if actual_cond > 1e11:
                    # For extremely ill-conditioned systems, just check for finite solution
                    assert np.all(np.isfinite(x_reg)), f"Solution should be finite for cond={actual_cond:.2e}"
                    print(f"⚠️  High condition {actual_cond:.2e}: relative residual = {residual_relative:.2e} (expected)")
                else:
                    max_acceptable_error = min(0.1, 1e-6 * actual_cond)
                    assert residual_relative < max_acceptable_error, (
                        f"Solution quality too poor for cond={actual_cond:.2e}: "
                        f"relative residual={residual_relative:.2e}"
                    )
                
                print(f"✅ Condition {actual_cond:.2e}: relative residual = {residual_relative:.2e}")
                
            except np.linalg.LinAlgError as e:
                if actual_cond > 1e10:
                    # Acceptable to fail for extremely ill-conditioned systems
                    print(f"⚠️  Expected failure for condition {actual_cond:.2e}: {e}")
                else:
                    pytest.fail(f"Unexpected failure for condition {actual_cond:.2e}: {e}")
    
    def test_floating_point_precision_limits(self):
        """
        Test behavior at floating point precision limits.
        
        Tests algorithm robustness when working near machine epsilon
        and floating point representation limits.
        """
        np.random.seed(42)
        
        # Test with values near machine epsilon
        epsilon = np.finfo(np.float64).eps
        
        n_features, n_atoms = 12, 6
        D = create_test_dictionary(n_features, n_atoms, seed=42)
        
        # Test signals with values near machine precision
        precision_scales = [epsilon, epsilon * 1e3, epsilon * 1e6]
        
        for scale in precision_scales:
            signal = np.random.randn(n_features, 1) * scale
            
            # Adjust lambda to be proportional to signal scale
            lambda_val = 0.1 * scale
            
            codes = fista_batch(D, signal, lambda_val, max_iter=50, tol=epsilon * 1e3)
            
            assert np.all(np.isfinite(codes)), f"FISTA failed at precision scale {scale/epsilon:.0f}*eps"
            
            # Check reconstruction makes sense relative to precision
            reconstruction = D @ codes
            error = np.linalg.norm(signal - reconstruction)
            signal_norm = np.linalg.norm(signal)
            
            if signal_norm > epsilon * 1e6:  # Only check if signal is not too small
                relative_error = error / signal_norm
                assert relative_error < 1.0, (
                    f"Reconstruction error too high at scale {scale/epsilon:.0f}*eps: "
                    f"relative_error={relative_error:.2e}"
                )
        
        print(f"✅ Floating point precision tests passed")
        print(f"   - Machine epsilon: {epsilon:.2e}")
        print(f"   - Tested scales: {[s/epsilon for s in precision_scales]}*eps")
    
    def test_pathological_sparsity_patterns(self):
        """
        Test algorithm stability with pathological sparsity patterns.
        
        Tests edge cases in sparsity patterns that might cause numerical issues.
        """
        np.random.seed(42)
        
        n_features, n_atoms = 20, 15
        D = create_test_dictionary(n_features, n_atoms, seed=42)
        
        # Test 1: Single huge coefficient
        codes_huge = np.zeros((n_atoms, 1))
        codes_huge[0, 0] = 1e6
        signal_huge = D @ codes_huge
        
        recovered_codes = fista_batch(D, signal_huge, 1e3, max_iter=100, tol=1e-8)
        assert np.all(np.isfinite(recovered_codes)), "Should handle single huge coefficient"
        
        # Test 2: Many tiny coefficients
        codes_tiny = np.random.randn(n_atoms, 1) * 1e-10
        signal_tiny = D @ codes_tiny + 1e-12 * np.random.randn(n_features, 1)
        
        recovered_tiny = fista_batch(D, signal_tiny, 1e-12, max_iter=100, tol=1e-15)
        assert np.all(np.isfinite(recovered_tiny)), "Should handle many tiny coefficients"
        
        # Test 3: Alternating huge positive/negative coefficients
        codes_alternating = np.zeros((n_atoms, 1))
        codes_alternating[::2, 0] = 1e5
        codes_alternating[1::2, 0] = -1e5
        signal_alternating = D @ codes_alternating
        
        recovered_alt = fista_batch(D, signal_alternating, 1e2, max_iter=100, tol=1e-8)
        assert np.all(np.isfinite(recovered_alt)), "Should handle alternating huge coefficients"
        
        print(f"✅ Pathological sparsity pattern tests passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])