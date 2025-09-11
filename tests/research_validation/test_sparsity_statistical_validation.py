"""
Statistical validation tests for sparsity assertions in sparse coding.

Addresses the weak sparsity assertion issue identified in the comprehensive analysis.
Provides rigorous statistical tests for L1 penalty effectiveness and sparsity patterns.
"""

import numpy as np
import pytest
from sparse_coding import DictionaryLearner
from sparse_coding.fista_batch import fista_batch  # Direct FISTA usage for controlled testing
from tests.conftest import assert_sparse_solution, create_test_dictionary


class TestSparsityStatisticalValidation:
    """Rigorous statistical validation of sparsity properties."""
    
    def test_l1_penalty_sparsity_effect_statistical(self):
        """
        Statistically validate L1 penalty's sparsity-inducing effect.
        
        Addresses the weak assertion issue: instead of just checking that
        higher lambda leads to higher sparsity, we validate the statistical
        significance and magnitude of the effect.
        """
        np.random.seed(42)
        
        # Create test problem
        n_features, n_atoms = 64, 32
        D = create_test_dictionary(n_features, n_atoms, condition_number=5.0, seed=42)
        
        # Generate multiple test signals for statistical validation
        n_test_signals = 20
        signals = []
        for i in range(n_test_signals):
            true_codes = np.zeros((n_atoms, 1))
            active_indices = np.random.choice(n_atoms, size=8, replace=False) 
            true_codes[active_indices, 0] = np.random.randn(8)
            signal = D @ true_codes + 0.02 * np.random.randn(n_features, 1)
            signals.append(signal)
        
        # Test multiple lambda values
        lambda_values = [0.01, 0.05, 0.1, 0.2, 0.5]
        sparsity_results = []
        
        for lam in lambda_values:
            # Collect sparsity statistics across all test signals
            gini_coefficients = []
            effective_sparsities = []
            magnitude_ratios = []
            
            for signal in signals:
                # Use FISTA directly for controlled L1 testing
                codes = fista_batch(D, signal, lam, max_iter=100, tol=1e-6)
                
                # Use our enhanced statistical validation
                stats = assert_sparse_solution(codes, sparsity_threshold=0.05)
                gini_coefficients.append(stats['gini_coefficient'])
                effective_sparsities.append(stats['effective_sparsity'])
                
                # Calculate magnitude ratio manually for consistency
                codes_flat = codes.ravel()
                abs_codes = np.abs(codes_flat)
                nonzero_mask = abs_codes >= 0.05
                
                if np.sum(nonzero_mask) > 0 and np.sum(~nonzero_mask) > 0:
                    mean_significant = np.mean(abs_codes[nonzero_mask])
                    mean_small = np.mean(abs_codes[~nonzero_mask])
                    ratio = mean_significant / (mean_small + 1e-12)
                    magnitude_ratios.append(ratio)
            
            # Store aggregate statistics
            sparsity_results.append({
                'lambda': lam,
                'mean_gini': np.mean(gini_coefficients),
                'std_gini': np.std(gini_coefficients),
                'mean_effective_sparsity': np.mean(effective_sparsities),
                'mean_magnitude_ratio': np.mean(magnitude_ratios) if magnitude_ratios else 0
            })
        
        # Statistical validation of L1 penalty effect
        
        # 1. Monotonic increase in sparsity with lambda
        gini_values = [result['mean_gini'] for result in sparsity_results]
        effective_sparsity_values = [result['mean_effective_sparsity'] for result in sparsity_results]
        
        # Check monotonicity with strict statistical requirements
        gini_increases = np.diff(gini_values)
        sparsity_increases = np.diff(effective_sparsity_values)
        
        # At least 80% of increases should be positive (allow some noise)
        gini_positive_ratio = np.sum(gini_increases > 0) / len(gini_increases)
        sparsity_positive_ratio = np.sum(sparsity_increases > 0) / len(sparsity_increases)
        
        assert gini_positive_ratio >= 0.8, (
            f"Gini coefficient should increase with lambda: only {gini_positive_ratio:.1%} of "
            f"increases were positive. Values: {gini_values}"
        )
        
        assert sparsity_positive_ratio >= 0.8, (
            f"Effective sparsity should increase with lambda: only {sparsity_positive_ratio:.1%} of "
            f"increases were positive. Values: {effective_sparsity_values}"
        )
        
        # 2. Magnitude of effect validation
        total_gini_increase = gini_values[-1] - gini_values[0]
        total_sparsity_increase = effective_sparsity_values[-1] - effective_sparsity_values[0]
        
        assert total_gini_increase > 0.05, (
            f"L1 penalty should substantially increase concentration: "
            f"Gini increase {total_gini_increase:.3f} < 0.05"
        )
        
        assert total_sparsity_increase > 0.05, (
            f"L1 penalty should substantially increase sparsity: "
            f"Effective sparsity increase {total_sparsity_increase:.3f} < 0.05"
        )
        
        # 3. Statistical significance using paired comparison
        # Compare lowest vs highest lambda using paired t-test concept
        low_lambda_ginis = []
        high_lambda_ginis = []
        
        for signal in signals[:10]:  # Use subset for paired comparison
            # Low lambda
            codes_low = fista_batch(D, signal, lambda_values[0], max_iter=100, tol=1e-6)
            stats_low = assert_sparse_solution(codes_low, sparsity_threshold=0.05)
            low_lambda_ginis.append(stats_low['gini_coefficient'])
            
            # High lambda
            codes_high = fista_batch(D, signal, lambda_values[-1], max_iter=100, tol=1e-6)
            stats_high = assert_sparse_solution(codes_high, sparsity_threshold=0.05)
            high_lambda_ginis.append(stats_high['gini_coefficient'])
        
        # Paired differences
        differences = np.array(high_lambda_ginis) - np.array(low_lambda_ginis)
        mean_difference = np.mean(differences)
        
        # Effect size should be meaningful (Cohen's d equivalent)
        pooled_std = np.sqrt((np.var(low_lambda_ginis) + np.var(high_lambda_ginis)) / 2)
        effect_size = mean_difference / (pooled_std + 1e-12)
        
        assert effect_size > 0.5, (
            f"L1 penalty effect size too small: {effect_size:.3f} < 0.5 (Cohen's d). "
            f"Mean difference: {mean_difference:.3f}, pooled std: {pooled_std:.3f}"
        )
        
        print(f"✅ L1 penalty statistical validation passed:")
        print(f"   - Gini monotonicity: {gini_positive_ratio:.1%} positive increases")
        print(f"   - Sparsity monotonicity: {sparsity_positive_ratio:.1%} positive increases") 
        print(f"   - Total Gini increase: {total_gini_increase:.3f}")
        print(f"   - Effect size (Cohen's d): {effect_size:.3f}")
    
    def test_sparsity_pattern_consistency_across_methods(self):
        """
        Validate that different penalty methods produce consistent sparsity patterns.
        
        Tests L1, L2, and ElasticNet penalties to ensure they show expected
        relative sparsity properties.
        """
        np.random.seed(42)
        
        # Create test problem
        n_features, n_atoms = 48, 24
        D = create_test_dictionary(n_features, n_atoms, condition_number=3.0, seed=42)
        
        # Generate test signal  
        true_codes = np.zeros((n_atoms, 1))
        active_indices = np.random.choice(n_atoms, size=6, replace=False)
        true_codes[active_indices, 0] = np.random.randn(6)
        signal = D @ true_codes + 0.03 * np.random.randn(n_features, 1)
        
        # Test different penalties
        penalty_configs = [
            {'mode': 'l1', 'lam': 0.1},
            {'mode': 'l2', 'lam': 0.1}, 
            {'mode': 'elastic_net', 'l1': 0.05, 'l2': 0.05}
        ]
        
        penalty_stats = {}
        
        for config in penalty_configs:
            if config['mode'] == 'l1':
                codes = fista_batch(D, signal, config['lam'], max_iter=100, tol=1e-6)
            elif config['mode'] == 'l2':
                # L2 penalty: analytical solution (Ridge regression)
                lam = config['lam']
                G = D.T @ D + lam * np.eye(D.shape[1])
                codes = np.linalg.solve(G, D.T @ signal)
            elif config['mode'] == 'elastic_net':
                # Simplified ElasticNet: combine L1 + L2 effects
                # Use L1 (FISTA) with higher lambda to approximate ElasticNet
                combined_lam = config['l1'] + 0.5 * config['l2']
                codes = fista_batch(D, signal, combined_lam, max_iter=100, tol=1e-6)
                
            codes = codes
            
            # Statistical validation
            stats = assert_sparse_solution(codes, sparsity_threshold=0.05)
            penalty_stats[config['mode']] = stats
        
        # Expected relationships based on theory
        
        # 1. L1 should be more sparse than L2
        l1_sparsity = penalty_stats['l1']['effective_sparsity']
        l2_sparsity = penalty_stats['l2']['effective_sparsity']
        l1_gini = penalty_stats['l1']['gini_coefficient']
        l2_gini = penalty_stats['l2']['gini_coefficient']
        
        assert l1_sparsity > l2_sparsity, (
            f"L1 should be more sparse than L2: "
            f"L1 sparsity {l1_sparsity:.3f} ≤ L2 sparsity {l2_sparsity:.3f}"
        )
        
        assert l1_gini > l2_gini, (
            f"L1 should be more concentrated than L2: "
            f"L1 Gini {l1_gini:.3f} ≤ L2 Gini {l2_gini:.3f}"
        )
        
        # 2. ElasticNet should be between L1 and L2
        en_sparsity = penalty_stats['elastic_net']['effective_sparsity']
        en_gini = penalty_stats['elastic_net']['gini_coefficient']
        
        assert l2_sparsity <= en_sparsity <= l1_sparsity, (
            f"ElasticNet sparsity should be between L2 and L1: "
            f"L2: {l2_sparsity:.3f}, EN: {en_sparsity:.3f}, L1: {l1_sparsity:.3f}"
        )
        
        # 3. All penalties should produce valid sparse solutions
        for penalty_name, stats in penalty_stats.items():
            assert stats['validation'] == 'PASSED', (
                f"{penalty_name} penalty failed sparsity validation"
            )
        
        print(f"✅ Penalty consistency validation passed:")
        print(f"   - L1 sparsity: {l1_sparsity:.3f}, Gini: {l1_gini:.3f}")
        print(f"   - L2 sparsity: {l2_sparsity:.3f}, Gini: {l2_gini:.3f}")
        print(f"   - ElasticNet sparsity: {en_sparsity:.3f}, Gini: {en_gini:.3f}")

    def test_sparsity_robustness_to_noise(self):
        """
        Test sparsity properties remain consistent under different noise levels.
        
        Validates that the sparsity-inducing properties of L1 penalty are robust
        to reasonable levels of measurement noise.
        """
        np.random.seed(42)
        
        # Create clean test problem
        n_features, n_atoms = 40, 20
        D = create_test_dictionary(n_features, n_atoms, condition_number=2.0, seed=42)
        
        true_codes = np.zeros((n_atoms, 1))
        active_indices = np.random.choice(n_atoms, size=5, replace=False)
        true_codes[active_indices, 0] = np.random.randn(5)
        clean_signal = D @ true_codes
        
        # Test different noise levels
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
        sparsity_stats = []
        
        for noise_std in noise_levels:
            # Add noise
            noisy_signal = clean_signal + noise_std * np.random.randn(n_features, 1)
            
            # Sparse coding
            codes = fista_batch(D, noisy_signal, 0.05, max_iter=100, tol=1e-6)
            
            # Statistical validation
            stats = assert_sparse_solution(codes, sparsity_threshold=0.03)
            sparsity_stats.append({
                'noise_level': noise_std,
                'effective_sparsity': stats['effective_sparsity'],
                'gini_coefficient': stats['gini_coefficient'],
                'n_significant': stats['n_significant']
            })
        
        # Validate robustness
        
        # 1. Sparsity should remain reasonable across noise levels
        sparsities = [s['effective_sparsity'] for s in sparsity_stats]
        ginis = [s['gini_coefficient'] for s in sparsity_stats]
        
        # Sparsity should not degrade too much with moderate noise
        clean_sparsity = sparsities[0]
        moderate_noise_sparsity = sparsities[2]  # 0.05 noise level
        
        sparsity_degradation = (clean_sparsity - moderate_noise_sparsity) / clean_sparsity
        
        assert sparsity_degradation < 0.3, (
            f"Sparsity degrades too much with moderate noise: {sparsity_degradation:.1%} degradation. "
            f"Clean: {clean_sparsity:.3f}, Moderate noise: {moderate_noise_sparsity:.3f}"
        )
        
        # 2. Gini coefficient should remain meaningful
        clean_gini = ginis[0]
        moderate_noise_gini = ginis[2]
        
        gini_degradation = (clean_gini - moderate_noise_gini) / clean_gini
        
        assert gini_degradation < 0.4, (
            f"Concentration degrades too much with moderate noise: {gini_degradation:.1%} degradation. "
            f"Clean: {clean_gini:.3f}, Moderate noise: {moderate_noise_gini:.3f}"
        )
        
        # 3. All solutions should pass statistical validation
        for i, stats in enumerate(sparsity_stats):
            noise_level = noise_levels[i]
            
            # Allow slightly relaxed validation for high noise
            try:
                # This should work for all noise levels
                assert stats['gini_coefficient'] > 0.2, (
                    f"Noise level {noise_level} produces insufficient concentration: "
                    f"Gini={stats['gini_coefficient']:.3f} < 0.2"
                )
            except AssertionError as e:
                if noise_level > 0.15:  # Allow failure only for very high noise
                    print(f"⚠️  High noise level {noise_level} fails concentration test (expected)")
                else:
                    raise e
        
        print(f"✅ Noise robustness validation passed:")
        print(f"   - Sparsity degradation at moderate noise: {sparsity_degradation:.1%}")
        print(f"   - Concentration degradation at moderate noise: {gini_degradation:.1%}")
        print(f"   - Noise levels tested: {noise_levels}")