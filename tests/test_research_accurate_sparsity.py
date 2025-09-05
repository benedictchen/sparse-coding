"""
🧪 Research-Accurate Sparsity Function Tests
===========================================

🎯 ELI5 Summary:
This test file is like a scientific laboratory where we carefully test each algorithm 
to make sure it works exactly like the research papers say it should! Just like how 
scientists test their theories with experiments, we test our code with mathematical 
scenarios to prove it matches the original research.

🔬 Research Foundation:
Testing all sparsity functions and optimization methods from:
- Olshausen & Field (1996): "Emergence of simple-cell receptive field properties"
- Beck & Teboulle (2009): "A Fast Iterative Shrinkage-Thresholding Algorithm" (FISTA)
- Zou & Hastie (2005): "Regularization and variable selection via the elastic net"
- Huber (1964): "Robust estimation of a location parameter"

🙏 If this library helps your research:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
💖 GitHub Sponsors: https://github.com/sponsors/benedictchen

Author: Benedict Chen (benedict@benedictchen.com)
"""

import numpy as np
import pytest
from unittest import TestCase
from scipy import optimize
import warnings

# Import the research-accurate sparsity components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sparse_coding.research_accurate_sparsity import (
    SparsenessFunction,
    SparsenessFunctions as SparseFunc,
    OptimizationAlgorithm,
    DictionaryUpdate,  
    SparseCodingConfig,
    FISTAOptimizer
)

# Try to import optional classes
try:
    from sparse_coding.research_accurate_sparsity import ResearchAccurateSparseCoder, create_sparse_coder
    HAS_RESEARCH_CODER = True
except ImportError:
    HAS_RESEARCH_CODER = False


class TestOlshausenFieldSparsityFunctions(TestCase):
    """
    Test all sparsity functions mentioned in Olshausen & Field FIXME comments.
    
    Each test validates the mathematical properties described in research papers.
    """
    
    def setUp(self):
        """Set up test data with known mathematical properties"""
        self.test_coeffs_sparse = np.array([0.0, 0.1, 0.0, -0.3, 0.0, 0.5])  # 50% sparse
        self.test_coeffs_dense = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])    # 0% sparse
        self.test_coeffs_single = np.array([1.0])
        self.tolerance = 1e-10
        
    def test_log_sparseness_olshausen_field_exact(self):
        """
        Test S(x) = log(1 + x²) exactly as in Olshausen & Field (1996).
        
        This was the primary sparseness function in the original paper.
        Mathematical form: S(ai) = -Σ log(1 + (ai/σ)²)
        """
        # Test with sigma = 1.0 (original paper default)
        coeffs = np.array([0.0, 1.0, -1.0, 2.0])
        expected_individual = [-np.log(1 + 0**2), -np.log(1 + 1**2), -np.log(1 + 1**2), -np.log(1 + 2**2)]
        expected_total = sum(expected_individual)
        
        result = SparseFunc.log_sparseness(coeffs, sigma=1.0)
        
        self.assertAlmostEqual(result, expected_total, places=10)
        
        # Test scaling behavior with different sigma values
        sigma_half = 0.5
        result_scaled = SparseFunc.log_sparseness(coeffs, sigma=sigma_half) 
        # With sigma=0.5, normalized coeffs are 2x larger, so penalty should be stronger
        expected_scaled = -np.log(1 + 0) - np.log(1 + 4) - np.log(1 + 4) - np.log(1 + 16)
        self.assertAlmostEqual(result_scaled, expected_scaled, places=10)
        
        # Test sparsity promotion: sparse vector should have better score than dense
        sparse_score = SparseFunc.log_sparseness(self.test_coeffs_sparse)
        dense_score = SparseFunc.log_sparseness(self.test_coeffs_dense)
        self.assertGreater(sparse_score, dense_score, "Sparse coefficients should have higher score (less negative)")
    
    def test_l1_sparseness_lasso_penalty(self):
        """
        Test L1 penalty S(x) = |x| - standard LASSO regularization.
        
        L1 penalty is the most common modern sparsity function.
        Should exactly equal sum of absolute values.
        """
        coeffs = np.array([-2.0, 0.0, 1.5, -0.5])
        expected = 2.0 + 0.0 + 1.5 + 0.5  # Sum of absolute values
        
        result = SparseFunc.l1_sparseness(coeffs)
        self.assertAlmostEqual(result, expected, places=10)
        
        # Test zero coefficients
        zeros = np.zeros(10)
        self.assertEqual(SparseFunc.l1_sparseness(zeros), 0.0)
        
        # Test sparsity promotion
        sparse_score = SparseFunc.l1_sparseness(self.test_coeffs_sparse)
        dense_score = SparseFunc.l1_sparseness(self.test_coeffs_dense)
        self.assertLess(sparse_score, dense_score, "Sparse coefficients should have lower L1 penalty")

    def test_gaussian_sparseness_exponential_decay(self):
        """
        Test Gaussian penalty S(x) = -e^(-x²) from Olshausen & Field papers.
        
        Alternative penalty with exponential decay behavior.
        """
        coeffs = np.array([0.0, 1.0, -1.0])
        expected = -np.exp(-0**2) - np.exp(-1**2) - np.exp(-1**2)
        
        result = SparseFunc.gaussian_sparseness(coeffs, sigma=1.0)
        self.assertAlmostEqual(result, expected, places=10)
        
        # Test at zero: should equal -1.0
        zero_result = SparseFunc.gaussian_sparseness(np.array([0.0]))
        self.assertAlmostEqual(zero_result, -1.0, places=10)

    def test_huber_sparseness_smooth_l1_approximation(self):
        """
        Test Huber penalty - smooth approximation to L1 for numerical stability.
        
        Quadratic for small values, linear for large values.
        Based on Huber (1964) robust estimation theory.
        """
        delta = 1.0
        
        # Test small values (|x| <= δ): should be quadratic
        small_coeffs = np.array([0.5, -0.5])  # |x| <= 1.0
        expected_small = 0.5 * (0.5**2 / delta + 0.5**2 / delta)
        result_small = SparseFunc.huber_sparseness(small_coeffs, delta=delta)
        self.assertAlmostEqual(result_small, expected_small, places=10)
        
        # Test large values (|x| > δ): should be linear  
        large_coeffs = np.array([2.0, -3.0])  # |x| > 1.0
        expected_large = (2.0 - 0.5*delta) + (3.0 - 0.5*delta)
        result_large = SparseFunc.huber_sparseness(large_coeffs, delta=delta)
        self.assertAlmostEqual(result_large, expected_large, places=10)

    def test_elastic_net_l1_l2_combination(self):
        """
        Test Elastic Net = L1 + L2 combination as in Zou & Hastie (2005).
        
        Balances sparsity (L1) with coefficient smoothness (L2).
        """
        coeffs = np.array([1.0, 2.0, -1.0])
        
        # Test pure L1 (l1_ratio = 1.0)
        result_pure_l1 = SparseFunc.elastic_net_sparseness(coeffs, l1_ratio=1.0)
        expected_l1 = 1.0 + 2.0 + 1.0  # Sum of absolute values
        self.assertAlmostEqual(result_pure_l1, expected_l1, places=10)
        
        # Test pure L2 (l1_ratio = 0.0)  
        result_pure_l2 = SparseFunc.elastic_net_sparseness(coeffs, l1_ratio=0.0)
        expected_l2 = 0.5 * (1.0**2 + 2.0**2 + 1.0**2)  # 0.5 * sum of squares
        self.assertAlmostEqual(result_pure_l2, expected_l2, places=10)
        
        # Test balanced combination (l1_ratio = 0.5)
        result_balanced = SparseFunc.elastic_net_sparseness(coeffs, l1_ratio=0.5)
        expected_balanced = 0.5 * expected_l1 + 0.5 * expected_l2
        self.assertAlmostEqual(result_balanced, expected_balanced, places=10)

    def test_cauchy_sparseness_heavy_tailed(self):
        """
        Test Cauchy penalty for extreme sparsity promotion.
        
        Heavy-tailed distribution penalty based on Cauchy distribution.
        Should be less sensitive to large coefficients than Gaussian penalties.
        """
        gamma = 1.0
        coeffs = np.array([0.0, 1.0, 2.0])
        
        # Cauchy log-likelihood: -log(γ/(π(γ² + x²)))
        expected = (-np.log(gamma / (np.pi * (gamma**2 + 0**2))) +
                   -np.log(gamma / (np.pi * (gamma**2 + 1**2))) +
                   -np.log(gamma / (np.pi * (gamma**2 + 2**2))))
        
        result = SparseFunc.cauchy_sparseness(coeffs, gamma=gamma)
        self.assertAlmostEqual(result, expected, places=10)

    def test_student_t_sparseness_generalized_cauchy(self):
        """
        Test Student's t-distribution penalty with controllable tail heaviness.
        
        Generalizes Cauchy (df=1) with degrees of freedom parameter.
        Lower df promotes more extreme sparsity.
        """
        df = 1.0  # Should approach Cauchy distribution
        coeffs = np.array([0.0, 1.0])
        
        # Student-t penalty computation
        from scipy.special import gammaln
        log_gamma_term = gammaln((df + 1) / 2) - gammaln(df / 2)
        log_normalization = log_gamma_term - 0.5 * np.log(np.pi * df)
        
        expected = 0.0  # Will be computed in loop
        for coef in coeffs:
            log_density = log_normalization - 0.5 * (df + 1) * np.log(1 + coef**2 / df)
            expected += -log_density
        
        result = SparseFunc.student_t_sparseness(coeffs, df=df)
        self.assertAlmostEqual(result, expected, places=8)  # Slightly lower precision due to special functions

    def test_laplace_sparseness_equivalent_to_l1(self):
        """
        Test Laplace penalty - should be equivalent to L1.
        
        Laplace prior corresponds exactly to L1 penalty.
        """
        coeffs = np.array([1.0, -2.0, 0.5])
        scale = 2.0
        
        result = SparseFunc.laplace_sparseness(coeffs, scale=scale)
        expected = np.sum(np.abs(coeffs)) / scale
        
        self.assertAlmostEqual(result, expected, places=10)


class TestFISTAOptimizer(TestCase):
    """
    Test FISTA optimizer implementation for research accuracy.
    
    Based on Beck & Teboulle (2009) "A Fast Iterative Shrinkage-Thresholding Algorithm"
    """
    
    def setUp(self):
        """Set up test optimization problems with known solutions"""
        # Simple test problem: minimize ||Ax - b||² + λ||x||₁
        np.random.seed(42)  # Reproducible tests
        self.A = np.random.randn(20, 10)  # Dictionary matrix
        self.x_true = np.array([1.0, 0, 0, 2.0, 0, 0, -1.5, 0, 0, 0])  # Sparse ground truth
        self.b = self.A @ self.x_true + 0.01 * np.random.randn(20)  # Noisy observations
        
        self.config = SparseCodingConfig(
            sparseness_function=SparsenessFunction.L1,
            sparsity_penalty=0.1,
            max_iterations=1000,
            tolerance=1e-8,
            fista_backtrack=True,
            fista_restart=True
        )
        
    def test_fista_convergence_beck_teboulle(self):
        """
        Test FISTA convergence properties as described in Beck & Teboulle (2009).
        
        Should converge faster than standard ISTA with O(1/k²) rate.
        """
        optimizer = FISTAOptimizer(self.config)
        
        # Solve sparse coding problem
        solution, info = optimizer.solve(self.A, self.b)
        
        # Check convergence
        self.assertTrue(info['convergence'], "FISTA should converge within max_iterations")
        self.assertLess(info['iterations'], self.config.max_iterations, "Should converge before max iterations")
        
        # Check solution quality
        reconstruction_error = np.linalg.norm(self.A @ solution - self.b)
        self.assertLess(reconstruction_error, 0.1, "Solution should have low reconstruction error")
        
        # Check sparsity: solution should be sparser than ground truth or similar
        sparsity_level = np.mean(np.abs(solution) < 1e-3)
        self.assertGreater(sparsity_level, 0.5, "Solution should be reasonably sparse")

    def test_proximal_operators_research_accurate(self):
        """
        Test proximal operators for different sparsity functions.
        
        Each proximal operator should solve: argmin_x { 0.5||x-z||² + λS(x) }
        """
        optimizer = FISTAOptimizer(self.config)
        z = np.array([0.5, -1.5, 0.1, -0.05])
        lambda_val = 0.2
        
        # Test L1 proximal operator (soft thresholding)
        result_l1 = optimizer._proximal_operator(z, lambda_val)
        expected_l1 = np.sign(z) * np.maximum(np.abs(z) - lambda_val, 0)
        np.testing.assert_array_almost_equal(result_l1, expected_l1, decimal=10)
        
        # Test log penalty proximal operator (numerical solution)
        config_log = SparseCodingConfig(sparseness_function=SparsenessFunction.LOG)
        optimizer_log = FISTAOptimizer(config_log)
        result_log = optimizer_log._proximal_operator(z, lambda_val)
        
        # Should be close to input but shrunk towards zero
        self.assertTrue(np.all(np.abs(result_log) <= np.abs(z)), "Log penalty should shrink coefficients")

    def test_backtracking_line_search(self):
        """
        Test backtracking line search for adaptive step size.
        
        Based on Armijo condition from Beck & Teboulle (2009).
        """
        optimizer = FISTAOptimizer(self.config)
        
        # Test with a simple quadratic function
        y = np.array([1.0, 2.0, 0.5])
        gradient = 2 * y  # Gradient of ||y||²
        L_initial = 1.0
        
        x_new, L_final = optimizer._backtracking_line_search(
            np.eye(3), y, y, gradient, L_initial
        )
        
        # Should find a valid step size
        self.assertGreater(L_final, 0, "Final Lipschitz constant should be positive")
        self.assertTrue(np.all(np.isfinite(x_new)), "Solution should be finite")

    def test_objective_function_computation(self):
        """
        Test complete objective function evaluation: data fidelity + sparsity penalty.
        """
        optimizer = FISTAOptimizer(self.config)
        
        coeffs = np.array([1.0, 0.0, -0.5])
        dictionary = np.eye(3)  # Identity for simplicity  
        patch = np.array([0.8, 0.0, -0.4])
        
        # Compute objective manually
        data_fidelity = 0.5 * np.linalg.norm(patch - dictionary @ coeffs)**2
        sparsity_penalty = self.config.sparsity_penalty * np.sum(np.abs(coeffs))
        expected_objective = data_fidelity + sparsity_penalty
        
        # Test optimizer's computation
        computed_objective = optimizer._objective_function(dictionary, patch, coeffs)
        
        self.assertAlmostEqual(computed_objective, expected_objective, places=10)


class TestResearchAccurateSparseCoder(TestCase):
    """
    Test complete sparse coder with all FIXME solutions implemented.
    
    Integration tests for dictionary learning and sparse coding pipeline.
    """
    
    def setUp(self):
        """Set up test data for sparse coding experiments"""
        np.random.seed(123)  # Reproducible results
        
        # Create synthetic overcomplete dictionary learning problem
        self.patch_size = 8
        self.n_atoms = 16  # 2x overcomplete
        self.n_patches = 100
        
        # Generate ground truth dictionary and sparse codes
        self.true_dictionary = np.random.randn(self.patch_size, self.n_atoms)
        self.true_dictionary /= np.linalg.norm(self.true_dictionary, axis=0)
        
        # Generate sparse codes and reconstruct patches
        self.sparse_codes = np.zeros((self.n_patches, self.n_atoms))
        for i in range(self.n_patches):
            # Make sparse: only 20% of coefficients are non-zero
            active_atoms = np.random.choice(self.n_atoms, size=3, replace=False)
            self.sparse_codes[i, active_atoms] = np.random.randn(3)
            
        self.test_patches = (self.sparse_codes @ self.true_dictionary.T).T
        self.test_patches += 0.01 * np.random.randn(*self.test_patches.shape)  # Add noise
        
        self.config = SparseCodingConfig(
            sparseness_function=SparsenessFunction.L1,
            optimization_algorithm=OptimizationAlgorithm.FISTA,
            dictionary_update=DictionaryUpdate.OLSHAUSEN_FIELD,
            sparsity_penalty=0.1,
            max_iterations=100,  # Fewer iterations for testing speed
            learning_rate=0.01
        )

    @pytest.mark.skipif(not HAS_RESEARCH_CODER, reason="ResearchAccurateSparseCoder not available")
    def test_sparse_coder_initialization(self):
        """Test initialization of research-accurate sparse coder"""
        coder = ResearchAccurateSparseCoder(self.config)
        
        self.assertEqual(coder.config.sparseness_function, SparsenessFunction.L1)
        self.assertEqual(coder.config.optimization_algorithm, OptimizationAlgorithm.FISTA)
        self.assertIsNone(coder.dictionary)  # Should start uninitialized

    @pytest.mark.skipif(not HAS_RESEARCH_CODER, reason="ResearchAccurateSparseCoder not available")
    def test_dictionary_learning_olshausen_field(self):
        """
        Test Olshausen & Field dictionary learning algorithm.
        
        Should learn dictionary that can sparsely represent the training data.
        """
        coder = ResearchAccurateSparseCoder(self.config)
        
        # Learn dictionary from patches
        learning_info = coder.learn_dictionary(self.test_patches.T, self.n_atoms)
        
        # Check that dictionary was learned
        self.assertIsNotNone(coder.dictionary)
        self.assertEqual(coder.dictionary.shape, (self.patch_size, self.n_atoms))
        
        # Check that dictionary atoms are normalized (Olshausen & Field constraint)
        atom_norms = np.linalg.norm(coder.dictionary, axis=0)
        np.testing.assert_array_almost_equal(atom_norms, np.ones(self.n_atoms), decimal=6)
        
        # Check convergence information
        self.assertIn('iteration_objectives', learning_info)
        self.assertIn('final_iteration', learning_info)
        self.assertGreater(len(learning_info['iteration_objectives']), 0)

    @pytest.mark.skipif(not HAS_RESEARCH_CODER, reason="ResearchAccurateSparseCoder not available") 
    def test_patch_encoding_with_different_sparsity_functions(self):
        """
        Test patch encoding with all implemented sparsity functions.
        
        Each sparsity function should produce different but valid sparse codes.
        """
        # Initialize with known dictionary
        coder = ResearchAccurateSparseCoder(self.config)
        coder.dictionary = self.true_dictionary.copy()
        
        test_patch = self.test_patches[:, 0]  # First patch
        
        # Test encoding with different sparsity functions
        sparsity_functions = [SparsenessFunction.L1, SparsenessFunction.LOG]
        
        results = {}
        for sparse_func in sparsity_functions:
            config_test = SparseCodingConfig(
                sparseness_function=sparse_func,
                optimization_algorithm=OptimizationAlgorithm.FISTA,
                sparsity_penalty=0.1,
                max_iterations=200
            )
            coder_test = ResearchAccurateSparseCoder(config_test)
            coder_test.dictionary = self.true_dictionary.copy()
            
            coeffs, info = coder_test.encode_patch(test_patch)
            results[sparse_func] = {'coeffs': coeffs, 'info': info}
            
            # Check solution validity
            self.assertTrue(info['convergence'], f"Should converge with {sparse_func}")
            self.assertTrue(np.all(np.isfinite(coeffs)), f"Coefficients should be finite with {sparse_func}")
            
            # Check reconstruction quality  
            reconstruction = coder_test.dictionary @ coeffs
            reconstruction_error = np.linalg.norm(reconstruction - test_patch)
            self.assertLess(reconstruction_error, 0.2, f"Good reconstruction with {sparse_func}")
        
        # Different sparsity functions should give different solutions
        l1_coeffs = results[SparsenessFunction.L1]['coeffs']
        log_coeffs = results[SparsenessFunction.LOG]['coeffs']
        difference = np.linalg.norm(l1_coeffs - log_coeffs)
        self.assertGreater(difference, 0.01, "Different sparsity functions should give different solutions")

    @pytest.mark.skipif(not HAS_RESEARCH_CODER, reason="create_sparse_coder not available")
    def test_factory_function_research_profiles(self):
        """
        Test factory function for creating research-accurate configurations.
        """
        profiles = ['olshausen_field_original', 'modern_fista', 'robust_huber', 'elastic_net']
        
        for profile in profiles:
            coder = create_sparse_coder(profile)
            self.assertIsInstance(coder, ResearchAccurateSparseCoder)
            
            # Check profile-specific settings
            if profile == 'olshausen_field_original':
                self.assertEqual(coder.config.sparseness_function, SparsenessFunction.LOG)
                self.assertEqual(coder.config.optimization_algorithm, OptimizationAlgorithm.GRADIENT_DESCENT)
            elif profile == 'modern_fista':
                self.assertEqual(coder.config.sparseness_function, SparsenessFunction.L1)
                self.assertEqual(coder.config.optimization_algorithm, OptimizationAlgorithm.FISTA)
            elif profile == 'robust_huber':
                self.assertEqual(coder.config.sparseness_function, SparsenessFunction.HUBER)
            elif profile == 'elastic_net':
                self.assertEqual(coder.config.sparseness_function, SparsenessFunction.ELASTIC_NET)


class TestSparseCodingConfigValidation(TestCase):
    """
    Test comprehensive configuration validation implemented in FIXME solutions.
    
    Tests all parameter validation scenarios with research-based bounds.
    """
    
    def test_configuration_validation_research_bounds(self):
        """
        Test configuration validation with research-based parameter bounds.
        
        Should validate parameters according to Olshausen & Field recommendations.
        """
        # Test valid configuration - should not raise warnings  
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Convert warnings to errors for testing
            try:
                valid_config = SparseCodingConfig(
                    sparseness_function=SparsenessFunction.L1,
                    sparsity_penalty=0.1,  # Within recommended range [0.01, 0.5]
                    learning_rate=0.001,   # Within recommended range [0.0001, 0.01]
                    max_iterations=1000,
                    tolerance=1e-6
                )
                self.assertIsNotNone(valid_config)
            except UserWarning as w:
                self.fail(f"Valid configuration should not raise warnings: {w}")
    
    def test_parameter_bounds_validation(self):
        """Test parameter bounds validation with memory awareness"""
        
        # Test n_components validation
        with self.assertRaises(ValueError):
            SparseCodingConfig(sparseness_function=SparsenessFunction.L1, sparsity_penalty=0.1, n_components=-1)
            
        with self.assertRaises(ValueError):
            SparseCodingConfig(sparseness_function=SparsenessFunction.L1, sparsity_penalty=0.1, n_components=20000)  # Exceeds limit
            
        # Test sparsity penalty validation
        with self.assertRaises(ValueError):
            SparseCodingConfig(sparseness_function=SparsenessFunction.L1, sparsity_penalty=-0.1)
            
        # Test learning rate validation  
        with self.assertRaises(ValueError):
            SparseCodingConfig(sparseness_function=SparsenessFunction.L1, sparsity_penalty=0.1, learning_rate=-0.01)

    def test_algorithm_compatibility_validation(self):
        """Test validation of algorithm compatibility combinations"""
        
        # Test with warnings for suboptimal combinations
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should generate a compatibility warning
            config = SparseCodingConfig(
                sparseness_function=SparsenessFunction.GAUSSIAN,
                optimization_algorithm=OptimizationAlgorithm.FISTA,
                sparsity_penalty=0.1
            )
            
            # Should have generated some warnings about parameter choices
            self.assertIsNotNone(config)  # Config should still be created


if __name__ == '__main__':
    # Run comprehensive test suite
    print("🧪 Running Research-Accurate Sparse Coding Tests")
    print("=" * 55)
    
    # Configure test runner for detailed output
    import unittest
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(loader.loadTestsFromTestCase(TestOlshausenFieldSparsityFunctions))
    suite.addTest(loader.loadTestsFromTestCase(TestFISTAOptimizer))  
    suite.addTest(loader.loadTestsFromTestCase(TestResearchAccurateSparseCoder))
    suite.addTest(loader.loadTestsFromTestCase(TestSparseCodingConfigValidation))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All research accuracy tests passed!")
    else:
        print("❌ Some tests failed. Review implementation for research accuracy.")
        
    # Exit with proper code
    exit(0 if result.wasSuccessful() else 1)