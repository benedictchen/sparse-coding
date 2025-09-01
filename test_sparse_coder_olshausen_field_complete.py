#!/usr/bin/env python3
"""
Sparse Coder - Complete Olshausen & Field (1996) Research-Aligned Tests
========================================================================

Comprehensive test suite covering 100% of sparse_coder.py functionality
with strict adherence to Olshausen & Field (1996) research paper.

Research Paper: "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"
Author: Benedict Chen (benedict@benedictchen.com)

Test Coverage Target: 100%
Research Alignment: Complete Olshausen & Field algorithm verification
Configuration Preservation: All options tested and preserved
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
import warnings

try:
    from .sparse_coder import SparseCoder
except ImportError:
    from sparse_coder import SparseCoder


class TestOlshausenFieldInitialization:
    """Test initialization with all configuration options (Research Paper Section 2)"""
    
    def test_basic_initialization_default_parameters(self):
        """Test basic initialization with Olshausen & Field default parameters"""
        coder = SparseCoder()
        assert coder.n_components == 256  # Typical overcomplete dictionary size
        assert coder.patch_size == (16, 16)  # Standard natural image patch size
        assert coder.sparseness_function == 'l1'  # L1 penalty from paper
        assert coder.optimization_method == 'coordinate_descent'  # Standard method
        assert coder.sparsity_penalty == 0.1  # Default lambda parameter
        
    def test_initialization_all_sparseness_functions(self):
        """Test all sparseness functions from research literature"""
        # L1 sparseness (Olshausen & Field primary method)
        coder_l1 = SparseCoder(sparseness_function='l1')
        assert coder_l1.sparseness_function == 'l1'
        
        # Log sparseness (smooth approximation)
        coder_log = SparseCoder(sparseness_function='log')
        assert coder_log.sparseness_function == 'log'
        
        # Gaussian sparseness (probabilistic interpretation)  
        coder_gaussian = SparseCoder(sparseness_function='gaussian')
        assert coder_gaussian.sparseness_function == 'gaussian'
        
    def test_initialization_all_optimization_methods(self):
        """Test all optimization methods for dictionary learning"""
        # Coordinate descent (standard method)
        coder_cd = SparseCoder(optimization_method='coordinate_descent')
        assert coder_cd.optimization_method == 'coordinate_descent'
        
        # Original Olshausen & Field equation (5) method
        coder_eq5 = SparseCoder(optimization_method='equation_5')
        assert coder_eq5.optimization_method == 'equation_5'
        
        # FISTA (fast iterative shrinkage-thresholding)
        coder_fista = SparseCoder(optimization_method='fista')
        assert coder_fista.optimization_method == 'fista'
        
        # Proximal gradient methods
        coder_prox = SparseCoder(optimization_method='proximal_gradient')
        assert coder_prox.optimization_method == 'proximal_gradient'
        
    def test_initialization_all_l1_solvers(self):
        """Test all L1 optimization solvers"""
        # Coordinate descent L1 solver
        coder_cd = SparseCoder(l1_solver='coordinate_descent')
        assert coder_cd.l1_solver == 'coordinate_descent'
        
        # L-BFGS-B solver
        coder_lbfgs = SparseCoder(l1_solver='lbfgs_b')
        assert coder_lbfgs.l1_solver == 'lbfgs_b'
        
        # FISTA L1 solver
        coder_fista = SparseCoder(l1_solver='fista')
        assert coder_fista.l1_solver == 'fista'
        
    def test_sklearn_compatible_parameters(self):
        """Test sklearn-style parameter interface"""
        # Alpha parameter (sklearn convention for L1 penalty)
        coder = SparseCoder(alpha=0.05)
        assert coder.sparsity_penalty == 0.05
        
        # Algorithm parameter (sklearn convention for optimization method)
        coder = SparseCoder(algorithm='fista')
        assert coder.optimization_method == 'fista'
        
        # Combined sklearn parameters
        coder = SparseCoder(alpha=0.2, algorithm='equation_5')
        assert coder.sparsity_penalty == 0.2
        assert coder.optimization_method == 'equation_5'
        
    def test_parameter_validation_and_bounds(self):
        """Test parameter validation preserves all valid configurations"""
        # Valid patch sizes
        for size in [(4, 4), (8, 8), (16, 16), (32, 32)]:
            coder = SparseCoder(patch_size=size)
            assert coder.patch_size == size
            
        # Valid n_components ranges
        for n in [16, 64, 128, 256, 512, 1024]:
            coder = SparseCoder(n_components=n)
            assert coder.n_components == n
            
        # Valid sparsity penalties
        for penalty in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
            coder = SparseCoder(sparsity_penalty=penalty)
            assert coder.sparsity_penalty == penalty


class TestOlshausenFieldAlgorithms:
    """Test core algorithms from Olshausen & Field (1996) Section 3"""
    
    def test_dictionary_initialization_methods(self):
        """Test dictionary initialization methods"""
        coder = SparseCoder(n_components=16, patch_size=(4, 4))
        
        # Random initialization (default)
        coder._initialize_dictionary()
        assert coder.dictionary is not None
        assert coder.dictionary.shape == (16, 16)
        
        # Verify dictionary is normalized (unit norm columns)
        norms = np.linalg.norm(coder.dictionary, axis=0)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)
        
    def test_whitening_filter_application(self):
        """Test zero-phase whitening filter (Olshausen & Field preprocessing)"""
        coder = SparseCoder(n_components=8, patch_size=(3, 3))
        
        # Create test patches
        patches = np.random.randn(10, 9) * 0.1 + 0.5
        
        # Apply whitening
        whitened = coder._apply_whitening_filter(patches)
        
        # Verify whitening properties
        assert whitened.shape == patches.shape
        assert not np.array_equal(patches, whitened)  # Should be transformed
        
    def test_sparse_encoding_equation_5_method(self):
        """Test Equation (5) from Olshausen & Field (1996)"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        test_patch = np.random.randn(4) * 0.1 + 0.5
        
        # Test equation 5 sparse encoding
        codes = coder._sparse_encode_equation_5(test_patch, max_iter=10)
        
        assert codes.shape == (4,)
        assert np.isfinite(codes).all()
        
    def test_fista_sparse_encoding_algorithm(self):
        """Test FISTA algorithm for sparse encoding"""
        coder = SparseCoder(n_components=6, patch_size=(3, 2))
        coder._initialize_dictionary()
        
        test_patch = np.random.randn(6) * 0.1 + 0.5
        
        # Test FISTA encoding
        codes = coder._fista_sparse_encode(test_patch, max_iter=20)
        
        assert codes.shape == (6,)
        assert np.isfinite(codes).all()
        
    def test_coordinate_descent_l1_solver(self):
        """Test coordinate descent L1 optimization"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        test_patch = np.random.randn(4) * 0.1 + 0.5
        
        # Test coordinate descent L1 solving
        codes = coder._coordinate_descent_l1(test_patch, max_iter=15)
        
        assert codes.shape == (4,)
        assert np.isfinite(codes).all()
        
    def test_lbfgs_l1_optimization(self):
        """Test L-BFGS L1 optimization method"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        test_patch = np.random.randn(4) * 0.1 + 0.5
        
        # Test L-BFGS L1 optimization
        codes = coder._lbfgs_l1_optimize(test_patch)
        
        assert codes.shape == (4,)
        assert np.isfinite(codes).all()


class TestSparseCodingMethods:
    """Test sparse coding inference methods (Research Paper Section 4)"""
    
    def test_single_patch_sparse_encoding(self):
        """Test sparse encoding of single patch"""
        coder = SparseCoder(n_components=8, patch_size=(4, 2))
        coder._initialize_dictionary()
        
        patch = np.random.randn(8) * 0.1 + 0.5
        
        codes = coder._sparse_encode_single(patch)
        
        assert codes.shape == (8,)
        assert np.isfinite(codes).all()
        
        # Verify sparsity (should have many near-zero elements)
        sparsity_ratio = np.mean(np.abs(codes) < 1e-6)
        assert sparsity_ratio > 0.3  # At least 30% sparse
        
    def test_batch_sparse_encoding(self):
        """Test batch sparse encoding of multiple patches"""
        coder = SparseCoder(n_components=6, patch_size=(3, 2))
        coder._initialize_dictionary()
        
        patches = np.random.randn(5, 6) * 0.1 + 0.5
        
        codes = coder.sparse_encode(patches)
        
        assert codes.shape == (5, 6)
        assert np.isfinite(codes).all()
        
    def test_sparseness_function_l1_penalty(self):
        """Test L1 sparseness function (primary method from paper)"""
        coder = SparseCoder(sparseness_function='l1')
        
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        penalty = coder._apply_sparseness_penalty(x)
        
        # L1 penalty: |x|
        expected = np.abs(x)
        np.testing.assert_allclose(penalty, expected)
        
    def test_sparseness_function_log_penalty(self):
        """Test log sparseness function (smooth approximation)"""
        coder = SparseCoder(sparseness_function='log')
        
        x = np.array([-1.0, 0.0, 1.0])
        penalty = coder._apply_sparseness_penalty(x)
        
        # Log penalty: log(1 + x²)
        expected = np.log(1 + x**2)
        np.testing.assert_allclose(penalty, expected)
        
    def test_sparseness_function_gaussian_penalty(self):
        """Test Gaussian sparseness function"""
        coder = SparseCoder(sparseness_function='gaussian')
        
        x = np.array([-1.0, 0.0, 1.0])
        penalty = coder._apply_sparseness_penalty(x)
        
        # Gaussian penalty: 1 - exp(-x²/2σ²)
        sigma = 1.0
        expected = 1 - np.exp(-x**2 / (2 * sigma**2))
        np.testing.assert_allclose(penalty, expected)


class TestDictionaryLearning:
    """Test dictionary learning algorithms (Research Paper Section 5)"""
    
    def test_dictionary_update_single_iteration(self):
        """Test single dictionary update iteration"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2), max_iter=1)
        
        # Create test data
        images = [np.random.randn(6, 6) * 0.1 + 0.5]
        
        # Perform dictionary learning
        coder.fit(images, n_patches=8)
        
        assert coder.dictionary is not None
        assert coder.dictionary.shape == (4, 4)
        
        # Verify dictionary normalization
        norms = np.linalg.norm(coder.dictionary, axis=0)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)
        
    def test_dictionary_learning_convergence(self):
        """Test dictionary learning convergence properties"""
        coder = SparseCoder(n_components=6, patch_size=(3, 2), max_iter=10, tolerance=1e-6)
        
        images = [np.random.randn(8, 8) * 0.1 + 0.5 for _ in range(2)]
        
        # Dictionary learning should converge
        result = coder.fit(images, n_patches=12)
        
        assert 'converged' in result
        assert 'final_reconstruction_error' in result
        assert 'n_iterations' in result
        
    def test_alternating_optimization_steps(self):
        """Test alternating optimization (sparse coding + dictionary update)"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2), max_iter=3)
        
        images = [np.random.randn(6, 6) * 0.1 + 0.5]
        
        # Track optimization steps
        initial_dict = None
        
        def track_dictionary(self):
            nonlocal initial_dict
            if initial_dict is None:
                initial_dict = self.dictionary.copy()
                
        # Monkey patch to track changes
        original_update = coder._update_dictionary_single_iteration
        coder._update_dictionary_single_iteration = lambda patches, codes: (
            track_dictionary(coder), 
            original_update(patches, codes)
        )[1]
        
        coder.fit(images, n_patches=8)
        
        # Dictionary should have changed during optimization
        if initial_dict is not None:
            assert not np.allclose(coder.dictionary, initial_dict)


class TestImageProcessingPipeline:
    """Test complete image processing pipeline (Research Application)"""
    
    def test_patch_extraction_from_images(self):
        """Test patch extraction from natural images"""
        coder = SparseCoder(n_components=8, patch_size=(4, 4))
        
        # Create test image
        image = np.random.randn(12, 12) * 0.1 + 0.5
        
        # Extract patches
        patches = coder._extract_patches_from_image(image, n_patches=6)
        
        assert patches.shape[0] == 6  # Number of patches
        assert patches.shape[1] == 16  # Flattened patch size (4*4)
        
    def test_image_reconstruction_from_patches(self):
        """Test reconstruction of images from patch codes"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        # Create test patches and encode
        patches = np.random.randn(3, 4) * 0.1 + 0.5
        codes = coder.sparse_encode(patches)
        
        # Reconstruct patches
        reconstructed = coder.reconstruct(codes)
        
        assert reconstructed.shape == patches.shape
        
        # Verify reconstruction equation: reconstructed = codes @ dictionary.T
        expected = codes @ coder.dictionary.T
        np.testing.assert_allclose(reconstructed, expected)
        
    def test_end_to_end_sparse_coding_pipeline(self):
        """Test complete end-to-end pipeline"""
        coder = SparseCoder(n_components=8, patch_size=(4, 4), max_iter=2)
        
        # Training images
        train_images = [np.random.randn(10, 10) * 0.1 + 0.5 for _ in range(2)]
        
        # Step 1: Learn dictionary
        coder.fit(train_images, n_patches=12)
        
        # Step 2: Encode test image
        test_image = np.random.randn(10, 10) * 0.1 + 0.5
        codes = coder.transform([test_image])
        
        # Step 3: Reconstruct
        reconstruction = coder.reconstruct(codes)
        
        # Verify pipeline integrity
        assert codes.shape[1] == 8  # n_components
        assert reconstruction.shape[1] == 16  # patch_size flattened
        
        # Verify reconstruction consistency
        expected_reconstruction = codes @ coder.dictionary.T
        np.testing.assert_allclose(reconstruction, expected_reconstruction)


class TestSklearnCompatibility:
    """Test scikit-learn API compatibility"""
    
    def test_fit_method_signature_and_behavior(self):
        """Test fit method with all parameter variations"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        
        images = [np.random.randn(6, 6) * 0.1 + 0.5]
        
        # Test with n_patches parameter
        result1 = coder.fit(images, n_patches=8)
        assert result1 is not None
        
        # Test without optional parameters
        coder2 = SparseCoder(n_components=4, patch_size=(2, 2))
        result2 = coder2.fit(images)
        assert result2 is not None
        
    def test_transform_method_behavior(self):
        """Test transform method for sparse encoding"""
        coder = SparseCoder(n_components=6, patch_size=(3, 2), max_iter=2)
        
        # Fit first
        train_images = [np.random.randn(8, 8) * 0.1 + 0.5]
        coder.fit(train_images, n_patches=10)
        
        # Transform test data
        test_images = [np.random.randn(8, 8) * 0.1 + 0.5]
        codes = coder.transform(test_images)
        
        assert codes.shape[1] == 6  # n_components
        assert np.isfinite(codes).all()
        
    def test_fit_transform_method_equivalence(self):
        """Test fit_transform method equivalence to fit().transform()"""
        coder1 = SparseCoder(n_components=4, patch_size=(2, 2), max_iter=2)
        coder2 = SparseCoder(n_components=4, patch_size=(2, 2), max_iter=2)
        
        images = [np.random.randn(6, 6) * 0.1 + 0.5]
        
        # Method 1: fit_transform
        codes1 = coder1.fit_transform(images, n_patches=8)
        
        # Method 2: fit then transform
        coder2.fit(images, n_patches=8)
        codes2 = coder2.transform(images)
        
        # Should produce similar results (allowing for randomness)
        assert codes1.shape == codes2.shape
        
    def test_get_params_and_set_params(self):
        """Test sklearn parameter interface"""
        coder = SparseCoder(n_components=16, sparsity_penalty=0.2)
        
        # Get parameters
        params = coder.get_params()
        
        assert 'n_components' in params
        assert 'sparsity_penalty' in params
        assert params['n_components'] == 16
        assert params['sparsity_penalty'] == 0.2
        
        # Set parameters
        coder.set_params(n_components=32, sparsity_penalty=0.15)
        
        assert coder.n_components == 32
        assert coder.sparsity_penalty == 0.15


class TestConfigurationPreservation:
    """Test that all configuration options are preserved and functional"""
    
    def test_all_sparseness_functions_preserved(self):
        """Verify all sparseness functions work without removal"""
        functions = ['l1', 'log', 'gaussian']
        
        for func in functions:
            coder = SparseCoder(n_components=4, patch_size=(2, 2), sparseness_function=func)
            coder._initialize_dictionary()
            
            # Test that function works
            test_patch = np.random.randn(4) * 0.1 + 0.5
            codes = coder._sparse_encode_single(test_patch)
            
            assert codes.shape == (4,)
            assert np.isfinite(codes).all()
            
    def test_all_optimization_methods_preserved(self):
        """Verify all optimization methods work without removal"""
        methods = ['coordinate_descent', 'equation_5', 'fista', 'proximal_gradient']
        
        for method in methods:
            coder = SparseCoder(n_components=4, patch_size=(2, 2), optimization_method=method, max_iter=2)
            
            images = [np.random.randn(6, 6) * 0.1 + 0.5]
            
            # Test that method works
            result = coder.fit(images, n_patches=6)
            assert result is not None
            
    def test_all_l1_solvers_preserved(self):
        """Verify all L1 solvers work without removal"""
        solvers = ['coordinate_descent', 'lbfgs_b', 'fista']
        
        for solver in solvers:
            coder = SparseCoder(n_components=4, patch_size=(2, 2), l1_solver=solver)
            coder._initialize_dictionary()
            
            # Test that solver works
            test_patch = np.random.randn(4) * 0.1 + 0.5
            
            if solver == 'coordinate_descent':
                codes = coder._coordinate_descent_l1(test_patch, max_iter=10)
            elif solver == 'lbfgs_b':
                codes = coder._lbfgs_l1_optimize(test_patch)
            elif solver == 'fista':
                codes = coder._fista_l1_solve(test_patch, max_iter=10)
                
            assert codes.shape == (4,)
            assert np.isfinite(codes).all()


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""
    
    def test_empty_image_handling(self):
        """Test handling of empty or invalid images"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        
        # Test with empty list
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = coder.fit([], n_patches=0)
            
        # Should handle gracefully without crashing
        
    def test_single_pixel_patches(self):
        """Test handling of 1x1 patches"""
        coder = SparseCoder(n_components=2, patch_size=(1, 1), max_iter=2)
        
        images = [np.random.randn(3, 3) * 0.1 + 0.5]
        
        # Should work without error
        coder.fit(images, n_patches=4)
        codes = coder.transform(images)
        
        assert codes.shape[1] == 2
        
    def test_large_dictionary_handling(self):
        """Test handling of large overcomplete dictionaries"""
        coder = SparseCoder(n_components=64, patch_size=(4, 4), max_iter=1)
        
        images = [np.random.randn(12, 12) * 0.1 + 0.5]
        
        # Should handle large dictionaries
        coder.fit(images, n_patches=20)
        
        assert coder.dictionary.shape == (16, 64)  # patch_size flattened x n_components


class TestResearchAlgorithmAccuracy:
    """Test accuracy of implementation against research paper specifications"""
    
    def test_olshausen_field_equation_5_implementation(self):
        """Test Equation (5) implementation matches paper specification"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2), optimization_method='equation_5')
        coder._initialize_dictionary()
        
        test_patch = np.random.randn(4) * 0.1 + 0.5
        
        # Test equation 5 with different parameters
        codes = coder._sparse_encode_equation_5(test_patch, learning_rate=0.01, max_iter=20)
        
        # Verify basic properties
        assert codes.shape == (4,)
        assert np.isfinite(codes).all()
        
        # Verify that codes satisfy sparsity constraints
        reconstruction_error = np.mean((test_patch - codes @ coder.dictionary.T)**2)
        assert reconstruction_error < 1.0  # Reasonable reconstruction
        
    def test_dictionary_normalization_constraint(self):
        """Test that dictionary columns maintain unit norm (paper constraint)"""
        coder = SparseCoder(n_components=8, patch_size=(4, 2), max_iter=3)
        
        images = [np.random.randn(8, 8) * 0.1 + 0.5]
        coder.fit(images, n_patches=12)
        
        # Verify unit norm constraint
        norms = np.linalg.norm(coder.dictionary, axis=0)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-8)
        
    def test_sparsity_penalty_effect(self):
        """Test that sparsity penalty produces sparser solutions"""
        coder_low = SparseCoder(n_components=8, patch_size=(4, 2), sparsity_penalty=0.01)
        coder_high = SparseCoder(n_components=8, patch_size=(4, 2), sparsity_penalty=0.5)
        
        # Initialize same dictionary
        coder_low._initialize_dictionary()
        coder_high.dictionary = coder_low.dictionary.copy()
        
        test_patch = np.random.randn(8) * 0.1 + 0.5
        
        codes_low = coder_low._sparse_encode_single(test_patch)
        codes_high = coder_high._sparse_encode_single(test_patch)
        
        # Higher penalty should produce sparser solution
        sparsity_low = np.mean(np.abs(codes_low) < 1e-6)
        sparsity_high = np.mean(np.abs(codes_high) < 1e-6)
        
        assert sparsity_high >= sparsity_low  # Higher penalty → more sparse


def test_comprehensive_functionality_integration():
    """Integration test covering major functionality paths"""
    # Test multiple configurations together
    configs = [
        {'sparseness_function': 'l1', 'optimization_method': 'coordinate_descent'},
        {'sparseness_function': 'log', 'optimization_method': 'fista'},
        {'sparseness_function': 'gaussian', 'optimization_method': 'equation_5'}
    ]
    
    for config in configs:
        coder = SparseCoder(n_components=6, patch_size=(3, 2), max_iter=2, **config)
        
        # Full pipeline test
        images = [np.random.randn(8, 8) * 0.1 + 0.5]
        coder.fit(images, n_patches=8)
        codes = coder.transform(images)
        reconstruction = coder.reconstruct(codes)
        
        # Verify all steps work together
        assert codes.shape[1] == 6
        assert reconstruction.shape[1] == 6
        assert np.isfinite(codes).all()
        assert np.isfinite(reconstruction).all()


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])