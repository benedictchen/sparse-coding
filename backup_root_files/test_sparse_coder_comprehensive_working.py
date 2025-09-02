#!/usr/bin/env python3
"""
Sparse Coder - Comprehensive Working Test Suite for 100% Coverage
=================================================================

Systematic test suite building from working functionality to 100% coverage.
Tests are designed to work with actual implementation methods.

Research Paper: Olshausen & Field (1996)
Author: Benedict Chen (benedict@benedictchen.com)
"""

import numpy as np
import pytest
import warnings

try:
    from .sparse_coder import SparseCoder
except ImportError:
    from sparse_coder import SparseCoder


class TestBasicInitialization:
    """Test basic initialization and parameter handling"""
    
    def test_default_initialization(self):
        """Test default parameter initialization"""
        coder = SparseCoder()
        assert coder.n_components == 256
        assert coder.patch_size == (16, 16)
        assert coder.sparseness_function == 'l1'
        assert coder.optimization_method == 'coordinate_descent'
        
    def test_custom_parameters(self):
        """Test custom parameter initialization"""
        coder = SparseCoder(
            n_components=64,
            patch_size=(8, 8),
            sparsity_penalty=0.2,
            max_iter=20
        )
        assert coder.n_components == 64
        assert coder.patch_size == (8, 8)
        assert coder.sparsity_penalty == 0.2
        assert coder.max_iter == 20
        
    def test_sklearn_parameter_compatibility(self):
        """Test sklearn-style parameters"""
        # Test alpha parameter (maps to sparsity_penalty)
        coder1 = SparseCoder(alpha=0.15)
        assert coder1.sparsity_penalty == 0.15
        
        # Test algorithm parameter (maps to optimization_method)  
        coder2 = SparseCoder(algorithm='fista')
        assert coder2.optimization_method == 'fista'


class TestConfigurationOptions:
    """Test all configuration options work"""
    
    def test_all_sparseness_functions(self):
        """Test all sparseness function options"""
        functions = ['l1', 'log', 'gaussian']
        for func in functions:
            coder = SparseCoder(sparseness_function=func)
            assert coder.sparseness_function == func
            
    def test_all_optimization_methods(self):
        """Test all optimization method options"""
        methods = ['coordinate_descent', 'equation_5', 'fista', 'proximal_gradient']
        for method in methods:
            coder = SparseCoder(optimization_method=method)
            assert coder.optimization_method == method
            
    def test_all_l1_solvers(self):
        """Test all L1 solver options"""
        solvers = ['coordinate_descent', 'lbfgs_b', 'fista']
        for solver in solvers:
            coder = SparseCoder(l1_solver=solver)
            assert coder.l1_solver == solver


class TestDictionaryOperations:
    """Test dictionary initialization and operations"""
    
    def test_dictionary_initialization(self):
        """Test dictionary initialization"""
        coder = SparseCoder(n_components=8, patch_size=(4, 2))
        coder._initialize_dictionary()
        
        assert coder.dictionary is not None
        assert coder.dictionary.shape == (8, 8)  # patch_size flattened x n_components
        
        # Check normalization
        norms = np.linalg.norm(coder.dictionary, axis=0)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)
        
    def test_dictionary_coherence_calculation(self):
        """Test dictionary coherence calculation"""
        coder = SparseCoder(n_components=6, patch_size=(3, 2))
        coder._initialize_dictionary()
        
        coherence = coder._calculate_dictionary_coherence()
        assert 0 <= coherence <= 1
        assert np.isfinite(coherence)


class TestSparseEncodingMethods:
    """Test sparse encoding algorithms"""
    
    def test_single_patch_encoding(self):
        """Test encoding single patch"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        patch = np.random.randn(4) * 0.1 + 0.5
        codes = coder._sparse_encode_single(patch)
        
        assert codes.shape == (4,)
        assert np.isfinite(codes).all()
        
    def test_fista_sparse_encoding(self):
        """Test FISTA sparse encoding"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        patch = np.random.randn(4) * 0.1 + 0.5
        codes = coder._fista_sparse_encode(patch, max_iter=10)
        
        assert codes.shape == (4,)
        assert np.isfinite(codes).all()
        
    def test_equation_5_encoding(self):
        """Test Olshausen & Field equation 5 encoding"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        patch = np.random.randn(4) * 0.1 + 0.5
        codes = coder._sparse_encode_equation_5(patch)
        
        assert codes.shape == (4,)
        assert np.isfinite(codes).all()
        
    def test_batch_sparse_encoding(self):
        """Test batch sparse encoding"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        patches = np.random.randn(3, 4) * 0.1 + 0.5
        codes = coder.sparse_encode(patches)
        
        assert codes.shape == (3, 4)
        assert np.isfinite(codes).all()


class TestNewSparsityMethods:
    """Test newly added sparsity penalty methods"""
    
    def test_apply_sparseness_penalty_l1(self):
        """Test L1 sparseness penalty"""
        coder = SparseCoder(sparseness_function='l1')
        coeffs = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        penalty = coder._apply_sparseness_penalty(coeffs)
        expected = np.abs(coeffs)
        np.testing.assert_allclose(penalty, expected)
        
    def test_apply_sparseness_penalty_log(self):
        """Test log sparseness penalty"""
        coder = SparseCoder(sparseness_function='log')
        coeffs = np.array([-1.0, 0.0, 1.0])
        penalty = coder._apply_sparseness_penalty(coeffs)
        expected = np.log(1 + coeffs**2)
        np.testing.assert_allclose(penalty, expected)
        
    def test_apply_sparseness_penalty_gaussian(self):
        """Test Gaussian sparseness penalty"""
        coder = SparseCoder(sparseness_function='gaussian')
        coeffs = np.array([-1.0, 0.0, 1.0])
        penalty = coder._apply_sparseness_penalty(coeffs)
        
        # Should be finite and between 0 and 1
        assert np.isfinite(penalty).all()
        assert (penalty >= 0).all()
        assert (penalty <= 1).all()


class TestL1SolverMethods:
    """Test newly added L1 solver methods"""
    
    def test_coordinate_descent_l1(self):
        """Test coordinate descent L1 solver"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        signal = np.random.randn(4) * 0.1 + 0.5
        codes = coder._coordinate_descent_l1(signal, max_iter=10)
        
        assert codes.shape == (4,)
        assert np.isfinite(codes).all()
        
    def test_lbfgs_l1_optimize(self):
        """Test L-BFGS L1 optimization"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        signal = np.random.randn(4) * 0.1 + 0.5
        codes = coder._lbfgs_l1_optimize(signal)
        
        assert codes.shape == (4,)
        assert np.isfinite(codes).all()
        
    def test_fista_l1_solve(self):
        """Test FISTA L1 solver"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        signal = np.random.randn(4) * 0.1 + 0.5
        codes = coder._fista_l1_solve(signal, max_iter=10)
        
        assert codes.shape == (4,)
        assert np.isfinite(codes).all()


class TestImageProcessing:
    """Test image processing methods"""
    
    def test_extract_patches_from_image(self):
        """Test patch extraction from single image"""
        coder = SparseCoder(n_components=4, patch_size=(3, 3))
        
        image = np.random.randn(8, 8) * 0.1 + 0.5
        patches = coder._extract_patches_from_image(image, n_patches=5)
        
        assert patches.shape == (5, 9)  # 5 patches of 3x3 = 9
        assert np.isfinite(patches).all()
        
    def test_extract_patches_from_image_empty(self):
        """Test patch extraction with zero patches"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        
        image = np.random.randn(6, 6) * 0.1 + 0.5
        patches = coder._extract_patches_from_image(image, n_patches=0)
        
        assert patches.shape == (0, 4)
        
    def test_whitening_filter_application(self):
        """Test whitening filter with valid input"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        
        patches = np.random.randn(10, 4) * 0.1 + 0.5
        whitened = coder._apply_whitening_filter(patches)
        
        assert whitened.shape == patches.shape
        assert not np.array_equal(patches, whitened)  # Should be different
        
    def test_whitening_filter_empty(self):
        """Test whitening filter with empty input"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        
        patches = np.empty((0, 4))
        whitened = coder._apply_whitening_filter(patches)
        
        assert whitened.shape == patches.shape


class TestSklearnCompatibility:
    """Test sklearn-compatible methods"""
    
    def test_get_params(self):
        """Test get_params method"""
        coder = SparseCoder(n_components=32, sparsity_penalty=0.15)
        params = coder.get_params()
        
        assert isinstance(params, dict)
        assert 'n_components' in params
        assert 'sparsity_penalty' in params
        assert params['n_components'] == 32
        assert params['sparsity_penalty'] == 0.15
        
    def test_set_params(self):
        """Test set_params method"""
        coder = SparseCoder()
        
        # Set parameters
        result = coder.set_params(n_components=64, sparsity_penalty=0.25)
        
        assert result is coder  # Should return self
        assert coder.n_components == 64
        assert coder.sparsity_penalty == 0.25
        
    def test_set_params_invalid(self):
        """Test set_params with invalid parameter"""
        coder = SparseCoder()
        
        with pytest.raises(ValueError):
            coder.set_params(invalid_parameter=123)


class TestDictionaryLearning:
    """Test dictionary learning and fitting"""
    
    def test_dictionary_update_single_iteration(self):
        """Test single dictionary update iteration"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        patches = np.random.randn(3, 4) * 0.1 + 0.5
        coefficients = np.random.randn(3, 4) * 0.1
        
        change = coder._update_dictionary_single_iteration(patches, coefficients)
        
        assert np.isfinite(change)
        assert change >= 0
        
    def test_dictionary_update_empty_input(self):
        """Test dictionary update with empty input"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        patches = np.empty((0, 4))
        coefficients = np.empty((0, 4))
        
        change = coder._update_dictionary_single_iteration(patches, coefficients)
        assert change == 0.0
        
    def test_fit_basic(self):
        """Test basic fitting functionality"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2), max_iter=1)
        
        images = [np.random.randn(6, 6) * 0.1 + 0.5]
        result = coder.fit(images, n_patches=8)
        
        assert isinstance(result, dict)
        assert coder.dictionary is not None
        assert coder.dictionary.shape == (4, 4)
        
    def test_fit_transform_integration(self):
        """Test fit_transform method"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2), max_iter=1)
        
        images = [np.random.randn(6, 6) * 0.1 + 0.5]
        codes = coder.fit_transform(images, n_patches=8)
        
        assert codes is not None
        assert codes.shape[1] == 4  # n_components
        
    def test_transform_after_fit(self):
        """Test transform after fitting"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2), max_iter=1)
        
        # Fit first
        images = [np.random.randn(6, 6) * 0.1 + 0.5]
        coder.fit(images, n_patches=8)
        
        # Transform test data
        test_images = [np.random.randn(6, 6) * 0.1 + 0.5]
        codes = coder.transform(test_images)
        
        assert codes is not None
        assert codes.shape[1] == 4
        
    def test_reconstruct(self):
        """Test reconstruction from coefficients"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        coder._initialize_dictionary()
        
        coefficients = np.random.randn(2, 4) * 0.1
        reconstruction = coder.reconstruct(coefficients)
        
        assert reconstruction.shape == (2, 4)
        
        # Verify reconstruction equation
        expected = coefficients @ coder.dictionary.T
        np.testing.assert_allclose(reconstruction, expected)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_small_patches(self):
        """Test with 1x1 patches"""
        coder = SparseCoder(n_components=2, patch_size=(1, 1), max_iter=1)
        
        images = [np.random.randn(3, 3) * 0.1 + 0.5]
        coder.fit(images, n_patches=4)
        
        assert coder.dictionary.shape == (1, 2)
        
    def test_empty_image_list_handling(self):
        """Test handling of empty image lists"""
        coder = SparseCoder(n_components=4, patch_size=(2, 2))
        
        # Should handle gracefully without crashing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = coder.fit([], n_patches=0)
                # If it doesn't crash, that's good
            except Exception as e:
                # Some exceptions are expected for empty input
                assert "patches" in str(e).lower() or "empty" in str(e).lower()


def test_configuration_preservation():
    """Integration test verifying all configurations work together"""
    configs = [
        {
            'sparseness_function': 'l1', 
            'optimization_method': 'coordinate_descent',
            'l1_solver': 'coordinate_descent'
        },
        {
            'sparseness_function': 'log', 
            'optimization_method': 'fista',
            'l1_solver': 'fista'
        },
        {
            'sparseness_function': 'gaussian', 
            'optimization_method': 'equation_5',
            'l1_solver': 'lbfgs_b'
        }
    ]
    
    for config in configs:
        coder = SparseCoder(
            n_components=4, 
            patch_size=(2, 2), 
            max_iter=1,
            **config
        )
        
        # Test that configuration is preserved
        for key, value in config.items():
            assert getattr(coder, key) == value
            
        # Test that basic functionality works with this config
        images = [np.random.randn(6, 6) * 0.1 + 0.5]
        
        try:
            coder.fit(images, n_patches=6)
            codes = coder.transform(images)
            reconstruction = coder.reconstruct(codes)
            
            # Basic sanity checks
            assert codes is not None
            assert reconstruction is not None
            assert np.isfinite(codes).all()
            assert np.isfinite(reconstruction).all()
        except Exception as e:
            # Some configurations might fail, but we want to know which ones
            pytest.fail(f"Configuration {config} failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])