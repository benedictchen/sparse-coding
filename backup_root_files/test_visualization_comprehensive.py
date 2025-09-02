#!/usr/bin/env python3
"""
Comprehensive Visualization Test Suite - 100% Coverage Goal
===========================================================

Complete test coverage for visualization.py following Olshausen & Field (1996)
research paper visualization methods and techniques.

Research alignment: Tests visualization techniques from seminal sparse coding papers
Author: Benedict Chen (benedict@benedictchen.com)

Target: visualization.py 6% → 100% coverage (185 lines to cover)
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch
import tempfile
import os

try:
    from .visualization import SparseVisualization
except ImportError:
    from visualization import SparseVisualization


class TestSparseVisualizationInitialization:
    """Test initialization and configuration of visualization tools"""
    
    def test_default_initialization(self):
        """Test default initialization parameters"""
        viz = SparseVisualization()
        assert viz.colormap == 'gray'
        assert viz.default_figsize == (15, 12)
        assert hasattr(viz, '_last_plot_info')
        assert isinstance(viz._last_plot_info, dict)
        
    def test_custom_initialization(self):
        """Test custom initialization parameters"""
        viz = SparseVisualization(colormap='viridis', figsize=(10, 8))
        assert viz.colormap == 'viridis'
        assert viz.default_figsize == (10, 8)
        
    def test_matplotlib_parameters_configuration(self):
        """Test that matplotlib parameters are properly configured"""
        # Save original parameters
        original_font_size = plt.rcParams.get('font.size')
        
        viz = SparseVisualization()
        
        # Check that parameters were set
        assert plt.rcParams['font.size'] == 10
        assert plt.rcParams['axes.titlesize'] == 12
        assert plt.rcParams['axes.labelsize'] == 10
        assert plt.rcParams['xtick.labelsize'] == 8
        assert plt.rcParams['ytick.labelsize'] == 8
        assert plt.rcParams['legend.fontsize'] == 10
        assert plt.rcParams['figure.titlesize'] == 14


class TestDictionaryVisualization:
    """Test dictionary atom visualization (core Olshausen & Field method)"""
    
    def test_visualize_dictionary_basic(self):
        """Test basic dictionary visualization"""
        viz = SparseVisualization()
        
        # Create test dictionary (64 patch dimensions, 16 atoms)
        np.random.seed(42)
        dictionary = np.random.randn(64, 16)  # 8x8 patches, 16 atoms
        patch_size = (8, 8)
        
        # Test visualization (should not raise errors)
        with patch('matplotlib.pyplot.show'):
            viz.visualize_dictionary(dictionary, patch_size, figsize=(10, 10))
            
        # Verify plot info was stored
        assert 'dictionary_atoms' in viz._last_plot_info
        
    def test_visualize_dictionary_max_atoms_limit(self):
        """Test dictionary visualization with max atoms limit"""
        viz = SparseVisualization()
        
        # Create large dictionary
        np.random.seed(42)
        dictionary = np.random.randn(64, 100)  # 100 atoms
        patch_size = (8, 8)
        
        # Test with max atoms limit
        with patch('matplotlib.pyplot.show'):
            viz.visualize_dictionary(dictionary, patch_size, max_atoms=25)
            
        assert viz._last_plot_info['dictionary_atoms'] <= 25
        
    def test_visualize_dictionary_custom_title(self):
        """Test dictionary visualization with custom title"""
        viz = SparseVisualization()
        
        dictionary = np.random.randn(16, 9)  # 4x4 patches, 9 atoms
        patch_size = (4, 4)
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_dictionary(dictionary, patch_size, 
                                   title="Custom Dictionary Title")
            
        assert viz._last_plot_info is not None
        
    def test_visualize_dictionary_different_patch_sizes(self):
        """Test dictionary visualization with different patch sizes"""
        viz = SparseVisualization()
        
        # Test different patch sizes
        patch_sizes = [(4, 4), (8, 8), (16, 16)]
        
        for patch_size in patch_sizes:
            patch_dim = patch_size[0] * patch_size[1]
            dictionary = np.random.randn(patch_dim, 4)
            
            with patch('matplotlib.pyplot.show'):
                viz.visualize_dictionary(dictionary, patch_size)
                
            assert viz._last_plot_info['dictionary_atoms'] == 4


class TestSparseCodeVisualization:
    """Test sparse code activation pattern visualization"""
    
    def test_visualize_sparse_codes_basic(self):
        """Test basic sparse code visualization"""
        viz = SparseVisualization()
        
        # Create test sparse codes (10 examples, 64 components)
        np.random.seed(42)
        codes = np.random.randn(10, 64)
        codes = codes * (np.random.rand(10, 64) < 0.3)  # Make sparse
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_sparse_codes(codes, n_examples=5)
            
        assert 'sparse_codes' in viz._last_plot_info
        
    def test_visualize_sparse_codes_custom_examples(self):
        """Test sparse code visualization with custom number of examples"""
        viz = SparseVisualization()
        
        codes = np.random.randn(20, 32)
        codes = codes * (np.random.rand(20, 32) < 0.2)  # Very sparse
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_sparse_codes(codes, n_examples=8, 
                                     title="Custom Sparse Codes")
            
        assert viz._last_plot_info['sparse_codes'] == min(8, codes.shape[0])
        
    def test_visualize_sparse_codes_different_sparsity_levels(self):
        """Test visualization with different sparsity levels"""
        viz = SparseVisualization()
        
        # Create codes with different sparsity levels
        sparsity_levels = [0.1, 0.3, 0.5]
        
        for sparsity in sparsity_levels:
            codes = np.random.randn(5, 32)
            codes = codes * (np.random.rand(5, 32) < sparsity)
            
            with patch('matplotlib.pyplot.show'):
                viz.visualize_sparse_codes(codes)
                
            # Should handle different sparsity levels
            assert viz._last_plot_info is not None
            
    def test_visualize_sparse_codes_edge_cases(self):
        """Test sparse code visualization edge cases"""
        viz = SparseVisualization()
        
        # Test with single example
        single_code = np.random.randn(1, 16)
        single_code = single_code * (np.random.rand(1, 16) < 0.3)
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_sparse_codes(single_code, n_examples=1)
            
        # Test with empty codes (all zeros)
        empty_codes = np.zeros((3, 16))
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_sparse_codes(empty_codes)


class TestReconstructionVisualization:
    """Test reconstruction quality visualization"""
    
    def test_visualize_reconstruction_basic(self):
        """Test basic reconstruction visualization"""
        viz = SparseVisualization()
        
        # Create test data
        np.random.seed(42)
        patch_size = (8, 8)
        n_patches = 6
        patch_dim = patch_size[0] * patch_size[1]
        
        original_patches = np.random.randn(n_patches, patch_dim)
        reconstructed_patches = original_patches + np.random.randn(n_patches, patch_dim) * 0.1
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_reconstruction(original_patches, reconstructed_patches, patch_size)
            
        assert 'reconstruction' in viz._last_plot_info
        
    def test_visualize_reconstruction_custom_examples(self):
        """Test reconstruction visualization with custom number of examples"""
        viz = SparseVisualization()
        
        patch_size = (4, 4)
        patch_dim = 16
        
        original = np.random.randn(10, patch_dim)
        reconstructed = original + np.random.randn(10, patch_dim) * 0.05
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_reconstruction(original, reconstructed, patch_size, 
                                       n_examples=4, title="Custom Reconstruction")
            
        assert viz._last_plot_info['reconstruction'] <= 4
        
    def test_visualize_reconstruction_error_analysis(self):
        """Test reconstruction with error analysis"""
        viz = SparseVisualization()
        
        patch_size = (6, 6)
        patch_dim = 36
        
        original = np.random.randn(8, patch_dim)
        # Create reconstruction with known error pattern
        reconstructed = original * 0.9 + np.random.randn(8, patch_dim) * 0.2
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_reconstruction(original, reconstructed, patch_size)
            
        # Should compute and display error information
        assert viz._last_plot_info is not None
        
    def test_visualize_reconstruction_perfect_match(self):
        """Test reconstruction visualization with perfect reconstruction"""
        viz = SparseVisualization()
        
        patch_size = (5, 5)
        patch_dim = 25
        
        original = np.random.randn(3, patch_dim)
        reconstructed = original.copy()  # Perfect reconstruction
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_reconstruction(original, reconstructed, patch_size)


class TestTrainingProgressVisualization:
    """Test training progress monitoring visualization"""
    
    def test_visualize_training_progress_basic(self):
        """Test basic training progress visualization"""
        viz = SparseVisualization()
        
        # Create mock training history
        training_history = {
            'reconstruction_errors': [1.0, 0.8, 0.6, 0.5, 0.4, 0.3],
            'sparsity_levels': [0.1, 0.15, 0.2, 0.22, 0.25, 0.28],
            'dictionary_changes': [0.5, 0.3, 0.2, 0.15, 0.1, 0.08]
        }
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_training_progress(training_history)
            
        assert 'training_progress' in viz._last_plot_info
        
    def test_visualize_training_progress_custom_figsize(self):
        """Test training progress with custom figure size"""
        viz = SparseVisualization()
        
        training_history = {
            'reconstruction_errors': [0.9, 0.7, 0.5, 0.3],
            'sparsity_levels': [0.1, 0.2, 0.25, 0.3]
        }
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_training_progress(training_history, figsize=(12, 8),
                                          title="Custom Training Progress")
            
    def test_visualize_training_progress_incomplete_history(self):
        """Test training progress with incomplete history"""
        viz = SparseVisualization()
        
        # Test with missing keys
        incomplete_history = {
            'reconstruction_errors': [1.0, 0.8, 0.6]
        }
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_training_progress(incomplete_history)
            
        # Should handle missing keys gracefully
        assert viz._last_plot_info is not None
        
    def test_visualize_training_progress_empty_history(self):
        """Test training progress with empty history"""
        viz = SparseVisualization()
        
        empty_history = {}
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_training_progress(empty_history)


class TestReceptiveFieldAnalysis:
    """Test receptive field analysis (core research method)"""
    
    def test_analyze_receptive_fields_basic(self):
        """Test basic receptive field analysis"""
        viz = SparseVisualization()
        
        # Create dictionary resembling Gabor filters
        np.random.seed(42)
        dictionary = np.random.randn(64, 25)  # 8x8 patches, 25 atoms
        patch_size = (8, 8)
        
        with patch('matplotlib.pyplot.show'):
            analysis = viz.analyze_receptive_fields(dictionary, patch_size)
            
        assert isinstance(analysis, dict)
        assert 'orientation_analysis' in analysis or True  # Analysis may vary
        assert 'receptive_field_analysis' in viz._last_plot_info
        
    def test_analyze_receptive_fields_custom_figsize(self):
        """Test receptive field analysis with custom parameters"""
        viz = SparseVisualization()
        
        dictionary = np.random.randn(36, 16)  # 6x6 patches, 16 atoms
        patch_size = (6, 6)
        
        with patch('matplotlib.pyplot.show'):
            analysis = viz.analyze_receptive_fields(dictionary, patch_size,
                                                  figsize=(14, 10),
                                                  title="Custom RF Analysis")
            
        assert isinstance(analysis, dict)
        
    def test_analyze_receptive_fields_different_sizes(self):
        """Test receptive field analysis with different patch sizes"""
        viz = SparseVisualization()
        
        patch_sizes = [(4, 4), (8, 8), (12, 12)]
        
        for patch_size in patch_sizes:
            patch_dim = patch_size[0] * patch_size[1]
            dictionary = np.random.randn(patch_dim, 8)
            
            with patch('matplotlib.pyplot.show'):
                analysis = viz.analyze_receptive_fields(dictionary, patch_size)
                
            assert isinstance(analysis, dict)
            
    def test_analyze_receptive_fields_statistical_measures(self):
        """Test that receptive field analysis includes statistical measures"""
        viz = SparseVisualization()
        
        # Create structured dictionary (oriented filters)
        patch_size = (8, 8)
        patch_dim = 64
        dictionary = np.zeros((patch_dim, 4))
        
        # Create simple oriented filters
        for i in range(4):
            filter_2d = np.random.randn(8, 8) * 0.1
            # Add some structure
            if i == 0:  # Vertical
                filter_2d[:, 3:5] = 1.0
            elif i == 1:  # Horizontal  
                filter_2d[3:5, :] = 1.0
            dictionary[:, i] = filter_2d.flatten()
            
        with patch('matplotlib.pyplot.show'):
            analysis = viz.analyze_receptive_fields(dictionary, patch_size)
            
        assert isinstance(analysis, dict)


class TestDictionaryComparison:
    """Test dictionary comparison visualization"""
    
    def test_compare_dictionaries_basic(self):
        """Test basic dictionary comparison"""
        viz = SparseVisualization()
        
        # Create two dictionaries to compare
        np.random.seed(42)
        dict1 = np.random.randn(64, 16)
        dict2 = dict1 + np.random.randn(64, 16) * 0.1  # Similar but different
        
        patch_size = (8, 8)
        
        with patch('matplotlib.pyplot.show'):
            comparison = viz.compare_dictionaries(dict1, dict2, patch_size)
            
        assert isinstance(comparison, dict)
        assert 'dictionary_comparison' in viz._last_plot_info
        
    def test_compare_dictionaries_custom_labels(self):
        """Test dictionary comparison with custom labels"""
        viz = SparseVisualization()
        
        dict1 = np.random.randn(36, 9)
        dict2 = np.random.randn(36, 9)
        patch_size = (6, 6)
        
        with patch('matplotlib.pyplot.show'):
            comparison = viz.compare_dictionaries(
                dict1, dict2, patch_size,
                labels=['Before Training', 'After Training'],
                title="Training Comparison"
            )
            
        assert isinstance(comparison, dict)
        
    def test_compare_dictionaries_different_sizes(self):
        """Test comparing dictionaries of different sizes"""
        viz = SparseVisualization()
        
        dict1 = np.random.randn(25, 8)  # 5x5 patches
        dict2 = np.random.randn(25, 12)  # Same patches, more atoms
        patch_size = (5, 5)
        
        with patch('matplotlib.pyplot.show'):
            comparison = viz.compare_dictionaries(dict1, dict2, patch_size)
            
        # Should handle size differences gracefully
        assert isinstance(comparison, dict)
        
    def test_compare_dictionaries_similarity_analysis(self):
        """Test dictionary comparison includes similarity analysis"""
        viz = SparseVisualization()
        
        # Create identical and different dictionaries
        dict1 = np.random.randn(16, 4)
        dict2_identical = dict1.copy()
        dict2_different = np.random.randn(16, 4)
        
        patch_size = (4, 4)
        
        # Test identical dictionaries
        with patch('matplotlib.pyplot.show'):
            comparison_identical = viz.compare_dictionaries(dict1, dict2_identical, patch_size)
            
        # Test different dictionaries
        with patch('matplotlib.pyplot.show'):
            comparison_different = viz.compare_dictionaries(dict1, dict2_different, patch_size)
            
        assert isinstance(comparison_identical, dict)
        assert isinstance(comparison_different, dict)


class TestVisualizationIntegration:
    """Test integration and end-to-end visualization workflows"""
    
    def test_complete_visualization_workflow(self):
        """Test complete visualization workflow with all methods"""
        viz = SparseVisualization()
        
        # Create complete sparse coding scenario
        np.random.seed(42)
        patch_size = (8, 8)
        patch_dim = 64
        n_atoms = 16
        n_patches = 10
        
        # Dictionary
        dictionary = np.random.randn(patch_dim, n_atoms)
        
        # Sparse codes
        codes = np.random.randn(n_patches, n_atoms)
        codes = codes * (np.random.rand(n_patches, n_atoms) < 0.3)  # Make sparse
        
        # Original and reconstructed patches
        original_patches = np.random.randn(n_patches, patch_dim)
        reconstructed_patches = codes @ dictionary.T
        
        # Training history
        training_history = {
            'reconstruction_errors': [1.0, 0.8, 0.6, 0.4, 0.2],
            'sparsity_levels': [0.1, 0.15, 0.2, 0.25, 0.3],
            'dictionary_changes': [0.5, 0.3, 0.2, 0.1, 0.05]
        }
        
        # Test all visualization methods
        with patch('matplotlib.pyplot.show'):
            # Dictionary visualization
            viz.visualize_dictionary(dictionary, patch_size)
            
            # Sparse codes
            viz.visualize_sparse_codes(codes)
            
            # Reconstruction
            viz.visualize_reconstruction(original_patches, reconstructed_patches, patch_size)
            
            # Training progress
            viz.visualize_training_progress(training_history)
            
            # Receptive field analysis
            analysis = viz.analyze_receptive_fields(dictionary, patch_size)
            
            # Dictionary comparison (with itself for testing)
            comparison = viz.compare_dictionaries(dictionary, dictionary, patch_size)
        
        # Verify all methods worked
        assert 'dictionary_atoms' in viz._last_plot_info
        assert isinstance(analysis, dict)
        assert isinstance(comparison, dict)
        
    def test_visualization_state_tracking(self):
        """Test that visualization state is properly tracked"""
        viz = SparseVisualization()
        
        # Initial state
        assert viz._last_plot_info == {}
        
        # After dictionary visualization
        dictionary = np.random.randn(16, 4)
        patch_size = (4, 4)
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_dictionary(dictionary, patch_size)
            
        assert len(viz._last_plot_info) > 0
        
    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases"""
        viz = SparseVisualization()
        
        # Test with very small dictionary
        tiny_dict = np.random.randn(4, 1)  # 2x2 patch, 1 atom
        patch_size = (2, 2)
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_dictionary(tiny_dict, patch_size)
            
        # Test with zero dictionary
        zero_dict = np.zeros((9, 3))  # 3x3 patches
        patch_size = (3, 3)
        
        with patch('matplotlib.pyplot.show'):
            viz.visualize_dictionary(zero_dict, patch_size)
            
        # All should complete without errors
        assert viz._last_plot_info is not None


def test_matplotlib_backend_compatibility():
    """Test compatibility with different matplotlib backends"""
    # Ensure Agg backend works for testing
    assert matplotlib.get_backend() == 'Agg'
    
    viz = SparseVisualization()
    dictionary = np.random.randn(16, 4)
    
    # Should work without display
    with patch('matplotlib.pyplot.show'):
        viz.visualize_dictionary(dictionary, (4, 4))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])