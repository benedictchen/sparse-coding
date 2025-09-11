"""
Quantitative visualization correctness tests for sparse coding.

These tests ensure visualization functions produce mathematically correct outputs
and handle edge cases properly. Critical for research reproducibility where
visualization errors can lead to misinterpretation of results.

References:
    Cleveland & McGill (1984). Graphical Perception: Theory, Experimentation, and Application.
    Tufte (2001). The Visual Display of Quantitative Information.
    Hunter (2007). Matplotlib: A 2D graphics environment.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import io
import warnings

from sparse_coding.sparse_coding_visualization import (
    plot_dictionary_atoms, plot_training_progress, plot_sparse_codes,
    plot_reconstruction_quality, plot_sparsity_statistics, create_visualization_report
)
from sparse_coding import SparseCoder


class TestVisualizationCorrectness:
    """Test mathematical correctness of visualization functions."""
    
    def setup_method(self):
        """Set up test data with known mathematical properties."""
        np.random.seed(42)
        
        # Create test dictionary with known properties
        self.patch_size = (8, 8)
        self.n_atoms = 16
        self.n_patches = 32
        
        # Dictionary: normalized atoms with known structure
        self.dictionary = np.random.randn(64, self.n_atoms)
        self.dictionary = self.dictionary / np.linalg.norm(self.dictionary, axis=0, keepdims=True)
        
        # Sparse codes with controlled sparsity
        self.sparse_codes = np.zeros((self.n_atoms, self.n_patches))
        # Ensure exactly 3 non-zero coefficients per patch
        for i in range(self.n_patches):
            active_indices = np.random.choice(self.n_atoms, 3, replace=False)
            self.sparse_codes[active_indices, i] = np.random.randn(3)
        
        # Original patches
        self.original_patches = np.random.randn(64, self.n_patches) * 0.1
        
        # Training history with known properties
        self.training_history = {
            'reconstruction_errors': [1.0, 0.5, 0.25, 0.125, 0.0625],  # Exponential decay
            'sparsity_levels': [0.1, 0.15, 0.18, 0.19, 0.2],  # Monotonic increase
            'dictionary_changes': [10.0, 5.0, 2.5, 1.25, 0.625]  # Exponential decay
        }
    
    def test_plot_dictionary_atoms_mathematical_correctness(self):
        """Test dictionary atom plotting produces mathematically correct normalization."""
        fig = plot_dictionary_atoms(self.dictionary, self.patch_size, n_show=16)
        
        # Verify figure structure
        assert len(fig.axes) == 16, "Should create 4x4 grid for 16 atoms"
        
        # Test normalization correctness for each displayed atom
        atoms = self.dictionary.T.reshape(-1, *self.patch_size)
        for i, ax in enumerate(fig.axes[:min(16, self.n_atoms)]):
            # Get the image data from the plot
            image_data = ax.images[0].get_array()
            original_atom = atoms[i]
            
            # Verify normalization formula: (x - min) / (max - min + eps)
            expected_min = original_atom.min()
            expected_max = original_atom.max()
            expected_normalized = (original_atom - expected_min) / (expected_max - expected_min + 1e-8)
            
            np.testing.assert_allclose(image_data, expected_normalized, rtol=1e-10,
                                     err_msg=f"Atom {i} normalization incorrect")
        
        # Verify colormap is correctly applied
        for ax in fig.axes[:self.n_atoms]:
            assert ax.images[0].get_cmap().name == 'RdBu_r', "Should use RdBu_r colormap"
        
        plt.close(fig)
    
    def test_plot_training_progress_curve_accuracy(self):
        """Test training progress plots show mathematically correct curves."""
        fig = plot_training_progress(self.training_history)
        
        # Should have exactly 3 subplots
        assert len(fig.axes) == 3, "Should create 3 subplots for 3 metrics"
        
        # Test reconstruction error plot (log scale)
        recon_ax = fig.axes[0]
        recon_line = recon_ax.get_lines()[0]
        plotted_y = recon_line.get_ydata()
        expected_y = np.array(self.training_history['reconstruction_errors'])
        
        np.testing.assert_allclose(plotted_y, expected_y, rtol=1e-12,
                                 err_msg="Reconstruction error data incorrect")
        assert recon_ax.get_yscale() == 'log', "Reconstruction error should use log scale"
        
        # Test sparsity level plot (linear scale)
        sparsity_ax = fig.axes[1]
        sparsity_line = sparsity_ax.get_lines()[0]
        plotted_y = sparsity_line.get_ydata()
        expected_y = np.array(self.training_history['sparsity_levels'])
        
        np.testing.assert_allclose(plotted_y, expected_y, rtol=1e-12,
                                 err_msg="Sparsity level data incorrect")
        assert sparsity_ax.get_yscale() == 'linear', "Sparsity should use linear scale"
        
        # Test dictionary changes plot (log scale)
        dict_ax = fig.axes[2]
        dict_line = dict_ax.get_lines()[0]
        plotted_y = dict_line.get_ydata()
        expected_y = np.array(self.training_history['dictionary_changes'])
        
        np.testing.assert_allclose(plotted_y, expected_y, rtol=1e-12,
                                 err_msg="Dictionary changes data incorrect")
        assert dict_ax.get_yscale() == 'log', "Dictionary changes should use log scale"
        
        plt.close(fig)
    
    def test_plot_sparse_codes_sparsity_calculation_accuracy(self):
        """Test sparse codes plotting calculates sparsity correctly."""
        fig = plot_sparse_codes(self.sparse_codes, n_show=8)
        
        # Verify correct number of visible subplots for codes
        visible_subplots = len([ax for ax in fig.axes if ax.get_visible()])
        assert visible_subplots == 8, f"Should create 8 visible subplots for 8 codes, got {visible_subplots}"
        
        # Test sparsity calculation accuracy for each subplot
        threshold = 1e-6
        for i, ax in enumerate(fig.axes):
            if not ax.get_visible():  # Skip hidden subplots
                continue
                
            # Extract title to get sparsity value
            title = ax.get_title()
            sparsity_str = title.split('sparsity: ')[1].rstrip(')')
            plotted_sparsity = float(sparsity_str)
            
            # Calculate expected sparsity
            code_vector = self.sparse_codes[:, i]
            calculated_sparsity = np.mean(np.abs(code_vector) > threshold)
            
            # The displayed sparsity is rounded to 2 decimal places, so we need to check
            # that the displayed value matches the rounded calculation
            expected_displayed = round(calculated_sparsity, 2)
            
            np.testing.assert_allclose(plotted_sparsity, expected_displayed, rtol=1e-10,
                                     err_msg=f"Displayed sparsity incorrect for code {i}: got {plotted_sparsity}, expected {expected_displayed}")
            
            # Verify bar heights match coefficient values
            bars = ax.patches
            bar_heights = [bar.get_height() for bar in bars]
            np.testing.assert_allclose(bar_heights, code_vector, rtol=1e-12,
                                     err_msg=f"Bar heights incorrect for code {i}")
        
        plt.close(fig)
    
    def test_plot_reconstruction_quality_error_calculation(self):
        """Test reconstruction quality plot calculates MSE correctly."""
        reconstructed = self.dictionary @ self.sparse_codes
        fig = plot_reconstruction_quality(self.original_patches, reconstructed, n_show=4)
        
        # Should have 3 rows (original, reconstruction, error) x 4 columns
        assert len(fig.axes) == 12, "Should create 3x4 grid (12 subplots)"
        
        # Test MSE calculation accuracy
        patch_dim = self.original_patches.shape[0]
        patch_size = (int(np.sqrt(patch_dim)), int(np.sqrt(patch_dim)))
        
        orig_patches = self.original_patches.T.reshape(-1, *patch_size)
        recon_patches = reconstructed.T.reshape(-1, *patch_size)
        
        # Check error subplot titles (row 2, indices 8-11)
        for i in range(4):
            error_ax = fig.axes[8 + i]  # Third row
            title = error_ax.get_title()
            
            # Extract MSE from title
            mse_str = title.split('MSE: ')[1].rstrip(')')
            plotted_mse = float(mse_str)
            
            # Calculate expected MSE
            error = np.abs(orig_patches[i] - recon_patches[i])
            expected_mse = np.mean(error**2)
            
            np.testing.assert_allclose(plotted_mse, expected_mse, rtol=1e-6,
                                     err_msg=f"MSE calculation incorrect for patch {i}")
            
            # Verify error image data matches calculated error
            error_image = error_ax.images[0].get_array()
            np.testing.assert_allclose(error_image, error, rtol=1e-12,
                                     err_msg=f"Error image data incorrect for patch {i}")
        
        plt.close(fig)
    
    def test_plot_sparsity_statistics_quantitative_accuracy(self):
        """Test sparsity statistics plots calculate metrics correctly."""
        fig = plot_sparsity_statistics(self.sparse_codes)
        
        # Should have 2x2 grid (4 subplots)
        assert len(fig.axes) == 4, "Should create 2x2 grid (4 subplots)"
        
        threshold = 1e-6
        
        # Test usage frequency calculation (subplot 2, bottom-left)
        usage_ax = fig.axes[2]
        bars = usage_ax.patches
        plotted_usage = [bar.get_height() for bar in bars]
        
        # Calculate expected usage frequency
        expected_usage = np.mean(np.abs(self.sparse_codes) > threshold, axis=1)
        
        np.testing.assert_allclose(plotted_usage, expected_usage, rtol=1e-12,
                                 err_msg="Usage frequency calculation incorrect")
        
        # Test sparsity per patch histogram (subplot 1, top-right)
        sparsity_ax = fig.axes[1]
        expected_sparsity_per_patch = np.mean(np.abs(self.sparse_codes) > threshold, axis=0)
        
        # Get histogram data
        n, bins, patches = sparsity_ax.hist(expected_sparsity_per_patch, bins=30, alpha=0.0)
        
        # Verify our known sparsity: exactly 3/16 = 0.1875 for all patches
        expected_sparsity_value = 3.0 / self.n_atoms
        np.testing.assert_allclose(expected_sparsity_per_patch, expected_sparsity_value, rtol=1e-12,
                                 err_msg="Sparsity per patch should be exactly 3/16")
        
        # Test scatter plot correlation (subplot 3, bottom-right)
        scatter_ax = fig.axes[3]
        scatter_data = scatter_ax.collections[0].get_offsets()
        
        expected_mean_magnitude = np.mean(np.abs(self.sparse_codes), axis=1)
        plotted_x = scatter_data[:, 0]  # usage frequency
        plotted_y = scatter_data[:, 1]  # mean magnitude
        
        np.testing.assert_allclose(plotted_x, expected_usage, rtol=1e-12,
                                 err_msg="Scatter plot x-axis (usage) incorrect")
        np.testing.assert_allclose(plotted_y, expected_mean_magnitude, rtol=1e-12,
                                 err_msg="Scatter plot y-axis (magnitude) incorrect")
        
        plt.close(fig)
    
    def test_create_visualization_report_comprehensive_correctness(self):
        """Test complete visualization report creates all expected components."""
        # Test with all optional parameters
        figures = create_visualization_report(
            dictionary=self.dictionary,
            sparse_codes=self.sparse_codes,
            signals=self.original_patches,
            training_history=self.training_history,
            patch_size=self.patch_size
        )
        
        # Should create 5 figures (dictionary, progress, codes, stats, reconstruction)
        assert len(figures) == 5, "Should create 5 figures with all components"
        
        # Test parameter validation
        with pytest.raises(ValueError, match="Either 'codes' or 'sparse_codes' parameter is required"):
            create_visualization_report(
                dictionary=self.dictionary,
                patch_size=self.patch_size
            )
        
        with pytest.raises(ValueError, match="patch_size parameter is required"):
            create_visualization_report(
                dictionary=self.dictionary,
                sparse_codes=self.sparse_codes
            )
        
        # Test backward compatibility with legacy parameter names
        figures_legacy = create_visualization_report(
            dictionary=self.dictionary,
            codes=self.sparse_codes,  # Legacy name
            history=self.training_history,  # Legacy name
            original_patches=self.original_patches,  # Legacy name
            patch_size=self.patch_size
        )
        
        assert len(figures_legacy) == 5, "Legacy parameter names should work"
        
        # Clean up
        for fig in figures + figures_legacy:
            plt.close(fig)
    
    def test_visualization_handles_edge_cases_gracefully(self):
        """Test visualization functions handle edge cases without errors."""
        
        # Test with single atom dictionary
        single_atom_dict = self.dictionary[:, :1]
        fig1 = plot_dictionary_atoms(single_atom_dict, self.patch_size, n_show=1)
        assert len(fig1.axes) == 1, "Should handle single atom correctly"
        plt.close(fig1)
        
        # Test with empty training history
        fig2 = plot_training_progress({})
        assert len(fig2.axes) == 3, "Should create subplots even with empty history"
        plt.close(fig2)
        
        # Test with all-zero sparse codes
        zero_codes = np.zeros_like(self.sparse_codes)
        fig3 = plot_sparse_codes(zero_codes, n_show=4)
        
        # All sparsity values should be 0.0
        for ax in fig3.axes:
            title = ax.get_title()
            sparsity_str = title.split('sparsity: ')[1].rstrip(')')
            sparsity_value = float(sparsity_str)
            assert sparsity_value == 0.0, "All-zero codes should have 0.0 sparsity"
        
        plt.close(fig3)
        
        # Test with identical original and reconstructed (perfect reconstruction)
        fig4 = plot_reconstruction_quality(self.original_patches, self.original_patches, n_show=2)
        
        # All MSE values should be 0.0
        for i in range(2):
            error_ax = fig4.axes[4 + i]  # Third row
            title = error_ax.get_title()
            mse_str = title.split('MSE: ')[1].rstrip(')')
            mse_value = float(mse_str)
            assert mse_value < 1e-10, f"Perfect reconstruction should have MSE ≈ 0, got {mse_value}"
        
        plt.close(fig4)
    
    def test_visualization_numerical_stability(self):
        """Test visualization functions maintain numerical stability."""
        
        # Test with extreme values
        extreme_dict = np.array([[1e10, -1e10], [-1e-10, 1e-10]])
        fig1 = plot_dictionary_atoms(extreme_dict, (2, 1), n_show=2)
        
        # Verify normalization doesn't produce NaN or inf
        for ax in fig1.axes[:2]:
            image_data = ax.images[0].get_array()
            assert np.all(np.isfinite(image_data)), "Extreme values should not produce NaN/inf"
            assert np.min(image_data) >= 0.0, "Normalized values should be non-negative"
            assert np.max(image_data) <= 1.0, "Normalized values should not exceed 1.0"
        
        plt.close(fig1)
        
        # Test with very sparse codes (single non-zero per column)
        ultra_sparse = np.zeros((10, 5))
        ultra_sparse[0, 0] = 1.0
        ultra_sparse[5, 1] = -2.0
        
        fig2 = plot_sparsity_statistics(ultra_sparse)
        
        # Usage frequency should be exactly correct
        usage_ax = fig2.axes[2]
        bars = usage_ax.patches
        plotted_usage = [bar.get_height() for bar in bars]
        
        expected_usage = [0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0]  # 1/5 for indices 0,5
        np.testing.assert_allclose(plotted_usage, expected_usage, rtol=1e-12,
                                 err_msg="Ultra-sparse usage frequency incorrect")
        
        plt.close(fig2)
    
    def test_visualization_integration_with_sparse_coder(self):
        """Test visualization functions work correctly with real SparseCoder output."""
        # Create a small sparse coding problem
        np.random.seed(42)
        data = np.random.randn(25, 50) * 0.1  # 5x5 patches, 50 samples
        
        coder = SparseCoder(n_atoms=16, mode='l1', lam=0.1, seed=42)
        coder.fit(data, n_steps=3, verbose=False)
        
        # Get sparse codes
        codes = coder.encode(data[:, :10])
        
        # Test dictionary visualization
        fig1 = plot_dictionary_atoms(coder.D, (5, 5), n_show=16)
        assert len(fig1.axes) == 16, "Should visualize all 16 dictionary atoms"
        
        # Verify dictionary atoms are normalized
        for i, ax in enumerate(fig1.axes):
            image_data = ax.images[0].get_array()
            assert 0.0 <= np.min(image_data) <= np.max(image_data) <= 1.0, f"Atom {i} not properly normalized"
        
        plt.close(fig1)
        
        # Test sparse codes visualization
        fig2 = plot_sparse_codes(codes, n_show=8)
        
        # Verify sparsity calculations are reasonable
        for i, ax in enumerate(fig2.axes):
            title = ax.get_title()
            sparsity_str = title.split('sparsity: ')[1].rstrip(')')
            sparsity_value = float(sparsity_str)
            assert 0.0 <= sparsity_value <= 1.0, f"Code {i} sparsity {sparsity_value} outside [0,1]"
        
        plt.close(fig2)
        
        # Test reconstruction quality
        reconstructed = coder.D @ codes
        fig3 = plot_reconstruction_quality(data[:, :4], reconstructed[:, :4], n_show=4)
        
        # Verify MSE calculations are reasonable
        for i in range(4):
            error_ax = fig3.axes[8 + i]
            title = error_ax.get_title()
            mse_str = title.split('MSE: ')[1].rstrip(')')
            mse_value = float(mse_str)
            assert mse_value >= 0.0, f"MSE {mse_value} should be non-negative"
            assert mse_value < 10.0, f"MSE {mse_value} suspiciously high"
        
        plt.close(fig3)


class TestVisualizationRobustness:
    """Test visualization robustness against various input conditions."""
    
    def test_memory_efficiency_large_datasets(self):
        """Test visualization functions don't consume excessive memory with large inputs."""
        # Create moderately large dataset
        large_dict = np.random.randn(1024, 256)  # 32x32 patches, 256 atoms
        large_dict = large_dict / np.linalg.norm(large_dict, axis=0, keepdims=True)
        
        # Test memory-efficient visualization (showing subset)
        fig = plot_dictionary_atoms(large_dict, (32, 32), n_show=64)  # Show only 64/256
        
        # Should create exactly 64 subplots, not 256
        assert len(fig.axes) == 64, "Should limit display to requested number"
        
        plt.close(fig)
    
    def test_matplotlib_backend_compatibility(self):
        """Test visualization works with different matplotlib backends."""
        # Save current backend
        original_backend = plt.get_backend()
        
        try:
            # Test with Agg backend (non-interactive)
            plt.switch_backend('Agg')
            
            dict_test = np.random.randn(16, 4)
            dict_test = dict_test / np.linalg.norm(dict_test, axis=0, keepdims=True)
            
            fig = plot_dictionary_atoms(dict_test, (4, 4), n_show=4)
            assert fig is not None, "Should work with Agg backend"
            plt.close(fig)
            
        finally:
            # Restore original backend
            plt.switch_backend(original_backend)
    
    def test_figure_cleanup_memory_management(self):
        """Test proper memory cleanup when closing figures."""
        # Test that creating and closing figures doesn't accumulate memory
        figures = []
        
        for i in range(5):
            dict_data = np.random.randn(25, 8)
            dict_data = dict_data / np.linalg.norm(dict_data, axis=0, keepdims=True)
            
            fig = plot_dictionary_atoms(dict_data, (5, 5), n_show=8)
            figures.append(fig)
        
        # Close all figures
        for fig in figures:
            plt.close(fig)
        
        # Test should complete without memory issues
        assert len(figures) == 5, "Should create 5 figures successfully"


class TestVisualizationPerformance:
    """Test visualization performance characteristics."""
    
    def test_rendering_time_reasonable(self):
        """Test visualization functions complete within reasonable time."""
        import time
        
        # Create moderately complex visualization task
        dict_data = np.random.randn(256, 64)  # 16x16 patches, 64 atoms
        dict_data = dict_data / np.linalg.norm(dict_data, axis=0, keepdims=True)
        
        start_time = time.time()
        fig = plot_dictionary_atoms(dict_data, (16, 16), n_show=64)
        end_time = time.time()
        
        rendering_time = end_time - start_time
        assert rendering_time < 5.0, f"Rendering took {rendering_time:.2f}s, should be < 5s"
        
        plt.close(fig)
    
    def test_memory_usage_bounded(self):
        """Test visualization doesn't cause memory leaks."""
        import gc
        
        # Get initial memory usage (approximate)
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and close multiple figures
        for i in range(10):
            dict_data = np.random.randn(64, 16)
            dict_data = dict_data / np.linalg.norm(dict_data, axis=0, keepdims=True)
            
            fig = plot_dictionary_atoms(dict_data, (8, 8), n_show=16)
            plt.close(fig)
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count shouldn't grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Memory leak detected: {object_growth} new objects"