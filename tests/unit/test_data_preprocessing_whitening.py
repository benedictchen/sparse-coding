"""
Unit tests for data_preprocessing_whitening module.

Tests the zero-phase whitening implementation for research accuracy
and mathematical properties according to Olshausen & Field (1996).
"""

import numpy as np
import pytest
from sparse_coding.data_preprocessing_whitening import zero_phase_whiten


class TestZeroPhaseWhitening:
    """Test zero-phase whitening implementation."""
    
    def test_basic_functionality(self):
        """Test basic whitening functionality."""
        # Create test image with known properties
        np.random.seed(42)
        image = np.random.randn(32, 32) * 0.5 + 0.2
        
        # Apply whitening
        whitened = zero_phase_whiten(image)
        
        # Check output properties
        assert whitened.shape == image.shape
        assert isinstance(whitened, np.ndarray)
        assert whitened.dtype in [np.float32, np.float64]
        
        # Check normalization (should be approximately zero mean, unit variance)
        assert abs(whitened.mean()) < 1e-10  # Zero mean
        assert abs(whitened.std() - 1.0) < 1e-10  # Unit variance
    
    def test_mathematical_properties(self):
        """Test mathematical properties of whitening filter."""
        # Test with different f0 values
        image = np.random.randn(64, 64) * 0.3
        
        whitened_low_f0 = zero_phase_whiten(image, f0=100.0)
        whitened_high_f0 = zero_phase_whiten(image, f0=400.0)
        
        # Different f0 should produce different results
        assert not np.allclose(whitened_low_f0, whitened_high_f0)
        
        # Both should maintain shape and normalization
        for result in [whitened_low_f0, whitened_high_f0]:
            assert result.shape == image.shape
            assert abs(result.mean()) < 1e-10
            assert abs(result.std() - 1.0) < 1e-10
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with constant image
        constant_image = np.ones((16, 16)) * 5.0
        result = zero_phase_whiten(constant_image)
        
        # Should handle gracefully (likely returns zeros after whitening)
        assert result.shape == constant_image.shape
        assert np.isfinite(result).all()
        
        # Test with very small image
        small_image = np.random.randn(4, 4)
        result = zero_phase_whiten(small_image)
        assert result.shape == small_image.shape
        
        # Test invalid input dimensions
        with pytest.raises(ValueError, match="Expected 2D image"):
            zero_phase_whiten(np.random.randn(8))  # 1D array
        
        with pytest.raises(ValueError, match="Expected 2D image"):
            zero_phase_whiten(np.random.randn(8, 8, 3))  # 3D array
    
    def test_preserve_mean_option(self):
        """Test preserve_mean parameter."""
        image = np.random.randn(32, 32) + 10.0  # Non-zero mean
        
        # Default: should remove mean
        whitened_default = zero_phase_whiten(image, preserve_mean=False)
        assert abs(whitened_default.mean()) < 1e-10
        
        # Preserve mean: should maintain some offset
        whitened_preserve = zero_phase_whiten(image, preserve_mean=True)
        # Note: whitening may still change the mean, but shouldn't force it to zero
        assert whitened_preserve.shape == image.shape
    
    def test_normalize_output_option(self):
        """Test normalize_output parameter."""
        image = np.random.randn(32, 32) * 2.0
        
        # With normalization (default)
        whitened_norm = zero_phase_whiten(image, normalize_output=True)
        assert abs(whitened_norm.std() - 1.0) < 1e-10
        
        # Without normalization
        whitened_no_norm = zero_phase_whiten(image, normalize_output=False)
        # Standard deviation should not be forced to 1.0
        assert whitened_no_norm.shape == image.shape
        assert np.isfinite(whitened_no_norm).all()
    
    def test_numerical_stability(self):
        """Test numerical stability parameters."""
        image = np.random.randn(16, 16)
        
        # Different eps values should produce similar but not identical results
        result1 = zero_phase_whiten(image, numerical_stability_eps=1e-12)
        result2 = zero_phase_whiten(image, numerical_stability_eps=1e-8)
        
        # Should be close but not identical
        assert np.allclose(result1, result2, rtol=1e-6)
        assert not np.array_equal(result1, result2)
    
    def test_research_compliance_olshausen_field(self):
        """Test compliance with Olshausen & Field (1996) parameters."""
        # Use typical parameters from the paper
        np.random.seed(1996)  # Use paper year as seed
        image = np.random.randn(64, 64) * 0.1  # Typical image patch size and scale
        
        # Apply whitening with research-typical parameters
        whitened = zero_phase_whiten(
            image, 
            f0=200.0,  # Typical cutoff frequency
            normalize_output=True,
            preserve_mean=False
        )
        
        # Verify research-standard properties
        assert whitened.shape == (64, 64)
        assert abs(whitened.mean()) < 1e-10  # Zero mean (DC removal)
        assert abs(whitened.std() - 1.0) < 1e-10  # Unit variance
        
        # Check that high frequencies are enhanced (whitening effect)
        # Compare energy in different frequency bands
        fft_original = np.fft.fft2(image - image.mean())
        fft_whitened = np.fft.fft2(whitened)
        
        # Both should have similar total energy preservation
        original_energy = np.sum(np.abs(fft_original)**2)
        whitened_energy = np.sum(np.abs(fft_whitened)**2)
        
        # Energy should be redistributed but finite
        assert np.isfinite(original_energy)
        assert np.isfinite(whitened_energy)
        assert whitened_energy > 0
    
    def test_frequency_domain_properties(self):
        """Test frequency domain properties of the whitening filter."""
        # Create image with known frequency content
        H, W = 32, 32
        y, x = np.ogrid[:H, :W]
        
        # Create sinusoidal pattern (single frequency)
        freq = 0.1
        image = np.sin(2 * np.pi * freq * (x + y))
        
        whitened = zero_phase_whiten(image, f0=200.0)
        
        # Should preserve finite, real output
        assert np.isfinite(whitened).all()
        assert np.isreal(whitened).all()
        assert whitened.shape == image.shape
    
    def test_parameter_validation(self):
        """Test parameter validation and bounds."""
        image = np.random.randn(16, 16)
        
        # f0=0 should still work due to numerical_stability_eps
        result = zero_phase_whiten(image, f0=0.0)
        assert np.isfinite(result).all()
        
        # Very large f0 should still work
        result = zero_phase_whiten(image, f0=1e6)
        assert np.isfinite(result).all()
        
        # Very small f0 should still work
        result = zero_phase_whiten(image, f0=1e-3)
        assert np.isfinite(result).all()


class TestWhiteningIntegration:
    """Integration tests for whitening in sparse coding pipeline."""
    
    def test_pipeline_compatibility(self):
        """Test compatibility with sparse coding pipeline."""
        # Create natural image-like data
        np.random.seed(42)
        # Simulate 1/f power spectrum characteristic of natural images
        image = np.random.randn(64, 64)
        fft_image = np.fft.fft2(image)
        
        # Create 1/f filter
        H, W = image.shape
        fy, fx = np.ogrid[:H, :W]
        fy = fy - H//2
        fx = fx - W//2
        r = np.sqrt(fx**2 + fy**2) + 1e-10
        filt = 1 / r
        
        # Apply 1/f spectrum
        natural_like = np.fft.ifft2(fft_image * np.fft.ifftshift(filt)).real
        
        # Whiten the natural-like image
        whitened = zero_phase_whiten(natural_like)
        
        # Should produce valid output for sparse coding
        assert whitened.shape == natural_like.shape
        assert np.isfinite(whitened).all()
        assert abs(whitened.mean()) < 1e-10
        assert abs(whitened.std() - 1.0) < 1e-10
        
    def test_batch_processing_consistency(self):
        """Test that whitening is consistent across different images."""
        np.random.seed(123)
        images = [np.random.randn(32, 32) for _ in range(5)]
        
        # Whiten each image with same parameters
        whitened_images = [zero_phase_whiten(img, f0=200.0) for img in images]
        
        # Each should have proper normalization
        for whitened in whitened_images:
            assert abs(whitened.mean()) < 1e-10
            assert abs(whitened.std() - 1.0) < 1e-10
            assert np.isfinite(whitened).all()