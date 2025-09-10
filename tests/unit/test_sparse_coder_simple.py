"""
Simplified unit tests for SparseCoder class that match the actual implementation.
"""

import numpy as np
import pytest
from sparse_coding import SparseCoder
from tests.conftest import assert_dictionary_normalized


class TestSparseCoderBasic:
    """Test basic SparseCoder functionality."""
    
    def test_initialization(self):
        """Test SparseCoder initialization."""
        coder = SparseCoder(n_atoms=64, mode="l1", seed=42)
        
        assert coder.n_atoms == 64
        assert coder.mode == "l1"
        assert coder.seed == 42
        assert coder.D is None  # Not initialized yet
    
    def test_fit_and_encode_l1_mode(self, synthetic_data):
        """Test fit and encode with L1 mode."""
        data = synthetic_data
        X = data['signals']
        
        coder = SparseCoder(n_atoms=data['n_components'], mode="l1", seed=42)
        coder.fit(X)
        
        # Check dictionary is initialized and normalized
        assert coder.D is not None
        assert coder.D.shape == (data['n_features'], data['n_components'])
        assert_dictionary_normalized(coder.D)
        
        # Test encoding
        A = coder.encode(X)
        assert A.shape == (data['n_components'], data['n_samples'])
        assert np.all(np.isfinite(A))
    
    def test_fit_and_encode_paper_mode(self, synthetic_data):
        """Test fit and encode with paper mode."""
        data = synthetic_data
        X = data['signals']  # Use full dataset
        
        # Use fewer atoms than samples to avoid sampling error
        n_atoms = min(data['n_components'], data['n_samples'] - 5)
        coder = SparseCoder(n_atoms=n_atoms, mode="paper", seed=42)
        coder.fit(X, n_steps=3)  # Fewer steps for speed
        
        # Check dictionary
        assert coder.D is not None
        assert coder.D.shape == (data['n_features'], n_atoms)
        assert_dictionary_normalized(coder.D)
        
        # Test encoding  
        A = coder.encode(X)
        assert A.shape == (n_atoms, X.shape[1])
        assert np.all(np.isfinite(A))
    
    def test_decode_functionality(self, synthetic_data):
        """Test decode functionality."""
        data = synthetic_data
        X = data['signals'][:, :3]
        
        coder = SparseCoder(n_atoms=data['n_components'], mode="l1", seed=42)
        coder.fit(X)
        
        A = coder.encode(X)
        X_reconstructed = coder.decode(A)
        
        # Basic sanity checks
        assert X_reconstructed.shape == X.shape
        assert np.all(np.isfinite(X_reconstructed))
        
        # Should achieve reasonable reconstruction
        mse = np.mean((X - X_reconstructed)**2)
        signal_power = np.mean(X**2)
        relative_mse = mse / max(signal_power, 1e-12)
        
        assert relative_mse < 1.0, f"Reconstruction error too high: {relative_mse:.3f}"
    
    def test_invalid_mode_error(self):
        """Test that invalid mode raises error during operation."""
        X = np.random.randn(20, 10)
        coder = SparseCoder(n_atoms=8, mode="invalid")
        
        with pytest.raises(ValueError, match="mode must be"):
            coder.fit(X)
    
    def test_encode_without_fit_error(self):
        """Test that encoding without fitting raises error."""
        X = np.random.randn(20, 10)
        coder = SparseCoder(n_atoms=8)
        
        with pytest.raises(AssertionError, match="Dictionary not initialized"):
            coder.encode(X)
    
    def test_auto_lambda_selection(self, synthetic_data):
        """Test automatic lambda selection."""
        data = synthetic_data
        X = data['signals']
        
        # L1 mode with auto lambda
        coder_l1 = SparseCoder(n_atoms=data['n_components'], mode="l1", lam=None, seed=42)
        coder_l1.fit(X)
        
        # Should determine reasonable lambda during fit
        A_l1 = coder_l1.encode(X)
        assert np.all(np.isfinite(A_l1))
        
        # Paper mode with auto lambda  
        coder_paper = SparseCoder(n_atoms=data['n_components'], mode="paper", lam=None, seed=42)
        coder_paper.fit(X[:, :5], n_steps=2)  # Smaller/faster
        
        A_paper = coder_paper.encode(X[:, :5])
        assert np.all(np.isfinite(A_paper))
    
    def test_reproducibility_with_seed(self, synthetic_data):
        """Test that results are reproducible with same seed."""
        data = synthetic_data
        X = data['signals'][:, :5]
        
        # Two coders with same seed
        coder1 = SparseCoder(n_atoms=data['n_components'], mode="l1", seed=42)
        coder2 = SparseCoder(n_atoms=data['n_components'], mode="l1", seed=42)
        
        coder1.fit(X)
        coder2.fit(X)
        
        A1 = coder1.encode(X)
        A2 = coder2.encode(X)
        
        # Should be very close (allowing for numerical precision)
        np.testing.assert_allclose(A1, A2, rtol=1e-10, atol=1e-12)
    
    def test_different_input_shapes(self, synthetic_data):
        """Test handling of different input shapes."""
        data = synthetic_data
        
        coder = SparseCoder(n_atoms=data['n_components'], mode="l1", seed=42)
        coder.fit(data['signals'])
        
        # Single signal
        X_single = data['signals'][:, 0:1]
        A_single = coder.encode(X_single)
        assert A_single.shape == (data['n_components'], 1)
        
        # Multiple signals
        X_multi = data['signals'][:, :5] 
        A_multi = coder.encode(X_multi)
        assert A_multi.shape == (data['n_components'], 5)
        
        # First signal should match
        np.testing.assert_allclose(A_single[:, 0], A_multi[:, 0], rtol=1e-10)