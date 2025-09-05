"""
Test input validation and error handling.
"""

import numpy as np
import pytest
from sparse_coding import SparseCoder

def test_invalid_input_shapes():
    """Test validation of input array shapes."""
    coder = SparseCoder(n_atoms=32, seed=42)
    
    # 1D array should fail
    with pytest.raises(ValueError, match="X must be 2D array"):
        coder.fit(np.array([1, 2, 3, 4]))
    
    # 3D array should fail
    with pytest.raises(ValueError, match="X must be 2D array"):
        coder.fit(np.random.randn(10, 20, 5))
    
    # Empty arrays should fail
    with pytest.raises(ValueError, match="Invalid data dimensions"):
        coder.fit(np.empty((0, 10)))
    
    with pytest.raises(ValueError, match="Invalid data dimensions"):
        coder.fit(np.empty((10, 0)))

def test_non_finite_inputs():
    """Test handling of NaN and infinite values."""
    coder = SparseCoder(n_atoms=16, seed=42)
    
    # NaN in training data
    X_nan = np.random.randn(64, 100)
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="X contains non-finite values"):
        coder.fit(X_nan)
    
    # Inf in training data  
    X_inf = np.random.randn(64, 100)
    X_inf[0, 0] = np.inf
    with pytest.raises(ValueError, match="X contains non-finite values"):
        coder.fit(X_inf)
    
    # Train valid coder first
    X_valid = np.random.randn(64, 100)
    coder.fit(X_valid, n_steps=1)
    
    # NaN in encoding data
    X_test_nan = np.random.randn(64, 10)
    X_test_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="X contains non-finite values"):
        coder.encode(X_test_nan)
    
    # NaN in coefficients for decoding
    A_nan = np.random.randn(16, 10)
    A_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="A contains non-finite values"):
        coder.decode(A_nan)

def test_incompatible_dimensions():
    """Test dimension compatibility checks."""
    coder = SparseCoder(n_atoms=32, seed=42)
    
    # Fit with 64-dimensional patches
    X_train = np.random.randn(64, 100)
    coder.fit(X_train, n_steps=1)
    
    # Try to encode patches of different dimension
    X_wrong = np.random.randn(32, 50)  # Wrong patch size
    with pytest.raises(ValueError, match="X shape .* incompatible with dictionary shape"):
        coder.encode(X_wrong)
    
    # Try to decode coefficients with wrong number of atoms
    A_wrong = np.random.randn(16, 50)  # Wrong number of atoms
    with pytest.raises(ValueError, match="A shape .* incompatible with dictionary shape"):
        coder.decode(A_wrong)

def test_invalid_parameters():
    """Test validation of algorithm parameters."""
    coder = SparseCoder(n_atoms=16, seed=42)
    X = np.random.randn(64, 100)
    
    # Negative n_steps
    with pytest.raises(ValueError, match="n_steps must be positive"):
        coder.fit(X, n_steps=-1)
    
    # Zero n_steps  
    with pytest.raises(ValueError, match="n_steps must be positive"):
        coder.fit(X, n_steps=0)
    
    # Negative learning rate
    with pytest.raises(ValueError, match="lr must be positive"):
        coder.fit(X, lr=-0.1)
    
    # Zero learning rate
    with pytest.raises(ValueError, match="lr must be positive"):
        coder.fit(X, lr=0.0)

def test_encode_before_fit():
    """Test that encoding fails if dictionary not fitted."""
    coder = SparseCoder(n_atoms=32, seed=42)
    X = np.random.randn(64, 10)
    
    with pytest.raises(ValueError, match="Dictionary not initialized. Call fit"):
        coder.encode(X)

def test_decode_before_fit():
    """Test that decoding fails if dictionary not fitted."""
    coder = SparseCoder(n_atoms=32, seed=42)
    A = np.random.randn(32, 10)
    
    with pytest.raises(ValueError, match="Dictionary not initialized"):
        coder.decode(A)

def test_invalid_mode():
    """Test handling of invalid sparse coding mode."""
    coder = SparseCoder(n_atoms=32, mode='l1', seed=42)  # Start with valid mode
    X = np.random.randn(64, 100)
    coder.fit(X, n_steps=1)
    
    # Change mode to invalid after fit
    coder.mode = 'invalid_mode'
    
    # Encoding should fail with invalid mode
    with pytest.raises(ValueError, match="Unknown mode: invalid_mode"):
        coder.encode(X[:, :10])

def test_robust_to_input_types():
    """Test that API accepts different input types gracefully.""" 
    coder = SparseCoder(n_atoms=16, seed=42)
    
    # Lists should be converted to arrays
    X_list = np.random.randn(32, 50).tolist()
    X_array = np.array(X_list).T  # Make it (32, 50)
    
    coder.fit(X_array, n_steps=1)  # Should work
    
    # Integer arrays should be converted to float (same dimensions as training: p=50)
    X_int = np.random.randint(-10, 10, size=(50, 20))  # Match p=50 from training
    codes = coder.encode(X_int)  # Should work
    assert codes.dtype == float

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    # Very small dictionary
    coder_small = SparseCoder(n_atoms=1, seed=42)
    X_small = np.random.randn(4, 10)
    coder_small.fit(X_small, n_steps=1)  # Should work
    codes = coder_small.encode(X_small)
    assert codes.shape == (1, 10)
    
    # Single patch
    coder = SparseCoder(n_atoms=8, seed=42) 
    X_single = np.random.randn(16, 1)
    coder.fit(X_single, n_steps=1)
    codes = coder.encode(X_single)
    assert codes.shape == (8, 1)
    
    # Very small patches
    coder_tiny = SparseCoder(n_atoms=2, seed=42)
    X_tiny = np.random.randn(1, 20)  # 1-pixel "patches"
    coder_tiny.fit(X_tiny, n_steps=1)
    codes = coder_tiny.encode(X_tiny)
    assert codes.shape == (2, 20)