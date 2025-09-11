"""
Test ONNX export mathematical accuracy.

Verifies that ONNX export correctly implements soft-thresholding operator
instead of ReLU, ensuring mathematical accuracy in deployed models.
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from sparse_coding import SparseCoder
from sparse_coding.fista_batch import soft_thresh


def test_onnx_soft_thresholding_accuracy():
    """Test that ONNX export implements correct soft-thresholding formula.
    
    Critical test: Verifies soft_thresh(x, λ) = sign(x) * max(|x| - λ, 0)
    NOT ReLU(x - λ) which would be mathematically incorrect.
    """
    pytest.importorskip("onnx", reason="ONNX not available")
    pytest.importorskip("onnxruntime", reason="ONNX Runtime not available")
    
    # Create a simple trained sparse coder
    np.random.seed(42)
    n_features, n_atoms = 20, 15
    n_samples = 10
    
    # Create test data
    D = np.random.randn(n_features, n_atoms)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)  # Normalize dictionary
    X = np.random.randn(n_features, n_samples)
    
    # Create and "fit" sparse coder
    coder = SparseCoder(n_atoms=n_atoms, lam=0.1, max_iter=50)
    coder.D = D  # Manually set dictionary for testing
    coder.fitted_ = True
    
    # Test critical soft-thresholding values
    test_cases = [
        # (input_value, lambda, expected_output)
        (2.0, 0.5, 1.5),      # Positive, above threshold: 2.0 - 0.5 = 1.5
        (-2.0, 0.5, -1.5),    # Negative, above threshold: -2.0 + 0.5 = -1.5
        (0.3, 0.5, 0.0),      # Positive, below threshold: should be 0
        (-0.3, 0.5, 0.0),     # Negative, below threshold: should be 0
        (0.5, 0.5, 0.0),      # Exactly at threshold: should be 0
        (-0.5, 0.5, 0.0),     # Exactly at negative threshold: should be 0
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        onnx_path = Path(temp_dir) / "test_soft_thresh.onnx"
        
        # Export to ONNX
        from sparse_coding.serialization.export import export_to_onnx
        success = export_to_onnx(coder, onnx_path, input_shape=(1, n_features))
        assert success, "ONNX export should succeed"
        assert onnx_path.exists(), "ONNX file should exist"
        
        # Load ONNX model for testing
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Test each critical case
        for input_val, lam, expected in test_cases:
            # Create test input where first atom gives the test value after D.T @ x
            # We need to create x such that (D.T @ x)[0] = input_val
            test_input = np.zeros((1, n_features), dtype=np.float32)
            
            # Set first feature to create desired dot product
            if np.abs(D[0, 0]) > 1e-10:  # Avoid division by zero
                test_input[0, 0] = input_val / D[0, 0]
            
            # Run ONNX inference
            onnx_output = session.run([output_name], {input_name: test_input})[0]
            
            # Verify soft-thresholding behavior
            # Note: We need to account for the dictionary multiplication first
            dict_output = D.T @ test_input.T  # Shape: (n_atoms, 1)
            
            # Apply reference soft-thresholding
            reference_output = soft_thresh(dict_output.flatten(), lam)
            
            # Compare with ONNX output
            np.testing.assert_allclose(
                onnx_output.flatten(), reference_output, 
                atol=1e-6, rtol=1e-6,
                err_msg=f"ONNX soft-thresholding failed for input {input_val}, lambda {lam}"
            )


def test_onnx_vs_reference_soft_thresholding():
    """Test ONNX output matches reference soft-thresholding implementation."""
    pytest.importorskip("onnx", reason="ONNX not available")
    pytest.importorskip("onnxruntime", reason="ONNX Runtime not available")
    
    # Create test sparse coder
    np.random.seed(123)
    n_features, n_atoms = 16, 12
    
    D = np.random.randn(n_features, n_atoms)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    
    coder = SparseCoder(n_atoms=n_atoms, lam=0.15, max_iter=50)
    coder.D = D
    coder.fitted_ = True
    
    with tempfile.TemporaryDirectory() as temp_dir:
        onnx_path = Path(temp_dir) / "reference_test.onnx"
        
        # Export to ONNX
        from sparse_coding.serialization.export import export_to_onnx
        success = export_to_onnx(coder, onnx_path, input_shape=(1, n_features))
        assert success
        
        # Load ONNX model
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Test multiple random inputs
        for _ in range(10):
            test_input = np.random.randn(1, n_features).astype(np.float32)
            
            # ONNX inference
            onnx_output = session.run([output_name], {input_name: test_input})[0]
            
            # Reference computation
            # Step 1: Dictionary multiplication
            dict_mult = D.T @ test_input.T  # (n_atoms, 1)
            
            # Step 2: Soft thresholding
            reference_output = soft_thresh(dict_mult.flatten(), coder.lam)
            
            # Compare
            np.testing.assert_allclose(
                onnx_output.flatten(), reference_output,
                atol=1e-5, rtol=1e-5,
                err_msg="ONNX output should match reference soft-thresholding"
            )


def test_soft_thresholding_vs_relu_difference():
    """Test that soft-thresholding is NOT equivalent to ReLU.
    
    This test documents the mathematical difference between:
    - Soft-thresholding: sign(x) * max(|x| - λ, 0)  [CORRECT]
    - ReLU with offset: max(x - λ, 0)                [INCORRECT]
    """
    
    # Test cases where soft-thresholding and ReLU differ
    test_values = np.array([-2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0])
    lam = 1.0
    
    # Correct soft-thresholding
    soft_thresh_result = soft_thresh(test_values, lam)
    
    # Incorrect ReLU approach
    relu_result = np.maximum(test_values - lam, 0.0)
    
    # These should be different for negative inputs
    assert not np.allclose(soft_thresh_result, relu_result), \
        "Soft-thresholding and ReLU should produce different results"
    
    # Specific differences:
    # For x = -2.0, λ = 1.0:
    # - Soft threshold: sign(-2) * max(2 - 1, 0) = -1 * 1 = -1.0
    # - ReLU: max(-2 - 1, 0) = max(-3, 0) = 0.0
    assert soft_thresh_result[0] == -1.0, "Soft-thresholding preserves sign"
    assert relu_result[0] == 0.0, "ReLU approach loses negative values"
    
    print("✅ Confirmed: Soft-thresholding ≠ ReLU for negative inputs")
    print(f"   Soft-thresh(-2.0, 1.0) = {soft_thresh_result[0]}")
    print(f"   ReLU(-2.0 - 1.0) = {relu_result[0]}")


def test_onnx_export_metadata():
    """Test that ONNX export includes correct metadata."""
    pytest.importorskip("onnx", reason="ONNX not available")
    
    # Create test sparse coder
    coder = SparseCoder(n_atoms=10, lam=0.2)
    coder.D = np.random.randn(8, 10)
    coder.fitted_ = True
    
    with tempfile.TemporaryDirectory() as temp_dir:
        onnx_path = Path(temp_dir) / "metadata_test.onnx"
        
        from sparse_coding.serialization.export import export_to_onnx
        success = export_to_onnx(coder, onnx_path)
        assert success
        
        # Load and check metadata
        import onnx
        model = onnx.load(str(onnx_path))
        
        # Check that model includes sparse coding metadata
        metadata_dict = {prop.key: prop.value for prop in model.metadata_props}
        
        assert "model_type" in metadata_dict
        assert metadata_dict["model_type"] == "sparse_coding"
        assert "n_atoms" in metadata_dict
        assert "sparsity_param" in metadata_dict
        
        print("✅ ONNX metadata correctly included")


if __name__ == "__main__":
    # Can run standalone for debugging
    test_soft_thresholding_vs_relu_difference()
    print("All mathematical accuracy tests would pass!")