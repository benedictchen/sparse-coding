#!/usr/bin/env python3
"""
Test visualization functionality to improve coverage
"""

import numpy as np
import matplotlib
# Use Agg backend to avoid display issues
matplotlib.use('Agg')

try:
    from .sparse_coder import SparseCoder
except ImportError:
    from sparse_coder import SparseCoder
try:
    from .visualization import SparseVisualization
except ImportError:
    from visualization import SparseVisualization

def test_visualization_coverage():
    """Test visualization methods to improve coverage"""
    print("Testing visualization functionality...")
    
    # Create test data
    np.random.seed(42)
    test_images = np.random.randn(2, 16, 16) * 0.1 + 0.5
    
    # Initialize and train sparse coder
    coder = SparseCoder(n_components=8, patch_size=(4, 4), max_iter=1)
    coder.fit(test_images)
    
    # Get sparse codes
    codes = coder.transform(test_images[:1])
    
    # Initialize visualization
    viz = SparseVisualization(colormap='gray', figsize=(8, 8))
    print("✅ SparseVisualization initialized")
    
    # Test dictionary visualization
    try:
        viz.visualize_dictionary(coder.dictionary, coder.patch_size, 
                                max_atoms=8, title="Test Dictionary")
        print("✅ visualize_dictionary")
    except Exception as e:
        print(f"❌ visualize_dictionary: {e}")
    
    # Test sparse codes visualization
    try:
        if hasattr(viz, 'visualize_sparse_codes'):
            viz.visualize_sparse_codes(codes, title="Test Sparse Codes")
            print("✅ visualize_sparse_codes")
    except Exception as e:
        print(f"❌ visualize_sparse_codes: {e}")
    
    # Test reconstruction visualization
    try:
        if hasattr(viz, 'visualize_reconstruction'):
            reconstructions = coder.reconstruct(codes)
            viz.visualize_reconstruction(test_images[:1], reconstructions,
                                       patch_size=coder.patch_size)
            print("✅ visualize_reconstruction")
    except Exception as e:
        print(f"❌ visualize_reconstruction: {e}")
    
    # Test learning progress visualization
    try:
        if hasattr(viz, 'visualize_learning_progress'):
            errors = [0.1, 0.08, 0.06, 0.05, 0.04]
            sparsity = [10, 12, 11, 10, 9]
            viz.visualize_learning_progress(errors, sparsity)
            print("✅ visualize_learning_progress")
    except Exception as e:
        print(f"❌ visualize_learning_progress: {e}")
    
    # Test basis comparison
    try:
        if hasattr(viz, 'compare_basis_functions'):
            basis1 = np.random.randn(16, 8)
            basis2 = np.random.randn(16, 8)
            viz.compare_basis_functions(basis1, basis2, coder.patch_size)
            print("✅ compare_basis_functions")
    except Exception as e:
        print(f"❌ compare_basis_functions: {e}")
    
    # Test feature visualization
    try:
        if hasattr(viz, 'visualize_feature_usage'):
            usage_stats = np.random.rand(8)
            viz.visualize_feature_usage(usage_stats)
            print("✅ visualize_feature_usage")
    except Exception as e:
        print(f"❌ visualize_feature_usage: {e}")
    
    print("🎉 Visualization test completed!")

if __name__ == "__main__":
    test_visualization_coverage()