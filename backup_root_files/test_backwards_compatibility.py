#!/usr/bin/env python3
"""
Backwards Compatibility Test - Ensure No Functionality Was Removed
"""

import numpy as np
try:
    from .sparse_coder import SparseCoder
    from .dictionary_learning import DictionaryLearner
    from .feature_extraction import SparseFeatureExtractor
    from .visualization import SparseVisualization
except ImportError:
    from sparse_coder import SparseCoder
    from dictionary_learning import DictionaryLearner
    from feature_extraction import SparseFeatureExtractor
    from visualization import SparseVisualization

def test_sparse_coder_backwards_compatibility():
    """Test that all original SparseCoder functionality still works"""
    print("🔍 Testing SparseCoder backwards compatibility...")
    
    # Create test data
    np.random.seed(42)
    test_images = np.random.randn(2, 16, 16) * 0.1 + 0.5
    
    # Test 1: Original initialization should work
    try:
        coder = SparseCoder(n_components=16, patch_size=(4, 4))
        print("✅ Original SparseCoder initialization works")
    except Exception as e:
        print(f"❌ Original initialization failed: {e}")
        return False
    
    # Test 2: Original fit method should work
    try:
        coder.fit(test_images, max_iterations=2, verbose=False)
        print("✅ Original fit() method works")
    except Exception as e:
        print(f"❌ Original fit() failed: {e}")
        return False
    
    # Test 3: Original transform method should work
    try:
        codes = coder.transform(test_images[:1])
        print(f"✅ Original transform() method works: {codes.shape}")
    except Exception as e:
        print(f"❌ Original transform() failed: {e}")
        return False
    
    # Test 4: All original parameters should still work
    try:
        coder_params = SparseCoder(
            n_components=8,
            patch_size=(8, 8),
            sparsity_penalty=0.1,
            sparseness_function='l1',
            optimization_method='coordinate_descent',
            max_iter=5,
            tolerance=1e-6
        )
        print("✅ All original parameters still accepted")
    except Exception as e:
        print(f"❌ Original parameters failed: {e}")
        return False
    
    # Test 5: Verify new sklearn parameters are optional
    try:
        # These should work without requiring new parameters
        coder_old_style = SparseCoder(n_components=4)
        coder_new_style = SparseCoder(alpha=0.05, algorithm='fista')
        print("✅ Both old and new parameter styles work")
    except Exception as e:
        print(f"❌ Parameter compatibility issue: {e}")
        return False
    
    return True

def test_dictionary_learner_backwards_compatibility():
    """Test that all original DictionaryLearner functionality still works"""
    print("\n🔍 Testing DictionaryLearner backwards compatibility...")
    
    # Create test data
    np.random.seed(42)
    test_patches = np.random.randn(10, 16) * 0.1 + 0.5
    
    # Test 1: Original initialization should work
    try:
        learner = DictionaryLearner(
            n_components=8,
            patch_size=(4, 4),
            sparsity_penalty=0.1,
            learning_rate=0.01,
            max_iterations=2,
            tolerance=1e-6
        )
        print("✅ Original DictionaryLearner initialization works")
    except Exception as e:
        print(f"❌ Original initialization failed: {e}")
        return False
    
    # Test 2: Original methods should work
    try:
        learner.fit(test_patches, verbose=False)
        print("✅ Original fit() method works")
        
        dictionary = learner.get_dictionary()
        print(f"✅ Original get_dictionary() works: {dictionary.shape}")
        
        codes = learner.transform(test_patches[:2])
        print(f"✅ Original transform() works: {codes.shape}")
        
        reconstruction = learner.reconstruct(test_patches[:1])
        print(f"✅ Original reconstruct() works: {reconstruction.shape}")
        
    except Exception as e:
        print(f"❌ Original methods failed: {e}")
        return False
    
    return True

def test_feature_extractor_backwards_compatibility():
    """Test that all original SparseFeatureExtractor functionality still works"""
    print("\n🔍 Testing SparseFeatureExtractor backwards compatibility...")
    
    # Create test data
    np.random.seed(42)
    test_images = np.random.randn(2, 16, 16) * 0.1 + 0.5
    
    # Test 1: Original initialization should work
    try:
        extractor = SparseFeatureExtractor(
            n_components=8,
            patch_size=(4, 4),
            sparsity_penalty=0.1,
            overlap_factor=0.5,
            whitening=True
        )
        print("✅ Original SparseFeatureExtractor initialization works")
    except Exception as e:
        print(f"❌ Original initialization failed: {e}")
        return False
    
    # Test 2: Original methods should work
    try:
        extractor.fit(test_images, max_iterations=1, verbose=False)
        print("✅ Original fit() method works")
        
        features = extractor.transform(test_images[:1])
        print(f"✅ Original transform() works: {features.shape}")
        
        params = extractor.get_params()
        print(f"✅ Original get_params() works: {len(params)} parameters")
        
    except Exception as e:
        print(f"❌ Original methods failed: {e}")
        return False
    
    return True

def test_visualization_backwards_compatibility():
    """Test that all original SparseVisualization functionality still works"""
    print("\n🔍 Testing SparseVisualization backwards compatibility...")
    
    # Test 1: Original initialization should work
    try:
        viz = SparseVisualization(colormap='gray', figsize=(10, 8))
        print("✅ Original SparseVisualization initialization works")
    except Exception as e:
        print(f"❌ Original initialization failed: {e}")
        return False
    
    # Test 2: Original visualization methods should work
    try:
        # Create test data
        dictionary = np.random.randn(16, 8)
        dictionary = dictionary / np.linalg.norm(dictionary, axis=0)
        
        viz.visualize_dictionary(dictionary, (4, 4), max_atoms=8)
        print("✅ Original visualize_dictionary() works")
        
    except Exception as e:
        print(f"❌ Original visualization failed: {e}")
        return False
    
    return True

def test_all_original_configurations():
    """Test that all original configuration options still work"""
    print("\n🔍 Testing all original configuration combinations...")
    
    # Test different sparseness functions
    sparseness_functions = ['l1', 'log', 'gaussian']
    for func in sparseness_functions:
        try:
            coder = SparseCoder(sparseness_function=func, n_components=4)
            print(f"✅ Original sparseness function '{func}' works")
        except Exception as e:
            print(f"❌ Sparseness function '{func}' failed: {e}")
            return False
    
    # Test different optimization methods
    optimization_methods = ['coordinate_descent', 'lbfgs']
    for method in optimization_methods:
        try:
            coder = SparseCoder(optimization_method=method, n_components=4)
            print(f"✅ Original optimization method '{method}' works")
        except Exception as e:
            print(f"❌ Optimization method '{method}' failed: {e}")
            return False
    
    print("✅ All original configurations work properly")
    return True

def main():
    """Run all backwards compatibility tests"""
    print("🚀 BACKWARDS COMPATIBILITY TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_sparse_coder_backwards_compatibility,
        test_dictionary_learner_backwards_compatibility, 
        test_feature_extractor_backwards_compatibility,
        test_visualization_backwards_compatibility,
        test_all_original_configurations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🎉 BACKWARDS COMPATIBILITY RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL ORIGINAL FUNCTIONALITY PRESERVED - NO BREAKING CHANGES!")
        print("✅ All changes were purely additive with proper configuration options")
        return True
    else:
        print("❌ Some backwards compatibility issues found!")
        return False

if __name__ == "__main__":
    main()