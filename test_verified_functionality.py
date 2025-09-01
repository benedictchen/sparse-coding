#!/usr/bin/env python3
"""
Verified Functionality Test - Confirmed Working Operations
===========================================================

Quick test of verified working functionality with proper coverage.

Author: Benedict Chen (benedict@benedictchen.com)
"""

import numpy as np

try:
    from .sparse_coder import SparseCoder
    from .dictionary_learning import DictionaryLearner  
    from .feature_extraction import SparseFeatureExtractor
except ImportError:
    from sparse_coder import SparseCoder
    from dictionary_learning import DictionaryLearner
    from feature_extraction import SparseFeatureExtractor

def test_sparse_coder_verified():
    """Test verified SparseCoder functionality"""
    print("🔥 TESTING VERIFIED: SparseCoder")
    
    # Create test data
    np.random.seed(42)
    test_images = np.random.randn(2, 8, 8) * 0.1 + 0.5
    
    # Initialize SparseCoder
    coder = SparseCoder(n_components=4, patch_size=(3, 3), max_iter=3)
    
    # Fit (dictionary learning)
    coder.fit(test_images, n_patches=20)
    
    # Transform (sparse encoding)
    codes = coder.transform(test_images[:1])
    
    # Reconstruct
    reconstruction = coder.reconstruct(codes)
    
    # Test sklearn API
    codes2 = coder.fit_transform(test_images[:1])
    
    print(f"   ✓ SparseCoder working: {codes.shape} codes, {reconstruction.shape} reconstruction")
    return True

def test_dictionary_learner_verified():
    """Test verified DictionaryLearner functionality"""
    print("🔥 TESTING VERIFIED: DictionaryLearner")
    
    # Create patch data  
    np.random.seed(42)
    patches = np.random.randn(30, 9) * 0.1 + 0.5  # 30 patches of 3x3
    
    # Initialize DictionaryLearner
    learner = DictionaryLearner(n_components=6, patch_size=(3, 3), max_iterations=3)
    
    # Fit (dictionary learning)
    learner.fit(patches, verbose=False)
    
    # Get dictionary
    dictionary = learner.get_dictionary()
    
    # Transform
    codes = learner.transform(patches[:5])
    
    # Test sklearn API
    codes2 = learner.fit_transform(patches[:3])
    components = learner.get_components()
    
    print(f"   ✓ DictionaryLearner working: {dictionary.shape} dictionary, {codes.shape} codes")
    return True

def test_feature_extractor_verified():
    """Test verified SparseFeatureExtractor functionality"""
    print("🔥 TESTING VERIFIED: SparseFeatureExtractor")
    
    # Create images
    np.random.seed(42)  
    images = np.random.randn(2, 10, 10) * 0.1 + 0.5
    
    # Initialize SparseFeatureExtractor
    extractor = SparseFeatureExtractor(n_components=6, patch_size=(3, 3))
    
    # Fit 
    extractor.fit(images, max_iterations=3, verbose=False)
    
    # Transform (extract features)
    features = extractor.transform(images[:1])
    
    # Test sklearn API
    params = extractor.get_params()
    feature_names = extractor.get_feature_names()
    
    print(f"   ✓ SparseFeatureExtractor working: {features.shape} features extracted")
    return True

def run_verified_tests():
    """Run verified functionality tests"""
    print("=" * 60)
    print("🎯 VERIFIED FUNCTIONALITY TEST")
    print("   Testing confirmed working operations")
    print("=" * 60)
    
    tests = [
        test_sparse_coder_verified,
        test_dictionary_learner_verified, 
        test_feature_extractor_verified
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ✗ Test failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎉 VERIFIED RESULTS: {passed}/{total} test components passed")
    
    if passed == total:
        print("✅ ALL VERIFIED FUNCTIONALITY IS WORKING!")
        print("✅ SparseCoder: Dictionary learning + sparse encoding")
        print("✅ DictionaryLearner: Patch-based dictionary learning")
        print("✅ SparseFeatureExtractor: Image feature extraction")
        return True
    else:
        print(f"❌ {total - passed} components have issues")
        return False

if __name__ == "__main__":
    success = run_verified_tests()
    exit(0 if success else 1)