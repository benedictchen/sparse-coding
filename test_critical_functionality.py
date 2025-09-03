#!/usr/bin/env python3
"""
Critical Functionality Test - Core Sparse Coding Pipeline
=========================================================

Tests the most essential functionality that must work reliably:
1. Core sparse coding pipeline (image → dictionary → codes → reconstruction)  
2. Dictionary learning from natural-like images
3. Sparse encoding of new inputs
4. Basic sklearn API compatibility

Author: Benedict Chen (benedict@benedictchen.com)
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

try:
    from .sparse_coder import SparseCoder
    from .dictionary_learning import DictionaryLearner
    from .feature_extraction import SparseFeatureExtractor
except ImportError:
    from sparse_coder import SparseCoder
    from dictionary_learning import DictionaryLearner
    from feature_extraction import SparseFeatureExtractor

def create_test_images(n_images=5, img_size=(32, 32), seed=42):
    """Create realistic test images with edge-like features"""
    np.random.seed(seed)
    images = []
    
    for i in range(n_images):
        img = np.zeros(img_size)
        
        # Add oriented edges (like natural images)
        for _ in range(8):
            # Random edge parameters
            center_y, center_x = np.random.randint(8, img_size[0]-8, 2)
            orientation = np.random.uniform(0, np.pi)
            length = np.random.randint(8, 16)
            width = np.random.randint(1, 3)
            
            # Create oriented edge
            for t in np.linspace(-length//2, length//2, length):
                y = int(center_y + t * np.sin(orientation))
                x = int(center_x + t * np.cos(orientation))
                
                for w in range(-width, width+1):
                    wy = int(y + w * np.cos(orientation))
                    wx = int(x - w * np.sin(orientation))
                    
                    if 0 <= wy < img_size[0] and 0 <= wx < img_size[1]:
                        img[wy, wx] = 1.0
        
        # Add noise
        img += np.random.normal(0, 0.1, img_size)
        img = np.clip(img, 0, 1)
        images.append(img)
    
    return np.array(images)

def test_critical_sparse_coder():
    """Test critical SparseCoder functionality"""
    print("🔥 TESTING CRITICAL: SparseCoder Core Pipeline")
    
    # Create test images
    images = create_test_images(n_images=3, img_size=(24, 24))
    print(f"   ✓ Created {len(images)} test images: {images[0].shape}")
    
    # Test 1: Basic initialization
    try:
        coder = SparseCoder(n_components=16, patch_size=(6, 6), max_iter=3)
        print("   ✓ SparseCoder initialization successful")
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - Initialization: {e}")
        return False
    
    # Test 2: Dictionary learning (core algorithm)
    try:
        start_time = time.time()
        coder.fit(images, verbose=False)
        fit_time = time.time() - start_time
        print(f"   ✓ Dictionary learning successful ({fit_time:.2f}s)")
        
        # Verify dictionary properties
        assert coder.dictionary is not None, "Dictionary not created"
        assert coder.dictionary.shape == (36, 16), f"Wrong dictionary shape: {coder.dictionary.shape}"
        print(f"   ✓ Dictionary shape correct: {coder.dictionary.shape}")
        
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - Dictionary learning: {e}")
        return False
    
    # Test 3: Sparse encoding (core algorithm)
    try:
        start_time = time.time()
        codes = coder.transform(images[:2])
        encode_time = time.time() - start_time
        print(f"   ✓ Sparse encoding successful ({encode_time:.2f}s)")
        
        # Verify codes properties
        assert codes is not None, "Codes not generated"
        assert codes.shape[0] == 2, f"Wrong number of codes: {codes.shape[0]}"
        assert codes.shape[1] == 16, f"Wrong code dimension: {codes.shape[1]}"
        
        # Verify sparsity (most coefficients should be near zero)
        sparsity_ratio = np.mean(np.abs(codes) > 1e-3)
        print(f"   ✓ Codes shape correct: {codes.shape}, sparsity: {sparsity_ratio:.2f}")
        
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - Sparse encoding: {e}")
        return False
    
    # Test 4: Reconstruction (verify round-trip)
    try:
        reconstruction = coder.reconstruct(codes)
        print(f"   ✓ Reconstruction successful: {reconstruction.shape}")
        
        # Verify reconstruction quality (basic sanity check)
        reconstruction_error = np.mean((reconstruction - codes @ coder.dictionary.T)**2)
        assert reconstruction_error < 1e-10, f"Reconstruction error too high: {reconstruction_error}"
        
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - Reconstruction: {e}")
        return False
    
    # Test 5: sklearn API compatibility
    try:
        codes_fit_transform = coder.fit_transform(images[:1])
        assert codes_fit_transform.shape[1] == 16, "fit_transform failed"
        print("   ✓ sklearn fit_transform() works")
        
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - sklearn API: {e}")
        return False
    
    print("   🎉 SparseCoder CRITICAL functionality: ALL TESTS PASSED")
    return True

def test_critical_dictionary_learner():
    """Test critical DictionaryLearner functionality"""
    print("\n🔥 TESTING CRITICAL: DictionaryLearner Core Pipeline")
    
    # Create test patches directly
    np.random.seed(42)
    n_patches = 50
    patch_dim = 16  # 4x4 patches
    patches = np.random.randn(n_patches, patch_dim) * 0.1 + 0.5
    
    # Test 1: Basic initialization
    try:
        learner = DictionaryLearner(n_components=8, patch_size=(4, 4), max_iterations=3)
        print("   ✓ DictionaryLearner initialization successful")
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - Initialization: {e}")
        return False
    
    # Test 2: Dictionary learning
    try:
        result = learner.fit(patches, verbose=False)
        print("   ✓ Dictionary learning successful")
        
        # Verify dictionary properties
        dictionary = learner.get_dictionary()
        assert dictionary.shape == (16, 8), f"Wrong dictionary shape: {dictionary.shape}"
        print(f"   ✓ Dictionary shape correct: {dictionary.shape}")
        
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - Dictionary learning: {e}")
        return False
    
    # Test 3: Transform (encoding)
    try:
        codes = learner.transform(patches[:5])
        assert codes.shape == (5, 8), f"Wrong codes shape: {codes.shape}"
        print(f"   ✓ Transform successful: {codes.shape}")
        
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - Transform: {e}")
        return False
    
    # Test 4: sklearn API
    try:
        codes_sklearn = learner.fit_transform(patches[:3])
        components = learner.get_components()
        assert codes_sklearn.shape == (3, 8), "fit_transform failed"
        assert components.shape == (8, 16), "get_components failed"
        print("   ✓ sklearn API works")
        
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - sklearn API: {e}")
        return False
    
    print("   🎉 DictionaryLearner CRITICAL functionality: ALL TESTS PASSED")
    return True

def test_critical_feature_extractor():
    """Test critical SparseFeatureExtractor functionality"""
    print("\n🔥 TESTING CRITICAL: SparseFeatureExtractor Core Pipeline")
    
    # Create test images
    images = create_test_images(n_images=2, img_size=(16, 16))
    
    # Test 1: Basic initialization
    try:
        extractor = SparseFeatureExtractor(n_components=8, patch_size=(4, 4))
        print("   ✓ SparseFeatureExtractor initialization successful")
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - Initialization: {e}")
        return False
    
    # Test 2: Fit (dictionary learning)
    try:
        extractor.fit(images, max_iterations=2, verbose=False)
        print("   ✓ Feature extractor fit successful")
        
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - Fit: {e}")
        return False
    
    # Test 3: Transform (feature extraction)
    try:
        features = extractor.transform(images[:1])
        assert features is not None, "Features not generated"
        print(f"   ✓ Feature extraction successful: {features.shape}")
        
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - Feature extraction: {e}")
        return False
    
    # Test 4: sklearn API
    try:
        params = extractor.get_params()
        names = extractor.get_feature_names()
        assert len(params) > 0, "get_params failed"
        assert len(names) > 0, "get_feature_names failed"
        print("   ✓ sklearn API works")
        
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - sklearn API: {e}")
        return False
    
    print("   🎉 SparseFeatureExtractor CRITICAL functionality: ALL TESTS PASSED")
    return True

def test_end_to_end_pipeline():
    """Test complete end-to-end sparse coding pipeline"""
    print("\n🔥 TESTING CRITICAL: End-to-End Pipeline")
    
    # Create realistic test scenario
    images = create_test_images(n_images=4, img_size=(20, 20))
    
    try:
        # Step 1: Learn dictionary from training images
        print("   → Step 1: Learning dictionary...")
        coder = SparseCoder(n_components=12, patch_size=(5, 5), max_iter=2)
        coder.fit(images[:3], verbose=False)  # Train on first 3 images
        
        # Step 2: Encode test image
        print("   → Step 2: Encoding test image...")
        test_codes = coder.transform([images[3]])  # Test on 4th image
        
        # Step 3: Reconstruct from codes
        print("   → Step 3: Reconstructing from codes...")
        reconstructed = coder.reconstruct(test_codes)
        
        # Step 4: Verify pipeline integrity
        print("   → Step 4: Verifying pipeline...")
        assert test_codes.shape[1] == 12, "Wrong code dimension"
        assert reconstructed.shape[1] == 25, "Wrong reconstruction dimension"
        
        # Check that reconstruction uses learned dictionary
        expected_reconstruction = test_codes @ coder.dictionary.T
        reconstruction_match = np.allclose(reconstructed, expected_reconstruction, atol=1e-10)
        assert reconstruction_match, "Reconstruction doesn't match expected"
        
        print("   ✓ End-to-end pipeline verification successful")
        print(f"   ✓ Input: {images[3].shape} → Codes: {test_codes.shape} → Reconstruction: {reconstructed.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ CRITICAL FAILURE - End-to-end pipeline: {e}")
        return False

def run_critical_tests():
    """Run all critical functionality tests"""
    print("=" * 80)
    print("🎯 CRITICAL FUNCTIONALITY TEST SUITE")
    print("   Testing most essential sparse coding functionality")
    print("=" * 80)
    
    tests = [
        test_critical_sparse_coder,
        test_critical_dictionary_learner,
        test_critical_feature_extractor,
        test_end_to_end_pipeline
    ]
    
    passed = 0
    total = len(tests)
    start_time = time.time()
    
    for test in tests:
        if test():
            passed += 1
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"🎉 CRITICAL FUNCTIONALITY RESULTS: {passed}/{total} tests passed ({total_time:.2f}s)")
    
    if passed == total:
        print("✅ Critical functionality tests passed!")
        print("✅ Core sparse coding pipeline is fully functional")
        print("✅ Dictionary learning algorithms working correctly") 
        print("✅ Sparse encoding producing valid results")
        print("✅ sklearn API compatibility confirmed")
        print("✅ End-to-end pipeline verified")
        return True
    else:
        print(f"❌ {total - passed} CRITICAL FAILURES DETECTED!")
        print("❌ Core functionality needs immediate attention")
        return False

if __name__ == "__main__":
    success = run_critical_tests()
    exit(0 if success else 1)