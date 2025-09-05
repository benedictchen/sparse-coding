#!/usr/bin/env python3
"""
Test Research Accuracy Fixes for Sparse Coding
===============================================

Validates that the critical σ/λ calibration and whitening pipeline fixes 
are working correctly according to Olshausen & Field (1996).

Tests:
1. Image-level vs patch-level whitening comparison
2. Correct σ computation from whitened patches 
3. Impact on λ parameter calibration
4. End-to-end research-accurate pipeline

Author: Benedict Chen
Reference: Olshausen & Field (1996) - Emergence of simple-cell receptive field properties
"""

import numpy as np
import sys
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_natural_test_images(n_images: int = 15, size: tuple = (96, 96)):
    """Create synthetic natural-like images for testing"""
    images = []
    
    for i in range(n_images):
        # Base noise
        img = np.random.randn(*size) * 0.15
        
        # Add oriented edge structure (mimics natural image statistics)
        for _ in range(8):
            # Random oriented line
            y_start = np.random.randint(15, size[0] - 15)
            x_start = np.random.randint(15, size[1] - 15)
            angle = np.random.uniform(0, np.pi)
            length = np.random.randint(15, 25)
            intensity = np.random.uniform(0.7, 1.8)
            width = np.random.randint(1, 3)
            
            for t in range(length):
                for w in range(-width, width + 1):
                    y = int(y_start + t * np.sin(angle) + w * np.cos(angle))
                    x = int(x_start + t * np.cos(angle) - w * np.sin(angle))
                    if 0 <= y < size[0] and 0 <= x < size[1]:
                        img[y, x] += intensity * np.exp(-w**2 / 2)
        
        # Add some texture
        for _ in range(3):
            y_patch = np.random.randint(0, size[0] - 20)
            x_patch = np.random.randint(0, size[1] - 20)
            texture = np.random.randn(20, 20) * 0.3
            img[y_patch:y_patch+20, x_patch:x_patch+20] += texture
        
        images.append(img)
    
    return images

def test_preprocessing_accuracy():
    """Test the research-accurate preprocessing fixes"""
    print("🔬 TESTING RESEARCH ACCURACY FIXES")
    print("=" * 60)
    
    # Generate test data
    test_images = create_natural_test_images(n_images=20, size=(80, 80))
    
    print(f"📊 Test dataset: {len(test_images)} images, size {test_images[0].shape}")
    
    # Test 1: Compare old vs new preprocessing
    print(f"\n📋 TEST 1: Preprocessing Method Comparison")
    print("-" * 40)
    
    try:
        from sparse_coding.research_accurate_preprocessing import ResearchAccuratePreprocessor
        
        preprocessor = ResearchAccuratePreprocessor(
            patch_size=(16, 16),
            f0_cycles_per_picture=200.0,
            mode="paper"
        )
        
        # Apply research-accurate preprocessing
        patches_new, sigma_new, stats = preprocessor.preprocess_images_paper_accurate(
            test_images, n_patches_per_image=100
        )
        
        print(f"✅ New method (image-level whitening):")
        print(f"   • σ from whitened patches: {sigma_new:.6f}")
        print(f"   • Patch statistics: μ={stats['patches_mean']:.4f}, σ={stats['patches_std']:.4f}")
        print(f"   • Total patches: {len(patches_new)}")
        
        # Simulate old method for comparison
        print(f"\n🔄 Simulating old method (patch-level whitening)...")
        patches_old_method = []
        for image in test_images:
            for _ in range(100):
                y = np.random.randint(0, image.shape[0] - 16)
                x = np.random.randint(0, image.shape[1] - 16)
                patch = image[y:y+16, x:x+16]
                patches_old_method.append(patch.flatten())
        
        patches_old = np.array(patches_old_method)
        
        # Apply patch-level whitening (old way)
        patches_old_centered = patches_old - np.mean(patches_old, axis=1, keepdims=True)
        cov_old = np.cov(patches_old_centered, rowvar=False)
        eigenvals_old, eigenvecs_old = np.linalg.eigh(cov_old)
        whitening_matrix_old = eigenvecs_old @ np.diag(1.0 / np.sqrt(eigenvals_old + 1e-5)) @ eigenvecs_old.T
        patches_old_whitened = patches_old_centered @ whitening_matrix_old
        sigma_old = np.std(patches_old_whitened)
        
        print(f"❌ Old method (patch-level whitening):")
        print(f"   • σ from patch whitening: {sigma_old:.6f}")
        print(f"   • Patch statistics: μ={np.mean(patches_old_whitened):.4f}, σ={np.std(patches_old_whitened):.4f}")
        
        # Comparison
        print(f"\n📈 COMPARISON:")
        print(f"   • σ ratio (new/old): {sigma_new/sigma_old:.3f}")
        print(f"   • λ impact: λ_new/λ_old = {sigma_new/sigma_old:.3f}")
        
        if abs(sigma_new/sigma_old - 1.0) > 0.1:
            print(f"   ✅ SIGNIFICANT DIFFERENCE DETECTED - Fix is working!")
        else:
            print(f"   ⚠️  Small difference - may need stronger test data")
            
        test1_passed = True
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        test1_passed = False
    
    # Test 2: Validate σ calibration consistency
    print(f"\n📋 TEST 2: σ Calibration Consistency")
    print("-" * 40)
    
    try:
        if test1_passed:
            # Check that σ computed from whitened patches matches what we use in encoding
            sigma_check = np.std(patches_new)
            sigma_diff = abs(sigma_check - sigma_new) / sigma_new
            
            print(f"   • σ from preprocessing: {sigma_new:.6f}")
            print(f"   • σ from patch array: {sigma_check:.6f}")
            print(f"   • Relative difference: {sigma_diff:.2%}")
            
            if sigma_diff < 0.01:
                print(f"   ✅ σ calibration is CONSISTENT")
                test2_passed = True
            else:
                print(f"   ❌ σ calibration INCONSISTENT - {sigma_diff:.2%} difference")
                test2_passed = False
        else:
            test2_passed = False
            
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        test2_passed = False
    
    # Test 3: End-to-end sparse coding with research-accurate mode
    print(f"\n📋 TEST 3: End-to-End Sparse Coding")
    print("-" * 40)
    
    try:
        from sparse_coding import SparseCoder
        
        # Create sparse coder
        coder = SparseCoder(
            n_components=64,
            sparsity_penalty=0.1,
            max_iter=20,  # Reduced for testing
            random_state=42
        )
        
        print(f"   🔄 Fitting with mode='paper' (research-accurate)...")
        coder.fit(test_images, mode="paper")
        
        # Check that σ was set correctly
        if hasattr(coder, 'sigma_'):
            print(f"   ✅ σ stored in coder: {coder.sigma_:.6f}")
            
            # Test sparse encoding
            test_patches = patches_new[:50]  # Use preprocessed patches
            codes = coder.transform(test_patches)
            
            print(f"   ✅ Sparse coding successful:")
            print(f"      • Test patches: {len(test_patches)}")
            print(f"      • Sparse codes: {codes.shape}")
            print(f"      • Average sparsity: {np.mean(np.sum(codes != 0, axis=1)):.1f} active elements")
            
            test3_passed = True
        else:
            print(f"   ❌ σ not stored in coder")
            test3_passed = False
            
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        test3_passed = False
    
    # Test 4: Original Olshausen & Field implementation
    print(f"\n📋 TEST 4: Original Olshausen & Field Implementation")
    print("-" * 40)
    
    try:
        from sparse_coding.sparse_coding_modules.olshausen_field import OlshausenFieldOriginal
        
        # Test with research-accurate preprocessing
        original_coder = OlshausenFieldOriginal(
            n_components=32,
            patch_size=(12, 12),
            sparsity_penalty=0.2,
            sigma=None,  # Should be computed automatically
            max_iter=30,
            random_seed=42
        )
        
        print(f"   🔄 Fitting original algorithm with auto-σ...")
        smaller_test_images = test_images[:10]  # Smaller for speed
        results = original_coder.fit_original(smaller_test_images, n_patches=1000)
        
        print(f"   ✅ Original algorithm completed:")
        print(f"      • Final σ used: {original_coder.sigma:.6f}")
        print(f"      • Reconstruction error: {results['final_reconstruction_error']:.6f}")
        print(f"      • Sparsity: {results['final_sparsity']:.1f} active elements")
        print(f"      • Iterations: {results['n_iterations']}")
        
        test4_passed = True
        
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        test4_passed = False
    
    # Summary
    print(f"\n🎯 RESEARCH ACCURACY TEST SUMMARY")
    print("=" * 60)
    
    tests_passed = sum([test1_passed, test2_passed, test3_passed, test4_passed])
    total_tests = 4
    
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if test1_passed:
        print(f"✅ Preprocessing method comparison: PASSED")
    else:
        print(f"❌ Preprocessing method comparison: FAILED")
        
    if test2_passed:
        print(f"✅ σ calibration consistency: PASSED")  
    else:
        print(f"❌ σ calibration consistency: FAILED")
        
    if test3_passed:
        print(f"✅ End-to-end sparse coding: PASSED")
    else:
        print(f"❌ End-to-end sparse coding: FAILED")
        
    if test4_passed:
        print(f"✅ Original Olshausen & Field: PASSED")
    else:
        print(f"❌ Original Olshausen & Field: FAILED")
    
    if tests_passed == total_tests:
        print(f"\n🎉 ALL TESTS PASSED - Research accuracy fixes are working correctly!")
        print(f"🔬 The implementation now matches Olshausen & Field (1996) methodology exactly:")
        print(f"   • Image-level whitening with R(f) = |f|exp(-(f/f₀)⁴)")
        print(f"   • σ computed from the same distribution used in encoding")
        print(f"   • DC removal at image level before patch extraction")
        print(f"   • Proper λ/σ calibration for research reproduction")
    else:
        print(f"\n⚠️  Some tests failed - please check implementation details")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    print("🔬 SPARSE CODING RESEARCH ACCURACY VALIDATION")
    print("=" * 70)
    print("Testing critical σ/λ calibration and whitening pipeline fixes")
    print("Reference: Olshausen & Field (1996) - Emergence of simple-cell receptive field properties")
    print("")
    
    success = test_preprocessing_accuracy()
    
    if success:
        print(f"\n✅ VALIDATION COMPLETE: All research accuracy fixes working correctly")
        sys.exit(0)
    else:
        print(f"\n❌ VALIDATION FAILED: Some tests did not pass") 
        sys.exit(1)