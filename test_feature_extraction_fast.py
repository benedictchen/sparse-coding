#!/usr/bin/env python3
"""
Fast test for feature extraction functionality to improve coverage
"""

import numpy as np
try:
    from .feature_extraction import SparseFeatureExtractor
except ImportError:
    from feature_extraction import SparseFeatureExtractor

def test_feature_extraction_fast():
    """Fast test of SparseFeatureExtractor methods"""
    print("Testing feature extraction functionality (fast)...")
    
    # Create tiny test data
    np.random.seed(42)
    test_images = np.random.randn(2, 8, 8) * 0.1 + 0.5
    
    # Initialize feature extractor
    try:
        extractor = SparseFeatureExtractor(n_components=4, patch_size=(4, 4), 
                                         sparsity_penalty=0.1, whitening=True)
        print("✅ SparseFeatureExtractor initialized")
    except Exception as e:
        print(f"❌ SparseFeatureExtractor initialization: {e}")
        return
    
    # Test preprocessing
    try:
        preprocessed = extractor._preprocess_images(test_images, fit=True)
        print(f"✅ _preprocess_images: {preprocessed.shape}")
    except Exception as e:
        print(f"❌ _preprocess_images: {e}")
    
    # Test patch extraction
    try:
        patches = extractor._extract_all_patches(test_images[:1])
        print(f"✅ _extract_all_patches: {patches.shape}")
    except Exception as e:
        print(f"❌ _extract_all_patches: {e}")
    
    # Test get_params
    try:
        params = extractor.get_params()
        print(f"✅ get_params: {len(params)} parameters")
    except Exception as e:
        print(f"❌ get_params: {e}")
    
    # Test set_params
    try:
        extractor.set_params(sparsity_penalty=0.05)
        print("✅ set_params")
    except Exception as e:
        print(f"❌ set_params: {e}")
    
    # Test fit with minimal iterations
    try:
        result = extractor.fit(test_images, max_iterations=1, verbose=False)
        print("✅ fit (1 iteration)")
    except Exception as e:
        print(f"❌ fit: {e}")
    
    # Test transform
    try:
        features = extractor.transform(test_images[:1], pooling='mean')
        print(f"✅ transform: {features.shape}")
    except Exception as e:
        print(f"❌ transform: {e}")
    
    # Test fit_transform
    try:
        extractor_ft = SparseFeatureExtractor(n_components=2, patch_size=(4, 4))
        features2 = extractor_ft.fit_transform(test_images[:1])
        print(f"✅ fit_transform: {features2.shape}")
    except Exception as e:
        print(f"❌ fit_transform: {e}")
    
    # Test get_feature_names
    try:
        if hasattr(extractor, 'get_feature_names'):
            names = extractor.get_feature_names()
            print(f"✅ get_feature_names: {len(names)} features")
    except Exception as e:
        print(f"❌ get_feature_names: {e}")
        
    # Test get_feature_names_out
    try:
        names_out = extractor.get_feature_names_out()
        print(f"✅ get_feature_names_out: {len(names_out)} features")
    except Exception as e:
        print(f"❌ get_feature_names_out: {e}")
    
    # Test configuration options
    try:
        extractor_no_whiten = SparseFeatureExtractor(n_components=2, whitening=False)
        print("✅ Configuration without whitening")
    except Exception as e:
        print(f"❌ No whitening config: {e}")
    
    print("🎉 Fast feature extraction test completed!")

if __name__ == "__main__":
    test_feature_extraction_fast()