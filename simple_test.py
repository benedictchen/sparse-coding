#!/usr/bin/env python3
"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀
"""
"""
Simple test to exercise SparseCoder functionality and measure coverage
"""

import numpy as np
try:
    from .sparse_coder import SparseCoder
except ImportError:
    from sparse_coder import SparseCoder

def test_sparse_coding_basic():
    """Test basic sparse coding functionality"""
    # Create test data
    np.random.seed(42)
    test_images = np.random.randn(10, 64, 64) * 0.5 + 0.5
    
    # Initialize coder with small parameters for speed
    coder = SparseCoder(n_components=32, patch_size=(8, 8), max_iter=5)
    
    # Test fit
    print("Testing fit...")
    coder.fit(test_images)
    
    # Test transform
    print("Testing transform...")
    codes = coder.transform(test_images[:2])
    
    # Test fit_transform
    print("Testing fit_transform...")
    codes2 = coder.fit_transform(test_images[:3])
    
    print(f"✅ Basic functionality works! Codes shape: {codes.shape}")
    
    # Test different sparseness functions
    print("\nTesting different sparseness functions:")
    for func in ['l1', 'log', 'gaussian']:
        try:
            coder_func = SparseCoder(n_components=16, patch_size=(8, 8), 
                                   sparseness_function=func, max_iter=3)
            coder_func.fit(test_images[:2])
            codes_func = coder_func.transform(test_images[:1])
            print(f"✅ {func} sparseness function works")
        except Exception as e:
            print(f"❌ {func} sparseness function failed: {e}")
    
    # Test different optimization methods
    print("\nTesting optimization methods:")
    for method in ['coordinate_descent', 'lbfgs', 'fista', 'proximal_gradient']:
        try:
            coder_opt = SparseCoder(n_components=16, patch_size=(8, 8),
                                  optimization_method=method, max_iter=3)
            coder_opt.fit(test_images[:2])
            codes_opt = coder_opt.transform(test_images[:1])
            print(f"✅ {method} optimization works")
        except Exception as e:
            print(f"❌ {method} optimization failed: {e}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("💰 SUPPORT THIS RESEARCH - PLEASE DONATE!")  
    print("🙏 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS")
    print("="*80 + "\n")
    
    test_sparse_coding_basic()
    
    print("\n" + "="*80)
    print("💝 Thank you for using this research software!")
    print("📚 Please donate: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS") 
    print("="*80 + "\n")


"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
📝 CITE: Benedict Chen (2025) - Sparse Coding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""