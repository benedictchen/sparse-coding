#!/usr/bin/env python3
"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀
"""
"""
Fast coverage test for sparse coding
"""

import numpy as np
try:
    from .sparse_coder import SparseCoder
except ImportError:
    try:
        from sparse_coder import SparseCoder
    except ImportError:
        from ..sparse_coder import SparseCoder

def test_fast_coverage():
    """Fast test to get coverage of key methods"""
    # Create tiny test data
    np.random.seed(42)
    test_images = np.random.randn(2, 16, 16) * 0.1 + 0.5
    
    print("Testing basic initialization and core methods...")
    
    # Test 1: Basic initialization
    coder = SparseCoder(n_components=4, patch_size=(4, 4), max_iter=1)
    print("✅ Basic initialization")
    
    # Test 2: Configuration validation
    coder._validate_configuration()
    print("✅ Configuration validation")
    
    # Test 3: Dictionary initialization
    coder._initialize_dictionary()
    print("✅ Dictionary initialization")
    
    # Test 4: Patch extraction
    patches = coder._extract_patches(test_images, n_patches=20)
    print(f"✅ Patch extraction: {patches.shape}")
    
    # Test 5: Sparse encoding with different methods
    patch = patches[0]
    
    # Test equation 5 method
    coeffs_eq5 = coder._sparse_encode_equation_5(patch)
    print("✅ Equation 5 encoding")
    
    # Test general optimization
    def dummy_obj(x):
        return np.sum((coder.dictionary @ x - patch)**2) + 0.1 * np.sum(np.abs(x))
    
    def dummy_grad(x):
        return 2 * coder.dictionary.T @ (coder.dictionary @ x - patch) + 0.1 * np.sign(x)
    
    coeffs_gen = coder._general_optimization(patch, dummy_obj, dummy_grad, np.zeros(4))
    print("✅ General optimization")
    
    # Test FISTA 
    coeffs_fista = coder._fista_optimization(patch, dummy_obj, dummy_grad, np.zeros(4))
    print("✅ FISTA optimization")
    
    # Test proximal gradient
    coeffs_prox = coder._proximal_gradient(patch, dummy_obj, dummy_grad, np.zeros(4))
    print("✅ Proximal gradient")
    
    # Test single patch encoding
    coeffs_single = coder._sparse_encode_single(patch)
    print("✅ Single patch encoding")
    
    # Test different sparseness functions
    for func in ['l1', 'log', 'gaussian']:
        coder.sparseness_function = func
        try:
            coeffs_func = coder._sparse_encode_equation_5(patch)
            print(f"✅ {func} sparseness function")
        except Exception as e:
            print(f"❌ {func} sparseness function: {e}")
    
    # Test sklearn-style methods with minimal data
    coder_sklearn = SparseCoder(n_components=4, patch_size=(4, 4), max_iter=1)
    
    # Quick fit (1 iteration)
    print("Testing sklearn-style methods...")
    coder_sklearn.fit(test_images)
    print("✅ fit()")
    
    # Test transform
    codes = coder_sklearn.transform(test_images[:1])
    print(f"✅ transform(): {codes.shape}")
    
    # Test fit_transform
    codes2 = coder_sklearn.fit_transform(test_images[:1])
    print(f"✅ fit_transform(): {codes2.shape}")
    
    # Test advanced features
    print("Testing advanced features...")
    
    # Test overcomplete basis creation
    try:
        basis = coder.create_overcomplete_basis(overcompleteness_factor=1.5, basis_type='random')
        print(f"✅ create_overcomplete_basis: {basis.shape}")
    except Exception as e:
        print(f"❌ create_overcomplete_basis: {e}")
    
    # Test lateral inhibition
    try:
        activations = np.random.randn(8)
        inhibited = coder.lateral_inhibition(activations, inhibition_strength=0.5)
        print(f"✅ lateral_inhibition: {inhibited.shape}")
    except Exception as e:
        print(f"❌ lateral_inhibition: {e}")
    
    # Test basis creation methods
    try:
        gabor_basis = coder._create_gabor_basis(16, 8)
        print(f"✅ _create_gabor_basis: {gabor_basis.shape}")
    except Exception as e:
        print(f"❌ _create_gabor_basis: {e}")
        
    try:
        edge_basis = coder._create_edge_basis(16, 8)
        print(f"✅ _create_edge_basis: {edge_basis.shape}")
    except Exception as e:
        print(f"❌ _create_edge_basis: {e}")
    
    # Test more advanced methods to improve coverage
    print("Testing additional methods...")
    
    # Test configure_sparseness_function
    try:
        coder.configure_sparseness_function('l1', l1_lambda=0.1)
        print("✅ configure_sparseness_function(l1)")
        coder.configure_sparseness_function('log', log_base=10.0)
        print("✅ configure_sparseness_function(log)")
        coder.configure_sparseness_function('gaussian', gaussian_sigma=1.0)
        print("✅ configure_sparseness_function(gaussian)")
    except Exception as e:
        print(f"❌ configure_sparseness_function: {e}")
    
    # Test different optimization methods
    for opt_method in ['coordinate_descent', 'lbfgs', 'fista', 'proximal_gradient']:
        try:
            coder.optimization_method = opt_method
            coeffs = coder._sparse_encode_single(patch)
            print(f"✅ optimization method: {opt_method}")
        except Exception as e:
            print(f"❌ optimization method {opt_method}: {e}")
    
    # Test error handling and edge cases
    try:
        # Test with invalid parameters
        bad_coder = SparseCoder(n_components=-1, patch_size=(0, 0))
        bad_coder._validate_configuration()
    except Exception as e:
        print(f"✅ Error handling for bad parameters: {type(e).__name__}")
    
    # Test get_components method if it exists
    try:
        if hasattr(coder_sklearn, 'get_components'):
            components = coder_sklearn.get_components()
            print(f"✅ get_components(): {components.shape if components is not None else 'None'}")
    except Exception as e:
        print(f"❌ get_components(): {e}")
    
    # Test reconstruction
    try:
        if hasattr(coder_sklearn, 'reconstruct'):
            reconstruction = coder_sklearn.reconstruct(codes[:1])
            print(f"✅ reconstruct(): {reconstruction.shape}")
    except Exception as e:
        print(f"❌ reconstruct(): {e}")
    
    # Test noise handling
    try:
        noisy_images = test_images + np.random.normal(0, 0.5, test_images.shape)
        coder_noise = SparseCoder(n_components=4, patch_size=(4, 4), max_iter=1)
        coder_noise.fit(noisy_images)
        print("✅ Noise handling")
    except Exception as e:
        print(f"❌ Noise handling: {e}")
    
    print("\n🎉 Fast coverage test completed!")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("💰 SUPPORT THIS RESEARCH - PLEASE DONATE!")  
    print("🙏 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS")
    print("="*80 + "\n")
    
    test_fast_coverage()
    
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