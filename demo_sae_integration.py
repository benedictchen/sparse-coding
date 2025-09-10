#!/usr/bin/env python3
"""
Comprehensive demo of SAE (Sparse Autoencoder) integration.

Demonstrates the unified feature interface bridging classical dictionary
learning and modern sparse autoencoders. Shows interoperability between
methods and interpretability analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add package to path for development
sys.path.insert(0, str(Path(__file__).parent))

from sparse_coding.sae import (
    fit_features, encode_features, decode_features, compare_features,
    FeatureExtractor, create_feature_atlas, summarize_atlas
)


def generate_demo_data(n_samples=1000, n_features=64, n_true_atoms=32, noise_level=0.1):
    """Generate synthetic sparse data for demonstration."""
    print("🎲 Generating synthetic sparse data...")
    
    # Create ground truth dictionary (random orthogonal)
    np.random.seed(42)
    D_true = np.random.randn(n_features, n_true_atoms)
    D_true = D_true / np.linalg.norm(D_true, axis=0, keepdims=True)
    
    # Generate sparse codes
    A_true = np.zeros((n_samples, n_true_atoms))
    for i in range(n_samples):
        # Each sample uses only 3-5 atoms
        n_active = np.random.randint(3, 6)
        active_indices = np.random.choice(n_true_atoms, n_active, replace=False)
        A_true[i, active_indices] = np.random.randn(n_active) * 2
    
    # Generate observations
    X = A_true @ D_true.T + noise_level * np.random.randn(n_samples, n_features)
    
    print(f"✅ Generated {n_samples} samples with {n_features} features")
    print(f"   Ground truth: {n_true_atoms} atoms, sparsity: {np.mean(A_true == 0):.1%}")
    
    return X, D_true, A_true


def demo_unified_interface():
    """Demonstrate unified feature interface across methods."""
    print("\n" + "="*60)
    print("🔬 UNIFIED FEATURE INTERFACE DEMO")
    print("="*60)
    
    # Generate data
    X, D_true, A_true = generate_demo_data(n_samples=500, n_features=32, n_true_atoms=16)
    
    # Split into train/test
    X_train, X_test = X[:400], X[400:]
    
    print(f"\n📊 Data shapes:")
    print(f"   Training: {X_train.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Method 1: Classical Dictionary Learning
    print(f"\n🔍 Method 1: Classical Dictionary Learning")
    features_dict = fit_features(
        X_train, 
        method='dict',
        n_atoms=16,
        sparsity=0.1,
        max_iter=50
    )
    print(f"   ✅ Fitted {features_dict.n_atoms} dictionary atoms")
    
    # Method 2: Sparse Autoencoder (if PyTorch available)
    try:
        print(f"\n🧠 Method 2: Sparse Autoencoder (L1)")
        features_sae = fit_features(
            X_train,
            method='sae', 
            n_atoms=16,
            sparsity=1e-3,
            sae_type='L1SAE',
            n_epochs=100,
            lr=1e-3
        )
        print(f"   ✅ Fitted {features_sae.n_atoms} SAE features")
        has_sae = True
        
    except ImportError as e:
        print(f"   ⚠️  SAE unavailable: {e}")
        features_sae = None
        has_sae = False
    
    # Method 3: Hybrid Approach
    if has_sae:
        print(f"\n⚡ Method 3: Hybrid (Dict → SAE)")
        features_hybrid = fit_features(
            X_train,
            method='hybrid',
            n_atoms=16, 
            sparsity=0.1,
            n_epochs=50
        )
        print(f"   ✅ Fitted {features_hybrid.n_atoms} hybrid features")
    
    # Encode test data with all methods
    print(f"\n📈 Encoding test data...")
    
    # Dictionary method
    A_dict = encode_features(X_test, features_dict)
    X_recon_dict = decode_features(A_dict, features_dict)
    mse_dict = np.mean((X_test - X_recon_dict)**2)
    sparsity_dict = np.mean(np.abs(A_dict) < 1e-6)
    
    print(f"   Dict MSE: {mse_dict:.6f}, Sparsity: {sparsity_dict:.1%}")
    
    if has_sae:
        # SAE method
        A_sae = encode_features(X_test, features_sae) 
        X_recon_sae = decode_features(A_sae, features_sae)
        mse_sae = np.mean((X_test - X_recon_sae)**2)
        sparsity_sae = np.mean(np.abs(A_sae) < 1e-6)
        
        print(f"   SAE MSE:  {mse_sae:.6f}, Sparsity: {sparsity_sae:.1%}")
        
        # Hybrid method
        A_hybrid = encode_features(X_test, features_hybrid)
        X_recon_hybrid = decode_features(A_hybrid, features_hybrid)  
        mse_hybrid = np.mean((X_test - X_recon_hybrid)**2)
        sparsity_hybrid = np.mean(np.abs(A_hybrid) < 1e-6)
        
        print(f"   Hybrid MSE: {mse_hybrid:.6f}, Sparsity: {sparsity_hybrid:.1%}")
    
    return X_test, features_dict, features_sae if has_sae else None


def demo_feature_extractor_class():
    """Demonstrate FeatureExtractor sklearn-style interface."""
    print("\n" + "="*60)
    print("🔧 FEATUREEXTRACTOR CLASS DEMO")  
    print("="*60)
    
    # Generate data
    X, _, _ = generate_demo_data(n_samples=300, n_features=24, n_true_atoms=12)
    
    # Create extractor
    extractor = FeatureExtractor(
        method='dict',
        n_atoms=12,
        sparsity=0.15
    )
    
    print(f"\n📚 Using sklearn-style interface:")
    print(f"   Method: {extractor.method}")
    print(f"   Atoms: {extractor.n_atoms}")
    print(f"   Sparsity: {extractor.sparsity}")
    
    # Fit and transform
    A = extractor.fit_transform(X)
    X_recon = extractor.inverse_transform(A)
    
    print(f"\n📊 Results:")
    print(f"   Input shape: {X.shape}")
    print(f"   Code shape: {A.shape}")
    print(f"   Reconstruction MSE: {np.mean((X - X_recon)**2):.6f}")
    print(f"   Sparsity achieved: {np.mean(np.abs(A) < 1e-6):.1%}")
    
    # Access fitted features
    features = extractor.features_
    print(f"   Dictionary shape: {features.dictionary.shape}")
    print(f"   Method: {features.method}")
    
    return extractor


def demo_method_comparison():
    """Demonstrate automatic method comparison."""
    print("\n" + "="*60)
    print("⚔️  METHOD COMPARISON DEMO")
    print("="*60)
    
    # Generate data  
    X, _, _ = generate_demo_data(n_samples=400, n_features=28, n_true_atoms=14)
    
    # Compare methods
    methods = ['dict']
    try:
        # Test if PyTorch is available
        import torch
        methods.append('sae')
        print("   Available methods: Dictionary Learning, SAE")
    except ImportError:
        print("   Available methods: Dictionary Learning only")
    
    print(f"\n🏁 Comparing methods on {X.shape[0]} samples...")
    
    results = compare_features(
        X,
        methods=methods,
        n_atoms=14,
        sparsity=0.1,
        n_epochs=50  # For SAE if available
    )
    
    print(f"\n📋 Comparison Results:")
    print(f"{'Method':<12} {'MSE':<12} {'Sparsity':<12} {'Avg Nonzero':<12}")
    print("-" * 50)
    
    for method, result in results.items():
        if 'error' in result:
            print(f"{method:<12} ERROR: {result['error']}")
        else:
            print(f"{method:<12} {result['mse']:<12.6f} {result['sparsity_level']:<12.1%} {result['n_nonzero']:<12.1f}")
    
    return results


def demo_interpretability_analysis():
    """Demonstrate interpretability and analysis tools."""
    print("\n" + "="*60)
    print("🔍 INTERPRETABILITY ANALYSIS DEMO")
    print("="*60)
    
    # Generate more structured data for better analysis
    X, _, _ = generate_demo_data(n_samples=600, n_features=36, n_true_atoms=18)
    
    # Fit features
    features = fit_features(
        X,
        method='dict', 
        n_atoms=18,
        sparsity=0.08,
        max_iter=100
    )
    
    print(f"✅ Fitted {features.n_atoms} features using {features.method}")
    
    # Create feature atlas
    print(f"\n📖 Creating feature atlas...")
    atlas = create_feature_atlas(
        features,
        X,
        n_examples=5,
        sort_by='usage'
    )
    
    # Print summary
    print(f"\n📋 Feature Atlas Summary:")
    summary = summarize_atlas(atlas, n_top=10)
    print(summary)
    
    # Detailed analysis of top features
    print(f"\n🔬 Detailed Analysis of Top 5 Features:")
    print("-" * 50)
    
    for i, feat_idx in enumerate(atlas['feature_ranking'][:5]):
        stats = atlas['feature_details'][feat_idx]
        examples = atlas['top_examples'][feat_idx]
        
        print(f"\n{i+1}. Feature {feat_idx}:")
        print(f"   Activation frequency: {stats.activation_freq:.1%}")
        print(f"   Mean activation: {stats.mean_activation:.4f}")
        print(f"   Max activation: {stats.max_activation:.4f}")
        print(f"   Selectivity: {stats.selectivity:.3f}")
        print(f"   Top activations: {examples['activations'][:3]}")
    
    return atlas


def demo_cross_framework_compatibility():
    """Demonstrate backend compatibility across frameworks."""
    print("\n" + "="*60)
    print("🔄 CROSS-FRAMEWORK COMPATIBILITY DEMO")
    print("="*60)
    
    # Generate data
    X_np, _, _ = generate_demo_data(n_samples=200, n_features=20, n_true_atoms=10)
    
    print(f"📊 Original data: NumPy array {X_np.shape}")
    
    # Fit with NumPy data
    features = fit_features(X_np, method='dict', n_atoms=10, sparsity=0.1, max_iter=30)
    print(f"✅ Fitted features with NumPy backend")
    
    # Test PyTorch compatibility
    try:
        import torch
        print(f"\n🔥 Testing PyTorch compatibility...")
        
        # Convert to PyTorch
        X_torch = torch.from_numpy(X_np).float()
        print(f"   PyTorch tensor: {X_torch.shape}, dtype: {X_torch.dtype}")
        
        # Encode using same features (backend adaptation should be automatic)
        A_torch = encode_features(X_torch, features)
        print(f"   Encoded shape: {A_torch.shape}, type: {type(A_torch)}")
        
        # Decode back
        X_recon_torch = decode_features(A_torch, features) 
        mse = torch.mean((X_torch - X_recon_torch)**2).item()
        print(f"   Reconstruction MSE: {mse:.6f}")
        print(f"✅ PyTorch compatibility confirmed")
        
    except ImportError:
        print(f"⚠️  PyTorch not available, skipping compatibility test")
    
    # Test JAX compatibility
    try:
        import jax.numpy as jnp
        print(f"\n⚡ Testing JAX compatibility...")
        
        X_jax = jnp.array(X_np)
        print(f"   JAX array: {X_jax.shape}, dtype: {X_jax.dtype}")
        
        A_jax = encode_features(X_jax, features)
        print(f"   Encoded shape: {A_jax.shape}, type: {type(A_jax)}")
        
        X_recon_jax = decode_features(A_jax, features)
        mse = float(jnp.mean((X_jax - X_recon_jax)**2))
        print(f"   Reconstruction MSE: {mse:.6f}")
        print(f"✅ JAX compatibility confirmed")
        
    except ImportError:
        print(f"⚠️  JAX not available, skipping compatibility test")


def main():
    """Run comprehensive SAE integration demo."""
    print("🚀 SPARSE AUTOENCODER INTEGRATION DEMO")
    print("=" * 60)
    print("Demonstrates unified interface bridging classical dictionary")
    print("learning and modern sparse autoencoders for interpretability.")
    
    try:
        # Core demos
        demo_unified_interface()
        demo_feature_extractor_class()
        demo_method_comparison()
        demo_interpretability_analysis()
        demo_cross_framework_compatibility()
        
        print(f"\n🎉 All demos completed successfully!")
        print(f"\n💡 Key takeaways:")
        print(f"   • Unified interface works across dict/SAE methods")
        print(f"   • sklearn-style FeatureExtractor for easy integration")
        print(f"   • Comprehensive interpretability analysis tools")
        print(f"   • Automatic backend compatibility (NumPy/PyTorch/JAX)")
        print(f"   • Bridge between classical and modern sparse coding")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)