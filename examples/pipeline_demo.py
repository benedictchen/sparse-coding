#!/usr/bin/env python3
"""
Complete Sparse Coding Pipeline Demo

Demonstrates the full sparse coding workflow:
1. Dictionary Learning from natural images
2. Sparse feature extraction
3. Advanced optimization methods
4. Visualization and analysis
5. TensorBoard logging
6. Performance evaluation

This is a comprehensive example showing all restored functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add sparse_coding to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sparse_coding.dictionary_learner import DictionaryLearner
from sparse_coding.proximal_gradient_optimization import create_proximal_sparse_coder
from sparse_coding.sparse_coding_visualization import create_visualization_report
from sparse_coding.dashboard import DashboardLogger


def load_or_generate_data(n_images: int = 50, image_size: int = 64, 
                         patch_size: tuple = (12, 12), n_patches: int = 8000):
    """
    Load or generate natural image data for training
    
    In practice, you would load real images. Here we simulate natural image statistics.
    """
    
    print(f"📊 Generating {n_images} synthetic natural images...")
    
    images = []
    patches_list = []
    
    for img_idx in range(n_images):
        # Generate 1/f^2 noise (natural image statistics)
        freqs_x = np.fft.fftfreq(image_size)
        freqs_y = np.fft.fftfreq(image_size)
        fx, fy = np.meshgrid(freqs_x, freqs_y)
        freqs2d = np.sqrt(fx**2 + fy**2)
        freqs2d[0, 0] = 1  # Avoid division by zero
        
        # 1/f power spectrum with some randomness
        power = 1.0 / (freqs2d**(1.8 + 0.4 * np.random.random()))
        
        # Random phases
        phases = np.random.uniform(0, 2*np.pi, (image_size, image_size))
        complex_spectrum = np.sqrt(power) * np.exp(1j * phases)
        
        # Generate image
        image = np.real(np.fft.ifft2(complex_spectrum))
        image = (image - image.mean()) / image.std()
        
        # Add some structure (edges, textures)
        if np.random.random() > 0.5:
            # Add vertical edges
            edge_pos = np.random.randint(image_size // 4, 3 * image_size // 4)
            image[:, :edge_pos] += 0.5
            image[:, edge_pos:] -= 0.5
        
        if np.random.random() > 0.5:
            # Add horizontal edges  
            edge_pos = np.random.randint(image_size // 4, 3 * image_size // 4)
            image[:edge_pos, :] += 0.5
            image[edge_pos:, :] -= 0.5
        
        images.append(image)
        
        # Extract patches from this image
        for _ in range(n_patches // n_images):
            i = np.random.randint(0, image_size - patch_size[0])
            j = np.random.randint(0, image_size - patch_size[1])
            patch = image[i:i+patch_size[0], j:j+patch_size[1]]
            
            # Normalize patch
            patch = patch - patch.mean()
            if patch.std() > 1e-6:
                patch = patch / patch.std()
            
            patches_list.append(patch.flatten())
    
    patches = np.array(patches_list).T
    images = np.array(images)
    
    print(f"Generated {len(images)} images and {patches.shape[1]} patches")
    return images, patches


def main():
    """Run complete sparse coding pipeline demonstration"""
    
    print("🌟 Complete Sparse Coding Pipeline Demo")
    print("======================================")
    print("Demonstrating all restored functionality from v2 minimal to full-featured library")
    
    # Setup parameters
    params = {
        'n_components': 100,  # Dictionary size
        'patch_size': (12, 12),  # Larger patches for richer features
        'sparsity_penalty': 0.03,
        'n_images': 30,
        'n_patches': 6000,
        'max_iterations': 300
    }
    
    print(f"\n🔧 Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Step 1: Data preparation
    print(f"\n{'='*50}")
    print("STEP 1: DATA PREPARATION")
    print(f"{'='*50}")
    
    images, training_patches = load_or_generate_data(
        n_images=params['n_images'],
        patch_size=params['patch_size'],
        n_patches=params['n_patches']
    )
    
    # Show some example images
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Example Generated Images', fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f'Image {i+1}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Step 2: Dictionary Learning with Logging
    print(f"\n{'='*50}")
    print("STEP 2: DICTIONARY LEARNING WITH TENSORBOARD")
    print(f"{'='*50}")
    
    # Setup logging
    log_dir = Path("sparse_coding_logs")
    log_dir.mkdir(exist_ok=True)
    
    dashboard = DashboardLogger(
        tensorboard_dir=str(log_dir / "tensorboard"),
        csv_path=str(log_dir / "metrics.csv")
    )
    
    # Create enhanced dictionary learner
    learner = DictionaryLearner(
        n_components=params['n_components'],
        patch_size=params['patch_size'],
        sparsity_penalty=params['sparsity_penalty'],
        learning_rate=0.02,
        max_iterations=params['max_iterations'],
        random_seed=42
    )
    
    print("🚀 Training dictionary with logging...")
    
    # Custom training loop with logging
    patches = training_patches
    
    for iteration in range(params['max_iterations']):
        # Sparse coding step
        codes = []
        for i in range(patches.shape[1]):
            patch = patches[:, i]
            code = learner.sparse_coder.encode(patch)
            codes.append(code)
        codes = np.array(codes).T
        
        # Dictionary update step
        dict_change = learner._update_dictionary(patches, codes)
        learner.sparse_coder.dictionary = learner.dictionary
        
        # Compute and log metrics
        metrics = learner._compute_metrics(patches, codes)
        metrics['dictionary_change'] = dict_change
        
        # Log to dashboard
        dashboard.log_training_metrics(metrics)
        
        if iteration % 20 == 0:
            dashboard.log_dictionary_atoms(learner.dictionary, params['patch_size'])
            dashboard.log_sparsity_histogram(codes)
            
        dashboard.step_forward()
        
        # Print progress
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: Error={metrics['reconstruction_error']:.6f}, "
                  f"Sparsity={metrics['sparsity_level']:.3f}")
        
        # Check convergence
        if dict_change < 1e-6:
            print(f"Converged after {iteration} iterations")
            break
    
    dashboard.close()
    
    print(f"📊 TensorBoard logs saved to: {log_dir / 'tensorboard'}")
    print("💡 Run: tensorboard --logdir sparse_coding_logs/tensorboard")
    
    # Step 3: Advanced Optimization Comparison
    print(f"\n{'='*50}")
    print("STEP 3: SOLVER COMPARISON BENCHMARK")
    print(f"{'='*50}")
    
    # Test different optimization methods on a sample patch
    test_patch = patches[:, 0]
    
    methods = ['ista', 'fista', 'coordinate_descent', 'adaptive_fista']
    optimization_results = {}
    
    for method in methods:
        print(f"Testing {method.upper()}...")
        
        optimizer = create_proximal_sparse_coder(
            learner.dictionary,
            penalty_type='l1',
            penalty_params={'lam': params['sparsity_penalty']},
            max_iter=200
        )
        
        result = getattr(optimizer, method)(test_patch)
        optimization_results[method] = result
        
        print(f"  Iterations: {result['iterations']}, "
              f"Converged: {result['converged']}, "
              f"Sparsity: {np.mean(np.abs(result['solution']) > 1e-6):.3f}")
    
    # Step 4: Comprehensive Visualization
    print(f"\n{'='*50}")
    print("STEP 4: VISUALIZATION & ANALYSIS")
    print(f"{'='*50}")
    
    # Generate sparse codes for visualization
    test_patches = patches[:, :100]  # Use first 100 patches
    test_codes = []
    
    print("Generating sparse codes for visualization...")
    for i in range(test_patches.shape[1]):
        patch = test_patches[:, i]
        code = learner.sparse_coder.encode(patch)
        test_codes.append(code)
    
    test_codes = np.array(test_codes).T
    
    # Create visualization report
    print("Creating visualization report...")
    
    figures = create_visualization_report(
        dictionary=learner.dictionary,
        codes=test_codes,
        history=learner.training_history,
        patch_size=params['patch_size'],
        original_patches=test_patches,
        save_path=str(log_dir / "analysis")
    )
    
    # Show all figures
    for fig in figures:
        plt.show()
    
    # Step 5: Feature Extraction and Analysis
    print(f"\n{'='*50}")
    print("STEP 5: FEATURE EXTRACTION & PERFORMANCE")
    print(f"{'='*50}")
    
    # Transform all training images to features
    print("Extracting features from all images...")
    features = learner.transform(images, pooling='max')
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Average sparsity: {np.mean(np.abs(features) > 1e-3):.3f}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    
    # Analyze feature diversity
    feature_correlations = np.corrcoef(features.T)
    mean_correlation = np.mean(np.abs(feature_correlations[np.triu_indices_from(feature_correlations, k=1)]))
    print(f"Mean absolute correlation between features: {mean_correlation:.3f}")
    
    # Plot feature statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Feature Analysis', fontsize=16)
    
    # Feature magnitude distribution
    axes[0, 0].hist(features[features > 1e-6], bins=50, alpha=0.7)
    axes[0, 0].set_title('Non-zero Feature Magnitudes')
    axes[0, 0].set_xlabel('Magnitude')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sparsity per image
    sparsity_per_image = np.mean(np.abs(features) > 1e-3, axis=1)
    axes[0, 1].hist(sparsity_per_image, bins=20, alpha=0.7)
    axes[0, 1].set_title('Sparsity per Image')
    axes[0, 1].set_xlabel('Fraction of Active Features')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature correlation heatmap
    im = axes[1, 0].imshow(feature_correlations[:50, :50], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 0].set_title('Feature Correlations (first 50)')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Dictionary atom usage
    atom_usage = np.mean(np.abs(test_codes) > 1e-6, axis=1)
    axes[1, 1].bar(range(len(atom_usage)), atom_usage, alpha=0.7)
    axes[1, 1].set_title('Dictionary Atom Usage')
    axes[1, 1].set_xlabel('Atom Index')
    axes[1, 1].set_ylabel('Usage Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Final summary
    print(f"\n{'='*50}")
    print("📋 PIPELINE SUMMARY")
    print(f"{'='*50}")
    
    print(f"✅ Dictionary Learning: {params['n_components']} atoms learned")
    print(f"✅ Sparse Coding: {test_codes.shape[1]} patches encoded")  
    print(f"✅ Advanced Optimization: {len(methods)} methods tested")
    print(f"✅ Visualization: {len(figures)} comprehensive plots generated")
    print(f"✅ Feature Extraction: {features.shape[0]} images → {features.shape[1]} features")
    print(f"✅ Logging: TensorBoard + CSV metrics saved")
    
    final_metrics = {
        'Dictionary atoms': params['n_components'],
        'Training patches': patches.shape[1],
        'Final reconstruction error': learner.training_history['reconstruction_errors'][-1],
        'Final sparsity': learner.training_history['sparsity_levels'][-1],
        'Feature dimensionality': features.shape[1],
        'Average feature sparsity': np.mean(np.abs(features) > 1e-3),
        'Mean feature correlation': mean_correlation
    }
    
    print(f"\n📊 Final Metrics:")
    for metric, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print(f"\n💾 All results saved to: {log_dir.absolute()}")
    print(f"\n🎉 Complete sparse coding pipeline demonstration finished!")
    print(f"🚀 From minimal v2 ({549} lines) to full-featured library!")


if __name__ == "__main__":
    main()