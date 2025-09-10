#!/usr/bin/env python3
"""
Basic Dictionary Learning Example

Demonstrates how to use the DictionaryLearner to learn features from natural images.
This reproduces the classic Olshausen & Field (1996) experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add sparse_coding to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sparse_coding.dictionary_learner import DictionaryLearner
from sparse_coding.visualization import plot_dictionary_atoms, plot_training_progress


def generate_natural_image_patches(n_patches: int = 10000, patch_size: tuple = (8, 8)) -> np.ndarray:
    """
    Generate random patches from natural images (simulated)
    
    In practice, you would load real natural images and extract patches.
    Here we simulate with correlated noise that mimics natural image statistics.
    """
    
    # Create synthetic "natural" images with 1/f^2 power spectrum
    image_size = 64
    n_images = n_patches // 100
    
    patches = []
    
    for _ in range(n_images):
        # Generate 1/f^2 noise in frequency domain
        freqs = np.fft.fftfreq(image_size).reshape(-1, 1)
        freqs2d = np.sqrt(freqs**2 + freqs.T**2)
        freqs2d[0, 0] = 1  # Avoid division by zero
        
        # 1/f^2 power spectrum
        power_spectrum = 1.0 / (freqs2d**2)
        
        # Generate random phases
        phases = np.random.uniform(0, 2*np.pi, (image_size, image_size))
        complex_spectrum = np.sqrt(power_spectrum) * np.exp(1j * phases)
        
        # Convert to spatial domain
        image = np.real(np.fft.ifft2(complex_spectrum))
        
        # Normalize
        image = (image - image.mean()) / image.std()
        
        # Extract random patches
        for _ in range(100):
            i = np.random.randint(0, image_size - patch_size[0])
            j = np.random.randint(0, image_size - patch_size[1])
            patch = image[i:i+patch_size[0], j:j+patch_size[1]]
            patches.append(patch.flatten())
    
    return np.array(patches[:n_patches]).T


def main():
    """Run basic dictionary learning example"""
    
    print("🌟 Basic Dictionary Learning Example")
    print("===================================")
    
    # Parameters
    n_components = 64  # Number of dictionary atoms
    patch_size = (8, 8)  # 8x8 patches
    sparsity_penalty = 0.05  # L1 regularization
    n_patches = 5000  # Number of training patches
    
    print(f"Parameters:")
    print(f"  - Dictionary size: {n_components} atoms")
    print(f"  - Patch size: {patch_size}")
    print(f"  - Sparsity penalty: {sparsity_penalty}")
    print(f"  - Training patches: {n_patches}")
    
    # Generate training data
    print("\n📊 Generating training data...")
    patches = generate_natural_image_patches(n_patches, patch_size)
    print(f"Generated {patches.shape[1]} patches of dimension {patches.shape[0]}")
    
    # Create dictionary learner
    print("\n🔧 Initializing dictionary learner...")
    learner = DictionaryLearner(
        n_components=n_components,
        patch_size=patch_size,
        sparsity_penalty=sparsity_penalty,
        learning_rate=0.01,
        max_iterations=500,
        random_seed=42
    )
    
    # Train dictionary
    print("\n🚀 Training dictionary...")
    history = learner.fit(patches.T, verbose=True)
    
    # Visualize results
    print("\n📈 Creating visualizations...")
    
    # Plot dictionary atoms
    fig1 = plot_dictionary_atoms(
        learner.dictionary, 
        patch_size, 
        n_show=64,
        title=f"Learned Dictionary Atoms (Olshausen & Field Style)"
    )
    plt.show()
    
    # Plot training progress
    fig2 = plot_training_progress(history)
    plt.show()
    
    # Test feature extraction
    print("\n🎯 Testing feature extraction...")
    test_patches = generate_natural_image_patches(100, patch_size)
    features = learner.transform(test_patches.T)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Average sparsity: {np.mean(np.abs(features) > 1e-3):.3f}")
    
    # Show some example sparse codes
    fig3, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig3.suptitle("Example Sparse Codes")
    
    for i, ax in enumerate(axes.flat):
        if i < len(features):
            feature = features[i]
            active_indices = np.where(np.abs(feature) > 1e-3)[0]
            
            ax.bar(range(len(feature)), feature, alpha=0.7)
            ax.set_title(f"Patch {i} ({len(active_indices)}/{len(feature)} active)")
            ax.set_xlabel("Dictionary Index")
            ax.set_ylabel("Coefficient")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Dictionary learning example completed!")
    print(f"Final reconstruction error: {history['reconstruction_errors'][-1]:.6f}")
    print(f"Final sparsity level: {history['sparsity_levels'][-1]:.3f}")
    
    # Save results
    output_dir = Path("dictionary_learning_results")
    output_dir.mkdir(exist_ok=True)
    
    fig1.savefig(output_dir / "dictionary_atoms.png", dpi=150, bbox_inches='tight')
    fig2.savefig(output_dir / "training_progress.png", dpi=150, bbox_inches='tight')
    fig3.savefig(output_dir / "sparse_codes.png", dpi=150, bbox_inches='tight')
    
    print(f"Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()