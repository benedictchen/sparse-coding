"""
Reproduce Olshausen & Field (1996) Figure 4 results with paper-exact mode.

Usage:
    python examples/figure4_demo.py --images ./path/to/images --samples 50000 --out of_out

The script will:
* Sample 16x16 patches from natural images
* Apply zero-phase whitening R(f) = |f| * exp(-(f/f0)^4)  
* Train dictionary with paper-exact sparse coding
* Save: dictionary mosaic, coefficient histogram, reconstruction error curve
"""

import argparse
import os
import math
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sparse_coding import SparseCoder, OFigure4Preset, zero_phase_whiten

def load_grayscale_image(path):
    """Load image as grayscale array in [0,1]."""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=float) / 255.0

def sample_patches(images, patch_size, n_samples, rng):
    """Sample random patches from images."""
    H, W = images[0].shape
    patches = np.zeros((patch_size * patch_size, n_samples))
    
    for i in range(n_samples):
        # Random image and location
        img = images[rng.integers(0, len(images))]
        y = rng.integers(0, H - patch_size + 1)  
        x = rng.integers(0, W - patch_size + 1)
        
        # Extract patch
        patch = img[y:y+patch_size, x:x+patch_size]
        patches[:, i] = patch.reshape(-1)
    
    return patches

def whiten_images(images, f0):
    """Apply zero-phase whitening to each image."""
    return [zero_phase_whiten(img, f0=f0) for img in images]

def create_dictionary_mosaic(D, patch_size, cols=None):
    """Create mosaic visualization of dictionary atoms."""
    p, K = D.shape
    if cols is None:
        cols = int(math.ceil(math.sqrt(K)))
    rows = int(math.ceil(K / cols))
    
    # Create mosaic with spacing
    mosaic = np.zeros((rows * (patch_size + 1) - 1, cols * (patch_size + 1) - 1))
    
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= K:
                break
            
            # Get and normalize atom
            atom = D[:, k].reshape(patch_size, patch_size)
            atom_norm = (atom - atom.min()) / (atom.max() - atom.min() + 1e-12)
            
            # Place in mosaic
            y0 = r * (patch_size + 1)
            x0 = c * (patch_size + 1)
            mosaic[y0:y0+patch_size, x0:x0+patch_size] = atom_norm
            k += 1
    
    return mosaic

def compute_reconstruction_error_curve(coder, X, chunk_size=1000):
    """Compute reconstruction error on chunks of data."""
    N = X.shape[1]
    errors = []
    
    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        X_chunk = X[:, i:end_idx]
        
        # Encode and decode  
        A_chunk = coder.encode(X_chunk)
        X_hat_chunk = coder.decode(A_chunk)
        
        # Relative error
        error = np.linalg.norm(X_chunk - X_hat_chunk) / np.linalg.norm(X_chunk)
        errors.append(error)
    
    return errors

def main():
    parser = argparse.ArgumentParser(description="Reproduce Olshausen & Field Figure 4")
    parser.add_argument("--images", type=str, required=True, 
                       help="Folder containing natural images")
    parser.add_argument("--samples", type=int, default=50000,
                       help="Number of patches to sample") 
    parser.add_argument("--out", type=str, default="of_out",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Use research preset
    preset = OFigure4Preset()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Set random seed
    rng = np.random.default_rng(args.seed)
    
    # Load images
    image_files = sorted(glob.glob(os.path.join(args.images, "*")))
    if not image_files:
        raise FileNotFoundError(f"No images found in {args.images}")
    
    print(f"Loading {len(image_files)} images...")
    images = []
    for file in image_files:
        try:
            img = load_grayscale_image(file)
            images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
    
    if not images:
        raise RuntimeError("No valid images loaded")
    
    print(f"Loaded {len(images)} images successfully")
    
    # Apply image-level whitening (paper-accurate)
    print("Applying zero-phase whitening...")
    images_whitened = whiten_images(images, preset.f0)
    
    # Sample patches from whitened images
    print(f"Sampling {args.samples} patches of size {preset.patch_size}x{preset.patch_size}...")
    X = sample_patches(images_whitened, preset.patch_size, args.samples, rng)
    
    # Remove DC component (zero-mean patches)
    X = X - np.mean(X, axis=0, keepdims=True)
    
    print(f"Patch matrix shape: {X.shape}")
    print(f"Patch statistics: mean={np.mean(X):.6f}, std={np.std(X):.6f}")
    
    # Train sparse coder with paper-exact mode
    print("Training sparse coder (paper-exact mode)...")
    coder = SparseCoder(
        n_atoms=preset.n_atoms,
        mode='paper',
        ratio_lambda_over_sigma=preset.ratio_lambda_over_sigma,
        max_iter=preset.max_iter_inner,
        tol=preset.tol_inner,
        seed=args.seed
    )
    
    # Fit dictionary
    coder.fit(X, n_steps=preset.n_steps_outer, lr=preset.lr_dict)
    
    # Encode all patches
    print("Encoding patches...")
    A = coder.encode(X)
    
    # Compute diagnostics
    print("Computing diagnostics...")
    X_hat = coder.decode(A)
    sparsity = np.mean(np.abs(A) < 1e-8)
    recon_error = np.linalg.norm(X - X_hat) / np.linalg.norm(X)
    
    print(f"Final sparsity level: {sparsity:.3f}")
    print(f"Reconstruction error: {recon_error:.6f}")
    
    # Compute error curve  
    error_curve = compute_reconstruction_error_curve(coder, X, chunk_size=1000)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Dictionary mosaic
    mosaic = create_dictionary_mosaic(coder.D, preset.patch_size)
    
    # Save plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Dictionary mosaic
    ax1.imshow(mosaic, cmap='gray', interpolation='nearest')
    ax1.set_title(f'Learned Dictionary ({preset.n_atoms} atoms)')
    ax1.axis('off')
    
    # Coefficient histogram
    ax2.hist(A.flatten(), bins=200, density=True, alpha=0.7)
    ax2.set_title('Coefficient Distribution')
    ax2.set_xlabel('Coefficient Value')
    ax2.set_ylabel('Density')
    ax2.set_yscale('log')
    
    # Reconstruction error curve
    ax3.plot(error_curve, 'b-', marker='o', markersize=3)
    ax3.set_title('Reconstruction Error vs Batch')
    ax3.set_xlabel('Batch Index (1000 patches each)')
    ax3.set_ylabel('Relative Error')
    ax3.grid(True, alpha=0.3)
    
    # Sample reconstruction
    idx = rng.integers(0, min(16, X.shape[1]))
    original = X[:, idx].reshape(preset.patch_size, preset.patch_size)
    reconstructed = X_hat[:, idx].reshape(preset.patch_size, preset.patch_size)
    
    # Show side-by-side
    sample_vis = np.hstack([original, reconstructed])
    ax4.imshow(sample_vis, cmap='gray')
    ax4.set_title('Sample: Original | Reconstructed')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'figure4_results.png'), 
                dpi=200, bbox_inches='tight')
    plt.close()
    
    # Save data  
    np.save(os.path.join(args.out, 'dictionary.npy'), coder.D)
    np.save(os.path.join(args.out, 'coefficients.npy'), A)
    np.save(os.path.join(args.out, 'patches.npy'), X)
    
    # Save summary statistics
    stats = {
        'sparsity_level': float(sparsity),
        'reconstruction_error': float(recon_error),
        'n_atoms': preset.n_atoms,
        'patch_size': preset.patch_size,
        'n_samples': args.samples,
        'seed': args.seed,
        'mode': 'paper',
        'lambda_over_sigma': preset.ratio_lambda_over_sigma
    }
    
    import json
    with open(os.path.join(args.out, 'experiment_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nResults saved to: {args.out}/")
    print(f"  - figure4_results.png: Main visualization")  
    print(f"  - dictionary.npy: Learned basis functions")
    print(f"  - coefficients.npy: Sparse coefficients")
    print(f"  - patches.npy: Original patches")
    print(f"  - experiment_stats.json: Summary statistics")
    print(f"\nExperiment completed successfully!")

if __name__ == "__main__":
    main()