"""
🏗️ Sparse Coding - Data Processing Utilities Module
==================================================


Author: Benedict Chen (benedict@benedictchen.com)
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties"

🎯 MODULE PURPOSE:
=================
Data processing utilities for sparse coding including patch extraction,
normalization, whitening, and image reconstruction from patches.

🔬 RESEARCH FOUNDATION:
======================
Implements data preprocessing methods for sparse coding applications:
- Olshausen & Field (1996): Patch extraction from natural images
- Modern computer vision: ZCA whitening and patch normalization
- Image processing: Overlapping patch reconstruction methods

This module contains the data processing components, split from the
994-line monolith for specialized preprocessing functionality.
"""

import numpy as np
from typing import Tuple, Optional, List
import warnings


def extract_patches_2d(image: np.ndarray, patch_size: Tuple[int, int], 
                      max_patches: Optional[int] = None, 
                      random_state: Optional[int] = None) -> np.ndarray:
    """
    Extract random 2D patches from image
    
    Parameters
    ----------
    image : array-like, shape (height, width)
        Input image
    patch_size : tuple
        (patch_height, patch_width)
    max_patches : int, optional
        Maximum number of patches to extract
    random_state : int, optional
        Random seed
        
    Returns
    -------
    patches : array, shape (n_patches, patch_height * patch_width)
        Extracted and flattened patches
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    h, w = image.shape
    ph, pw = patch_size
    
    if ph > h or pw > w:
        raise ValueError(f"Patch size {patch_size} larger than image {(h, w)}")
    
    # Calculate maximum possible patches
    max_possible = (h - ph + 1) * (w - pw + 1)
    
    if max_patches is None or max_patches > max_possible:
        n_patches = max_possible
        # Extract all patches systematically
        patches = []
        for y in range(h - ph + 1):
            for x in range(w - pw + 1):
                patch = image[y:y+ph, x:x+pw].flatten()
                patches.append(patch)
    else:
        n_patches = max_patches
        # Extract random patches
        patches = []
        for _ in range(n_patches):
            y = np.random.randint(0, h - ph + 1)
            x = np.random.randint(0, w - pw + 1)
            patch = image[y:y+ph, x:x+pw].flatten()
            patches.append(patch)
    
    return np.array(patches, dtype=np.float32)


def extract_patches_from_images(images: List[np.ndarray], patch_size: Tuple[int, int],
                               patches_per_image: int = 100,
                               normalize_patches: bool = True,
                               random_state: Optional[int] = None) -> np.ndarray:
    """
    Extract patches from multiple images
    
    Parameters
    ----------
    images : list of arrays
        List of input images
    patch_size : tuple
        Size of patches to extract
    patches_per_image : int
        Number of patches per image
    normalize_patches : bool
        Whether to normalize patches (subtract mean)
    random_state : int, optional
        Random seed
        
    Returns
    -------
    patches : array, shape (n_images * patches_per_image, patch_dim)
        All extracted patches
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    all_patches = []
    
    for i, image in enumerate(images):
        # Convert to grayscale if needed
        if image.ndim == 3:
            image = np.mean(image, axis=2)
        
        # Extract patches from this image
        patches = extract_patches_2d(image, patch_size, patches_per_image, 
                                   random_state=random_state + i if random_state else None)
        
        # Normalize patches if requested
        if normalize_patches:
            patches = normalize_patch_batch(patches)
        
        all_patches.append(patches)
    
    return np.vstack(all_patches)


def normalize_patch_batch(patches: np.ndarray, method: str = 'subtract_mean') -> np.ndarray:
    """
    Normalize a batch of patches
    
    Parameters
    ----------
    patches : array, shape (n_patches, patch_dim)
        Input patches
    method : str
        Normalization method: 'subtract_mean', 'unit_variance', 'unit_norm', 'whiten'
        
    Returns
    -------
    normalized_patches : array, same shape as input
        Normalized patches
    """
    if method == 'subtract_mean':
        # Subtract mean from each patch
        means = np.mean(patches, axis=1, keepdims=True)
        return patches - means
    
    elif method == 'unit_variance':
        # Zero mean, unit variance
        means = np.mean(patches, axis=1, keepdims=True)
        stds = np.std(patches, axis=1, keepdims=True) + 1e-8
        return (patches - means) / stds
    
    elif method == 'unit_norm':
        # L2 normalize each patch
        norms = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8
        return patches / norms
    
    elif method == 'whiten':
        # ZCA whitening
        return whiten_patches(patches)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def whiten_patches(patches: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Apply ZCA whitening to patches
    
    Parameters
    ----------
    patches : array, shape (n_patches, patch_dim)
        Input patches
    epsilon : float
        Regularization parameter
        
    Returns
    -------
    whitened_patches : array, same shape as input
        Whitened patches
    """
    # Center the data
    mean = np.mean(patches, axis=0)
    centered = patches - mean
    
    # Compute covariance
    cov = np.cov(centered.T)
    
    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Whitening transform
    whitening_matrix = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals + epsilon)) @ eigenvecs.T
    
    return centered @ whitening_matrix.T


def reconstruct_image_from_patches(patches: np.ndarray, 
                                  image_shape: Tuple[int, int],
                                  patch_size: Tuple[int, int],
                                  overlap_method: str = 'average') -> np.ndarray:
    """
    Reconstruct image from overlapping patches
    
    Parameters
    ----------
    patches : array, shape (n_patches, patch_dim)
        Flattened patches
    image_shape : tuple
        (height, width) of target image
    patch_size : tuple
        (patch_height, patch_width)
    overlap_method : str
        How to handle overlaps: 'average', 'first', 'last'
        
    Returns
    -------
    reconstructed : array, shape image_shape
        Reconstructed image
    """
    h, w = image_shape
    ph, pw = patch_size
    
    reconstructed = np.zeros(image_shape)
    counts = np.zeros(image_shape)
    
    patch_idx = 0
    
    # Place patches back
    for y in range(h - ph + 1):
        for x in range(w - pw + 1):
            if patch_idx < len(patches):
                patch = patches[patch_idx].reshape(patch_size)
                
                if overlap_method == 'average':
                    reconstructed[y:y+ph, x:x+pw] += patch
                    counts[y:y+ph, x:x+pw] += 1
                elif overlap_method == 'first':
                    mask = counts[y:y+ph, x:x+pw] == 0
                    reconstructed[y:y+ph, x:x+pw][mask] = patch[mask]
                    counts[y:y+ph, x:x+pw] += 1
                else:  # last
                    reconstructed[y:y+ph, x:x+pw] = patch
                    counts[y:y+ph, x:x+pw] += 1
                
                patch_idx += 1
    
    # Average overlapping regions
    if overlap_method == 'average':
        counts[counts == 0] = 1  # Avoid division by zero
        reconstructed /= counts
    
    return reconstructed


# Export functions
__all__ = [
    'extract_patches_2d',
    'extract_patches_from_images', 
    'normalize_patch_batch',
    'whiten_patches',
    'reconstruct_image_from_patches'
]


if __name__ == "__main__":
    print("🏗️ Sparse Coding - Data Processing Utilities Module")
    print("=" * 50)
    print("📊 MODULE CONTENTS:")
    print("  • Patch extraction from images (2D and multi-image)")
    print("  • Patch normalization (mean subtraction, unit variance, unit norm)")
    print("  • ZCA whitening for preprocessing")
    print("  • Image reconstruction from overlapping patches")
    print("  • Research-accurate data processing for sparse coding")
    print("")
    print("✅ Data processing utilities module loaded successfully!")
    print("🔬 Essential preprocessing for sparse coding algorithms!")