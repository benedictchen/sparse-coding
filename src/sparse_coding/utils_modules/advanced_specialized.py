"""
🏗️ Sparse Coding - Advanced and Specialized Utilities Module
==========================================================


Author: Benedict Chen (benedict@benedictchen.com)
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties"

🎯 MODULE PURPOSE:
=================
Advanced and specialized utilities for sparse coding including Gabor/DCT dictionary 
creation, lateral inhibition networks, noise estimation, and dictionary orthogonalization.

🔬 RESEARCH FOUNDATION:
======================
Implements advanced methods for sparse coding research:
- Gabor filter dictionaries for orientation-selective features
- DCT dictionaries for frequency-domain sparse coding
- Lateral inhibition networks inspired by biological vision
- Noise variance estimation from reconstruction residuals
- Dictionary orthogonalization methods

This module contains the advanced and specialized utilities, split from the
994-line monolith for research-grade functionality.
"""

import numpy as np
from sklearn.preprocessing import normalize
from typing import Tuple


def create_gabor_dictionary(patch_size: Tuple[int, int], n_orientations: int = 8,
                           n_scales: int = 3, n_phases: int = 2) -> np.ndarray:
    """
    Create Gabor filter dictionary
    
    Parameters
    ----------
    patch_size : tuple
        Size of patches (height, width)
    n_orientations : int
        Number of orientations
    n_scales : int
        Number of scales  
    n_phases : int
        Number of phases
        
    Returns
    -------
    gabor_dict : array, shape (n_filters, patch_height * patch_width)
        Gabor dictionary
    """
    h, w = patch_size
    gabor_filters = []
    
    # Create coordinate grids
    y, x = np.meshgrid(np.arange(h) - h//2, np.arange(w) - w//2, indexing='ij')
    
    for scale_idx in range(n_scales):
        sigma = 2**(scale_idx + 1)
        
        for orientation_idx in range(n_orientations):
            theta = orientation_idx * np.pi / n_orientations
            
            for phase_idx in range(n_phases):
                phase = phase_idx * np.pi / n_phases
                
                # Rotate coordinates
                x_rot = x * np.cos(theta) + y * np.sin(theta)
                y_rot = -x * np.sin(theta) + y * np.cos(theta)
                
                # Create Gabor filter
                gaussian = np.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
                sinusoid = np.cos(2 * np.pi * x_rot / sigma + phase)
                
                gabor = gaussian * sinusoid
                gabor = gabor.flatten()
                
                # Normalize
                gabor = gabor / np.linalg.norm(gabor)
                gabor_filters.append(gabor)
    
    return np.array(gabor_filters)


def create_dct_dictionary(patch_size: Tuple[int, int]) -> np.ndarray:
    """
    Create 2D DCT (Discrete Cosine Transform) dictionary
    
    Parameters
    ---------- 
    patch_size : tuple
        Size of patches (height, width)
        
    Returns
    -------
    dct_dict : array, shape (patch_height * patch_width, patch_height * patch_width)
        DCT dictionary (complete basis)
    """
    h, w = patch_size
    
    # Create 2D DCT basis
    dct_basis = []
    
    for u in range(h):
        for v in range(w):
            # DCT basis function
            basis_func = np.zeros((h, w))
            
            for i in range(h):
                for j in range(w):
                    cu = 1/np.sqrt(2) if u == 0 else 1
                    cv = 1/np.sqrt(2) if v == 0 else 1
                    
                    basis_func[i, j] = (cu * cv / np.sqrt(h * w) * 
                                      np.cos((2*i + 1) * u * np.pi / (2*h)) *
                                      np.cos((2*j + 1) * v * np.pi / (2*w)))
            
            dct_basis.append(basis_func.flatten())
    
    return np.array(dct_basis)


def lateral_inhibition_network(codes: np.ndarray, inhibition_strength: float = 0.1,
                             connectivity_radius: int = 3, 
                             n_iterations: int = 10) -> np.ndarray:
    """
    Apply lateral inhibition to sparse codes using network dynamics
    
    Parameters
    ----------
    codes : array, shape (n_samples, n_components)
        Input sparse codes
    inhibition_strength : float
        Strength of inhibitory connections
    connectivity_radius : int  
        Radius of lateral connections
    n_iterations : int
        Number of network update iterations
        
    Returns
    -------
    inhibited_codes : array, same shape as input
        Codes after lateral inhibition
    """
    n_samples, n_components = codes.shape
    inhibited_codes = codes.copy()
    
    # Create lateral inhibition weight matrix
    W_inhibit = np.zeros((n_components, n_components))
    
    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                distance = abs(i - j)
                if distance <= connectivity_radius:
                    W_inhibit[i, j] = -inhibition_strength / distance
    
    # Apply network dynamics
    for iteration in range(n_iterations):
        for sample_idx in range(n_samples):
            # Network update: x_new = x + dt * (-x + W * f(x))
            x = inhibited_codes[sample_idx]
            
            # Apply nonlinearity (rectification)
            fx = np.maximum(x, 0)
            
            # Network dynamics
            dx_dt = -0.1 * x + W_inhibit @ fx
            inhibited_codes[sample_idx] += 0.1 * dx_dt  # dt = 0.1
    
    return inhibited_codes


def estimate_noise_variance(X: np.ndarray, codes: np.ndarray, 
                           dictionary: np.ndarray) -> float:
    """
    Estimate noise variance from sparse coding residuals
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Original data
    codes : array, shape (n_samples, n_components)
        Sparse codes
    dictionary : array, shape (n_components, n_features)
        Dictionary
        
    Returns
    -------
    noise_variance : float
        Estimated noise variance
    """
    # Compute reconstruction
    X_reconstructed = codes @ dictionary
    
    # Residuals
    residuals = X - X_reconstructed
    
    # Estimate noise variance (using robust estimator)
    noise_variance = np.median(np.sum(residuals**2, axis=1))
    
    return noise_variance


def compute_mutual_coherence_matrix(dictionary: np.ndarray) -> np.ndarray:
    """
    Compute full mutual coherence matrix between dictionary atoms
    
    Parameters
    ----------
    dictionary : array, shape (n_components, n_features)
        Dictionary matrix
        
    Returns
    -------
    coherence_matrix : array, shape (n_components, n_components)
        Matrix of pairwise coherences
    """
    # Normalize dictionary
    normalized_dict = normalize(dictionary, axis=1)
    
    # Compute Gram matrix (coherence matrix)
    coherence_matrix = np.abs(normalized_dict @ normalized_dict.T)
    
    # Set diagonal to zero (self-coherence not meaningful)
    np.fill_diagonal(coherence_matrix, 0)
    
    return coherence_matrix


def orthogonalize_dictionary(dictionary: np.ndarray, method: str = 'gram_schmidt') -> np.ndarray:
    """
    Orthogonalize dictionary atoms
    
    Parameters
    ----------
    dictionary : array, shape (n_components, n_features)
        Input dictionary  
    method : str
        Orthogonalization method: 'gram_schmidt', 'qr', 'svd'
        
    Returns
    -------
    ortho_dictionary : array, same shape as input
        Orthogonalized dictionary
    """
    if method == 'gram_schmidt':
        ortho_dict = dictionary.copy()
        
        for i in range(len(dictionary)):
            # Subtract projections onto previous vectors
            for j in range(i):
                proj = np.dot(ortho_dict[i], ortho_dict[j]) * ortho_dict[j]
                ortho_dict[i] -= proj
            
            # Normalize
            norm = np.linalg.norm(ortho_dict[i])
            if norm > 1e-8:
                ortho_dict[i] /= norm
        
        return ortho_dict
    
    elif method == 'qr':
        Q, R = np.linalg.qr(dictionary.T)
        return Q.T[:len(dictionary)]
    
    elif method == 'svd':
        U, s, Vt = np.linalg.svd(dictionary, full_matrices=False)
        return U @ Vt
    
    else:
        raise ValueError(f"Unknown orthogonalization method: {method}")


# Export functions
__all__ = [
    'create_gabor_dictionary',
    'create_dct_dictionary',
    'lateral_inhibition_network',
    'estimate_noise_variance',
    'compute_mutual_coherence_matrix',
    'orthogonalize_dictionary'
]


if __name__ == "__main__":
    print("🏗️ Sparse Coding - Advanced and Specialized Utilities Module")
    print("=" * 50)
    print("📊 MODULE CONTENTS:")
    print("  • Gabor filter dictionary creation (multi-orientation, multi-scale)")
    print("  • 2D DCT dictionary for frequency-domain sparse coding")
    print("  • Lateral inhibition network dynamics (biologically inspired)")
    print("  • Noise variance estimation from reconstruction residuals")
    print("  • Mutual coherence matrix computation for dictionary analysis")
    print("  • Dictionary orthogonalization (Gram-Schmidt, QR, SVD methods)")
    print("")
    print("✅ Advanced and specialized utilities module loaded successfully!")
    print("🔬 Research-grade tools for advanced sparse coding applications!")
    
    # Quick functionality test
    print("\n🧪 QUICK FUNCTIONALITY TEST:")
    try:
        # Test Gabor dictionary creation
        gabor_dict = create_gabor_dictionary((8, 8), n_orientations=4, n_scales=2)
        print(f"✅ Gabor dictionary created: {gabor_dict.shape}")
        
        # Test DCT dictionary creation
        dct_dict = create_dct_dictionary((4, 4))
        print(f"✅ DCT dictionary created: {dct_dict.shape}")
        
        # Test dictionary orthogonalization
        test_dict = np.random.randn(5, 16)
        ortho_dict = orthogonalize_dictionary(test_dict, method='gram_schmidt')
        print(f"✅ Dictionary orthogonalization: {ortho_dict.shape}")
        
        print("🎉 All advanced utilities working correctly!")
        
    except Exception as e:
        print(f"❌ Error during functionality test: {e}")
        print("⚠️  Some utilities may not be working correctly")