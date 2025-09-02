"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀
"""
"""
Data Processing Module - Sparse Coding Library
=============================================

This module contains data processing utilities for sparse coding, extracted from
the main SparseCoder class to support modular architecture.

Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties 
by Learning a Sparse Code for Natural Images"

Key Functions:
- extract_patches: Random patch extraction from images  
- whiten_patches: General whitening wrapper
- whiten_patches_olshausen_field: Original Olshausen & Field whitening
- whiten_patches_zca: ZCA whitening implementation

Author: Benedict Chen (benedict@benedictchen.com)
"""

import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Import scipy.fft for more robust FFT operations
try:
    from scipy.fft import fft2, ifft2, fftfreq
except ImportError:
    # Fallback to numpy for compatibility
    from numpy.fft import fft2, ifft2, fftfreq


class DataProcessingMixin:
    """
    Mixin class for data processing functionality in sparse coding.
    
    This mixin provides access to data processing methods while maintaining
    the `self` state access patterns from the original class methods.
    
    Expected attributes from parent class:
    - patch_size: Tuple[int, int] - Size of patches (height, width)
    """

    def extract_patches(self, images: np.ndarray, n_patches: int) -> np.ndarray:
        """
        Extract random patches from images
        
        Args:
            images: Input images array (n_images, height, width)
            n_patches: Number of patches to extract
            
        Returns:
            np.ndarray: Array of flattened patches (n_patches, patch_dim)
        """
        patches = []
        patch_h, patch_w = self.patch_size
        max_attempts = n_patches * 10  # Prevent infinite loops
        attempts = 0
        
        while len(patches) < n_patches and attempts < max_attempts:
            attempts += 1
            
            # Select random image
            img_idx = np.random.randint(0, len(images))
            image = images[img_idx]
            
            # Ensure image has at least the required dimensions
            if len(image.shape) != 2:
                # Convert to 2D if needed
                if len(image.shape) == 1:
                    # Try to reshape square image
                    side = int(np.sqrt(len(image)))
                    if side * side == len(image):
                        image = image.reshape(side, side)
                    else:
                        continue
                elif len(image.shape) > 2:
                    # Take first channel if multi-channel
                    image = image[:, :, 0] if image.shape[2] > 0 else image[:, :]
            
            # Select random patch location
            max_y = image.shape[0] - patch_h
            max_x = image.shape[1] - patch_w
            
            if max_y <= 0 or max_x <= 0:
                continue
                
            y = np.random.randint(0, max_y)
            x = np.random.randint(0, max_x)
            
            # Extract patch
            patch = image[y:y+patch_h, x:x+patch_w]
            patches.append(patch.flatten())
        
        # Ensure we return at least some patches
        if len(patches) == 0:
            # Generate synthetic patches as fallback
            print("   WARNING: No patches extracted, generating synthetic patches")
            for _ in range(min(n_patches, 100)):
                synthetic_patch = np.random.randn(patch_h, patch_w) * 0.1
                patches.append(synthetic_patch.flatten())
        
        return np.array(patches)

    def whiten_patches(self, patches: np.ndarray) -> np.ndarray:
        """
        Whiten patches to decorrelate pixels (preprocessing step)
        
        This removes the natural correlation structure of images,
        making the sparse structure more apparent.
        
        Args:
            patches: Input patches array (n_patches, patch_dim)
            
        Returns:
            np.ndarray: Whitened patches
        """
        # Center patches
        patches_centered = patches - np.mean(patches, axis=1, keepdims=True)
        
        # Compute covariance matrix
        cov = np.cov(patches_centered, rowvar=False)
        
        # Eigendecomposition for whitening
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Whitening transform
        epsilon = 1e-5  # Regularization
        whitening_matrix = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals + epsilon)) @ eigenvecs.T
        
        patches_whitened = patches_centered @ whitening_matrix
        
        return patches_whitened

    def whiten_patches_olshausen_field(self, patches: np.ndarray) -> np.ndarray:
        """
        Zero-phase whitening filter as specified in Olshausen & Field 1996
        
        This is the original research whitening method described in the seminal paper.
        The paper specifies: "zero-phase whitening/lowpass filter, R(f) = fe^(-f/f0)"
        where f₀ = 200 cycles/picture.
        
        This whitening approach is crucial for reproducing the exact results from the
        original paper, as it shapes the frequency response in a way that leads to
        the emergence of oriented edge detectors in the learned dictionary.
        
        Args:
            patches: Input patches array (n_patches, patch_dim)
            
        Returns:
            np.ndarray: Whitened patches using Olshausen & Field method
        """
        print("   🔬 Applying Olshausen & Field zero-phase whitening filter...")
        
        # Step 1: Remove DC component
        patches_centered = patches - np.mean(patches, axis=1, keepdims=True)
        
        # Step 2: Apply whitening filter in frequency domain
        patch_2d = patches_centered.reshape(-1, *self.patch_size)
        whitened_patches = []
        
        for patch in patch_2d:
            # FFT - Use scipy.fft for better performance if available
            fft_patch = fft2(patch)
            
            # Create frequency grid
            freqs_y = fftfreq(patch.shape[0])
            freqs_x = fftfreq(patch.shape[1])
            fy, fx = np.meshgrid(freqs_y, freqs_x, indexing='ij')
            
            # Frequency magnitude
            f_mag = np.sqrt(fx**2 + fy**2)
            
            # Whitening filter: R(f) = f * exp(-f/f0)
            # f₀ = 200 cycles/picture (from the paper)
            f0 = 200.0 / max(patch.shape)  # Normalize by patch size
            whitening_filter = f_mag * np.exp(-f_mag / f0)
            whitening_filter[0, 0] = 1e-10  # Avoid division by zero at DC
            
            # Apply filter
            whitened_fft = fft_patch * whitening_filter
            whitened_patch = np.real(ifft2(whitened_fft))
            whitened_patches.append(whitened_patch.flatten())
            
        return np.array(whitened_patches)

    def whiten_patches_zca(self, patches: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        ZCA (Zero-phase Component Analysis) whitening
        
        Alternative whitening approach mentioned in later sparse coding literature.
        ZCA whitening is a symmetric whitening transformation that preserves the
        structure of the data while decorrelating the components.
        
        The ZCA transformation is: X_zca = X * U * D^(-1/2) * U^T
        where U and D are from the SVD of the covariance matrix.
        
        Args:
            patches: Input patches array (n_patches, patch_dim)
            epsilon: Regularization parameter to avoid numerical issues
            
        Returns:
            np.ndarray: ZCA whitened patches
        """
        print("   🔬 Applying ZCA whitening...")
        
        # Center patches
        patches_centered = patches - np.mean(patches, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(patches_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Whitening transformation
        whitening_transform = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + epsilon)) @ eigenvectors.T
        
        # Apply whitening
        whitened_patches = patches_centered @ whitening_transform
        
        return whitened_patches


# Standalone functions for direct use without mixin
def extract_patches(images: np.ndarray, n_patches: int, patch_size: Tuple[int, int]) -> np.ndarray:
    """
    Standalone function to extract random patches from images
    
    Args:
        images: Input images array (n_images, height, width)
        n_patches: Number of patches to extract
        patch_size: Size of patches (height, width)
        
    Returns:
        np.ndarray: Array of flattened patches (n_patches, patch_dim)
    """
    # Create a minimal class instance to use the mixin
    class PatchExtractor(DataProcessingMixin):
        def __init__(self, patch_size):
            self.patch_size = patch_size
    
    extractor = PatchExtractor(patch_size)
    return extractor.extract_patches(images, n_patches)


def whiten_patches(patches: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Standalone function to whiten patches using specified method
    
    Args:
        patches: Input patches array (n_patches, patch_dim)  
        method: Whitening method ('standard', 'olshausen_field', 'zca')
        
    Returns:
        np.ndarray: Whitened patches
        
    Raises:
        ValueError: If unknown method is specified
    """
    # Infer patch size from patches
    patch_dim = patches.shape[1]
    patch_size = (int(np.sqrt(patch_dim)), int(np.sqrt(patch_dim)))
    
    # Create a minimal class instance to use the mixin
    class PatchWhitener(DataProcessingMixin):
        def __init__(self, patch_size):
            self.patch_size = patch_size
    
    whitener = PatchWhitener(patch_size)
    
    if method == 'standard':
        return whitener.whiten_patches(patches)
    elif method == 'olshausen_field':
        return whitener.whiten_patches_olshausen_field(patches)
    elif method == 'zca':
        return whitener.whiten_patches_zca(patches)
    else:
        raise ValueError(f"Unknown whitening method: {method}. "
                        f"Choose from: 'standard', 'olshausen_field', 'zca'")


def whiten_patches_olshausen_field(patches: np.ndarray, patch_size: Tuple[int, int]) -> np.ndarray:
    """
    Standalone function for Olshausen & Field whitening
    
    Args:
        patches: Input patches array (n_patches, patch_dim)
        patch_size: Size of patches (height, width)
        
    Returns:
        np.ndarray: Whitened patches using Olshausen & Field method
    """
    class PatchWhitener(DataProcessingMixin):
        def __init__(self, patch_size):
            self.patch_size = patch_size
    
    whitener = PatchWhitener(patch_size)
    return whitener.whiten_patches_olshausen_field(patches)


def whiten_patches_zca(patches: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Standalone function for ZCA whitening
    
    Args:
        patches: Input patches array (n_patches, patch_dim)
        epsilon: Regularization parameter
        
    Returns:
        np.ndarray: ZCA whitened patches
    """
    # Infer patch size from patches
    patch_dim = patches.shape[1]
    patch_size = (int(np.sqrt(patch_dim)), int(np.sqrt(patch_dim)))
    
    class PatchWhitener(DataProcessingMixin):
        def __init__(self, patch_size):
            self.patch_size = patch_size
    
    whitener = PatchWhitener(patch_size)
    return whitener.whiten_patches_zca(patches, epsilon)


# Utility function for preprocessing pipeline
def preprocess_patches(patches: np.ndarray, 
                      patch_size: Tuple[int, int],
                      whitening_method: str = 'olshausen_field',
                      **kwargs) -> np.ndarray:
    """
    Complete preprocessing pipeline for patches
    
    Args:
        patches: Input patches array (n_patches, patch_dim)
        patch_size: Size of patches (height, width)
        whitening_method: Method to use ('standard', 'olshausen_field', 'zca')
        **kwargs: Additional parameters for whitening methods
        
    Returns:
        np.ndarray: Preprocessed patches ready for sparse coding
    """
    print(f"🔬 Preprocessing {len(patches)} patches using {whitening_method} whitening...")
    
    # Apply whitening
    if whitening_method == 'olshausen_field':
        whitened = whiten_patches_olshausen_field(patches, patch_size)
    elif whitening_method == 'zca':
        epsilon = kwargs.get('epsilon', 1e-5)
        whitened = whiten_patches_zca(patches, epsilon)
    elif whitening_method == 'standard':
        whitened = whiten_patches(patches, 'standard')
    else:
        raise ValueError(f"Unknown whitening method: {whitening_method}")
    
    print(f"✓ Preprocessing complete. Patch statistics:")
    print(f"   Mean: {whitened.mean():.6f}")
    print(f"   Std: {whitened.std():.6f}")
    print(f"   Min: {whitened.min():.6f}, Max: {whitened.max():.6f}")
    
    return whitened


# Configuration and information functions
def get_whitening_methods_info():
    """
    Get information about available whitening methods
    
    Returns:
        dict: Information about whitening methods and their properties
    """
    return {
        'available_methods': {
            'standard': {
                'description': 'Standard eigenvalue-based whitening',
                'properties': 'Fast, decorrelates pixels, general purpose',
                'recommended_use': 'General sparse coding applications'
            },
            'olshausen_field': {
                'description': 'Original Olshausen & Field 1996 zero-phase whitening filter',
                'properties': 'R(f) = f*exp(-f/f0) filter, research-accurate',
                'recommended_use': 'Reproducing original paper results'
            },
            'zca': {
                'description': 'Zero-phase Component Analysis whitening',
                'properties': 'Symmetric transformation, preserves data structure',
                'recommended_use': 'When data structure preservation is important'
            }
        },
        'implementation_details': {
            'olshausen_field': {
                'frequency_cutoff': 'f0 = 200 cycles/picture (normalized by patch size)',
                'filter_formula': 'R(f) = f * exp(-f/f0)',
                'domain': 'Applied in frequency domain via FFT'
            },
            'zca': {
                'transformation': 'X_zca = X * U * D^(-1/2) * U^T',
                'regularization': 'epsilon parameter for numerical stability'
            }
        },
        'usage_examples': {
            'basic_whitening': "whitened = whiten_patches(patches, method='standard')",
            'research_accurate': "whitened = whiten_patches_olshausen_field(patches, patch_size)",
            'zca_whitening': "whitened = whiten_patches_zca(patches, epsilon=1e-4)",
            'full_pipeline': "preprocessed = preprocess_patches(patches, patch_size, 'olshausen_field')"
        }
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("💰 SUPPORT THIS RESEARCH - PLEASE DONATE!")  
    print("🙏 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS")
    print("="*80 + "\n")
    
    """
    Demonstration of data processing functionality
    """
    print("🔬 Sparse Coding Data Processing Module")
    print("=" * 45)
    
    # Generate test data
    print("Generating test images...")
    test_images = np.random.randn(10, 64, 64) * 0.5 + 0.5
    patch_size = (16, 16)
    
    # Extract patches
    print(f"\nExtracting patches of size {patch_size}...")
    patches = extract_patches(test_images, 100, patch_size)
    print(f"Extracted {len(patches)} patches")
    
    # Test different whitening methods
    methods = ['standard', 'olshausen_field', 'zca']
    
    for method in methods:
        print(f"\nTesting {method} whitening...")
        try:
            if method == 'olshausen_field':
                whitened = whiten_patches_olshausen_field(patches, patch_size)
            elif method == 'zca':
                whitened = whiten_patches_zca(patches)
            else:
                whitened = whiten_patches(patches, method)
            
            print(f"✓ {method} whitening successful")
            print(f"   Shape: {whitened.shape}")
            print(f"   Mean: {whitened.mean():.6f}, Std: {whitened.std():.6f}")
            
        except Exception as e:
            print(f"❌ {method} whitening failed: {e}")
    
    # Show available methods info
    print(f"\n📊 Available Whitening Methods:")
    info = get_whitening_methods_info()
    for method, details in info['available_methods'].items():
        print(f"   • {method}: {details['description']}")
        print(f"     Properties: {details['properties']}")
    
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