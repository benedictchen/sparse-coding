"""
Dictionary Learning Implementation
Based on: Olshausen & Field (1996) online dictionary learning algorithm

Implements adaptive dictionary learning for sparse coding where both
the sparse codes and dictionary atoms are learned simultaneously.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from .sparse_coder import SparseCoder


class DictionaryLearner:
    """
    Dictionary Learning for Sparse Coding
    
    Learns both the dictionary D and sparse codes α simultaneously:
    min_{D,α} ||X - Dα||_2^2 + λ||α||_1
    
    Uses alternating optimization between dictionary update and sparse coding.
    """
    
    def __init__(
        self,
        n_components: int = 100,
        patch_size: Tuple[int, int] = (8, 8),
        sparsity_penalty: float = 0.1,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Dictionary Learner
        
        Args:
            n_components: Number of dictionary atoms
            patch_size: Size of image patches
            sparsity_penalty: L1 regularization parameter
            learning_rate: Dictionary update learning rate
            max_iterations: Maximum training iterations
            tolerance: Convergence tolerance
            random_seed: Random seed for reproducibility
        """
        
        self.n_components = n_components
        self.patch_size = patch_size
        self.patch_dim = patch_size[0] * patch_size[1]
        self.sparsity_penalty = sparsity_penalty
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Initialize dictionary randomly
        self.dictionary = np.random.randn(self.patch_dim, n_components)
        self._normalize_dictionary()
        
        # Initialize sparse coder
        self.sparse_coder = SparseCoder(
            dictionary=self.dictionary,
            sparsity_penalty=sparsity_penalty
        )
        
        # Training history
        self.training_history = {
            'reconstruction_errors': [],
            'sparsity_levels': [],
            'dictionary_changes': []
        }
        
    def _normalize_dictionary(self):
        """Normalize dictionary atoms to unit norm"""
        norms = np.linalg.norm(self.dictionary, axis=0)
        norms[norms == 0] = 1  # Avoid division by zero
        self.dictionary = self.dictionary / norms[np.newaxis, :]
        
    def _extract_patches(self, images: np.ndarray) -> np.ndarray:
        """Extract patches from images"""
        
        if len(images.shape) == 2:
            images = images[np.newaxis, :, :]
            
        patches = []
        patch_h, patch_w = self.patch_size
        
        for image in images:
            h, w = image.shape
            for i in range(0, h - patch_h + 1, patch_h // 2):
                for j in range(0, w - patch_w + 1, patch_w // 2):
                    patch = image[i:i+patch_h, j:j+patch_w]
                    patches.append(patch.flatten())
                    
        return np.array(patches)
        
    def _update_dictionary(self, patches: np.ndarray, codes: np.ndarray) -> float:
        """Update dictionary using gradient descent"""
        
        old_dict = self.dictionary.copy()
        
        # Compute reconstruction error gradient
        reconstruction = self.dictionary @ codes.T
        error = patches.T - reconstruction
        
        # Dictionary gradient: -2 * error * codes^T
        gradient = -2 * error @ codes / len(patches)
        
        # Update dictionary
        self.dictionary += self.learning_rate * gradient
        
        # Normalize atoms
        self._normalize_dictionary()
        
        # Update sparse coder dictionary
        self.sparse_coder.dictionary = self.dictionary
        
        # Return change magnitude
        change = np.linalg.norm(self.dictionary - old_dict)
        return change
        
    def fit(self, images: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        """
        Learn dictionary from image data
        
        Args:
            images: Training images (n_images, height, width) or (height, width)
            verbose: Whether to print progress
            
        Returns:
            Training statistics
        """
        
        # Extract patches
        patches = self._extract_patches(images)
        n_patches = len(patches)
        
        if verbose:
            print(f"🎯 Learning dictionary from {n_patches} patches...")
            print(f"   Patch size: {self.patch_size}")
            print(f"   Dictionary size: {self.n_components} atoms")
            
        prev_error = float('inf')
        
        for iteration in range(self.max_iterations):
            
            # Step 1: Sparse coding (fix dictionary, optimize codes)
            codes = np.array([
                self.sparse_coder.encode_patch(patch) 
                for patch in patches
            ])
            
            # Step 2: Dictionary update (fix codes, optimize dictionary)
            dict_change = self._update_dictionary(patches, codes)
            
            # Calculate metrics
            reconstruction = self.dictionary @ codes.T
            recon_error = np.mean((patches.T - reconstruction) ** 2)
            sparsity = np.mean(np.sum(np.abs(codes) > 1e-6, axis=1))
            
            # Store history
            self.training_history['reconstruction_errors'].append(recon_error)
            self.training_history['sparsity_levels'].append(sparsity)
            self.training_history['dictionary_changes'].append(dict_change)
            
            # Check convergence
            if abs(prev_error - recon_error) < self.tolerance:
                if verbose:
                    print(f"   Converged at iteration {iteration+1}")
                break
                
            prev_error = recon_error
            
            # Progress update
            if verbose and (iteration + 1) % (self.max_iterations // 10) == 0:
                progress = (iteration + 1) / self.max_iterations * 100
                print(f"   Progress: {progress:5.1f}% | Error: {recon_error:.6f} | Sparsity: {sparsity:.1f} | Dict Change: {dict_change:.6f}")
                
        results = {
            'final_reconstruction_error': recon_error,
            'final_sparsity': sparsity,
            'final_dictionary_change': dict_change,
            'n_iterations': iteration + 1,
            'converged': iteration < self.max_iterations - 1
        }
        
        if verbose:
            print(f"✅ Dictionary learning complete!")
            print(f"   Final reconstruction error: {recon_error:.6f}")
            print(f"   Final sparsity level: {sparsity:.1f}")
            
        return results
        
    def get_dictionary(self) -> np.ndarray:
        """Get learned dictionary"""
        return self.dictionary.copy()
        
    def get_dictionary_images(self) -> np.ndarray:
        """Get dictionary atoms reshaped as images"""
        return self.dictionary.T.reshape(-1, *self.patch_size)
        
    def transform(self, images: np.ndarray) -> np.ndarray:
        """Transform images to sparse codes using learned dictionary"""
        patches = self._extract_patches(images)
        codes = np.array([
            self.sparse_coder.encode_patch(patch) 
            for patch in patches
        ])
        return codes
        
    def reconstruct(self, images: np.ndarray) -> np.ndarray:
        """
        🔄 Reconstruct Images from Sparse Representation - Olshausen & Field 1996!
        
        Reconstructs images from sparse codes using proper overlapping patch
        averaging to handle patch boundaries correctly.
        
        Args:
            images: Input images to reconstruct [n_images, height, width] or [height, width]
            
        Returns:
            Reconstructed images with same shape as input
            
        📚 **Reference**: Olshausen, B. A., & Field, D. J. (1996). 
        "Emergence of simple-cell receptive field properties by learning a sparse code"
        
        🎆 **Proper Reconstruction Process**:
        1. Transform images to sparse codes
        2. Reconstruct overlapping patches from dictionary
        3. Average overlapping regions for seamless reconstruction
        4. Handle boundary effects with proper normalization
        
        📊 **Quality Metrics**:
        - Maintains spatial continuity across patch boundaries
        - Preserves fine details through sparse representation
        - Minimizes reconstruction artifacts
        """
        # Handle single image case
        single_image = images.ndim == 2
        if single_image:
            images = images[np.newaxis]  # Add batch dimension
            
        reconstructed_images = []
        
        for img in images:
            # Extract patches and get sparse codes
            patches = self._extract_patches(img[np.newaxis])
            codes = np.array([
                self.sparse_coder.encode_patch(patch) 
                for patch in patches
            ])
            
            # Reconstruct patches
            reconstructed_patches = (self.dictionary @ codes.T).T
            
            # Proper patch-to-image reconstruction with overlap averaging
            reconstructed_img = self._reconstruct_from_patches(
                reconstructed_patches, img.shape, self.patch_size, self.stride
            )
            reconstructed_images.append(reconstructed_img)
            
        result = np.array(reconstructed_images)
        return result[0] if single_image else result
    
    def _reconstruct_from_patches(self, patches: np.ndarray, image_shape: tuple, 
                                patch_size: tuple, stride: int) -> np.ndarray:
        """
        🎨 Reconstruct Image from Overlapping Patches - Proper Averaging!
        
        Reconstructs an image from patches using proper overlapping patch
        averaging to ensure seamless reconstruction without artifacts.
        
        Args:
            patches: Reconstructed patches [n_patches, patch_height * patch_width]
            image_shape: Target image shape (height, width)
            patch_size: Size of each patch (height, width)
            stride: Stride between patch centers
            
        Returns:
            Reconstructed image with proper overlap averaging
            
        📊 **Averaging Algorithm**:
        1. Accumulate pixel values from all overlapping patches
        2. Track overlap count for each pixel
        3. Average by dividing accumulated values by overlap counts
        4. Handle edge cases with proper normalization
        """
        height, width = image_shape
        patch_h, patch_w = patch_size
        
        # Initialize accumulation arrays
        reconstructed = np.zeros((height, width))
        overlap_count = np.zeros((height, width))
        
        patch_idx = 0
        
        # Iterate through all patch positions
        for y in range(0, height - patch_h + 1, stride):
            for x in range(0, width - patch_w + 1, stride):
                if patch_idx >= len(patches):
                    break
                    
                # Reshape patch back to 2D
                patch = patches[patch_idx].reshape(patch_h, patch_w)
                
                # Add patch to accumulation
                reconstructed[y:y+patch_h, x:x+patch_w] += patch
                overlap_count[y:y+patch_h, x:x+patch_w] += 1
                
                patch_idx += 1
        
        # Average overlapping regions
        # Avoid division by zero
        overlap_count[overlap_count == 0] = 1
        reconstructed = reconstructed / overlap_count
        
        return reconstructed