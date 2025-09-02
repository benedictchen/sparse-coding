"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀

📚 Dictionary Learning - Teaching AI to See Like Babies Do
========================================================

Author: Benedict Chen (benedict@benedictchen.com)

📚 Research Paper:
Olshausen, B. A., & Field, D. J. (1996)
"Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"
Nature, 381(6583), 607-609

Additional Foundation:
Lee, H., Battle, A., Raina, R., & Ng, A. Y. (2007)
"Efficient sparse coding algorithms"
Advances in Neural Information Processing Systems, 19, 801-808

🎯 ELI5 Summary:
Imagine a baby seeing the world for the first time - they don't know what "edges," 
"corners," or "textures" are yet. Dictionary learning is like the baby's brain 
automatically discovering these basic building blocks by looking at lots of images! 
The "dictionary" is like a vocabulary of visual patterns (edges, spots, textures), 
and the baby learns to describe any image using just a few words from this dictionary. 
This is exactly how your visual cortex learned to see when you were a baby!

🧪 Research Background:
Olshausen & Field's revolutionary discovery: if you show an algorithm natural images
and ask it to find the most efficient way to represent them, it automatically discovers
edge detectors that look exactly like neurons in the primary visual cortex (V1)!

Key Insights:
- 🧠 **Biological Vision:** Real neurons use sparse coding to represent images
- ✨ **Emergent Features:** Edge detectors emerge naturally from efficiency principles
- 🎯 **Overcomplete Basis:** More dictionary atoms than input dimensions for flexibility
- 📊 **Sparsity Principle:** Most dictionary atoms should be inactive for any given image
- 🔄 **Adaptive Learning:** Dictionary and codes learned simultaneously through alternation

🎨 ASCII Diagram - Dictionary Learning Process:
==============================================

    Dictionary Learning Algorithm:
    
    📸 Natural Images → 🧩 Image Patches → 📚 Dictionary Learning
    
    Step 1: Extract patches from natural images
    ┌─────────────┐    ┌───┬───┬───┐
    │ 🏔️ Mountain  │ →  │ / │ \ │ - │  (Edge patches)
    │    Image     │    │ | │ ○ │ ~ │  (Various patterns)
    └─────────────┘    └───┴───┴───┘
    
    Step 2: Initialize random dictionary
    📚 Dictionary D = [atom₁, atom₂, ..., atom_k]
    ┌─────┬─────┬─────┬─────┐
    │ 🔲  │ 🔳  │ ⚫  │ ▲   │  ← Random patterns initially
    │noise│noise│noise│noise│
    └─────┴─────┴─────┴─────┘
    
    Step 3: Alternating Optimization
    
    🔄 ITERATION LOOP:
    
    A) Sparse Coding Step (Fix D, optimize α):
    min_α ||x - Dα||² + λ||α||₁
    
    Image patch: x = [0.8, 0.2, -0.5, 0.1, ...]
    Dictionary:  D = [d₁, d₂, d₃, d₄, ...]
    Sparse code: α = [0, 0.7, 0, 0.3, 0, 0, ...] ← Mostly zeros!
    
    B) Dictionary Update Step (Fix α, optimize D):
    Update each dictionary atom: d_i ← d_i + η(x - Dα)αᵢ
    
    📈 Learning Progress:
    Iteration 0:    Random noise patterns
    Iteration 100:  ≈≈≈ Wavy patterns emerging
    Iteration 500:  /// \\\ Clear edge detectors
    Iteration 1000: ═══ ║║║ Perfect Gabor-like filters!
    
    Final Dictionary (Like V1 Neurons!):
    ┌─────┬─────┬─────┬─────┬─────┐
    │  │  │ ╱╲  │ ═══ │ ┴┴┴ │ ○○○ │
    │ |||  │╱  ╲ │ ═══ │ ┴┴┴ │ ○○○ │  ← Gabor-like filters
    │  │  │ ╲╱  │ ═══ │ ┴┴┴ │ ○○○ │
    └─────┴─────┴─────┴─────┴─────┘
    Vertical Diagonal Horizontal Spots Textures
    
    Sparse Representation Example:
    Original patch: [complex 64-dim vector]
    Sparse code:   [0,0,0.8,0,0,0.3,0,0,0,0.1,0,0,...] ← 3 active out of 100!
    Reconstruction: 0.8×(vertical) + 0.3×(diagonal) + 0.1×(texture) ≈ original

🔬 Mathematical Framework:
=========================
Optimization Problem: min_{D,α} Σᵢ [½||xᵢ - Dαᵢ||₂² + λ||αᵢ||₁]

Dictionary Update: dⱼ ← dⱼ + η Σᵢ (xᵢ - Dαᵢ)αᵢⱼ  
Sparse Coding: α ← argmin_α ½||x - Dα||₂² + λ||α||₁

Normalization: ||dⱼ||₂ = 1 (unit norm dictionary atoms)
Sparsity: ||α||₀ << k (few active coefficients)

🌟 Why This Is Profound:
=======================
This isn't just an algorithm - it's a theory of how biological vision works!
- 👁️ **Neuroscience:** V1 neurons actually look like learned dictionary atoms
- 🧠 **Development:** Explains how infant visual systems self-organize  
- 💡 **Efficiency:** Sparse representations are metabolically efficient for brains
- 🎯 **Universality:** Works across species - cats, monkeys, humans all similar
- 🔬 **Predictive:** Predicted cortical properties before they were measured

This bridges neuroscience, computer vision, and information theory!
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
try:
    from .sparse_coder import SparseCoder
except ImportError:
    from sparse_coder import SparseCoder


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
                self.sparse_coder._sparse_encode_single(patch) 
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
            self.sparse_coder._sparse_encode_single(patch) 
            for patch in patches
        ])
        return codes
    
    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        """Fit the model and transform the data in one step (sklearn-style)"""
        self.fit(images)
        return self.transform(images)
    
    def get_components(self) -> np.ndarray:
        """Get dictionary components (sklearn-style)"""
        return self.dictionary.T  # sklearn convention: (n_components, n_features)
        
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

"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
📝 CITE: Benedict Chen (2025) - Sparse Coding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""