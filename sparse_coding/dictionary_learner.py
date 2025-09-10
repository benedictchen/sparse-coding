"""
Dictionary Learning for Sparse Coding

Implements the complete dictionary learning algorithm from Olshausen & Field (1996).
Learns both the dictionary D and sparse codes α simultaneously:
min_{D,α} ||X - Dα||_2^2 + λ||α||_1

Uses alternating optimization between dictionary update and sparse coding.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
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
            n_atoms=n_components,
            lam=sparsity_penalty,
            mode="l1",
            max_iter=1000,
            seed=random_seed or 0
        )
        # Set the dictionary directly
        self.sparse_coder.D = self.dictionary
        
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
        
    def _extract_patches(self, images: np.ndarray, overlap_factor: float = 0.5) -> np.ndarray:
        """Extract patches from images with specified overlap"""
        
        if len(images.shape) == 2:
            images = images[np.newaxis, :, :]
            
        patches = []
        patch_h, patch_w = self.patch_size
        step_h = max(1, int(patch_h * (1 - overlap_factor)))
        step_w = max(1, int(patch_w * (1 - overlap_factor)))
        
        for image in images:
            h, w = image.shape
            for i in range(0, h - patch_h + 1, step_h):
                for j in range(0, w - patch_w + 1, step_w):
                    patch = image[i:i+patch_h, j:j+patch_w]
                    patches.append(patch.flatten())
                    
        return np.array(patches).T
    
    def _update_dictionary(self, patches: np.ndarray, codes: np.ndarray) -> float:
        """Update dictionary using gradient descent"""
        
        old_dict = self.dictionary.copy()
        
        # Compute reconstruction error
        reconstruction = self.dictionary @ codes
        residual = patches - reconstruction
        
        # Gradient descent update
        gradient = -residual @ codes.T / codes.shape[1]
        self.dictionary = self.dictionary - self.learning_rate * gradient
        
        # Normalize dictionary atoms
        self._normalize_dictionary()
        
        # Return change in dictionary
        change = np.linalg.norm(self.dictionary - old_dict)
        return change
    
    def _compute_metrics(self, patches: np.ndarray, codes: np.ndarray) -> Dict[str, float]:
        """Compute training metrics"""
        
        reconstruction = self.dictionary @ codes
        reconstruction_error = np.mean((patches - reconstruction) ** 2)
        sparsity_level = np.mean(np.abs(codes) > 1e-6)
        
        return {
            'reconstruction_error': reconstruction_error,
            'sparsity_level': sparsity_level
        }
    
    def fit(self, images: np.ndarray, overlap_factor: float = 0.5, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train dictionary on image patches
        
        Args:
            images: Input images (can be single image or batch)
            overlap_factor: Patch overlap (0=no overlap, 0.5=50% overlap)
            verbose: Print training progress
            
        Returns:
            Dict containing training history
        """
        
        # Extract patches
        patches = self._extract_patches(images, overlap_factor)
        
        if verbose:
            print(f"Training on {patches.shape[1]} patches of size {self.patch_dim}")
        
        # Training loop
        for iteration in range(self.max_iterations):
            
            # Step 1: Sparse coding - update codes given dictionary
            codes = []
            for i in range(patches.shape[1]):
                patch = patches[:, i:i+1]  # Keep as 2D for compatibility
                code = self.sparse_coder.encode(patch)
                codes.append(code[:, 0])  # Extract 1D result
            codes = np.array(codes).T
            
            # Step 2: Dictionary update - update dictionary given codes
            dict_change = self._update_dictionary(patches, codes)
            
            # Update sparse coder with new dictionary
            self.sparse_coder.D = self.dictionary
            
            # Compute metrics
            metrics = self._compute_metrics(patches, codes)
            
            # Store history
            self.training_history['reconstruction_errors'].append(metrics['reconstruction_error'])
            self.training_history['sparsity_levels'].append(metrics['sparsity_level'])
            self.training_history['dictionary_changes'].append(dict_change)
            
            # Print progress
            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration}: Error={metrics['reconstruction_error']:.6f}, "
                      f"Sparsity={metrics['sparsity_level']:.3f}, "
                      f"Dict_change={dict_change:.6f}")
            
            # Check convergence
            if dict_change < self.tolerance:
                if verbose:
                    print(f"Converged after {iteration} iterations")
                break
        
        return self.training_history
    
    def transform(self, images: np.ndarray, pooling: str = 'max') -> np.ndarray:
        """
        Transform images to sparse features
        
        Args:
            images: Input images
            pooling: Pooling method ('max', 'mean', 'sum')
            
        Returns:
            Feature vectors for each image
        """
        
        if len(images.shape) == 2:
            images = images[np.newaxis, :, :]
        
        features = []
        
        for image in images:
            # Extract patches
            patches = self._extract_patches(image[np.newaxis, :, :])
            
            # Encode each patch
            codes = []
            for i in range(patches.shape[1]):
                patch = patches[:, i:i+1]  # Keep as 2D for compatibility
                code = self.sparse_coder.encode(patch)
                codes.append(code[:, 0])  # Extract 1D result
            codes = np.array(codes)
            
            # Pool codes across spatial locations
            if pooling == 'max':
                feature = np.max(np.abs(codes), axis=0)
            elif pooling == 'mean':
                feature = np.mean(codes, axis=0)
            elif pooling == 'sum':
                feature = np.sum(codes, axis=0)
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
                
            features.append(feature)
        
        return np.array(features)
    
    def fit_transform(self, images: np.ndarray, **kwargs) -> np.ndarray:
        """Fit dictionary and transform images"""
        self.fit(images, **kwargs)
        return self.transform(images)
    
    def get_dictionary_atoms(self) -> np.ndarray:
        """Get dictionary atoms reshaped as patches"""
        return self.dictionary.T.reshape(self.n_components, *self.patch_size)