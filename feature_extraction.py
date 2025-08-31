"""
Sparse Feature Extraction Implementation
Based on: Olshausen & Field (1996) feature extraction pipeline

Provides high-level interface for extracting sparse features from images
using learned dictionaries.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from .dictionary_learning import DictionaryLearner
from .sparse_coder import SparseCoder


class SparseFeatureExtractor:
    """
    High-level sparse feature extraction interface
    
    Combines dictionary learning and sparse coding to provide
    a complete feature extraction pipeline for images.
    """
    
    def __init__(
        self,
        n_components: int = 100,
        patch_size: Tuple[int, int] = (8, 8),
        sparsity_penalty: float = 0.1,
        overlap_factor: float = 0.5,
        whitening: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Sparse Feature Extractor
        
        Args:
            n_components: Number of dictionary atoms/features
            patch_size: Size of image patches
            sparsity_penalty: L1 regularization strength
            overlap_factor: Patch overlap factor (0=no overlap, 1=full overlap)
            whitening: Whether to apply whitening preprocessing
            random_seed: Random seed for reproducibility
        """
        
        self.n_components = n_components
        self.patch_size = patch_size
        self.sparsity_penalty = sparsity_penalty
        self.overlap_factor = overlap_factor
        self.whitening = whitening
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Components
        self.dictionary_learner = None
        self.sparse_coder = None
        self.is_fitted = False
        
        # Preprocessing parameters
        self.mean_ = None
        self.std_ = None
        self.whitening_matrix_ = None
        
    def _preprocess_images(self, images: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply preprocessing (normalization, whitening)"""
        
        processed = images.copy()
        
        if fit:
            # Compute statistics
            self.mean_ = np.mean(processed)
            self.std_ = np.std(processed)
            
        # Normalize
        if self.mean_ is not None and self.std_ is not None:
            processed = (processed - self.mean_) / (self.std_ + 1e-8)
            
        if self.whitening and fit:
            # Simple whitening: decorrelate patches
            patches = self._extract_all_patches(processed)
            cov = np.cov(patches.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            # Regularize eigenvalues
            eigenvals = np.maximum(eigenvals, 0.01)
            self.whitening_matrix_ = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
            
        return processed
        
    def _extract_all_patches(self, images: np.ndarray) -> np.ndarray:
        """Extract all patches from images with specified overlap"""
        
        if len(images.shape) == 2:
            images = images[np.newaxis, :, :]
            
        patches = []
        patch_h, patch_w = self.patch_size
        step_h = max(1, int(patch_h * (1 - self.overlap_factor)))
        step_w = max(1, int(patch_w * (1 - self.overlap_factor)))
        
        for image in images:
            h, w = image.shape
            for i in range(0, h - patch_h + 1, step_h):
                for j in range(0, w - patch_w + 1, step_w):
                    patch = image[i:i+patch_h, j:j+patch_w]
                    patches.append(patch.flatten())
                    
        return np.array(patches)
        
    def fit(self, images: np.ndarray, max_iterations: int = 1000, verbose: bool = True) -> Dict[str, Any]:
        """
        Fit sparse feature extractor to training images
        
        Args:
            images: Training images (n_images, height, width) or (height, width)
            max_iterations: Maximum dictionary learning iterations
            verbose: Whether to print progress
            
        Returns:
            Training statistics
        """
        
        if verbose:
            print(f"🎯 Fitting Sparse Feature Extractor...")
            
        # Preprocess images
        processed_images = self._preprocess_images(images, fit=True)
        
        # Initialize dictionary learner
        self.dictionary_learner = DictionaryLearner(
            n_components=self.n_components,
            patch_size=self.patch_size,
            sparsity_penalty=self.sparsity_penalty,
            max_iterations=max_iterations
        )
        
        # Learn dictionary
        results = self.dictionary_learner.fit(processed_images, verbose=verbose)
        
        # Initialize sparse coder with learned dictionary
        self.sparse_coder = SparseCoder(
            dictionary=self.dictionary_learner.get_dictionary(),
            sparsity_penalty=self.sparsity_penalty
        )
        
        self.is_fitted = True
        
        if verbose:
            print(f"✅ Sparse Feature Extractor fitted successfully!")
            
        return results
        
    def transform(self, images: np.ndarray, pooling: str = 'max', 
                 grid_size: Tuple[int, int] = (4, 4)) -> np.ndarray:
        """
        Transform images to sparse feature representation
        
        Args:
            images: Input images
            pooling: Pooling method ('max', 'mean', 'sum')
            grid_size: Spatial pooling grid size
            
        Returns:
            Feature vectors (n_images, n_features)
        """
        
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform!")
            
        # Preprocess images
        processed_images = self._preprocess_images(images)
        
        if len(processed_images.shape) == 2:
            processed_images = processed_images[np.newaxis, :, :]
            
        features = []
        
        for image in processed_images:
            # Extract patches and encode
            patches = self._extract_patches_with_positions(image)
            patch_codes = []
            positions = []
            
            for patch, pos in patches:
                code = self.sparse_coder.encode_patch(patch.flatten())
                patch_codes.append(code)
                positions.append(pos)
                
            # Apply spatial pooling
            if len(patch_codes) > 0:
                pooled_features = self._spatial_pooling(
                    patch_codes, positions, image.shape, pooling, grid_size
                )
            else:
                pooled_features = np.zeros(self.n_components * grid_size[0] * grid_size[1])
                
            features.append(pooled_features)
            
        return np.array(features)
        
    def _extract_patches_with_positions(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """Extract patches with their spatial positions"""
        
        patches_with_pos = []
        patch_h, patch_w = self.patch_size
        step_h = max(1, int(patch_h * (1 - self.overlap_factor)))
        step_w = max(1, int(patch_w * (1 - self.overlap_factor)))
        
        h, w = image.shape
        for i in range(0, h - patch_h + 1, step_h):
            for j in range(0, w - patch_w + 1, step_w):
                patch = image[i:i+patch_h, j:j+patch_w]
                patches_with_pos.append((patch, (i, j)))
                
        return patches_with_pos
        
    def _spatial_pooling(self, codes: List[np.ndarray], positions: List[Tuple[int, int]], 
                        image_shape: Tuple[int, int], pooling: str, 
                        grid_size: Tuple[int, int]) -> np.ndarray:
        """Apply spatial pooling to patch codes"""
        
        h, w = image_shape
        grid_h, grid_w = grid_size
        
        # Create spatial grid
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        # Initialize pooled features
        pooled = np.zeros((grid_h, grid_w, self.n_components))
        counts = np.zeros((grid_h, grid_w))
        
        # Assign codes to grid cells
        for code, (pos_i, pos_j) in zip(codes, positions):
            grid_i = min(pos_i // cell_h, grid_h - 1)
            grid_j = min(pos_j // cell_w, grid_w - 1)
            
            if pooling == 'max':
                pooled[grid_i, grid_j] = np.maximum(pooled[grid_i, grid_j], code)
            elif pooling == 'mean' or pooling == 'sum':
                pooled[grid_i, grid_j] += code
                counts[grid_i, grid_j] += 1
                
        # Finalize pooling
        if pooling == 'mean':
            mask = counts > 0
            pooled[mask] = pooled[mask] / counts[mask, np.newaxis]
            
        return pooled.flatten()
        
    def fit_transform(self, images: np.ndarray, **kwargs) -> np.ndarray:
        """Fit extractor and transform images in one step"""
        self.fit(images, **kwargs)
        return self.transform(images)
        
    def get_feature_images(self) -> np.ndarray:
        """Get learned features as images"""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted first!")
            
        return self.dictionary_learner.get_dictionary_images()
        
    def visualize_features(self, figsize: Tuple[int, int] = (12, 8)):
        """Visualize learned features"""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted first!")
            
        import matplotlib.pyplot as plt
        
        feature_images = self.get_feature_images()
        n_features = len(feature_images)
        
        # Determine grid size
        grid_size = int(np.ceil(np.sqrt(n_features)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i in range(n_features):
            ax = axes[i]
            ax.imshow(feature_images[i], cmap='gray')
            ax.set_title(f'Feature {i+1}')
            ax.axis('off')
            
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
        
        print(f"📊 Learned {n_features} sparse features")
        print(f"   Feature size: {self.patch_size}")
        print(f"   Sparsity penalty: {self.sparsity_penalty}")