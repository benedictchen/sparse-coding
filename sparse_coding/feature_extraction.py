"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀

╔════════════════════════════════════════════════════════════════════════════════╗
║                          Sparse Coding Feature Extraction                      ║
║                   Learning Meaningful Features Through Sparsity                 ║
╚════════════════════════════════════════════════════════════════════════════════╝

Created by: Benedict Chen
Contact: github.com/benedictchen • benedict@benedictchen.com

╔════════════════════════════════════════════════════════════════════════════════╗
║                                RESEARCH FOUNDATION                             ║
╚════════════════════════════════════════════════════════════════════════════════╝

🎓 FOUNDATIONAL PAPERS:
• Olshausen & Field (1996) "Emergence of simple-cell receptive field properties"
  - Discovered that sparse coding naturally learns edge detectors like V1 neurons
  - Established the connection between sparsity and biological visual processing

• Olshausen & Field (1997) "Sparse coding with an overcomplete basis set"
  - Introduced the mathematical framework for overcomplete sparse representations  
  - Showed how to learn redundant but efficient feature dictionaries

• Lee et al. (2006) "Efficient sparse coding algorithms"
  - Developed practical algorithms for large-scale sparse feature learning
  - Made sparse coding computationally feasible for real applications

• Coates & Ng (2011) "The importance of encoding versus training with sparse coding"
  - Demonstrated that good features matter more than classifier complexity
  - Showed sparse features achieve state-of-the-art performance on vision tasks

╔════════════════════════════════════════════════════════════════════════════════╗
║                                 ELI5 SUMMARY                                   ║
╚════════════════════════════════════════════════════════════════════════════════╝

🎨 Imagine you're an art teacher with a box of paintbrushes...

👨‍🎨 TRADITIONAL APPROACH (Dense):
"Use ALL the brushes for every painting!" 
Result: Muddy, overworked paintings 😵

🧙‍♂️ SPARSE CODING APPROACH:
"Use only the FEW brushes that matter most for each painting!"
Result: Clean, crisp, meaningful art! ✨

Here's how it works:
1. 📚 DICTIONARY LEARNING: "What are the most useful 'brushstrokes' (features)?"
   - Learn a library of fundamental patterns (edges, textures, shapes)
   - Like discovering the "alphabet" of visual elements

2. 🎯 SPARSE ENCODING: "For this specific image, which few features matter?"
   - Each image uses only a small subset of available features
   - Like writing words using only necessary letters, not the whole alphabet

3. 🔍 FEATURE EXTRACTION: "What does this image 'say' in our learned language?"
   - Represent complex images with simple, meaningful feature combinations
   - Like translating a novel into its key themes and concepts

WHY SPARSITY ROCKS:
• 🧠 Your brain works this way! Only a few neurons fire for each stimulus
• 🎯 Captures the most important information while ignoring noise  
• 🔧 Makes features interpretable - you can see what each one detects
• ⚡ More efficient - process only what matters, ignore the rest

╔════════════════════════════════════════════════════════════════════════════════╗
║                              VISUAL ARCHITECTURE                               ║
╚════════════════════════════════════════════════════════════════════════════════╝

                    SPARSE FEATURE EXTRACTION PIPELINE

    INPUT IMAGE                 PATCH EXTRACTION              LEARNED DICTIONARY
    ┌─────────────┐            ┌─────┬─────┬─────┐           ┌─────┬─────┬─────┐
    │ 🖼️  IMAGE    │    →      │ 8x8 │ 8x8 │ 8x8 │    ←     │  D₁ │  D₂ │  D₃ │ 
    │             │            ├─────┼─────┼─────┤           ├─────┼─────┼─────┤
    │    64x64    │            │ 8x8 │ 8x8 │ 8x8 │ Patches   │  D₄ │  D₅ │  D₆ │ Features
    │             │            ├─────┼─────┼─────┤           ├─────┼─────┼─────┤
    │             │            │ 8x8 │ 8x8 │ 8x8 │           │  D₇ │  D₈ │  D₉ │
    └─────────────┘            └─────┴─────┴─────┘           └─────┴─────┴─────┘
           │                           │                            │
           │                           ▼                            │
           │                   SPARSE CODING                       │
           │                 ┌─────────────────┐                   │
           │                 │ Find coefficients│  ◄───────────────┘
           │                 │     α such that: │
           │                 │   patch ≈ Σ αᵢDᵢ │
           │                 │   with few αᵢ≠0  │
           │                 └─────────────────┘
           │                           │
           ▼                           ▼
    SPATIAL POOLING             SPARSE COEFFICIENTS
    ┌─────┬─────┐               ┌───┬───┬───┬───┬───┐
    │ MAX │ MAX │               │ 0 │.8 │ 0 │ 0 │.6 │ ← Only few are non-zero!
    ├─────┼─────┤               ├───┼───┼───┼───┼───┤
    │ MAX │ MAX │               │ 0 │ 0 │.4 │ 0 │ 0 │
    └─────┴─────┘               └───┴───┴───┴───┴───┘
           │                           │
           └───────────┬───────────────┘
                       ▼
              FINAL FEATURE VECTOR
              [0.8, 0.6, 0.4, 0.0, ...]
               ↑    ↑    ↑    ↑
            Edge  Texture Corner None

╔════════════════════════════════════════════════════════════════════════════════╗
║                            MATHEMATICAL FRAMEWORK                              ║
╚════════════════════════════════════════════════════════════════════════════════╝

🔢 CORE OPTIMIZATION PROBLEM:

1. DICTIONARY LEARNING:
   min_{D,α} Σᵢ [½||xᵢ - Dαᵢ||₂² + λ||αᵢ||₁]
   
   Where:
   • xᵢ = input patch i
   • D = dictionary matrix (learned features)  
   • αᵢ = sparse coefficients for patch i
   • λ = sparsity penalty (bigger λ = more sparse)

2. SPARSE ENCODING (given learned D):
   αᵢ* = argmin_α [½||xᵢ - Dα||₂² + λ||α||₁]
   
   This is LASSO regression - balances reconstruction quality vs sparsity

3. FEATURE VECTOR CONSTRUCTION:
   f(image) = Pool({α₁*, α₂*, ..., αₙ*})
   
   Pool functions: max, mean, or sum across spatial locations

🎯 KEY HYPERPARAMETERS:
• n_components: Size of dictionary (more = richer representation)
• patch_size: Resolution of learned features (8x8 typical)
• sparsity_penalty (λ): Controls sparsity-quality tradeoff
• overlap_factor: How much patches overlap (more = better reconstruction)

⚡ COMPUTATIONAL INSIGHTS:
• Dictionary learning: Alternates between updating D and α
• Sparse coding: Each patch encoded independently (parallelizable!)
• Pooling: Provides translation invariance and dimensionality reduction

╔════════════════════════════════════════════════════════════════════════════════╗
║                             REAL-WORLD APPLICATIONS                            ║
╚════════════════════════════════════════════════════════════════════════════════╝

🌟 REVOLUTIONARY APPLICATIONS:

1. 🖼️  COMPUTER VISION:
   - Image classification: Sparse features → better than hand-crafted SIFT/HOG
   - Object detection: Learn part-based representations automatically
   - Image denoising: Separate signal from noise using sparsity prior
   - Super-resolution: Reconstruct high-res images from sparse representations

2. 🏥 MEDICAL IMAGING:
   - MRI reconstruction: Recover full images from undersampled k-space data
   - CT scan denoising: Reduce radiation dose while maintaining image quality  
   - Histology analysis: Automatically discover cellular patterns and structures
   - Brain imaging: Identify disease-specific activation patterns

3. 🎵 AUDIO & SPEECH:
   - Music source separation: Unmix instruments using learned audio dictionaries
   - Speech recognition: Robust features that work in noisy environments
   - Audio compression: Efficient encoding using sparse representations
   - Sound classification: Environmental sound recognition and analysis

4. 📊 DATA SCIENCE:
   - Anomaly detection: Identify outliers that can't be sparsely reconstructed
   - Dimensionality reduction: Alternative to PCA with better interpretability
   - Collaborative filtering: Recommend items using sparse user-item patterns
   - Text analysis: Learn semantic word features beyond simple bag-of-words

5. 🧬 SCIENTIFIC APPLICATIONS:
   - Gene expression analysis: Discover sparse regulatory patterns
   - Climate modeling: Identify key variables driving weather patterns
   - Materials science: Learn atomic structure patterns for property prediction
   - Astronomy: Detect rare astronomical events in survey data

🎯 WHY SPARSE FEATURES WIN:
• Interpretability: Each feature has clear meaning (unlike deep learning black boxes)
• Efficiency: Only compute/store non-zero coefficients
• Robustness: Sparse representations generalize better to new data
• Biological plausibility: Matches how the visual cortex actually works!

================================================================================
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
try:
    from .dictionary_learning import DictionaryLearner
    from .sparse_coder import SparseCoder
except ImportError:
    from dictionary_learning import DictionaryLearner
    from sparse_coder import SparseCoder


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
        
        for img in images:
            h, w = img.shape
            for i in range(0, h - patch_h + 1, step_h):
                for j in range(0, w - patch_w + 1, step_w):
                    patch = img[i:i+patch_h, j:j+patch_w]
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
            
        # Preprocess images (transform mode)
        processed_images = self._preprocess_images(images, fit=False)
        
        if len(processed_images.shape) == 2:
            processed_images = processed_images[np.newaxis, :, :]
            
        features = []
        
        for img in processed_images:
            # Extract patches from image
            patches = self._extract_patches_with_stride(img, grid_size)
            
            # Encode patches using sparse coding
            sparse_codes = self.sparse_coder.encode(patches)
            
            # Apply spatial pooling
            pooled_features = self._spatial_pooling(sparse_codes, pooling, grid_size)
            features.append(pooled_features)
            
        return np.array(features)
    
    def _extract_patches_with_stride(self, image: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
        """Extract patches with stride based on grid size"""
        h, w = image.shape
        patch_h, patch_w = self.patch_size
        
        grid_h, grid_w = grid_size
        stride_h = max(1, (h - patch_h) // (grid_h - 1)) if grid_h > 1 else h
        stride_w = max(1, (w - patch_w) // (grid_w - 1)) if grid_w > 1 else w
        
        patches = []
        for i in range(0, min(h - patch_h + 1, grid_h * stride_h), stride_h):
            for j in range(0, min(w - patch_w + 1, grid_w * stride_w), stride_w):
                patch = image[i:i+patch_h, j:j+patch_w]
                patches.append(patch.flatten())
                
        return np.array(patches)
    
    def _spatial_pooling(self, codes: np.ndarray, pooling: str, grid_size: Tuple[int, int]) -> np.ndarray:
        """Apply spatial pooling to sparse codes"""
        grid_h, grid_w = grid_size
        n_features = codes.shape[1]
        
        # Reshape codes to spatial grid
        codes_reshaped = codes.reshape(grid_h, grid_w, n_features)
        
        if pooling == 'max':
            pooled = np.max(codes_reshaped, axis=(0, 1))
        elif pooling == 'mean':
            pooled = np.mean(codes_reshaped, axis=(0, 1))
        elif pooling == 'sum':
            pooled = np.sum(codes_reshaped, axis=(0, 1))
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
            
        return pooled

    def fit_transform(self, images: np.ndarray, **fit_params) -> np.ndarray:
        """Fit extractor and transform images in one step"""
        self.fit(images, **fit_params)
        return self.transform(images)

    def get_feature_names_out(self) -> List[str]:
        """Get feature names for output"""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted first")
        
        feature_names = []
        for i in range(self.n_components):
            feature_names.append(f"sparse_feature_{i}")
        return feature_names

    def get_params(self) -> Dict[str, Any]:
        """Get parameter dictionary"""
        return {
            'n_components': self.n_components,
            'patch_size': self.patch_size,
            'sparsity_penalty': self.sparsity_penalty,
            'overlap_factor': self.overlap_factor,
            'whitening': self.whitening
        }

    def set_params(self, **params) -> 'SparseFeatureExtractor':
        """Set parameters"""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        return self

"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
📝 CITE: Benedict Chen (2025) - Sparse Coding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""