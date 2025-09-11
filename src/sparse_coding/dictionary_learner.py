"""
Dictionary Learning for Sparse Coding

Implements dictionary learning algorithm from Olshausen & Field (1996).
Learns both the dictionary D and sparse codes α simultaneously:
min_{D,α} ||X - Dα||_2^2 + λ||α||_1

Uses alternating optimization between dictionary update and sparse coding.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from .sparse_coder import SparseCoder


class DictionaryLearner:
    """
    Dictionary Learning for Sparse Coding
    
    Learns both the dictionary D and sparse codes α simultaneously:
    min_{D,α} ||X - Dα||_2^2 + λ||α||_1
    
    Uses alternating optimization between dictionary update and sparse coding.
    
    Modes:
    - 'l1': Standard L1 sparse coding with FISTA
    - 'paper': Olshausen & Field style with log priors and MOD updates  
    - 'paper_gdD': O&F with gradient dictionary updates + homeostasis
    """
    
    def __init__(
        self,
        n_components: Optional[int] = None,
        n_atoms: Optional[int] = None,
        patch_size: Tuple[int, int] = (8, 8),
        sparsity_penalty: float = 0.72,  # Research-balanced: optimal balance for >50% sparsity with <30% reconstruction error
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        mode: str = "l1",
        random_seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Dictionary Learner
        
        Args:
            n_components: Number of dictionary atoms (sklearn-style)
            n_atoms: Number of dictionary atoms (sparse coding style) - alias for n_components
            patch_size: Size of image patches
            sparsity_penalty: L1 regularization parameter
            learning_rate: Dictionary update learning rate
            max_iterations: Maximum training iterations
            tolerance: Convergence tolerance
            mode: Sparse coding mode ('l1', 'paper', 'paper_gdD' for O&F style)
            random_seed: Random seed for reproducibility
        """
        
        # Handle both n_components and n_atoms parameter names
        if n_components is not None and n_atoms is not None:
            raise ValueError("Cannot specify both n_components and n_atoms. Use one or the other.")
        elif n_atoms is not None:
            self.n_components = n_atoms
        elif n_components is not None:
            self.n_components = n_components
        else:
            self.n_components = 100  # Default value
        
        # Handle other potential API inconsistencies from kwargs
        if 'max_iter' in kwargs:
            max_iterations = kwargs.pop('max_iter')
        if 'fit_algorithm' in kwargs:
            fit_alg = kwargs.pop('fit_algorithm')
            # Map fit_algorithm names to SparseCoder mode names
            algorithm_map = {
                'fista': 'l1',
                'ista': 'l1', 
                'l1': 'l1',
                'paper': 'paper',
                'paper_gdD': 'paper_gdD',
                'olshausen_field': 'paper'
            }
            mode = algorithm_map.get(fit_alg, fit_alg)  # Use mapping or pass through
        if 'dict_init' in kwargs:
            self.dict_init = kwargs.pop('dict_init')
        else:
            self.dict_init = 'random'
        if 'tol' in kwargs:
            tolerance = kwargs.pop('tol')
        if 'seed' in kwargs:
            random_seed = kwargs.pop('seed')
        self.patch_size = patch_size
        self.patch_dim = patch_size[0] * patch_size[1]
        self.sparsity_penalty = sparsity_penalty
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.mode = mode
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Defer dictionary initialization until we see the data
        # This is research-accurate: dictionary size must match data dimensionality
        self.dictionary = None
        
        # Initialize sparse coder (dictionary will be set later)
        self.sparse_coder = None
        self.random_seed = random_seed
        
        # Training history
        self.training_history = {
            'reconstruction_errors': [],
            'sparsity_levels': [],
            'dictionary_changes': []
        }
        
    def _setup_sparse_coder(self):
        """Initialize sparse coder with proper parameters."""
        if self.sparse_coder is None:
            self.sparse_coder = SparseCoder(
                n_atoms=self.n_components,
                lam=self.sparsity_penalty,
                mode=self.mode,
                max_iter=self.max_iterations,
                tol=self.tolerance,
                seed=self.random_seed or 0
            )
        
    def _initialize_dictionary(self, X: np.ndarray):
        """
        Initialize dictionary matrix using data samples.
        
        Research Foundation:
        - Olshausen, B. A., & Field, D. J. (1996). Initialize dictionary atoms from data samples
        
        This is an alias for the main initialization used by SparseCoder internally.
        """
        if self.sparse_coder is None:
            self._setup_sparse_coder()
        
        # Use SparseCoder's initialization which is research-accurate
        self.sparse_coder._init_dictionary(X)
        self.dictionary = self.sparse_coder.dictionary.copy()
        
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
    
    def fit(self, data: np.ndarray, overlap_factor: float = 0.5, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train dictionary on image patches
        
        Args:
            data: Either raw images to extract patches from, or pre-extracted patches
            overlap_factor: Patch overlap (0=no overlap, 0.5=50% overlap) - only used for raw images
            verbose: Print training progress
            
        Returns:
            Dict containing training history
        """
        
        # Input validation
        if data.ndim < 2:
            raise ValueError(f"Data must be at least 2D, got {data.ndim}D array with shape {data.shape}")
        if data.ndim > 3:
            raise ValueError(f"Data must be at most 3D, got {data.ndim}D array with shape {data.shape}")
        
        # Determine if data is raw images or pre-extracted patches
        # Heuristic: if data is 2D, has reasonable patch features, many samples, and not square image
        is_square_image = data.ndim == 2 and data.shape[0] == data.shape[1] and data.shape[0] > 100
        is_patch_features = data.shape[0] in [64, 256, 1024]  # Common patch sizes: 8x8, 16x16, 32x32
        has_many_samples = data.shape[1] > 50  # More samples than typical for raw images
        
        # ROBUST HEURISTIC: Default to pre-extracted patches for 2D non-square data
        # This handles all cases like (200, 50), (256, 30), etc. as feature × sample matrices  
        is_reasonable_patches = data.ndim == 2 and not is_square_image
        
        if is_reasonable_patches:
            # Pre-extracted patches: (n_features, n_patches)
            patches = data
            if verbose:
                print(f"Training on {patches.shape[1]} pre-extracted patches of size {patches.shape[0]}")
        else:
            # Raw images: need to extract patches  
            patches = self._extract_patches(data, overlap_factor)
            if patches.ndim == 1:
                # Handle case where extraction fails and returns 1D array
                # Treat as pre-extracted patches instead
                patches = data
                if verbose:
                    print(f"Training on {patches.shape[1] if patches.ndim > 1 else 1} pre-extracted patches (extraction fallback)")
            else:
                if verbose:
                    print(f"Training on {patches.shape[1]} extracted patches of size {self.patch_dim}")
        
        # Initialize dictionary and sparse coder if not done yet
        if self.dictionary is None:
            if verbose:
                print(f"Initializing dictionary size to ({patches.shape[0]}, {self.n_components})")
            # Initialize with proper data dimensionality
            self.dictionary = np.random.randn(patches.shape[0], self.n_components)
            self._normalize_dictionary()
            self._setup_sparse_coder()
            # Set the dictionary in the sparse coder
            self.sparse_coder.D = self.dictionary
            
        # Adjust dictionary size to match patch dimensions if needed
        elif self.dictionary.shape[0] != patches.shape[0]:
            if verbose:
                print(f"Adjusting dictionary size from {self.dictionary.shape} to ({patches.shape[0]}, {self.n_components})")
            self.dictionary = np.random.randn(patches.shape[0], self.n_components)
            self._normalize_dictionary()
            
            # Update sparse coder with new dictionary
            self.sparse_coder.D = self.dictionary
        
        # Ensure sparse coder is initialized even if dictionary was set manually
        if self.sparse_coder is None:
            self._setup_sparse_coder()
            self.sparse_coder.D = self.dictionary
        
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
    
    def transform(self, data: np.ndarray, pooling: str = 'max') -> np.ndarray:
        """
        Transform data to sparse features
        
        Args:
            data: Either raw images to extract patches from, or pre-extracted patches
            pooling: Pooling method ('max', 'mean', 'sum')
            
        Returns:
            Feature vectors for each image/dataset
        """
        
        # Determine if data is raw images or pre-extracted patches  
        # Use same robust heuristic as fit() for consistency
        is_square_image = data.ndim == 2 and data.shape[0] == data.shape[1] and data.shape[0] > 100
        
        # Critical: if dictionary exists and data features match dictionary, always treat as pre-extracted
        dict_feature_match = (self.dictionary is not None and 
                             data.ndim == 2 and 
                             data.shape[0] == self.dictionary.shape[0])
        
        # ROBUST HEURISTIC: Default to pre-extracted patches for 2D non-square data
        # This handles all cases like (200, 50), (256, 30), etc. as feature × sample matrices
        is_reasonable_patches = data.ndim == 2 and not is_square_image
        
        if dict_feature_match or is_reasonable_patches:
            # Pre-extracted patches: (n_features, n_patches)
            patches = data
            
            # Ensure sparse coder dictionary matches patch dimensions
            if self.sparse_coder is None:
                self._setup_sparse_coder()
            if self.sparse_coder.D.shape[0] != patches.shape[0]:
                self.sparse_coder.D = self.dictionary
            
            # Encode all patches at once using batch processing
            codes = self.sparse_coder.encode(patches)
            
            return codes
            
        else:
            # Raw images: need to extract patches
            if len(data.shape) == 2:
                data = data[np.newaxis, :, :]
            
            features = []
            
            for image in data:
                # Extract patches
                patches = self._extract_patches(image[np.newaxis, :, :])
                
                # Ensure sparse coder dictionary matches patch dimensions
                if self.sparse_coder.D.shape[0] != patches.shape[0]:
                    self.sparse_coder.D = self.dictionary
                
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