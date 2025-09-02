"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀
"""
"""
Core Algorithm Module - Sparse Coding Library
==============================================

This module contains the main SparseCoder class that integrates all modular components
for sparse coding based on Olshausen & Field (1996).

The SparseCoder class inherits from all mixin classes to provide a comprehensive
sparse coding implementation while maintaining clean separation of concerns.

Author: Benedict Chen (benedict@benedictchen.com)
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"
"""

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Optional, Dict, Any

# Import all mixin classes
from .data_processing import DataProcessingMixin
from .optimization import OptimizationMixin
from .dictionary_update import DictionaryUpdateMixin
from .validation import ValidationMixin
from .visualization import VisualizationMixin
from .utilities import create_overcomplete_basis, lateral_inhibition


class SparseCoder(BaseEstimator, TransformerMixin, DataProcessingMixin, 
                  OptimizationMixin, DictionaryUpdateMixin, ValidationMixin, 
                  VisualizationMixin):
    """
    Sparse Coding Algorithm Based on Olshausen & Field (1996)
    
    This class implements the groundbreaking sparse coding algorithm that discovers
    edge detectors from natural images, demonstrating how simple cells in V1 could
    emerge from efficient coding principles.
    
    The implementation follows the original paper's approach with modern enhancements:
    - Multiple optimization algorithms (original equation 5, FISTA, coordinate descent)
    - Various sparseness functions (L1, log, gaussian, huber, etc.)
    - Configurable dictionary update methods
    - Comprehensive validation and visualization
    
    Attributes:
        n_components (int): Number of dictionary elements (basis functions)
        sparsity_penalty (float): λ parameter controlling sparsity vs reconstruction trade-off
        patch_size (Tuple[int, int]): Size of image patches to analyze
        max_iter (int): Maximum iterations for sparse coding
        tolerance (float): Convergence tolerance
        dictionary (np.ndarray): Learned dictionary matrix (patch_dim, n_components)
        components_ (np.ndarray): sklearn-style components matrix (n_components, patch_dim)
        
    Example:
        >>> from sparse_coding.sc_modules import SparseCoder
        >>> coder = SparseCoder(n_components=256, patch_size=(16, 16))
        >>> coder.fit(natural_images)
        >>> coefficients = coder.transform(test_images)
        >>> reconstructed = coder.reconstruct(coefficients)
    """
    
    def __init__(
        self,
        n_components: int = 256,
        sparsity_penalty: float = 0.1,
        patch_size: Tuple[int, int] = (16, 16),
        max_iter: int = 100,
        tolerance: float = 1e-6,
        dictionary: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
        # sklearn-style parameters
        alpha: Optional[float] = None,  # sklearn sparsity parameter (overrides sparsity_penalty if provided)
        algorithm: Optional[str] = None,  # sklearn algorithm parameter (maps to optimization_method)
        # Configuration options for implementations
        sparseness_function: str = 'l1',  # 'log', 'l1', 'gaussian', 'huber', 'elastic_net', 'cauchy', 'student_t'
        optimization_method: str = 'coordinate_descent',  # 'equation_5', 'fista', 'proximal_gradient'
        l1_solver: str = 'coordinate_descent'  # 'lbfgs_b', 'fista', 'coordinate_descent'
    ):
        """
        Initialize Sparse Coder
        
        Args:
            n_components: Number of dictionary elements (basis functions)
            sparsity_penalty: λ parameter controlling sparsity vs reconstruction trade-off
            patch_size: Size of image patches to analyze
            max_iter: Maximum iterations for sparse coding
            tolerance: Convergence tolerance
            dictionary: Pre-trained dictionary (optional, will initialize random if None)
            random_seed: Random seed for reproducibility
            alpha: sklearn-style sparsity parameter (overrides sparsity_penalty)
            algorithm: sklearn-style algorithm parameter
            sparseness_function: Type of sparseness function to use
            optimization_method: Optimization algorithm for sparse encoding
            l1_solver: Specific solver for L1 problems
        """
        
        # Handle sklearn-style parameters first
        if alpha is not None:
            sparsity_penalty = alpha  # sklearn convention overrides sparsity_penalty
        if algorithm is not None:
            # Map sklearn algorithm names to our optimization methods
            algorithm_mapping = {
                'ista': 'equation_5',
                'fista': 'fista', 
                'lars': 'coordinate_descent',  # approximate mapping
                'omp': 'coordinate_descent',   # approximate mapping
                'coordinate_descent': 'coordinate_descent'
            }
            optimization_method = algorithm_mapping.get(algorithm, optimization_method)
        
        self.n_components = n_components
        self.sparsity_penalty = sparsity_penalty
        self.patch_size = patch_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # Configuration options for implementations
        self.sparseness_function = sparseness_function
        self.optimization_method = optimization_method
        self.l1_solver = l1_solver
        
        # Additional configuration for whitening and dictionary updates
        self.whitening_method = 'olshausen_field'  # 'olshausen_field', 'zca', 'standard'
        self.dictionary_update_method = 'equation_6'  # 'equation_6', 'orthogonal', 'batch'
        self.learning_rate = 0.01
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        if dictionary is not None:
            # Use provided dictionary and infer patch_size from it
            patch_dim = dictionary.shape[0]
            # Try to infer square patch size first
            patch_size_inferred = int(np.sqrt(patch_dim))
            if patch_size_inferred * patch_size_inferred == patch_dim:
                self.patch_size = (patch_size_inferred, patch_size_inferred)
                print(f"✓ Inferred patch size {self.patch_size} from dictionary dimensions")
            else:
                # Keep original patch_size but warn about mismatch
                expected_dim = patch_size[0] * patch_size[1]
                if patch_dim != expected_dim:
                    print(f"Warning: Dictionary has {patch_dim} dimensions, adjusting patch_size to match")
                    # Find best rectangular factorization
                    for h in range(1, int(np.sqrt(patch_dim)) + 1):
                        if patch_dim % h == 0:
                            w = patch_dim // h
                            self.patch_size = (h, w)
                    print(f"✓ Adjusted patch size to {self.patch_size}")
            
            if dictionary.shape[1] != n_components:
                print(f"Warning: Using provided dictionary with {dictionary.shape[1]} components instead of {n_components}")
                self.n_components = dictionary.shape[1]
            self.dictionary = dictionary.copy()
        else:
            # Initialize random dictionary (will be learned)
            patch_dim = patch_size[0] * patch_size[1]
            self.dictionary = np.random.randn(patch_dim, n_components)
            
        # Normalize dictionary columns
        self.dictionary = normalize(self.dictionary, axis=0)
        
        # Add sklearn-style aliases for compatibility
        self.alpha = sparsity_penalty  # sklearn convention
        self.components_ = None  # Will be set after fitting
        
        # Training history
        self.training_history = {'reconstruction_error': [], 'sparsity': []}
        
        # Validate configuration
        self._validate_configuration()
        
        print(f"✓ Sparse Coder initialized: {n_components} components, {patch_size} patches")
        print(f"  Sparseness function: {sparseness_function}")
        print(f"  Optimization method: {optimization_method}")

    def fit(self, images: np.ndarray, n_patches: int = 10000) -> Dict[str, Any]:
        """
        Learn sparse dictionary from natural images
        
        This is the core algorithm that discovers edge detectors from natural images,
        implementing the alternating optimization approach from Olshausen & Field (1996).
        
        The algorithm alternates between:
        1. Sparse encoding: Find sparse coefficients for patches given current dictionary
        2. Dictionary update: Update dictionary elements to minimize reconstruction error
        
        Args:
            images: Natural images (n_images, height, width) 
            n_patches: Number of patches to extract for training
            
        Returns:
            Dict containing training statistics and final metrics
        """
        
        print(f"🎯 Learning sparse dictionary from {len(images)} images...")
        
        # Extract random patches from images
        patches = self.extract_patches(images, n_patches)
        print(f"   Extracted {len(patches)} patches of size {self.patch_size}")
        
        # Whitening preprocessing with configurable methods
        if self.whitening_method == 'olshausen_field':
            patches_whitened = self.whiten_patches_olshausen_field(patches)
        elif self.whitening_method == 'zca':
            patches_whitened = self.whiten_patches_zca(patches)
        else:
            patches_whitened = self.whiten_patches(patches)
        
        # Enhanced alternating optimization with adaptive batch sizes and convergence checking
        batch_size = min(1000, len(patches_whitened))
        convergence_threshold = 1e-6
        prev_error = float('inf')
        
        for iteration in range(50):  # More iterations for better convergence
            print(f"\n📚 Dictionary learning iteration {iteration + 1}/50")
            
            # Adaptive batch size - start small and increase
            current_batch_size = min(batch_size * (1 + iteration // 10), len(patches_whitened))
            batch_patches = patches_whitened[:current_batch_size]
            
            # 1. Sparse encode patches with current dictionary
            coefficients = self._enhanced_sparse_encode(batch_patches)
            
            # 2. Update dictionary using configurable method
            if self.dictionary_update_method == 'equation_6':
                self._update_dictionary_equation_6(batch_patches, coefficients)
            elif self.dictionary_update_method == 'orthogonal':
                self._update_dictionary_with_orthogonality(batch_patches, coefficients)
            elif self.dictionary_update_method == 'batch':
                self._update_dictionary_batch(batch_patches, coefficients)
            else:
                # Default to original Olshausen method
                self._update_dictionary_olshausen(batch_patches, coefficients)
            
            # 3. Calculate detailed statistics
            reconstruction = coefficients @ self.dictionary.T
            reconstruction_error = np.mean((batch_patches - reconstruction) ** 2)
            sparsity = np.mean(np.sum(np.abs(coefficients) > 1e-3, axis=1))
            
            # Calculate additional metrics
            dictionary_coherence = self.calculate_dictionary_coherence()
            feature_usage = np.mean(np.sum(np.abs(coefficients) > 1e-3, axis=0))  # How many patches use each feature
            
            self.training_history['reconstruction_error'].append(reconstruction_error)
            self.training_history['sparsity'].append(sparsity)
            
            print(f"   Batch size: {current_batch_size}")
            print(f"   Reconstruction error: {reconstruction_error:.6f}")
            print(f"   Average sparsity: {sparsity:.1f} active elements")
            print(f"   Dictionary coherence: {dictionary_coherence:.3f}")
            print(f"   Feature usage rate: {feature_usage:.1f}%")
            
            # Convergence check
            if abs(prev_error - reconstruction_error) < convergence_threshold:
                print(f"   ✓ Converged after {iteration + 1} iterations")
                break
                
            prev_error = reconstruction_error
            
            # Adaptive learning rate decay
            if iteration > 0 and reconstruction_error > prev_error:
                print("   ↓ Reducing sparsity penalty for better convergence")
                self.sparsity_penalty *= 0.95
            
        print(f"✅ Dictionary learning complete!")
        
        # Set sklearn-style components_ attribute after successful training
        self.components_ = self.dictionary.T  # sklearn expects (n_components, n_features) shape
        self.dictionary_ = self.dictionary  # Additional alias some tests might expect
        
        return {
            'final_reconstruction_error': reconstruction_error,
            'final_sparsity': sparsity,
            'n_dictionary_elements': self.n_components,
            'patch_size': self.patch_size
        }

    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Transform new images using learned sparse dictionary
        
        This method applies the learned dictionary to encode new images as sparse
        coefficient vectors. Each image is transformed into a sparse representation
        using the discovered edge detectors.
        
        Args:
            images: Input images (n_images, height, width) or single patch (height, width)
            
        Returns:
            np.ndarray: Sparse coefficients (n_images, n_components)
        """
        
        if self.dictionary is None:
            raise ValueError("Dictionary must be learned before transform!")
        
        # Handle single image case - directly flatten and encode
        if len(images) == 1:
            image = images[0]
            if image.shape == self.patch_size:
                # Single patch - encode directly
                patch_flat = image.flatten()
                
                # Whiten single patch
                if self.whitening_method == 'olshausen_field':
                    patch_whitened = self.whiten_patches_olshausen_field(patch_flat.reshape(1, -1))[0]
                elif self.whitening_method == 'zca':
                    patch_whitened = self.whiten_patches_zca(patch_flat.reshape(1, -1))[0]  
                else:
                    patch_whitened = self.whiten_patches(patch_flat.reshape(1, -1))[0]
                
                # Sparse encode single patch
                coeff = self._sparse_encode_single(patch_whitened)
                return coeff.reshape(1, -1)
            
        # Multiple images - extract patches and average coefficients per image
        all_coefficients = []
        for image in images:
            # Extract a few patches from this image
            patches = self.extract_patches([image], min(10, 50))  # Reasonable number
            
            # Whiten patches
            if self.whitening_method == 'olshausen_field':
                patches_whitened = self.whiten_patches_olshausen_field(patches)
            elif self.whitening_method == 'zca':
                patches_whitened = self.whiten_patches_zca(patches)
            else:
                patches_whitened = self.whiten_patches(patches)
            
            # Encode all patches from this image
            coefficients = self.sparse_encode(patches_whitened)
            
            # Average coefficients to get one representation per image
            image_coeff = np.mean(coefficients, axis=0)
            all_coefficients.append(image_coeff)
        
        return np.array(all_coefficients)
    
    def fit_transform(self, images: np.ndarray, n_patches: int = 10000) -> np.ndarray:
        """
        Fit the model and transform the data in one step (sklearn-style)
        
        This is a convenience method that combines dictionary learning and 
        transformation into a single call, following sklearn conventions.
        
        Args:
            images: Input images to fit and transform
            n_patches: Number of patches to use for dictionary learning
            
        Returns:
            np.ndarray: Sparse coefficient matrix for the input images
        """
        self.fit(images, n_patches)
        return self.transform(images)
        
    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Reconstruct patches from sparse coefficients
        
        This method reconstructs the original patches by multiplying the sparse
        coefficients with the learned dictionary. The quality of reconstruction
        indicates how well the dictionary captures the input structure.
        
        Args:
            coefficients: Sparse coefficient matrix (n_patches, n_components)
            
        Returns:
            np.ndarray: Reconstructed patches (n_patches, patch_dim)
        """
        return coefficients @ self.dictionary.T

    def sparse_encode(self, patches: np.ndarray) -> np.ndarray:
        """
        Sparse encode patches using the learned dictionary
        
        This is the main public interface for sparse encoding, supporting both
        single patches and batches of patches. The method delegates to the
        appropriate optimization algorithm based on configuration.
        
        Args:
            patches: Single patch (patch_dim,) or array of patches (n_patches, patch_dim)
            
        Returns:
            Sparse coefficients (n_components,) for single patch or (n_patches, n_components) for multiple patches
        """
        
        # Handle single patch case
        if patches.ndim == 1:
            return self._sparse_encode_single(patches)
        
        # Handle multiple patches
        n_patches = patches.shape[0]
        coefficients = np.zeros((n_patches, self.n_components))
        
        print(f"🔍 Sparse encoding {n_patches} patches...")
        
        for i in range(n_patches):
            coefficients[i] = self._sparse_encode_single(patches[i])
            
            if (i + 1) % 100 == 0:
                print(f"   Encoded {i + 1}/{n_patches} patches")
                
        return coefficients

    def calculate_dictionary_coherence(self) -> float:
        """
        Calculate dictionary coherence (mutual coherence)
        
        Measures how well-conditioned the dictionary is.
        Lower coherence is better for sparse coding.
        
        Returns:
            float: Dictionary coherence value
        """
        # Compute Gram matrix
        gram_matrix = self.dictionary.T @ self.dictionary
        
        # Remove diagonal elements
        off_diagonal = gram_matrix - np.eye(self.n_components)
        
        # Maximum off-diagonal element is the coherence
        coherence = np.max(np.abs(off_diagonal))
        
        return coherence

    # sklearn compatibility methods
    def get_params(self, deep=True):
        """Get parameters for this estimator (sklearn compatibility)"""
        return {
            'n_components': self.n_components,
            'sparsity_penalty': self.sparsity_penalty,
            'patch_size': self.patch_size,
            'max_iter': self.max_iter,
            'tolerance': self.tolerance,
            'alpha': self.alpha,
            'sparseness_function': self.sparseness_function,
            'optimization_method': self.optimization_method,
            'l1_solver': self.l1_solver
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator (sklearn compatibility)"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {type(self).__name__}")
        
        # Re-validate configuration after parameter changes
        if hasattr(self, '_validate_configuration'):
            self._validate_configuration()
            
        return self

    def score(self, X, y=None):
        """
        Return the mean reconstruction error (sklearn compatibility)
        
        Args:
            X: Input data
            y: Ignored (for sklearn compatibility)
            
        Returns:
            float: Negative mean reconstruction error (higher is better for sklearn)
        """
        if self.dictionary is None:
            raise ValueError("Dictionary must be learned before scoring!")
            
        # Transform and reconstruct
        coefficients = self.transform(X)
        reconstructed = self.reconstruct(coefficients)
        
        # Calculate mean squared error
        if len(X) == 1 and X[0].shape == self.patch_size:
            # Single patch case
            original_flat = X[0].flatten()
        else:
            # Multiple images - extract patches for comparison
            patches = self.extract_patches(X, min(100, 500))
            original_flat = patches.flatten()
            reconstructed = reconstructed.flatten()[:len(original_flat)]
        
        mse = np.mean((original_flat - reconstructed.flatten()[:len(original_flat)]) ** 2)
        
        # Return negative MSE (sklearn convention: higher score is better)
        return -mse

    def create_overcomplete_basis(self, overcompleteness_factor: float = 2.0, 
                                 basis_type: str = 'gabor',
                                 random_seed: Optional[int] = None) -> np.ndarray:
        """
        Create Overcomplete Basis (Olshausen & Field 1996 Key Concept)
        
        Creates an overcomplete dictionary where the number of basis functions
        exceeds the input dimensionality. This is fundamental to Olshausen & Field's
        approach for learning sparse representations of natural images.
        
        Args:
            overcompleteness_factor: Ratio of dictionary size to input dimension
                                   2.0 = twice as many basis functions as pixels
            basis_type: Type of initial basis ('gabor', 'dct', 'random', 'edges')
            random_seed: Random seed for reproducibility
            
        Returns:
            np.ndarray: Overcomplete dictionary matrix (input_dim, n_basis)
            
        Example:
            >>> coder = SparseCoder(patch_size=(8, 8))
            >>> basis = coder.create_overcomplete_basis(overcompleteness_factor=1.5, basis_type='gabor')
            >>> print(f"Created basis: {basis.shape}")  # (64, 96)
        """
        return create_overcomplete_basis(
            patch_size=self.patch_size,
            overcompleteness_factor=overcompleteness_factor,
            basis_type=basis_type,
            random_seed=random_seed
        )

    def lateral_inhibition(self, activations: np.ndarray, 
                          inhibition_strength: float = 0.5,
                          inhibition_radius: float = 1.0,
                          topology: str = 'linear') -> np.ndarray:
        """
        Apply Lateral Inhibition (Biologically-Inspired Competition)
        
        Implements competitive dynamics found in biological neural networks
        where active neurons suppress nearby neurons, promoting sparse representations.
        
        Args:
            activations: Neural activation values to apply inhibition to
            inhibition_strength: Strength of inhibitory connections (0-1)
            inhibition_radius: Radius of inhibitory influence
            topology: Type of inhibition topology ('linear', '2d_grid', 'gaussian')
            
        Returns:
            np.ndarray: Activations after lateral inhibition
            
        Example:
            >>> coder = SparseCoder()
            >>> activations = np.random.randn(100)
            >>> inhibited = coder.lateral_inhibition(activations, inhibition_strength=0.3)
        """
        return lateral_inhibition(
            activations=activations,
            inhibition_strength=inhibition_strength,
            inhibition_radius=inhibition_radius,
            topology=topology
        )

    def _coordinate_descent_l1(self, signal: np.ndarray, max_iter: int = 100, **kwargs) -> np.ndarray:
        """
        Coordinate Descent L1 Solver (Backward Compatibility Alias)
        
        This is an alias for the modular coordinate descent optimization.
        Maintained for backward compatibility with existing tests.
        
        Args:
            signal: Input signal to encode
            max_iter: Maximum iterations
            **kwargs: Additional arguments
            
        Returns:
            np.ndarray: Sparse coefficients
        """
        # Use the optimization mixin's coordinate descent method
        return self._coordinate_descent_lasso(signal, lambda x: 0.5 * np.sum((signal - self.dictionary @ x) ** 2) + self.sparsity_penalty * np.sum(np.abs(x)), np.zeros(self.n_components))


# Maintain backward compatibility
SparseCode = SparseCoder  # Legacy alias


"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
📝 CITE: Benedict Chen (2025) - Sparse Coding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""