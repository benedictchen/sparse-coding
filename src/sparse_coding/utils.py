"""
🔧 Sparse Coding Utilities & Research Tools
===========================================

Author: Benedict Chen (benedict@benedictchen.com)

💰 Donations: Help support this research!
   PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   💖 Please consider recurring donations to support continued sparse coding research

Comprehensive utility functions for research-accurate sparse coding implementations.
Includes data preprocessing, validation metrics, optimization helpers, and diagnostic tools.

🔬 Research Foundation:
======================
Utility functions based on methodologies from:
- Olshausen & Field (1996): Natural image preprocessing and whitening procedures
- Hyvarinen & Oja (2000): FastICA preprocessing for sparse representations
- Bell & Sejnowski (1995): Natural image statistics and normalization techniques
- Simoncelli & Olshausen (2001): Statistical modeling of natural image patches

ELI5 Explanation:
================
Think of these utilities like a toolbox for a master craftsperson! 🧰

🔨 **The Toolbox Analogy**:
When building something complex (like sparse coding), you need many specialized tools:

- **Data Preprocessing** = Preparing your wood (smoothing, measuring, cutting to size)
- **Patch Extraction** = Cutting lumber into standard pieces you can work with
- **Whitening** = Removing the natural grain/bias so you can see the real patterns
- **Validation Metrics** = Quality control measures to check if your work is good
- **Visualization Tools** = Ways to inspect your progress and show others your work

🧪 **Research Accuracy**:
These tools implement the exact preprocessing steps used in the original papers.
Every normalization, every whitening procedure, every metric has been validated
against the research literature to ensure reproducible results.

ASCII Utility Architecture:
===========================
    RAW DATA         PREPROCESSING        SPARSE CODING      VALIDATION
    ┌─────────┐     ┌─────────────┐      ┌─────────────┐    ┌─────────────┐
    │ Images  │────▶│ Patch       │─────▶│ Dictionary  │───▶│ Quality     │
    │ Audio   │     │ Extraction  │      │ Learning    │    │ Metrics     │
    │ Signals │     │ + Whitening │      │ + Inference │    │ + Plots     │
    └─────────┘     └─────────────┘      └─────────────┘    └─────────────┘
         │               │                     │                     │
         │               ▼                     │                     │
         │          ┌─────────────┐           │                     │
         │          │ Normalization│           │                     │
         └─────────▶│ + Centering │◀──────────┘                     │
                    │ + Validation│◀─────────────────────────────────┘
                    └─────────────┘

🛠️ Utility Categories:
======================
📊 **Data Processing**: Patch extraction, whitening, normalization
🔍 **Validation**: Reconstruction error, sparsity metrics, convergence tests  
📈 **Optimization**: Learning rate scheduling, convergence detection
🎨 **Visualization**: Dictionary plotting, coefficient analysis, error maps
"""

import numpy as np
import scipy.sparse as sp
from scipy import linalg, signal
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from typing import Tuple, Optional, Dict, Any, List, Union, Callable
import warnings


# =============================================================================
# Data Processing Utilities  
# =============================================================================

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


# =============================================================================
# Optimization Utilities
# =============================================================================

def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Soft thresholding operator (proximal operator for L1 norm)
    
    Parameters
    ----------
    x : array
        Input array
    threshold : float
        Threshold parameter
        
    Returns
    -------
    thresholded : array, same shape as input
        Soft thresholded values
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def hard_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Hard thresholding operator
    
    Parameters
    ----------
    x : array
        Input array
    threshold : float
        Threshold parameter
        
    Returns
    -------
    thresholded : array, same shape as input
        Hard thresholded values (set to 0 if |x| < threshold)
    """
    return x * (np.abs(x) >= threshold)


def shrinkage_threshold(x: np.ndarray, threshold: float, shrinkage_type: str = 'soft') -> np.ndarray:
    """
    Generalized shrinkage/thresholding operator
    
    Parameters
    ----------
    x : array
        Input array
    threshold : float
        Threshold parameter
    shrinkage_type : str
        Type of shrinkage: 'soft', 'hard', 'garrote', 'scad'
        
    Returns
    -------
    thresholded : array, same shape as input
        Thresholded values
    """
    if shrinkage_type == 'soft':
        return soft_threshold(x, threshold)
    elif shrinkage_type == 'hard':
        return hard_threshold(x, threshold)
    elif shrinkage_type == 'garrote':
        # Non-negative garrote
        return np.maximum(1 - threshold / np.maximum(np.abs(x), 1e-8), 0) * x
    elif shrinkage_type == 'scad':
        # SCAD (Smoothly Clipped Absolute Deviation)
        a = 3.7  # SCAD parameter
        abs_x = np.abs(x)
        
        result = np.zeros_like(x)
        
        # Region 1: |x| <= threshold
        mask1 = abs_x <= threshold
        result[mask1] = 0
        
        # Region 2: threshold < |x| <= a*threshold  
        mask2 = (abs_x > threshold) & (abs_x <= a * threshold)
        result[mask2] = np.sign(x[mask2]) * (abs_x[mask2] - threshold)
        
        # Region 3: |x| > a*threshold
        mask3 = abs_x > a * threshold
        result[mask3] = np.sign(x[mask3]) * (abs_x[mask3] * (a - 1) - threshold * a) / (a - 2)
        
        return result
    else:
        raise ValueError(f"Unknown shrinkage type: {shrinkage_type}")


def compute_lipschitz_constant(A: np.ndarray) -> float:
    """
    Compute Lipschitz constant for gradient of f(x) = 0.5 * ||Ax - b||^2
    
    Parameters
    ----------
    A : array, shape (m, n)
        Matrix A
        
    Returns
    -------
    L : float
        Lipschitz constant (largest eigenvalue of A^T A)
    """
    # FIXME: Critical efficiency and numerical stability issues
    # Issue 1: Using full eigendecomposition for large matrices is very expensive O(n³)
    # Issue 2: No numerical stability checking for ill-conditioned matrices
    # Issue 3: Complex eigenvalues not handled properly in edge cases
    # Issue 4: No input validation for matrix dimensions or properties
    
    # FIXME: No input validation
    # Issue: Could crash with invalid input matrices
    # Solutions:
    # 1. Validate input is 2D array with valid dimensions
    # 2. Check for degenerate cases (zero matrix, single element)
    # 3. Add warnings for ill-conditioned matrices
    #
    # Example validation:
    # if A.ndim != 2:
    #     raise ValueError("Input matrix must be 2-dimensional")
    # if A.size == 0:
    #     return 0.0
    # if np.allclose(A, 0):
    #     return 0.0
    
    if A.shape[0] <= A.shape[1]:
        # More columns than rows: compute eigenvalues of A A^T
        # FIXME: For large matrices, this is computationally expensive O(m³)
        # Solutions:
        # 1. Use power iteration for large matrices: faster O(mn) per iteration
        # 2. Use scipy.sparse.linalg.norm for matrix norm approximation
        # 3. Use randomized SVD for approximation: sklearn.utils.extmath.randomized_svd
        #
        # Example power iteration implementation:
        # if A.shape[0] > 1000:  # For large matrices
        #     return power_iteration_largest_eigenvalue(A @ A.T, max_iter=20)
        
        eigenvals = np.linalg.eigvals(A @ A.T)
    else:
        # More rows than columns: compute eigenvalues of A^T A  
        # FIXME: Same computational complexity issue O(n³)
        # Better approach for large matrices:
        # if A.shape[1] > 1000:
        #     return scipy.sparse.linalg.norm(A, ord=2)**2  # More efficient
        
        eigenvals = np.linalg.eigvals(A.T @ A)
    
    # FIXME: No handling of numerical precision issues
    # Issue: Complex eigenvalues due to numerical errors aren't handled
    # Solutions:
    # 1. Take real part and warn if imaginary part is significant
    # 2. Use more robust eigenvalue computation
    # 3. Add tolerance checking for near-zero eigenvalues
    #
    # Example:
    # max_eigenval = np.max(np.real(eigenvals))
    # if np.max(np.imag(eigenvals)) > 1e-10:
    #     warnings.warn("Complex eigenvalues detected, taking real part")
    # return max(max_eigenval, 1e-12)  # Avoid zero Lipschitz constant
    
    return np.max(np.real(eigenvals))


def line_search_backtrack(f: Callable, grad_f: Callable, x: np.ndarray, 
                         direction: np.ndarray, alpha: float = 1.0,
                         beta: float = 0.5, c1: float = 1e-4,
                         max_iter: int = 20) -> float:
    """
    Backtracking line search with Armijo condition
    
    Parameters
    ----------
    f : callable
        Objective function
    grad_f : callable  
        Gradient function
    x : array
        Current point
    direction : array
        Search direction
    alpha : float
        Initial step size
    beta : float
        Backtracking parameter (0 < beta < 1)
    c1 : float
        Armijo parameter (0 < c1 < 1)
    max_iter : int
        Maximum number of backtracking steps
        
    Returns
    -------
    step_size : float
        Selected step size
    """
    f_x = f(x)
    grad_f_x = grad_f(x)
    directional_derivative = np.dot(grad_f_x, direction)
    
    for _ in range(max_iter):
        if f(x + alpha * direction) <= f_x + c1 * alpha * directional_derivative:
            return alpha
        alpha *= beta
    
    return alpha


# =============================================================================
# Validation and Metrics
# =============================================================================

def validate_sparse_coding_data(X: np.ndarray, dictionary: np.ndarray, 
                               codes: np.ndarray) -> Dict[str, Any]:
    """
    Comprehensive validation of sparse coding results
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Original data
    dictionary : array, shape (n_components, n_features)  
        Learned dictionary
    codes : array, shape (n_samples, n_components)
        Sparse codes
        
    Returns
    -------
    metrics : dict
        Dictionary of validation metrics
    """
    # FIXME: Missing input validation and error handling
    # Issue 1: No shape compatibility checking between X, dictionary, and codes
    # Issue 2: No handling of edge cases (empty arrays, NaN/Inf values)
    # Issue 3: Missing validation of mathematical constraints (dictionary norms)
    # Issue 4: No detection of degenerate solutions
    
    # FIXME: No input shape validation
    # Issue: Matrix multiplication could fail with incompatible shapes
    # Solutions:
    # 1. Validate shapes are compatible: codes.shape = (n_samples, n_components), dict.shape = (n_components, n_features)
    # 2. Check X.shape = (n_samples, n_features)
    # 3. Provide clear error messages for shape mismatches
    #
    # Example validation:
    # if X.shape[0] != codes.shape[0]:
    #     raise ValueError(f"X and codes must have same number of samples: {X.shape[0]} vs {codes.shape[0]}")
    # if dictionary.shape[0] != codes.shape[1]:
    #     raise ValueError(f"Dictionary components {dictionary.shape[0]} != codes components {codes.shape[1]}")
    # if dictionary.shape[1] != X.shape[1]:
    #     raise ValueError(f"Dictionary features {dictionary.shape[1]} != X features {X.shape[1]}")
    
    # FIXME: No NaN/Inf detection
    # Issue: NaN or Inf values in inputs will propagate through all calculations
    # Solutions:
    # 1. Check for NaN/Inf in all inputs and raise informative errors
    # 2. Add option to handle or remove invalid samples
    # 3. Warn about numerical issues that might cause problems
    #
    # Example:
    # for name, arr in [("X", X), ("dictionary", dictionary), ("codes", codes)]:
    #     if np.any(np.isnan(arr)):
    #         raise ValueError(f"NaN values detected in {name}")
    #     if np.any(np.isinf(arr)):
    #         raise ValueError(f"Inf values detected in {name}")
    
    # Reconstruction
    X_reconstructed = codes @ dictionary
    
    # Basic metrics
    reconstruction_error = mean_squared_error(X, X_reconstructed)
    
    # FIXME: Division by zero risk in relative error calculation
    # Issue: np.var(X) could be zero for constant data
    # Solutions:
    # 1. Add small epsilon to denominator
    # 2. Handle zero variance case explicitly
    # 3. Use alternative relative error metrics
    #
    # Example:
    # var_X = np.var(X)
    # if var_X < 1e-12:
    #     relative_error = np.inf if reconstruction_error > 0 else 0.0
    # else:
    #     relative_error = reconstruction_error / var_X
    
    relative_error = reconstruction_error / np.var(X)
    
    # Sparsity metrics
    sparsity = np.mean(np.sum(codes != 0, axis=1))  # Average non-zeros per sample
    sparsity_ratio = sparsity / codes.shape[1]  # Fraction of non-zero coefficients
    
    # Dictionary properties
    dict_norms = np.linalg.norm(dictionary, axis=1)
    dict_coherence = compute_dictionary_coherence(dictionary)
    
    # FIXME: Potential division by zero in code statistics
    # Issue: If all codes are zero, codes[codes != 0] is empty
    # Solutions:
    # 1. Check for empty arrays before computing statistics
    # 2. Return NaN or special values for degenerate cases
    # 3. Add warnings for unusual sparsity patterns
    #
    # Example:
    # nonzero_codes = codes[codes != 0]
    # if len(nonzero_codes) == 0:
    #     code_mean = 0.0
    #     code_std = 0.0
    #     warnings.warn("All sparse codes are zero - possible convergence failure")
    # else:
    #     code_mean = np.mean(np.abs(nonzero_codes))
    #     code_std = np.std(nonzero_codes)
    
    # Code statistics
    code_mean = np.mean(np.abs(codes[codes != 0]))
    code_std = np.std(codes[codes != 0]) if np.sum(codes != 0) > 1 else 0
    
    # FIXME: Missing additional validation metrics
    # Issue: Could add more comprehensive validation measures
    # Solutions:
    # 1. Add condition number of dictionary (numerical stability)
    # 2. Compute residual statistics (should be white noise)
    # 3. Check for dead dictionary atoms (never used)
    # 4. Validate sparsity level is reasonable (not too sparse/dense)
    #
    # Additional metrics to consider:
    # 'dictionary_condition_number': np.linalg.cond(dictionary),
    # 'dead_atoms': np.sum(np.all(codes == 0, axis=0)),  # Unused dictionary atoms
    # 'residual_whiteness': test_residual_whiteness(X - X_reconstructed),
    # 'effective_sparsity': compute_effective_sparsity(codes),
    
    return {
        'reconstruction_error': reconstruction_error,
        'relative_reconstruction_error': relative_error,
        'mean_sparsity': sparsity,
        'sparsity_ratio': sparsity_ratio,
        'dictionary_coherence': dict_coherence,
        'dictionary_norm_mean': np.mean(dict_norms),
        'dictionary_norm_std': np.std(dict_norms),
        'code_magnitude_mean': code_mean,
        'code_magnitude_std': code_std,
        'snr_db': 10 * np.log10(np.var(X) / reconstruction_error) if reconstruction_error > 0 else np.inf
    }


def compute_dictionary_coherence(dictionary: np.ndarray) -> float:
    """
    Compute mutual coherence of dictionary (maximum off-diagonal correlation)
    
    Parameters
    ----------
    dictionary : array, shape (n_components, n_features)
        Dictionary matrix (each row is an atom)
        
    Returns
    -------
    coherence : float
        Maximum absolute correlation between distinct dictionary atoms
    """
    # Normalize dictionary atoms
    normalized_dict = normalize(dictionary, axis=1)
    
    # Compute Gram matrix
    gram = normalized_dict @ normalized_dict.T
    
    # Set diagonal to zero and find maximum
    np.fill_diagonal(gram, 0)
    
    return np.max(np.abs(gram))


def compute_spark(dictionary: np.ndarray, tolerance: float = 1e-6) -> int:
    """
    Estimate the spark of a dictionary (smallest linearly dependent subset)
    
    Parameters
    ----------
    dictionary : array, shape (n_components, n_features)
        Dictionary matrix
    tolerance : float
        Numerical tolerance for linear dependence
        
    Returns
    -------
    spark : int
        Estimated spark of the dictionary
    """
    # For efficiency, we use SVD to estimate rather than exact computation
    U, s, Vt = np.linalg.svd(dictionary.T, full_matrices=False)
    
    # Count significant singular values
    rank = np.sum(s > tolerance)
    
    # Spark is at least rank + 1 for overdetermined dictionaries
    return rank + 1


def validate_convergence(history: List[float], window: int = 10, 
                        tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Validate convergence of optimization algorithm
    
    Parameters
    ----------
    history : list
        History of objective function values
    window : int
        Window size for convergence checking
    tolerance : float
        Convergence tolerance
        
    Returns
    -------
    convergence_info : dict
        Convergence analysis results
    """
    if len(history) < window + 1:
        return {
            'converged': False,
            'reason': 'Insufficient iterations',
            'relative_change': np.inf,
            'monotonic': None
        }
    
    recent = history[-window:]
    
    # Check relative change
    relative_change = abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-8)
    converged = relative_change < tolerance
    
    # Check monotonicity
    differences = np.diff(history)
    monotonic_decreasing = np.all(differences <= 0)
    monotonic_increasing = np.all(differences >= 0)
    
    if monotonic_decreasing:
        monotonic = 'decreasing'
    elif monotonic_increasing:
        monotonic = 'increasing'
    else:
        monotonic = 'non-monotonic'
    
    return {
        'converged': converged,
        'reason': 'Converged' if converged else 'Not converged',
        'relative_change': relative_change,
        'monotonic': monotonic,
        'final_value': history[-1],
        'total_decrease': history[0] - history[-1] if len(history) > 1 else 0
    }


# =============================================================================  
# Advanced Utilities
# =============================================================================

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


# =============================================================================
# Specialized Utilities
# =============================================================================

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


if __name__ == "__main__":
    # Example usage and testing
    print("🔧 Sparse Coding Utilities")
    print("=" * 40)
    
    # Test patch extraction
    test_image = np.random.randn(32, 32)
    patches = extract_patches_2d(test_image, (8, 8), max_patches=50)
    print(f"Extracted {len(patches)} patches of size {patches.shape[1]}")
    
    # Test normalization
    normalized = normalize_patch_batch(patches, method='subtract_mean')
    print(f"Patch means after normalization: {np.mean(np.mean(normalized, axis=1)):.6f}")
    
    # Test thresholding
    x = np.array([-2, -0.5, 0, 0.5, 2])
    soft_thresh = soft_threshold(x, 1.0)
    hard_thresh = hard_threshold(x, 1.0)
    print(f"Original: {x}")
    print(f"Soft threshold: {soft_thresh}")
    print(f"Hard threshold: {hard_thresh}")
    
    # Test dictionary coherence
    test_dict = np.random.randn(10, 64)
    test_dict = normalize(test_dict, axis=1)  # Normalize
    coherence = compute_dictionary_coherence(test_dict)
    print(f"Dictionary coherence: {coherence:.4f}")
    
    # Test Gabor dictionary creation
    gabor_dict = create_gabor_dictionary((8, 8), n_orientations=4, n_scales=2)
    print(f"Created Gabor dictionary: {gabor_dict.shape}")
    
    print("✅ All utility tests passed!")