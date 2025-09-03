"""
🏗️ Sparse Coding - Utilities and Validation Module
==================================================


Author: Benedict Chen (benedict@benedictchen.com)
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties"

🎯 MODULE PURPOSE:
=================
Utility functions and validation methods for sparse coding including soft
thresholding, error calculation, data validation, and research accuracy checks.

🔬 RESEARCH FOUNDATION:
======================
Implements utility methods supporting:
- Olshausen & Field (1996): Soft thresholding and sparsity functions
- Modern sparse coding: Advanced thresholding and regularization
- Data preprocessing: Whitening, normalization, patch extraction
- Validation: Parameter checking, convergence monitoring

This module contains the utility and validation components, split from the
1544-line monolith for specialized support functionality.
"""

import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from typing import Tuple, Optional, Dict, Any, Union, List
import warnings


class UtilitiesValidationMixin:
    """
    🏗️ Utilities and Validation Mixin for Sparse Coding
    
    Contains utility functions including soft thresholding, error calculation,
    data validation, and research accuracy validation methods.
    
    Essential support functions for sparse coding algorithms.
    """
    
    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """
        Soft thresholding function (proximal operator for L1 norm).
        
        Implements: soft_threshold(x, λ) = sign(x) * max(|x| - λ, 0)
        
        This is the proximal operator for L1 regularization and is essential
        for ISTA/FISTA and coordinate descent algorithms.
        
        Args:
            x: Input array
            threshold: Thresholding parameter λ
            
        Returns:
            Soft-thresholded array
        """
        
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)
    
    def _soft_threshold_scalar(self, x: float, threshold: float) -> float:
        """Scalar version of soft thresholding for efficiency."""
        
        return np.sign(x) * max(abs(x) - threshold, 0.0)
    
    def _hard_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """
        Hard thresholding function.
        
        Implements: hard_threshold(x, λ) = x * I(|x| > λ)
        where I is the indicator function.
        
        Args:
            x: Input array
            threshold: Thresholding parameter
            
        Returns:
            Hard-thresholded array
        """
        
        return x * (np.abs(x) > threshold)
    
    def _reconstruction_error(self, X: np.ndarray, codes: np.ndarray) -> float:
        """
        Compute reconstruction error ||X - D @ codes.T||².
        
        Args:
            X: Data samples [n_samples, n_features]
            codes: Sparse coefficients [n_samples, n_components]
            
        Returns:
            Mean squared reconstruction error
        """
        
        reconstruction = X.T - self.dictionary_.T @ codes.T
        return np.mean(reconstruction ** 2)
    
    def _sparsity_cost(self, codes: np.ndarray) -> float:
        """
        Compute sparsity cost based on specified sparsity function.
        
        Implements various sparsity penalty functions:
        - L1: Σ|aᵢ| (promotes sparsity)
        - L2: Σaᵢ² (ridge regularization)
        - Log: Σlog(1 + aᵢ²) (Olshausen & Field 1996)
        - Student-t: Σlog(1 + aᵢ²/2) (heavy-tailed prior)
        
        Args:
            codes: Sparse coefficients [n_samples, n_components]
            
        Returns:
            Mean sparsity cost
        """
        
        sparsity_func = getattr(self, 'sparsity_func', 'l1')
        
        if sparsity_func == 'l1':
            return np.mean(np.abs(codes))
        elif sparsity_func == 'l2':
            return np.mean(codes ** 2)
        elif sparsity_func == 'log':
            return np.mean(np.log(1 + codes ** 2))
        elif sparsity_func == 'student_t':
            return np.mean(np.log(1 + codes ** 2 / 2))
        else:
            # Default to L1
            return np.mean(np.abs(codes))
    
    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and preprocess input data.
        
        Performs essential data validation and preprocessing including:
        - Shape validation
        - NaN/Inf checking
        - Data type conversion
        - Optional normalization and whitening
        
        Args:
            X: Input data array
            
        Returns:
            Validated and preprocessed data
        """
        
        # Convert to numpy array if needed
        X = np.asarray(X, dtype=np.float64)
        
        # Check for valid shape
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"X cannot be empty, got shape {X.shape}")
        
        # Check for NaN or Inf values
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values")
        
        # Check for reasonable data range
        if np.abs(X).max() > 1e6:
            warnings.warn("X contains very large values, consider normalization")
        
        # Optional preprocessing
        if hasattr(self, 'preprocess_data') and self.preprocess_data:
            X = self._preprocess_data(X)
        
        return X
    
    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Advanced data preprocessing for sparse coding.
        
        Implements preprocessing techniques commonly used in sparse coding:
        - Mean centering
        - Variance normalization
        - Whitening (decorrelation)
        - Patch normalization for image data
        
        Args:
            X: Input data [n_samples, n_features]
            
        Returns:
            Preprocessed data
        """
        
        preprocess_method = getattr(self, 'preprocess_method', 'standardize')
        
        if preprocess_method == 'standardize':
            # Z-score normalization
            scaler = StandardScaler()
            return scaler.fit_transform(X)
            
        elif preprocess_method == 'normalize':
            # L2 normalization per sample
            return normalize(X, axis=1, norm='l2')
            
        elif preprocess_method == 'whiten':
            # ZCA whitening (decorrelation)
            return self._zca_whiten(X)
            
        elif preprocess_method == 'mean_center':
            # Mean centering only
            return X - np.mean(X, axis=0, keepdims=True)
            
        else:
            return X
    
    def _zca_whiten(self, X: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
        """
        ZCA (Zero Component Analysis) whitening.
        
        Transforms data to have zero mean and unit covariance while
        preserving the structure as much as possible.
        
        Args:
            X: Input data [n_samples, n_features]
            regularization: Regularization parameter for numerical stability
            
        Returns:
            Whitened data
        """
        
        # Center the data
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        
        # Compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = X_centered.T @ X_centered / (n_samples - 1)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Add regularization to avoid numerical issues
        eigenvalues = eigenvalues + regularization
        
        # Compute whitening matrix
        whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        
        # Apply whitening
        return X_centered @ whitening_matrix
    
    def _compute_sparsity_level(self, codes: np.ndarray) -> Dict[str, float]:
        """
        Compute various sparsity metrics for coefficient analysis.
        
        Args:
            codes: Sparse coefficients [n_samples, n_components]
            
        Returns:
            Dictionary of sparsity metrics
        """
        
        # L0 "norm" (number of non-zero elements)
        l0_norm = np.mean(np.sum(np.abs(codes) > 1e-12, axis=1))
        
        # L1 norm
        l1_norm = np.mean(np.sum(np.abs(codes), axis=1))
        
        # L2 norm
        l2_norm = np.mean(np.sqrt(np.sum(codes ** 2, axis=1)))
        
        # Gini coefficient (inequality measure)
        gini_coeff = self._gini_coefficient(codes)
        
        # Hoyer sparsity measure
        hoyer_sparsity = self._hoyer_sparsity(codes)
        
        return {
            'l0_norm': l0_norm,
            'l1_norm': l1_norm,
            'l2_norm': l2_norm,
            'gini_coefficient': gini_coeff,
            'hoyer_sparsity': hoyer_sparsity,
            'sparsity_ratio': l0_norm / codes.shape[1]  # Fraction of active coefficients
        }
    
    def _gini_coefficient(self, codes: np.ndarray) -> float:
        """
        Compute Gini coefficient as sparsity measure.
        
        Gini coefficient measures inequality in coefficient magnitudes.
        Values close to 1 indicate high sparsity.
        
        Args:
            codes: Sparse coefficients
            
        Returns:
            Mean Gini coefficient across samples
        """
        
        gini_values = []
        
        for i in range(codes.shape[0]):
            # Sort absolute values
            sorted_abs = np.sort(np.abs(codes[i]))
            n = len(sorted_abs)
            
            # Compute Gini coefficient
            if np.sum(sorted_abs) > 1e-12:
                cumsum = np.cumsum(sorted_abs)
                gini = (2 * np.sum((np.arange(1, n+1) * sorted_abs))) / (n * np.sum(sorted_abs)) - (n+1)/n
                gini_values.append(gini)
            else:
                gini_values.append(1.0)  # Maximally sparse (all zeros)
        
        return np.mean(gini_values)
    
    def _hoyer_sparsity(self, codes: np.ndarray) -> float:
        """
        Compute Hoyer sparsity measure.
        
        Hoyer sparsity: (√n - ||x||₁/||x||₂) / (√n - 1)
        where n is the dimensionality. Values in [0, 1] with 1 being maximally sparse.
        
        Args:
            codes: Sparse coefficients
            
        Returns:
            Mean Hoyer sparsity across samples
        """
        
        hoyer_values = []
        n = codes.shape[1]
        sqrt_n = np.sqrt(n)
        
        for i in range(codes.shape[0]):
            x = codes[i]
            l1_norm = np.sum(np.abs(x))
            l2_norm = np.sqrt(np.sum(x ** 2))
            
            if l2_norm > 1e-12:
                hoyer = (sqrt_n - l1_norm / l2_norm) / (sqrt_n - 1)
                hoyer_values.append(max(0.0, min(1.0, hoyer)))  # Clip to [0, 1]
            else:
                hoyer_values.append(1.0)  # All zeros is maximally sparse
        
        return np.mean(hoyer_values)
    
    def _check_convergence(self, current_objective: float, previous_objectives: List[float],
                          patience: int = 5, min_improvement: float = 1e-6) -> bool:
        """
        Check convergence based on objective function improvements.
        
        Args:
            current_objective: Current objective function value
            previous_objectives: List of previous objective values
            patience: Number of iterations to wait for improvement
            min_improvement: Minimum improvement threshold
            
        Returns:
            True if converged, False otherwise
        """
        
        if len(previous_objectives) < patience:
            return False
        
        # Check if improvement is less than threshold for 'patience' iterations
        recent_objectives = previous_objectives[-patience:]
        improvements = [abs(recent_objectives[i] - recent_objectives[i-1]) 
                       for i in range(1, len(recent_objectives))]
        
        return all(imp < min_improvement for imp in improvements)
    
    def _validate_dictionary_quality(self) -> Dict[str, Any]:
        """
        Validate dictionary quality and detect potential issues.
        
        Returns:
            Dictionary quality metrics and warnings
        """
        
        if self.dictionary_ is None:
            return {'error': 'Dictionary not initialized'}
        
        quality_metrics = {}
        warnings_list = []
        
        # Check atom norms
        atom_norms = np.linalg.norm(self.dictionary_, axis=1)
        quality_metrics['atom_norms_mean'] = np.mean(atom_norms)
        quality_metrics['atom_norms_std'] = np.std(atom_norms)
        
        # Check for atoms with very small norms (dead atoms)
        dead_atoms = np.sum(atom_norms < 1e-6)
        quality_metrics['dead_atoms'] = dead_atoms
        if dead_atoms > 0:
            warnings_list.append(f"Found {dead_atoms} dead atoms with very small norms")
        
        # Check atom coherence (mutual coherence)
        gram_matrix = self.dictionary_ @ self.dictionary_.T
        np.fill_diagonal(gram_matrix, 0)  # Remove diagonal elements
        max_coherence = np.max(np.abs(gram_matrix))
        quality_metrics['max_coherence'] = max_coherence
        
        if max_coherence > 0.9:
            warnings_list.append(f"High atom coherence ({max_coherence:.3f}) may affect sparse recovery")
        
        # Check for duplicate atoms
        duplicate_threshold = 0.99
        duplicate_pairs = np.where(np.abs(gram_matrix) > duplicate_threshold)
        n_duplicates = len(duplicate_pairs[0])
        quality_metrics['duplicate_atoms'] = n_duplicates
        
        if n_duplicates > 0:
            warnings_list.append(f"Found {n_duplicates} nearly duplicate atom pairs")
        
        quality_metrics['warnings'] = warnings_list
        return quality_metrics
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator (scikit-learn compatibility)."""
        
        params = {
            'n_components': self.n_components,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tolerance': self.tolerance,
            'algorithm': self.algorithm,
            'dict_init': self.dict_init,
            'sparsity_func': self.sparsity_func,
            'lambda_schedule': self.lambda_schedule,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'batch_size': self.batch_size,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
        
        return params
    
    def set_params(self, **params) -> 'SparseCoder':
        """Set parameters for this estimator (scikit-learn compatibility)."""
        
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter: {param}")
        
        return self


# Export the mixin class
__all__ = ['UtilitiesValidationMixin']


if __name__ == "__main__":
    print("🏗️ Sparse Coding - Utilities and Validation Module")
    print("=" * 50)
    print("📊 MODULE CONTENTS:")
    print("  • UtilitiesValidationMixin - Essential utility functions")
    print("  • Soft/hard thresholding (proximal operators)")
    print("  • Reconstruction error and sparsity cost calculation")
    print("  • Advanced data preprocessing and validation")
    print("  • Sparsity metrics (Gini, Hoyer, L0/L1/L2 norms)")
    print("  • Dictionary quality validation and diagnostics")
    print("  • Convergence checking and parameter validation")
    print("  • ZCA whitening and preprocessing methods")
    print("")
    print("✅ Utilities and validation module loaded successfully!")
    print("🔬 Comprehensive sparse coding support functions!")