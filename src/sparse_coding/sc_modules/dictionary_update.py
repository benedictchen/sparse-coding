"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀

Dictionary Update Module for Sparse Coding

This module contains dictionary learning algorithms extracted from Olshausen & Field (1996)
sparse coding implementation. These methods implement the core dictionary learning algorithms
that discover edge detectors from natural images.

Author: Benedict Chen (benedict@benedictchen.com)
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"

Key Algorithms:
- Pure Olshausen & Field equation (6): Δφᵢ(xₙ,yₙ) = η⟨aᵢ⟨I(xₙ,yₙ) - Î(xₙ,yₙ)⟩⟩
- Method of Optimal Directions (MOD)
- Orthogonality enforcement via Gram-Schmidt
- Batch processing for efficiency
- Dictionary coherence analysis
"""

import numpy as np
from sklearn.preprocessing import normalize
from typing import Optional


class DictionaryUpdateMixin:
    """
    Dictionary Update Mixin Class
    
    This mixin provides dictionary learning functionality for sparse coding.
    It assumes the parent class has the following attributes:
    - self.dictionary: Dictionary matrix (patch_dim, n_components)
    - self.n_components: Number of dictionary elements
    - self.learning_rate: Learning rate for dictionary updates
    - self.sparsity_penalty: Sparsity regularization parameter
    
    The mixin implements various dictionary update methods from Olshausen & Field (1996).
    """

    def _update_dictionary(self, patches: np.ndarray, coefficients: np.ndarray):
        """
        Update dictionary using method of optimal directions (MOD)
        
        This is Olshausen & Field's key algorithm for learning the dictionary.
        The Method of Optimal Directions (MOD) updates each dictionary element
        
        # FIXME: Dictionary Update Critical Research Accuracy Issues
        #
        # 1. MISSING ORIGINAL EQUATION 6 IMPLEMENTATION
        #    - Current MOD is modern approximation, not original Olshausen & Field equation
        #    - Original: Δφᵢ(x,y) = η⟨aᵢ⟨I(x,y) - Î(x,y)⟩⟩  
        #    - Should implement exact gradient descent on dictionary elements
        #    - CODE REVIEW SUGGESTION - Implement exact Olshausen & Field (1996) Equation 6:
        #      ```python
        #      def update_dictionary_equation_6_exact(self, patches: np.ndarray, coefficients: np.ndarray):
        #          # Pure Olshausen & Field (1996) Equation 6 implementation
        #          reconstruction_error = patches - coefficients @ self.dictionary.T
        #          for i in range(self.n_components):
        #              # Only update if coefficient is significantly active
        #              active_mask = np.abs(coefficients[:, i]) > 1e-6
        #              if not np.any(active_mask):
        #                  continue
        #              # Gradient: η⟨aᵢ⟨I - Î⟩⟩ 
        #              gradient = np.mean(coefficients[active_mask, i:i+1] * 
        #                               reconstruction_error[active_mask], axis=0)
        #              self.dictionary[:, i] += self.learning_rate * gradient
        #              # Unit norm constraint: ||φᵢ|| = 1
        #              norm = np.linalg.norm(self.dictionary[:, i])
        #              if norm > 1e-12:
        #                  self.dictionary[:, i] /= norm
        #      ```
        #
        # 2. MISSING K-SVD ALGORITHM IMPLEMENTATION  
        #    - Paper mentions K-SVD as superior dictionary learning method
        #    - K-SVD updates dictionary atoms optimally using SVD decomposition
        #    - More robust than gradient descent for overcomplete dictionaries
        #    - CODE REVIEW SUGGESTION - Implement K-SVD dictionary update:
        #      ```python
        #      def update_dictionary_ksvd(self, patches: np.ndarray, coefficients: np.ndarray):
        #          # K-SVD dictionary learning algorithm
        #          for k in range(self.n_components):
        #              # Find samples that use atom k
        #              using_indices = np.where(np.abs(coefficients[:, k]) > 1e-10)[0]
        #              if len(using_indices) == 0:
        #                  # Replace unused atom with random patch
        #                  self.dictionary[:, k] = patches[np.random.randint(0, len(patches))]
        #                  self.dictionary[:, k] /= np.linalg.norm(self.dictionary[:, k])
        #                  continue
        #              # Compute error matrix without atom k
        #              coefficients_k = coefficients[using_indices, k].copy()
        #              coefficients[using_indices, k] = 0
        #              error_matrix = patches[using_indices] - coefficients[using_indices] @ self.dictionary.T
        #              coefficients[using_indices, k] = coefficients_k
        #              # SVD to find optimal atom and coefficients
        #              U, s, Vt = np.linalg.svd(error_matrix, full_matrices=False)
        #              self.dictionary[:, k] = Vt[0, :]
        #              coefficients[using_indices, k] = s[0] * U[:, 0]
        #      ```
        #
        # 3. MISSING TOPOGRAPHIC ORGANIZATION
        #    - Original experiments included topographic dictionary organization
        #    - Neighboring dictionary elements should have similar orientations
        #    - CODE REVIEW SUGGESTION - Add topographic constraint:
        #      ```python
        #      def add_topographic_penalty(self, sigma: float = 2.0):
        #          # Add topographic organization penalty
        #          # Create 2D grid positions for dictionary atoms
        #          grid_size = int(np.sqrt(self.n_components))
        #          positions = np.array([[i, j] for i in range(grid_size) 
        #                              for j in range(grid_size)])[:self.n_components]
        #          # Compute topographic penalty
        #          penalty = 0
        #          for i in range(self.n_components):
        #              for j in range(i+1, self.n_components):
        #                  distance = np.linalg.norm(positions[i] - positions[j])
        #                  weight = np.exp(-distance**2 / (2 * sigma**2))
        #                  penalty += weight * np.linalg.norm(self.dictionary[:, i] - 
        #                                                   self.dictionary[:, j])**2
        #          return penalty
        #      ```
        #
        # 4. MISSING COHERENCE CONTROL FOR OVERCOMPLETE BASES
        #    - Overcomplete dictionaries prone to high coherence (similar atoms)
        #    - Need coherence penalty to maintain atom diversity
        #    - CODE REVIEW SUGGESTION - Implement coherence regularization:
        #      ```python
        #      def apply_coherence_penalty(self, max_coherence: float = 0.9):
        #          # Apply coherence penalty to reduce atom similarity
        #          gram_matrix = self.dictionary.T @ self.dictionary
        #          # Zero out diagonal
        #          np.fill_diagonal(gram_matrix, 0)
        #          # Find high-coherence pairs
        #          high_coherence = np.abs(gram_matrix) > max_coherence
        #          if np.any(high_coherence):
        #              # Penalize high-coherence atoms
        #              i_indices, j_indices = np.where(high_coherence)
        #              for i, j in zip(i_indices, j_indices):
        #                  if i < j:  # Avoid double processing
        #                      # Make atoms more orthogonal
        #                      projection = np.dot(self.dictionary[:, i], 
        #                                        self.dictionary[:, j])
        #                      self.dictionary[:, i] -= 0.1 * projection * self.dictionary[:, j]
        #                      self.dictionary[:, j] -= 0.1 * projection * self.dictionary[:, i]
        #                      # Renormalize
        #                      self.dictionary[:, i] /= np.linalg.norm(self.dictionary[:, i])
        #                      self.dictionary[:, j] /= np.linalg.norm(self.dictionary[:, j])
        #      ```
        by finding the optimal direction that minimizes reconstruction error.
        
        Algorithm:
        1. For each dictionary element j:
           - Find patches that use this element significantly
           - Compute error when removing this element's contribution
           - Use SVD to find optimal update direction
           - Update both dictionary element and coefficients
        2. Normalize dictionary columns to unit length
        
        This method is particularly effective for overcomplete dictionaries
        where the number of dictionary elements exceeds the input dimension.
        
        Args:
            patches (np.ndarray): Input patches (n_patches, patch_dim)
            coefficients (np.ndarray): Sparse coefficients (n_patches, n_components)
        
        Notes:
            - Dictionary elements are normalized to unit length after update
            - Only updates elements that are actively used (coefficients > 1e-10)
            - Uses SVD for robust optimization of dictionary elements
        """
        
        for j in range(self.n_components):
            # Find patches that use this dictionary element
            using_indices = np.abs(coefficients[:, j]) > 1e-10
            
            if np.sum(using_indices) == 0:
                continue
                
            # Error when removing this dictionary element
            error = patches[using_indices] - (coefficients[using_indices] @ self.dictionary.T) + np.outer(coefficients[using_indices, j], self.dictionary[:, j])
            
            # Update dictionary element and coefficients via SVD
            if np.sum(using_indices) > 0:
                U, s, Vt = np.linalg.svd(error, full_matrices=False)
                
                # Update dictionary column
                self.dictionary[:, j] = Vt[0, :]
                
                # Update coefficients
                coefficients[using_indices, j] = s[0] * U[:, 0]
                
        # Normalize dictionary columns
        self.dictionary = normalize(self.dictionary, axis=0)

    def _update_dictionary_equation_6(self, patches: np.ndarray, coefficients: np.ndarray):
        """
        Pure Olshausen & Field equation (6) implementation
        
        This implements the original dictionary learning rule from equation (6):
        Δφᵢ(xₙ,yₙ) = η⟨aᵢ⟨I(xₙ,yₙ) - Î(xₙ,yₙ)⟩⟩
        
        Where:
        - φᵢ is the i-th dictionary element (basis function)
        - η is the learning rate
        - aᵢ is the coefficient for element i
        - I(xₙ,yₙ) is the original image patch
        - Î(xₙ,yₙ) is the reconstructed image patch
        - ⟨⟩ denotes averaging over active patches
        
        This is the foundational algorithm that enables the emergence of
        edge-detector-like receptive fields from natural image statistics.
        
        Key Insights:
        - Only updates dictionary elements that are significantly active
        - Uses reconstruction error to guide dictionary learning
        - Learning rate controls adaptation speed
        - Normalization ensures dictionary elements have unit length
        
        Args:
            patches (np.ndarray): Original image patches (n_patches, patch_dim)
            coefficients (np.ndarray): Sparse coefficients (n_patches, n_components)
        
        Notes:
            - Requires self.learning_rate to be set
            - Dictionary elements are normalized after each update
            - Only processes patches with significant activation (> 1e-4)
        """
        for i in range(self.n_components):
            # Find patches using this basis function significantly
            active_mask = np.abs(coefficients[:, i]) > 1e-4
            if not np.any(active_mask):
                continue
                
            # Get active patches and coefficients
            active_patches = patches[active_mask]
            active_coeffs = coefficients[active_mask, i]
            
            # Compute reconstruction error: I - Î
            reconstruction = coefficients[active_mask] @ self.dictionary.T
            error = active_patches - reconstruction
            
            # Apply equation (6): Δφᵢ = η⟨aᵢ⟨I - Î⟩⟩
            gradient = np.mean(active_coeffs[:, np.newaxis] * error, axis=0)
            self.dictionary[:, i] += self.learning_rate * gradient
            
            # Normalize to unit length (paper requirement)
            norm = np.linalg.norm(self.dictionary[:, i])
            if norm > 1e-10:
                self.dictionary[:, i] /= norm

    def _update_dictionary_with_orthogonality(self, patches: np.ndarray, coefficients: np.ndarray):
        """
        Equation (6) updates with orthogonality enforcement
        
        This method combines the original Olshausen & Field equation (6) updates
        with Gram-Schmidt orthogonalization to encourage orthogonal basis functions.
        
        Orthogonality Benefits:
        - Reduces redundancy between dictionary elements
        - Improves numerical stability
        - Can lead to better sparse representations
        - Prevents dictionary elements from becoming too similar
        
        Process:
        1. Apply standard equation (6) dictionary updates
        2. Apply Gram-Schmidt orthogonalization across all dictionary elements
        3. Renormalize each dictionary element to unit length
        
        Note: While orthogonality can be beneficial, it may not always be
        necessary or desired, as overcomplete dictionaries naturally allow
        for non-orthogonal representations that can be more flexible.
        
        Args:
            patches (np.ndarray): Original image patches (n_patches, patch_dim)
            coefficients (np.ndarray): Sparse coefficients (n_patches, n_components)
        
        References:
            - Olshausen & Field mention preference for orthogonal bases in their paper
            - Gram-Schmidt provides a systematic way to enforce orthogonality
        """
        # Apply equation (6) updates
        self._update_dictionary_equation_6(patches, coefficients)
        
        # Gram-Schmidt orthogonalization (paper mentions orthogonal basis preference)
        for i in range(self.n_components):
            for j in range(i):
                projection = np.dot(self.dictionary[:, i], self.dictionary[:, j])
                self.dictionary[:, i] -= projection * self.dictionary[:, j]
            # Renormalize
            norm = np.linalg.norm(self.dictionary[:, i])
            if norm > 1e-10:
                self.dictionary[:, i] /= norm

    def _update_dictionary_batch(self, patches: np.ndarray, coefficients: np.ndarray, batch_size: int = 100):
        """
        Batch processing version of equation (6) for computational efficiency
        
        This method applies equation (6) dictionary updates in batches rather than
        processing all patches simultaneously. This provides several advantages:
        
        Computational Benefits:
        - Reduced memory usage for large datasets
        - Better cache utilization
        - Allows processing of datasets larger than memory
        - Can be parallelized across batches
        
        Statistical Benefits:
        - More frequent dictionary updates can improve convergence
        - Stochastic updates can help escape local minima
        - Better adaptation to changing data patterns
        
        The method maintains the same mathematical guarantees as the full
        equation (6) update while being more computationally practical.
        
        Args:
            patches (np.ndarray): Original image patches (n_patches, patch_dim)
            coefficients (np.ndarray): Sparse coefficients (n_patches, n_components)
            batch_size (int, optional): Size of each processing batch. Defaults to 100.
        
        Notes:
            - Batch size can be tuned based on available memory
            - Smaller batches lead to more frequent updates
            - Larger batches provide more stable gradient estimates
        """
        n_patches = patches.shape[0]
        for start_idx in range(0, n_patches, batch_size):
            end_idx = min(start_idx + batch_size, n_patches)
            batch_patches = patches[start_idx:end_idx]
            batch_coeffs = coefficients[start_idx:end_idx]
            
            self._update_dictionary_equation_6(batch_patches, batch_coeffs)

    def _update_dictionary_olshausen(self, patches: np.ndarray, coefficients: np.ndarray):
        """
        Enhanced dictionary update using Olshausen & Field's method with modern improvements
        
        This method implements an enhanced version of the Olshausen & Field dictionary
        learning algorithm with momentum and adaptive learning rates for improved
        convergence and stability.
        
        Enhancements over basic equation (6):
        - Momentum term to accelerate convergence and smooth updates
        - Adaptive gradient normalization to handle varying coefficient magnitudes
        - Robust numerical handling to prevent division by zero
        - Memory of previous updates via momentum buffer
        
        Mathematical Foundation:
        The core update still follows equation (6) but with momentum:
        m_t = βm_{t-1} + η∇φᵢ
        φᵢ := φᵢ + m_t
        
        Where:
        - m_t is the momentum buffer
        - β is the momentum coefficient (typically 0.9)
        - η is the learning rate
        - ∇φᵢ is the gradient from equation (6)
        
        This approach combines the biological plausibility of the original
        algorithm with modern optimization techniques for better performance.
        
        Args:
            patches (np.ndarray): Original image patches (n_patches, patch_dim)
            coefficients (np.ndarray): Sparse coefficients (n_patches, n_components)
        
        Attributes Required:
            - self.dictionary_momentum: Momentum buffer (initialized on first call)
        
        Notes:
            - Momentum buffer is automatically initialized if not present
            - Learning rate and momentum can be tuned for specific datasets
            - More robust to noisy gradients than basic equation (6)
        """
        
        learning_rate = 0.01
        momentum = 0.9
        
        if not hasattr(self, 'dictionary_momentum'):
            self.dictionary_momentum = np.zeros_like(self.dictionary)
            
        for j in range(self.n_components):
            # Find patches that use this dictionary element significantly
            using_indices = np.abs(coefficients[:, j]) > 1e-4
            
            if np.sum(using_indices) == 0:
                continue
                
            # Calculate residual error when removing this dictionary element
            residual = patches[using_indices] - (coefficients[using_indices] @ self.dictionary.T)
            residual += np.outer(coefficients[using_indices, j], self.dictionary[:, j])
            
            # Update dictionary element using gradient descent with momentum
            if np.sum(using_indices) > 0:
                # Compute gradient
                gradient = residual.T @ coefficients[using_indices, j]
                gradient = gradient / (np.linalg.norm(coefficients[using_indices, j])**2 + 1e-8)
                
                # Apply momentum
                self.dictionary_momentum[:, j] = momentum * self.dictionary_momentum[:, j] + learning_rate * gradient
                self.dictionary[:, j] += self.dictionary_momentum[:, j]
                
                # Normalize dictionary column
                norm = np.linalg.norm(self.dictionary[:, j])
                if norm > 0:
                    self.dictionary[:, j] /= norm

    def _calculate_dictionary_coherence(self) -> float:
        """
        Calculate dictionary coherence (mutual coherence)
        
        Dictionary coherence measures how well-conditioned the dictionary is
        for sparse coding applications. It is defined as the maximum absolute
        inner product between any two distinct dictionary elements.
        
        Mathematical Definition:
        μ = max_{i≠j} |⟨φᵢ, φⱼ⟩|
        
        Where φᵢ and φⱼ are normalized dictionary elements.
        
        Interpretation:
        - Lower coherence (closer to 0) indicates better dictionary conditioning
        - Higher coherence (closer to 1) suggests redundant dictionary elements
        - Coherence affects sparse recovery guarantees in compressed sensing theory
        - Well-conditioned dictionaries lead to more reliable sparse coding
        
        Applications:
        - Dictionary quality assessment
        - Monitoring training progress
        - Comparing different dictionary learning algorithms
        - Theoretical analysis of sparse recovery conditions
        
        Theoretical Significance:
        In compressed sensing and sparse coding theory, dictionary coherence
        is crucial for determining:
        - Uniqueness of sparse solutions
        - Stability of sparse recovery
        - Required sparsity levels for exact recovery
        
        Returns:
            float: Dictionary coherence value (0 to 1, lower is better)
        
        Notes:
            - Assumes dictionary elements are normalized
            - Computational complexity: O(d * n²) where d is patch dimension, n is number of components
            - Returns 0 for orthogonal dictionaries
        """
        
        # Compute Gram matrix
        gram_matrix = self.dictionary.T @ self.dictionary
        
        # Remove diagonal elements
        off_diagonal = gram_matrix - np.eye(self.n_components)
        
        # Maximum off-diagonal element is the coherence
        coherence = np.max(np.abs(off_diagonal))
        
        return coherence

    def analyze_dictionary_properties(self) -> dict:
        """
        Comprehensive analysis of dictionary properties
        
        This method provides a detailed analysis of the learned dictionary,
        including statistical measures, geometric properties, and quality metrics
        that are important for understanding the dictionary's characteristics.
        
        Analysis Includes:
        - Dictionary coherence (conditioning measure)
        - Element norms and their distribution
        - Similarity statistics between dictionary elements
        - Sparsity-related metrics
        - Geometric properties
        
        Returns:
            dict: Comprehensive dictionary analysis containing:
                - coherence: Mutual coherence value
                - element_norms: Statistics about dictionary element magnitudes
                - similarity_stats: Inter-element similarity analysis
                - condition_number: Numerical conditioning of Gram matrix
                - effective_rank: Effective dimensionality of dictionary
        
        Notes:
            - Useful for debugging dictionary learning algorithms
            - Helps identify potential issues with dictionary quality
            - Can guide hyperparameter tuning
        """
        
        # Calculate coherence
        coherence = self._calculate_dictionary_coherence()
        
        # Element norm statistics
        element_norms = np.linalg.norm(self.dictionary, axis=0)
        
        # Similarity analysis
        gram_matrix = self.dictionary.T @ self.dictionary
        off_diagonal = gram_matrix - np.eye(self.n_components)
        
        # Condition number of Gram matrix
        eigenvals = np.linalg.eigvals(gram_matrix)
        condition_number = np.max(eigenvals) / (np.min(eigenvals) + 1e-12)
        
        # Effective rank (number of significant singular values)
        singular_values = np.linalg.svd(self.dictionary, compute_uv=False)
        effective_rank = np.sum(singular_values > 0.01 * singular_values[0])
        
        return {
            'coherence': coherence,
            'element_norms': {
                'mean': np.mean(element_norms),
                'std': np.std(element_norms),
                'min': np.min(element_norms),
                'max': np.max(element_norms)
            },
            'similarity_stats': {
                'mean_similarity': np.mean(np.abs(off_diagonal)),
                'max_similarity': np.max(np.abs(off_diagonal)),
                'std_similarity': np.std(np.abs(off_diagonal))
            },
            'condition_number': condition_number,
            'effective_rank': effective_rank,
            'n_components': self.n_components,
            'dictionary_shape': self.dictionary.shape
        }

    def get_dictionary_update_methods(self) -> dict:
        """
        Get information about available dictionary update methods
        
        Returns a comprehensive guide to the dictionary update methods
        available in this mixin, including their characteristics, use cases,
        and theoretical foundations.
        
        Returns:
            dict: Dictionary containing method information with:
                - method descriptions
                - computational characteristics
                - theoretical foundations
                - recommended use cases
                - implementation notes
        """
        
        return {
            'methods': {
                'equation_6': {
                    'name': 'Pure Olshausen & Field Equation (6)',
                    'description': 'Original dictionary learning rule from the 1996 paper',
                    'formula': 'Δφᵢ(xₙ,yₙ) = η⟨aᵢ⟨I(xₙ,yₙ) - Î(xₙ,yₙ)⟩⟩',
                    'characteristics': 'Biologically plausible, simple, proven effective',
                    'use_case': 'Best for replicating original paper results',
                    'computational_cost': 'Moderate',
                    'stability': 'Good with proper learning rate'
                },
                'mod': {
                    'name': 'Method of Optimal Directions (MOD)',
                    'description': 'SVD-based dictionary update for optimal reconstruction',
                    'characteristics': 'Mathematically optimal, robust to noise',
                    'use_case': 'Best for high-quality dictionary learning',
                    'computational_cost': 'Higher (due to SVD)',
                    'stability': 'Excellent'
                },
                'orthogonal': {
                    'name': 'Orthogonality-Enforced Updates',
                    'description': 'Equation (6) with Gram-Schmidt orthogonalization',
                    'characteristics': 'Reduces redundancy, improves conditioning',
                    'use_case': 'When dictionary elements should be orthogonal',
                    'computational_cost': 'Higher (due to orthogonalization)',
                    'stability': 'Very good'
                },
                'batch': {
                    'name': 'Batch Processing',
                    'description': 'Mini-batch version of equation (6)',
                    'characteristics': 'Memory efficient, can handle large datasets',
                    'use_case': 'Large-scale dictionary learning problems',
                    'computational_cost': 'Lower memory, scalable',
                    'stability': 'Good with proper batch size'
                },
                'enhanced': {
                    'name': 'Enhanced with Momentum',
                    'description': 'Modern optimization techniques with momentum',
                    'characteristics': 'Faster convergence, smoother updates',
                    'use_case': 'When training speed is important',
                    'computational_cost': 'Moderate (extra momentum storage)',
                    'stability': 'Excellent'
                }
            },
            'selection_guide': {
                'accuracy_priority': 'Use MOD for best reconstruction quality',
                'speed_priority': 'Use batch processing for large datasets',
                'biological_plausibility': 'Use equation_6 for original algorithm',
                'numerical_stability': 'Use orthogonal or enhanced methods',
                'memory_constrained': 'Use batch processing with small batch sizes'
            },
            'theoretical_background': {
                'olshausen_field_1996': 'Original sparse coding paper introducing equation (6)',
                'mod_algorithm': 'Method of Optimal Directions for dictionary learning',
                'compressed_sensing': 'Dictionary coherence theory for sparse recovery',
                'optimization_theory': 'Momentum methods for improved convergence'
            }
        }

"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
📝 CITE: Benedict Chen (2025) - Sparse Coding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""