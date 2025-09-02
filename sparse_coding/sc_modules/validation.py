"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀

Validation and Analysis Module for Sparse Coding

This module contains validation functions and analysis utilities for sparse coding
operations, extracted from the main SparseCoder class to provide modular
validation capabilities.

Author: Benedict Chen
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Optional, Any


class ValidationMixin:
    """
    Mixin class providing validation and analysis capabilities for sparse coding.
    
    This mixin can be used with sparse coding classes to add comprehensive
    validation of parameters, configuration, and analysis of learned dictionaries.
    
    Key Features:
    - Parameter validation with detailed error messages
    - Sparseness function configuration and validation
    - Dictionary analysis including orientation preferences and statistics
    - Comprehensive sparseness function information system
    """
    
    def _validate_configuration(self):
        """
        Validate all configuration parameters for sparse coding.
        
        Performs comprehensive validation of:
        - Patch size parameters (must be positive integers)
        - Number of components (must be positive)
        - Sparsity penalty (must be non-negative)
        - Optimization method validity
        - Sparseness function validity
        - Learning parameters
        
        Raises:
            ValueError: If any configuration parameter is invalid
            
        Note:
            This method should be called during initialization and whenever
            configuration parameters are modified.
        """
        # Validate patch size
        if not hasattr(self, 'patch_size') or len(self.patch_size) != 2:
            raise ValueError("patch_size must be a tuple of two integers")
        
        if self.patch_size[0] <= 0 or self.patch_size[1] <= 0:
            raise ValueError(f"Invalid patch size: {self.patch_size}. Both dimensions must be positive")
        
        if not isinstance(self.patch_size[0], int) or not isinstance(self.patch_size[1], int):
            raise ValueError(f"Patch size dimensions must be integers, got: {self.patch_size}")
        
        # Validate number of components
        if not hasattr(self, 'n_components'):
            raise ValueError("n_components must be set")
        
        if self.n_components <= 0:
            raise ValueError(f"Invalid number of components: {self.n_components}. Must be positive")
        
        if not isinstance(self.n_components, int):
            raise ValueError(f"n_components must be an integer, got: {type(self.n_components)}")
        
        # Validate sparsity penalty
        if not hasattr(self, 'sparsity_penalty'):
            raise ValueError("sparsity_penalty must be set")
        
        if self.sparsity_penalty < 0:
            raise ValueError(f"Invalid sparsity penalty: {self.sparsity_penalty}. Must be non-negative")
        
        # Validate optimization method
        if hasattr(self, 'optimization_method'):
            valid_optimization_methods = ['coordinate_descent', 'equation_5', 'fista', 'proximal_gradient']
            if self.optimization_method not in valid_optimization_methods:
                raise ValueError(f"Invalid optimization method: {self.optimization_method}. "
                               f"Must be one of: {valid_optimization_methods}")
        
        # Validate sparseness function
        if hasattr(self, 'sparseness_function'):
            valid_sparseness_functions = ['l1', 'log', 'gaussian', 'huber', 'elastic_net', 'cauchy', 'student_t']
            if self.sparseness_function not in valid_sparseness_functions:
                raise ValueError(f"Invalid sparseness function: {self.sparseness_function}. "
                               f"Must be one of: {valid_sparseness_functions}")
        
        # Validate L1 solver
        if hasattr(self, 'l1_solver'):
            valid_l1_solvers = ['coordinate_descent', 'lbfgs_b', 'fista']
            if self.l1_solver not in valid_l1_solvers:
                raise ValueError(f"Invalid L1 solver: {self.l1_solver}. "
                               f"Must be one of: {valid_l1_solvers}")
        
        # Validate learning parameters
        if hasattr(self, 'max_iter'):
            if self.max_iter <= 0:
                raise ValueError(f"Invalid max_iter: {self.max_iter}. Must be positive")
        
        if hasattr(self, 'tolerance'):
            if self.tolerance <= 0:
                raise ValueError(f"Invalid tolerance: {self.tolerance}. Must be positive")
        
        if hasattr(self, 'learning_rate'):
            if self.learning_rate <= 0:
                raise ValueError(f"Invalid learning_rate: {self.learning_rate}. Must be positive")
        
        # Validate dictionary update method
        if hasattr(self, 'dictionary_update_method'):
            valid_dict_methods = ['equation_6', 'orthogonal', 'batch']
            if self.dictionary_update_method not in valid_dict_methods:
                raise ValueError(f"Invalid dictionary_update_method: {self.dictionary_update_method}. "
                               f"Must be one of: {valid_dict_methods}")
        
        # Validate whitening method
        if hasattr(self, 'whitening_method'):
            valid_whitening_methods = ['olshausen_field', 'zca', 'standard']
            if self.whitening_method not in valid_whitening_methods:
                raise ValueError(f"Invalid whitening_method: {self.whitening_method}. "
                               f"Must be one of: {valid_whitening_methods}")
    
    def _validate_sparseness_function_parameters(self):
        """
        Validate parameters specific to the chosen sparseness function.
        
        Checks that all required parameters for the current sparseness function
        are present and within valid ranges.
        
        Raises:
            ValueError: If sparseness function parameters are invalid or missing
        """
        if not hasattr(self, 'sparseness_function'):
            return
        
        if self.sparseness_function == 'huber':
            if hasattr(self, 'huber_delta'):
                if self.huber_delta <= 0:
                    raise ValueError(f"huber_delta must be positive, got: {self.huber_delta}")
        
        elif self.sparseness_function == 'elastic_net':
            if hasattr(self, 'elastic_net_l1_ratio'):
                if not 0 <= self.elastic_net_l1_ratio <= 1:
                    raise ValueError(f"elastic_net_l1_ratio must be in [0, 1], got: {self.elastic_net_l1_ratio}")
        
        elif self.sparseness_function == 'cauchy':
            if hasattr(self, 'cauchy_gamma'):
                if self.cauchy_gamma <= 0:
                    raise ValueError(f"cauchy_gamma must be positive, got: {self.cauchy_gamma}")
        
        elif self.sparseness_function == 'student_t':
            if hasattr(self, 'student_t_nu'):
                if self.student_t_nu <= 0:
                    raise ValueError(f"student_t_nu must be positive, got: {self.student_t_nu}")
    
    def _validate_dictionary(self, dictionary: Optional[np.ndarray] = None) -> bool:
        """
        Validate dictionary matrix properties.
        
        Args:
            dictionary: Dictionary matrix to validate. If None, uses self.dictionary
            
        Returns:
            bool: True if dictionary is valid
            
        Raises:
            ValueError: If dictionary has invalid properties
        """
        if dictionary is None:
            if not hasattr(self, 'dictionary') or self.dictionary is None:
                raise ValueError("No dictionary to validate")
            dictionary = self.dictionary
        
        # Check dimensions
        if dictionary.ndim != 2:
            raise ValueError(f"Dictionary must be 2D matrix, got shape: {dictionary.shape}")
        
        expected_input_dim = self.patch_size[0] * self.patch_size[1]
        if dictionary.shape[0] != expected_input_dim:
            raise ValueError(f"Dictionary input dimension {dictionary.shape[0]} doesn't match "
                           f"expected patch dimension {expected_input_dim}")
        
        if dictionary.shape[1] != self.n_components:
            raise ValueError(f"Dictionary has {dictionary.shape[1]} components, expected {self.n_components}")
        
        # Check for NaN or infinite values
        if not np.isfinite(dictionary).all():
            raise ValueError("Dictionary contains NaN or infinite values")
        
        # Check if columns are normalized (within tolerance)
        column_norms = np.linalg.norm(dictionary, axis=0)
        tolerance = 1e-3  # Allow some numerical error
        if not np.allclose(column_norms, 1.0, atol=tolerance):
            warnings.warn(f"Dictionary columns are not normalized. "
                         f"Norms range: [{column_norms.min():.3f}, {column_norms.max():.3f}]")
        
        return True
    
    def _analyze_dictionary(self):
        """
        Analyze properties of the learned dictionary.
        
        Performs comprehensive analysis including:
        - Orientation preferences of dictionary elements
        - Element norms and normalization statistics
        - Inter-element similarity analysis
        - Sparsity and coverage statistics
        
        Prints detailed analysis to stdout and stores results in self._analysis_results
        if that attribute exists.
        
        Note:
            This analysis is particularly important for understanding whether
            the learned dictionary exhibits the expected properties (oriented
            edge detectors) described in Olshausen & Field 1996.
        """
        if not hasattr(self, 'dictionary') or self.dictionary is None:
            print("⚠️  No dictionary to analyze")
            return
        
        print(f"\n📊 Dictionary Analysis:")
        print(f"   Dictionary shape: {self.dictionary.shape}")
        
        # Calculate orientation preferences using gradient analysis
        orientations = []
        edge_strengths = []
        
        for i in range(self.n_components):
            element = self.dictionary[:, i].reshape(self.patch_size)
            
            # Calculate gradients to detect edge-like patterns
            grad_y = np.gradient(element, axis=0)
            grad_x = np.gradient(element, axis=1)
            
            # Edge strength as magnitude of dominant gradient
            grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
            edge_strength = np.max(grad_magnitude)
            edge_strengths.append(edge_strength)
            
            # Orientation calculation using dominant gradient direction
            if edge_strength > 0.1:  # Only consider elements with significant gradients
                # Find location of maximum gradient
                max_idx = np.unravel_index(np.argmax(grad_magnitude), grad_magnitude.shape)
                dominant_grad_y = grad_y[max_idx]
                dominant_grad_x = grad_x[max_idx]
                
                if abs(dominant_grad_x) + abs(dominant_grad_y) > 1e-6:
                    orientation = np.arctan2(dominant_grad_y, dominant_grad_x) * 180 / np.pi
                    # Normalize to [0, 180) degrees (since orientation is symmetric)
                    orientation = orientation % 180
                    orientations.append(orientation)
        
        # Analyze orientations
        if orientations:
            orientations = np.array(orientations)
            print(f"   • Oriented elements: {len(orientations)}/{self.n_components} "
                  f"({100*len(orientations)/self.n_components:.1f}%)")
            print(f"   • Orientation range: {np.min(orientations):.1f}° - {np.max(orientations):.1f}°")
            print(f"   • Mean orientation: {np.mean(orientations):.1f}° ± {np.std(orientations):.1f}°")
            
            # Check for good orientation coverage (should span most of 0-180°)
            orientation_coverage = (np.max(orientations) - np.min(orientations)) / 180.0
            print(f"   • Orientation coverage: {orientation_coverage:.1%}")
        else:
            print("   • No oriented elements detected (may indicate poor learning)")
        
        # Dictionary element statistics
        element_norms = np.linalg.norm(self.dictionary, axis=0)
        print(f"   • Element norms: {element_norms.mean():.3f} ± {element_norms.std():.3f}")
        print(f"   • Norm range: [{element_norms.min():.3f}, {element_norms.max():.3f}]")
        
        # Edge strength analysis
        if edge_strengths:
            edge_strengths = np.array(edge_strengths)
            print(f"   • Average edge strength: {edge_strengths.mean():.3f} ± {edge_strengths.std():.3f}")
            strong_edges = np.sum(edge_strengths > 0.2)  # Threshold for "strong" edges
            print(f"   • Strong edge detectors: {strong_edges}/{self.n_components} "
                  f"({100*strong_edges/self.n_components:.1f}%)")
        
        # Inter-element similarity analysis (dictionary coherence)
        similarity_matrix = self.dictionary.T @ self.dictionary
        off_diagonal = similarity_matrix - np.eye(self.n_components)
        avg_similarity = np.mean(np.abs(off_diagonal))
        max_similarity = np.max(np.abs(off_diagonal))
        
        print(f"   • Average element similarity: {avg_similarity:.3f}")
        print(f"   • Maximum element similarity: {max_similarity:.3f}")
        
        if max_similarity > 0.9:
            print("   ⚠️  High maximum similarity suggests redundant dictionary elements")
        elif max_similarity < 0.3:
            print("   ✓ Low maximum similarity indicates diverse dictionary elements")
        
        # Dictionary conditioning
        condition_number = np.linalg.cond(self.dictionary)
        print(f"   • Dictionary condition number: {condition_number:.2e}")
        
        if condition_number > 1e12:
            print("   ⚠️  Poor dictionary conditioning may cause numerical issues")
        
        # Store analysis results if the attribute exists
        if hasattr(self, '_analysis_results'):
            self._analysis_results = {
                'orientations': orientations,
                'edge_strengths': edge_strengths,
                'element_norms': element_norms,
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'condition_number': condition_number,
                'oriented_elements_fraction': len(orientations) / self.n_components if orientations else 0
            }
    
    def configure_sparseness_function(self, function_name: str, **kwargs):
        """
        Configure sparseness function and its parameters for maximum user flexibility.
        
        This method allows dynamic configuration of the sparseness penalty function
        used in sparse coding optimization. Different functions provide different
        trade-offs between sparsity, smoothness, and robustness.
        
        Args:
            function_name: One of ['l1', 'log', 'gaussian', 'huber', 'elastic_net', 'cauchy', 'student_t']
            **kwargs: Function-specific parameters:
                - huber_delta: Threshold for Huber penalty (default: 1.0)
                - elastic_net_l1_ratio: L1/L2 mixing ratio for elastic net (default: 0.5)
                - cauchy_gamma: Scale parameter for Cauchy penalty (default: 1.0)
                - student_t_nu: Degrees of freedom for Student-t penalty (default: 3.0)
        
        Raises:
            ValueError: If function_name is invalid or parameters are out of range
            
        Example:
            >>> # Configure L1 penalty (standard sparse coding)
            >>> coder.configure_sparseness_function('l1')
            
            >>> # Configure Huber penalty with custom threshold
            >>> coder.configure_sparseness_function('huber', huber_delta=2.0)
            
            >>> # Configure elastic net with 70% L1, 30% L2
            >>> coder.configure_sparseness_function('elastic_net', elastic_net_l1_ratio=0.7)
        """
        valid_functions = ['l1', 'log', 'gaussian', 'huber', 'elastic_net', 'cauchy', 'student_t']
        if function_name not in valid_functions:
            raise ValueError(f"Invalid sparseness function '{function_name}'. "
                           f"Choose from: {valid_functions}")
            
        self.sparseness_function = function_name
        
        # Set function-specific parameters with validation
        if function_name == 'huber':
            if 'huber_delta' in kwargs:
                delta = kwargs['huber_delta']
                if delta <= 0:
                    raise ValueError(f"huber_delta must be positive, got: {delta}")
                self.huber_delta = delta
            elif not hasattr(self, 'huber_delta'):
                self.huber_delta = 1.0  # Default value
                
        elif function_name == 'elastic_net':
            if 'elastic_net_l1_ratio' in kwargs:
                ratio = kwargs['elastic_net_l1_ratio']
                if not 0 <= ratio <= 1:
                    raise ValueError(f"elastic_net_l1_ratio must be in [0, 1], got: {ratio}")
                self.elastic_net_l1_ratio = ratio
            elif not hasattr(self, 'elastic_net_l1_ratio'):
                self.elastic_net_l1_ratio = 0.5  # Default value
                
        elif function_name == 'cauchy':
            if 'cauchy_gamma' in kwargs:
                gamma = kwargs['cauchy_gamma']
                if gamma <= 0:
                    raise ValueError(f"cauchy_gamma must be positive, got: {gamma}")
                self.cauchy_gamma = gamma
            elif not hasattr(self, 'cauchy_gamma'):
                self.cauchy_gamma = 1.0  # Default value
                
        elif function_name == 'student_t':
            if 'student_t_nu' in kwargs:
                nu = kwargs['student_t_nu']
                if nu <= 0:
                    raise ValueError(f"student_t_nu must be positive, got: {nu}")
                self.student_t_nu = nu
            elif not hasattr(self, 'student_t_nu'):
                self.student_t_nu = 3.0  # Default value
        
        # Validate the new configuration
        self._validate_sparseness_function_parameters()
        
        print(f"✓ Configured sparseness function: {function_name}")
        if kwargs:
            print(f"  Parameters: {kwargs}")
    
    def get_sparseness_function_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about available sparseness functions and current configuration.
        
        Returns detailed information about:
        - All available sparseness functions with descriptions
        - Mathematical properties of each function
        - Current configuration settings
        - Usage examples
        - Parameter ranges and recommendations
        
        Returns:
            Dict containing:
                - 'available_functions': Dict of function descriptions and properties
                - 'current_configuration': Current function and parameter settings
                - 'usage_examples': Code examples for each function
                - 'theoretical_properties': Mathematical properties and use cases
                
        Example:
            >>> info = coder.get_sparseness_function_info()
            >>> print(info['available_functions']['l1']['description'])
            >>> print(info['current_configuration'])
        """
        
        function_info = {
            'l1': {
                'description': 'L1 penalty: |x| - Standard sparse coding penalty from Olshausen & Field 1996',
                'mathematical_form': '||x||₁ = Σᵢ |xᵢ|',
                'parameters': {},
                'properties': [
                    'Sharp sparsity - produces exact zeros',
                    'Non-differentiable at zero (requires specialized solvers)',
                    'Convex optimization (global optimum guaranteed)',
                    'Scale-invariant penalty'
                ],
                'use_cases': [
                    'Standard sparse coding applications',
                    'When exact sparsity (zeros) is desired',
                    'Feature selection and compression'
                ],
                'computational_notes': 'Requires proximal operators (soft thresholding)'
            },
            'log': {
                'description': 'Log penalty: log(1 + x²) - Smooth approximation to L1 from original paper',
                'mathematical_form': 'Σᵢ log(1 + xᵢ²)',
                'parameters': {},
                'properties': [
                    'Smooth and differentiable everywhere',
                    'Concave penalty (promotes sparsity)',
                    'No exact zeros (approximate sparsity)',
                    'Self-regularizing (bounded second derivative)'
                ],
                'use_cases': [
                    'When smooth optimization is preferred',
                    'Gradient-based optimization methods',
                    'Avoiding numerical issues with L1'
                ],
                'computational_notes': 'Works well with standard gradient descent'
            },
            'gaussian': {
                'description': 'Gaussian penalty: -exp(-x²) - Probabilistic interpretation, favors small coefficients',
                'mathematical_form': '-Σᵢ exp(-xᵢ²)',
                'parameters': {},
                'properties': [
                    'Very smooth penalty function',
                    'Probabilistic interpretation (log of Gaussian prior)',
                    'Less aggressive sparsity than L1',
                    'Symmetric around zero'
                ],
                'use_cases': [
                    'When moderate sparsity is desired',
                    'Bayesian sparse coding interpretations',
                    'Noise-robust applications'
                ],
                'computational_notes': 'Computationally efficient, smooth gradients'
            },
            'huber': {
                'description': 'Huber penalty: Smooth transition from quadratic to linear - robust to outliers',
                'mathematical_form': 'Σᵢ huber(xᵢ, δ) where huber(x,δ) = ½x² if |x|≤δ, δ|x|-½δ² otherwise',
                'parameters': {
                    'huber_delta': {
                        'description': 'Transition threshold between quadratic and linear regimes',
                        'default': 1.0,
                        'range': '(0, ∞)',
                        'recommendations': 'Smaller values → more L1-like, Larger values → more L2-like'
                    }
                },
                'properties': [
                    'Smooth everywhere (differentiable)',
                    'Quadratic for small values, linear for large values',
                    'Robust to outliers',
                    'Convex optimization'
                ],
                'use_cases': [
                    'Noisy data with outliers',
                    'When both smooth optimization and sparsity are needed',
                    'Robust regression applications'
                ],
                'computational_notes': 'Efficient gradient computation, good convergence'
            },
            'elastic_net': {
                'description': 'Elastic net: α·L1 + (1-α)·L2 - Combines L1 and L2 penalties',
                'mathematical_form': 'α·||x||₁ + (1-α)·½||x||₂²',
                'parameters': {
                    'elastic_net_l1_ratio': {
                        'description': 'Mixing ratio between L1 and L2 penalties',
                        'default': 0.5,
                        'range': '[0, 1]',
                        'recommendations': '1.0=pure L1, 0.0=pure L2, 0.5=balanced'
                    }
                },
                'properties': [
                    'Combines sparsity (L1) with grouping (L2)',
                    'Handles correlated features well',
                    'Convex optimization',
                    'Stable solutions'
                ],
                'use_cases': [
                    'High-dimensional data with correlated features',
                    'When both sparsity and stability are needed',
                    'Feature selection with grouping'
                ],
                'computational_notes': 'Well-conditioned optimization, stable solutions'
            },
            'cauchy': {
                'description': 'Cauchy penalty: log(1 + (x/γ)²) - Heavy-tailed for extreme sparsity',
                'mathematical_form': 'Σᵢ log(1 + (xᵢ/γ)²)',
                'parameters': {
                    'cauchy_gamma': {
                        'description': 'Scale parameter controlling penalty strength',
                        'default': 1.0,
                        'range': '(0, ∞)',
                        'recommendations': 'Smaller values → more aggressive sparsity'
                    }
                },
                'properties': [
                    'Very heavy-tailed distribution',
                    'Extremely sparse solutions',
                    'Robust to outliers',
                    'Non-convex optimization (local minima)'
                ],
                'use_cases': [
                    'When extreme sparsity is needed',
                    'Robust sparse coding',
                    'Outlier-resistant applications'
                ],
                'computational_notes': 'Non-convex (requires good initialization)'
            },
            'student_t': {
                'description': 'Student-t penalty: log(1 + x²/ν) - Robust heavy-tailed, adjustable via degrees of freedom',
                'mathematical_form': 'Σᵢ log(1 + xᵢ²/ν)',
                'parameters': {
                    'student_t_nu': {
                        'description': 'Degrees of freedom controlling tail behavior',
                        'default': 3.0,
                        'range': '(0, ∞)',
                        'recommendations': 'Lower ν → heavier tails, higher ν → more Gaussian-like'
                    }
                },
                'properties': [
                    'Flexible tail behavior via degrees of freedom',
                    'Interpolates between Cauchy (ν→0) and Gaussian (ν→∞)',
                    'Robust to outliers',
                    'Smooth optimization'
                ],
                'use_cases': [
                    'Adaptive robustness based on data characteristics',
                    'When tail behavior needs tuning',
                    'Robust sparse coding with controllable sparsity'
                ],
                'computational_notes': 'Smooth gradients, good convergence properties'
            }
        }
        
        # Current configuration
        current_config = {
            'current_function': getattr(self, 'sparseness_function', 'l1'),
            'current_parameters': {},
            'sparsity_penalty': getattr(self, 'sparsity_penalty', None)
        }
        
        # Add current parameter values
        parameter_attributes = {
            'huber_delta': 'huber_delta',
            'elastic_net_l1_ratio': 'elastic_net_l1_ratio', 
            'cauchy_gamma': 'cauchy_gamma',
            'student_t_nu': 'student_t_nu'
        }
        
        for param_name, attr_name in parameter_attributes.items():
            if hasattr(self, attr_name):
                current_config['current_parameters'][param_name] = getattr(self, attr_name)
        
        # Usage examples
        usage_examples = {
            'l1_standard': {
                'code': "coder.configure_sparseness_function('l1')",
                'description': "Standard L1 sparse coding from Olshausen & Field 1996"
            },
            'log_smooth': {
                'code': "coder.configure_sparseness_function('log')",
                'description': "Smooth sparsity penalty from original paper"
            },
            'huber_robust': {
                'code': "coder.configure_sparseness_function('huber', huber_delta=2.0)",
                'description': "Robust penalty with smooth transition at δ=2.0"
            },
            'elastic_net_balanced': {
                'code': "coder.configure_sparseness_function('elastic_net', elastic_net_l1_ratio=0.7)",
                'description': "70% L1 + 30% L2 penalty for correlated features"
            },
            'cauchy_very_sparse': {
                'code': "coder.configure_sparseness_function('cauchy', cauchy_gamma=0.5)",
                'description': "Extreme sparsity with heavy-tailed penalty"
            },
            'student_t_adaptive': {
                'code': "coder.configure_sparseness_function('student_t', student_t_nu=5.0)",
                'description': "Adaptive tail behavior with ν=5 degrees of freedom"
            }
        }
        
        # Theoretical properties and guidance
        theoretical_properties = {
            'sparsity_strength_ranking': [
                'cauchy (strongest)',
                'l1',
                'student_t',
                'huber', 
                'log',
                'elastic_net',
                'gaussian (weakest)'
            ],
            'computational_difficulty_ranking': [
                'l1, huber, elastic_net (easiest - convex)',
                'log, gaussian, student_t (moderate - smooth)',
                'cauchy (hardest - non-convex)'
            ],
            'robustness_ranking': [
                'cauchy, student_t (most robust)',
                'huber',
                'elastic_net',
                'l1',
                'log, gaussian (least robust)'
            ],
            'recommendations': {
                'beginners': 'Start with l1 (standard) or log (smooth)',
                'noisy_data': 'Use huber, cauchy, or student_t',
                'correlated_features': 'Use elastic_net',
                'extreme_sparsity': 'Use cauchy or student_t with low nu',
                'smooth_optimization': 'Use log, gaussian, or huber'
            }
        }
        
        return {
            'available_functions': function_info,
            'current_configuration': current_config,
            'usage_examples': usage_examples,
            'theoretical_properties': theoretical_properties
        }
    
    def validate_patches(self, patches: np.ndarray) -> bool:
        """
        Validate input patches for sparse coding.
        
        Args:
            patches: Array of patches to validate
            
        Returns:
            bool: True if patches are valid
            
        Raises:
            ValueError: If patches are invalid
        """
        if patches is None:
            raise ValueError("Patches cannot be None")
        
        if not isinstance(patches, np.ndarray):
            raise ValueError(f"Patches must be numpy array, got: {type(patches)}")
        
        if patches.ndim not in [1, 2]:
            raise ValueError(f"Patches must be 1D or 2D array, got shape: {patches.shape}")
        
        # Check for finite values
        if not np.isfinite(patches).all():
            raise ValueError("Patches contain NaN or infinite values")
        
        # Check dimensions match expected patch size
        expected_patch_dim = self.patch_size[0] * self.patch_size[1]
        
        if patches.ndim == 1:
            # Single patch
            if len(patches) != expected_patch_dim:
                raise ValueError(f"Single patch has {len(patches)} elements, "
                               f"expected {expected_patch_dim}")
        else:
            # Multiple patches
            if patches.shape[1] != expected_patch_dim:
                raise ValueError(f"Patches have {patches.shape[1]} elements per patch, "
                               f"expected {expected_patch_dim}")
        
        return True
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report of the current sparse coder state.
        
        Returns:
            Dict containing validation results for all components
        """
        report = {
            'timestamp': np.datetime64('now'),
            'configuration_valid': False,
            'dictionary_valid': False,
            'sparseness_function_valid': False,
            'errors': [],
            'warnings': []
        }
        
        # Test configuration
        try:
            self._validate_configuration()
            report['configuration_valid'] = True
        except ValueError as e:
            report['errors'].append(f"Configuration error: {e}")
        
        # Test dictionary if present
        if hasattr(self, 'dictionary') and self.dictionary is not None:
            try:
                self._validate_dictionary()
                report['dictionary_valid'] = True
            except ValueError as e:
                report['errors'].append(f"Dictionary error: {e}")
        else:
            report['warnings'].append("No dictionary present (not yet trained)")
        
        # Test sparseness function parameters
        try:
            self._validate_sparseness_function_parameters()
            report['sparseness_function_valid'] = True
        except ValueError as e:
            report['errors'].append(f"Sparseness function error: {e}")
        
        # Overall status
        report['overall_valid'] = (report['configuration_valid'] and 
                                 report['sparseness_function_valid'] and
                                 (report['dictionary_valid'] or not hasattr(self, 'dictionary')))
        
        return report


# Standalone validation functions for use without mixin
def validate_sparse_coding_parameters(n_components: int, patch_size: Tuple[int, int], 
                                    sparsity_penalty: float, **kwargs) -> bool:
    """
    Standalone function to validate sparse coding parameters.
    
    Args:
        n_components: Number of dictionary elements
        patch_size: Patch dimensions
        sparsity_penalty: Sparsity penalty parameter
        **kwargs: Additional parameters to validate
        
    Returns:
        bool: True if all parameters are valid
        
    Raises:
        ValueError: If any parameter is invalid
    """
    if not isinstance(patch_size, (tuple, list)) or len(patch_size) != 2:
        raise ValueError("patch_size must be a tuple/list of two integers")
    
    if patch_size[0] <= 0 or patch_size[1] <= 0:
        raise ValueError(f"Invalid patch size: {patch_size}")
    
    if n_components <= 0:
        raise ValueError(f"Invalid number of components: {n_components}")
    
    if sparsity_penalty < 0:
        raise ValueError(f"Invalid sparsity penalty: {sparsity_penalty}")
    
    return True


def analyze_dictionary_standalone(dictionary: np.ndarray, patch_size: Tuple[int, int]) -> Dict[str, Any]:
    """
    Standalone function to analyze a dictionary matrix.
    
    Args:
        dictionary: Dictionary matrix to analyze
        patch_size: Patch dimensions for reshaping dictionary elements
        
    Returns:
        Dict containing analysis results
    """
    if dictionary.ndim != 2:
        raise ValueError("Dictionary must be 2D matrix")
    
    results = {}
    
    # Basic properties
    results['shape'] = dictionary.shape
    results['n_elements'] = dictionary.shape[1]
    
    # Element norms
    element_norms = np.linalg.norm(dictionary, axis=0)
    results['element_norms'] = {
        'mean': float(element_norms.mean()),
        'std': float(element_norms.std()),
        'min': float(element_norms.min()),
        'max': float(element_norms.max())
    }
    
    # Similarity analysis
    similarity_matrix = dictionary.T @ dictionary
    off_diagonal = similarity_matrix - np.eye(dictionary.shape[1])
    results['similarity'] = {
        'avg_similarity': float(np.mean(np.abs(off_diagonal))),
        'max_similarity': float(np.max(np.abs(off_diagonal)))
    }
    
    # Condition number
    results['condition_number'] = float(np.linalg.cond(dictionary))
    
    return results

"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
📝 CITE: Benedict Chen (2025) - Sparse Coding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""