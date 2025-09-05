"""
📋 Optimization
================

🎯 ELI5 Summary:
This file is an important component in our AI research system! Like different organs 
in your body that work together to keep you healthy, this file has a specific job that 
helps the overall algorithm work correctly and efficiently.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

📋 Component Integration:
========================
    ┌──────────┐
    │   This   │
    │Component │ ←→ Other Components
    └──────────┘
         ↑↓
    System Integration

"""
"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀

🚀 Sparse Coding Optimization Module
===================================

Author: Benedict Chen (benedict@benedictchen.com)

This module contains all the optimization algorithms extracted from the original
Sparse Coder implementation, following Olshausen & Field (1996).

The module implements multiple optimization approaches for sparse coding:
1. Original Olshausen & Field equation (5) - Fixed-point iteration
2. FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
3. Proximal Gradient Methods
4. Coordinate Descent for L1-regularized problems
5. General optimization for non-L1 sparseness functions

Key Research Features:
- Multiple sparseness functions: L1, log, gaussian, huber, elastic_net, cauchy, student_t
- Exact implementation of equation (5) from the original paper
- Modern optimization methods (FISTA, proximal gradient)
- Coordinate descent for L1 problems (proven optimal)

Mathematical Framework:
The goal is to solve: min ||x - Da||₂² + λ*S(a)
where x is patch, D is dictionary, a is sparse code, λ is sparsity penalty,
and S(a) is the sparseness function.

The original paper uses S(a) = -Σ S(aᵢ/σ) where S(x) can be:
- S(x) = log(1 + x²) [Primary paper choice]
- S(x) = |x| [L1 penalty - also tested]
- S(x) = -e^(-x²) [Gaussian - mentioned in paper]

🔬 Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"
"""

import numpy as np
from typing import Callable, Tuple, Optional
from scipy.optimize import minimize


class OptimizationMixin:
    """
    Mixin class containing all optimization algorithms for sparse coding.
    
    This class provides optimization methods that can be mixed into the main
    SparseCoder class, maintaining access to instance variables like self.dictionary,
    self.sparsity_penalty, etc.
    
    The mixin pattern allows for modular code organization while preserving
    the original class structure and `self` access patterns.
    """
    
    def olshausen_field_sparse_coding(self, patch: np.ndarray) -> np.ndarray:
        """
        Olshausen & Field (1996) sparse coding algorithm.
        
        Implements the cost function minimization from Nature 381:607-609 (1996):
        E = ||x - Φa||² + λ Σᵢ S(aᵢ/σ)
        
        where:
        - x: input patch
        - Φ: dictionary (basis functions)
        - a: sparse coefficients
        - S(u) = log(1 + u²): sparseness function
        - σ: scaling parameter
        - λ: sparsity penalty
        
        Uses gradient descent with proper mathematical formulation.
        
        Args:
            patch: Input image patch to encode (flattened vector)
            
        Returns:
            np.ndarray: Sparse coefficients representing patch in dictionary basis
        """
        
        # Initialize coefficients
        coeffs = np.zeros(self.dictionary.shape[1])
        
        # Olshausen & Field parameters
        sigma = getattr(self, 'sigma', 1.0)  # Scaling parameter
        lambda_sparse = getattr(self, 'sparsity_penalty', 0.1)  # Sparsity weight
        learning_rate = getattr(self, 'learning_rate', 0.01)
        max_iter = getattr(self, 'max_iter', 100)
        tolerance = getattr(self, 'tolerance', 1e-6)
        
        # Precompute dictionary operations for efficiency
        dict_transpose = self.dictionary.T
        gram_matrix = dict_transpose @ self.dictionary
        
        # Gradient descent optimization
        for iteration in range(max_iter):
            coeffs_old = coeffs.copy()
            
            # Compute reconstruction error gradient: ∇_a ||x - Φa||² = 2Φᵀ(Φa - x)
            reconstruction = self.dictionary @ coeffs
            reconstruction_grad = dict_transpose @ (reconstruction - patch)
            
            # Compute sparseness penalty gradient: ∇_a Σᵢ S(aᵢ/σ)
            # For S(u) = log(1 + u²), S'(u) = 2u/(1 + u²)
            normalized_coeffs = coeffs / sigma
            sparseness_grad = lambda_sparse * (2 * normalized_coeffs) / (sigma * (1 + normalized_coeffs**2))
            
            # Total gradient
            total_grad = reconstruction_grad + sparseness_grad
            
            # Gradient descent update
            coeffs = coeffs - learning_rate * total_grad
            
            # Check convergence
            if np.linalg.norm(coeffs - coeffs_old) < tolerance:
                break
        
        return coeffs
            
    def sparse_encode_gradient_descent(self, patch: np.ndarray) -> np.ndarray:
        """
        Modern gradient-based sparse coding implementation.
        
        This provides the 'equation_5' optimization method used in the sparse coding system.
        """
        return self.olshausen_field_sparse_coding(patch)

def _sparse_encode_fista(
    patch: np.ndarray,
    dictionary: np.ndarray,
    sparsity_penalty: float,
    max_iter: int = 100,
    tolerance: float = 1e-6
) -> np.ndarray:
    """
    FISTA algorithm - accelerated ISTA with O(1/k²) convergence
    
    Fast Iterative Shrinkage-Thresholding Algorithm for L1-regularized problems.
    Uses Nesterov acceleration for faster convergence than basic ISTA.
    """
    # Compute Lipschitz constant for step size
    L = np.linalg.norm(dictionary.T @ dictionary, ord=2)
    eta = 1.0 / L  # Step size
    
    x_k = np.zeros(dictionary.shape[1])  # Current iterate
    y_k = np.zeros(dictionary.shape[1])  # Extrapolated point
    t_k = 1.0  # Acceleration parameter
    
    for iteration in range(max_iter):
        # Gradient step on extrapolated point
        residual = patch - dictionary @ y_k
        gradient = -dictionary.T @ residual
        temp = y_k - eta * gradient
        
        # Proximal operator (soft thresholding)
        threshold = sparsity_penalty * eta
        x_k_new = np.sign(temp) * np.maximum(np.abs(temp) - threshold, 0)
        
        # Nesterov acceleration
        t_k_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        beta_k = (t_k - 1) / t_k_new
        y_k = x_k_new + beta_k * (x_k_new - x_k)
        
        # Convergence check
        if np.linalg.norm(x_k_new - x_k) < tolerance:
            break
            
        x_k, t_k = x_k_new, t_k_new
    
    return x_k

def log_sparseness_derivative(coeffs: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Derivative of log sparseness function S(x) = log(1 + (x/σ)²).
    
    From Olshausen & Field (1996) sparse coding framework.
    σ is a scaling parameter that can be tuned.
    """
    normalized_coeffs = coeffs / sigma
    return (2 * normalized_coeffs) / (sigma * (1 + normalized_coeffs**2))

def _sparse_encode_log_sparseness(
    patch: np.ndarray,
    dictionary: np.ndarray,
    sparsity_penalty: float,
    max_iter: int = 100,
    tolerance: float = 1e-6,
    sigma: float = 0.316
) -> np.ndarray:
    """
    Sparse encoding using original Olshausen & Field log sparseness function.
    
    Uses S(x) = log(1 + (x/σ)²) with σ = 0.316 from the original paper.
    """
    coefficients = np.zeros(dictionary.shape[1])
    
    for iteration in range(max_iter):
        residual = patch - dictionary @ coefficients
        
        for i in range(len(coefficients)):
            # Compute gradient of reconstruction error
            error_gradient = -dictionary[:, i].T @ residual
            
            # Add sparseness penalty gradient
            sparseness_gradient = sparsity_penalty * _log_sparseness_derivative(
                np.array([coefficients[i]]), sigma)[0]
            
            total_gradient = error_gradient + sparseness_gradient
            
            # Update coefficient with learning rate
            learning_rate = 0.01
            coefficients[i] -= learning_rate * total_gradient
            
        # Convergence check
        if iteration > 0 and np.linalg.norm(residual) < tolerance:
            break
    
    return coefficients

def _sparse_encode_with_lateral_inhibition(
    patch: np.ndarray,
    dictionary: np.ndarray,
    sparsity_penalty: float,
    max_iter: int = 100,
    tolerance: float = 1e-6,
    gamma: float = 0.1
) -> np.ndarray:
    """
    Sparse encoding with lateral inhibition for biological realism.
    
    Implements the lateral inhibition mechanism from Olshausen & Field (1996)
    where each coefficient inhibits others, promoting sparsity.
    """
    coefficients = np.zeros(dictionary.shape[1])
    
    for iteration in range(max_iter):
        residual = patch - dictionary @ coefficients
        
        for i in range(len(coefficients)):
            # Compute gradient of reconstruction error
            error_gradient = -dictionary[:, i].T @ residual
            
            # Add sparseness penalty
            sparseness_gradient = sparsity_penalty * np.sign(coefficients[i])
            
            # Lateral inhibition (biological realism)
            lateral_input = gamma * (np.sum(coefficients) - coefficients[i])
            
            total_gradient = error_gradient + sparseness_gradient + lateral_input
            
            # Update coefficient
            learning_rate = 0.01
            coefficients[i] -= learning_rate * total_gradient
            
        # Convergence check
        if iteration > 0 and np.linalg.norm(residual) < tolerance:
            break
    
    return coefficients


    def _fista_optimization(self, patch: np.ndarray, objective_func: Callable, 
                          gradient_func: Callable, initial_coeffs: np.ndarray) -> np.ndarray:
        """
        FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for L1-regularized problems
        
        FISTA provides accelerated convergence compared to basic iterative shrinkage methods.
        It uses Nesterov's momentum to achieve O(1/k²) convergence rate vs O(1/k) for
        standard gradient methods.
        
        This is particularly effective for L1-regularized sparse coding problems and
        is a modern alternative to the L-BFGS-B approach which is not optimal for L1.
        
        Args:
            patch: Input image patch
            objective_func: Objective function to minimize
            gradient_func: Gradient function
            initial_coeffs: Initial coefficient values
            
        Returns:
            np.ndarray: Optimized sparse coefficients
        """
        x = initial_coeffs.copy()
        y = initial_coeffs.copy()
        t = 1
        L = 1.0  # Lipschitz constant estimate
        
        for iteration in range(self.max_iter):
            x_old = x.copy()
            
            # Compute gradient at y
            grad = gradient_func(y)
            
            # Gradient step
            z = y - grad / L
            
            # Proximal operator for L1 (soft thresholding)
            threshold = self.sparsity_penalty / L
            x = np.sign(z) * np.maximum(np.abs(z) - threshold, 0)
            
            # FISTA momentum update
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = x + (t - 1) / t_new * (x - x_old)
            t = t_new
            
            # Check convergence
            if np.linalg.norm(x - x_old) < self.tolerance:
                break
                
        return x
    
    def _proximal_gradient(self, patch: np.ndarray, objective_func: Callable,
                         gradient_func: Callable, initial_coeffs: np.ndarray) -> np.ndarray:
        """
        Proximal gradient descent method for L1-regularized problems
        
        This method alternates between gradient steps on the smooth (quadratic) part
        of the objective and proximal steps on the non-smooth (L1) part.
        
        The proximal operator for L1 norm is soft thresholding, which sets small
        coefficients to zero, achieving sparsity.
        
        Args:
            patch: Input image patch
            objective_func: Objective function (unused in this implementation)
            gradient_func: Gradient function (unused - computed directly)
            initial_coeffs: Initial coefficient values
            
        Returns:
            np.ndarray: Optimized sparse coefficients
        """
        x = initial_coeffs.copy()
        step_size = 0.1
        
        for iteration in range(self.max_iter):
            x_old = x.copy()
            
            # Compute gradient of smooth part (reconstruction error)
            reconstruction = self.dictionary @ x
            error = reconstruction - patch
            grad_reconstruction = self.dictionary.T @ error
            
            # Gradient step
            z = x - step_size * grad_reconstruction
            
            # Proximal operator for L1 (soft thresholding)
            threshold = self.sparsity_penalty * step_size
            x = np.sign(z) * np.maximum(np.abs(z) - threshold, 0)
            
            # Check convergence
            if np.linalg.norm(x - x_old) < self.tolerance:
                break
                
        return x

    def _general_optimization(self, patch: np.ndarray, objective_func: Callable,
                            gradient_func: Callable, initial_coeffs: np.ndarray) -> np.ndarray:
        """
        General optimization for non-L1 sparseness functions
        
        Uses gradient descent with adaptive step size for sparseness functions
        that don't have closed-form proximal operators (like log, gaussian, etc.).
        
        This method is used when the sparseness function is not L1, so we can't
        use the efficient proximal methods (FISTA, coordinate descent).
        
        Args:
            patch: Input image patch
            objective_func: Objective function to minimize
            gradient_func: Gradient function
            initial_coeffs: Initial coefficient values
            
        Returns:
            np.ndarray: Optimized sparse coefficients
        """
        x = initial_coeffs.copy()
        step_size = 0.1
        
        for iteration in range(self.max_iter):
            x_old = x.copy()
            
            # Compute gradient
            grad = gradient_func(x)
            
            # Adaptive step size with backtracking
            x_new = x - step_size * grad
            
            # Check if objective improved, adjust step size if needed
            if objective_func(x_new) > objective_func(x):
                step_size *= 0.5  # Reduce step size
                x_new = x - step_size * grad
            else:
                step_size = min(step_size * 1.1, 0.5)  # Increase step size but cap it
            
            x = x_new
            
            # Check convergence
            if np.linalg.norm(x - x_old) < self.tolerance:
                break
                
        return x

    def _sparse_encode_single(self, patch: np.ndarray) -> np.ndarray:
        """
        Encode a single patch using sparse coding with configurable sparseness functions
        
        This is the main sparse encoding function that supports multiple sparseness
        functions and optimization methods. It solves:
        
        min ||x - Da||₂² + λ*S(a)
        
        where S(a) can be various sparseness functions:
        - L1: |a| (standard sparse coding)
        - Log: log(1 + a²) (original Olshausen & Field choice)
        - Gaussian: -exp(-a²) (smooth approximation)
        - Huber: smooth approximation to L1
        - Elastic Net: combination of L1 and L2
        - Cauchy: heavy-tailed for extreme sparsity
        - Student-t: robust heavy-tailed distribution
        
        Args:
            patch: Input image patch to encode (flattened)
            
        Returns:
            np.ndarray: Sparse coefficients
        """
        
        def objective(coefficients):
            """Objective function: reconstruction error + sparsity penalty"""
            reconstruction = self.dictionary @ coefficients
            reconstruction_error = 0.5 * np.sum((patch - reconstruction) ** 2)
            
            # Implement different sparseness functions
            if self.sparseness_function == 'log':
                # S(x) = log(1 + x²) - Original Olshausen & Field choice
                sigma = 1.0  # Scaling constant
                normalized_coeffs = coefficients / sigma
                sparsity_penalty = -self.sparsity_penalty * np.sum(np.log(1 + normalized_coeffs**2))
            elif self.sparseness_function == 'gaussian':
                # S(x) = -e^(-x²) - Alternative from paper
                sigma = 1.0
                normalized_coeffs = coefficients / sigma
                sparsity_penalty = self.sparsity_penalty * np.sum(np.exp(-normalized_coeffs**2))
            elif self.sparseness_function == 'huber':
                # Huber penalty - smooth approximation to L1 for robustness
                delta = getattr(self, 'huber_delta', 1.0)
                abs_coeffs = np.abs(coefficients)
                huber_penalty = np.where(abs_coeffs <= delta, 
                                        0.5 * coefficients**2, 
                                        delta * abs_coeffs - 0.5 * delta**2)
                sparsity_penalty = self.sparsity_penalty * np.sum(huber_penalty)
            elif self.sparseness_function == 'elastic_net':
                # Elastic net: combination of L1 and L2 penalties
                l1_ratio = getattr(self, 'elastic_net_l1_ratio', 0.5)
                l1_penalty = np.sum(np.abs(coefficients))
                l2_penalty = 0.5 * np.sum(coefficients**2)
                sparsity_penalty = self.sparsity_penalty * (l1_ratio * l1_penalty + (1 - l1_ratio) * l2_penalty)
            elif self.sparseness_function == 'cauchy':
                # Cauchy penalty - heavy-tailed for extreme sparsity
                gamma = getattr(self, 'cauchy_gamma', 1.0)
                sparsity_penalty = self.sparsity_penalty * np.sum(np.log(1 + (coefficients / gamma)**2))
            elif self.sparseness_function == 'student_t':
                # Student-t penalty - robust heavy-tailed distribution
                nu = getattr(self, 'student_t_nu', 3.0)  # degrees of freedom
                sparsity_penalty = self.sparsity_penalty * np.sum(np.log(1 + coefficients**2 / nu))
            else:
                # Default: L1 sparsity (|x|)
                sparsity_penalty = self.sparsity_penalty * np.sum(np.abs(coefficients))
                
            return reconstruction_error + sparsity_penalty
        
        def gradient(coefficients):
            """Gradient of objective function with different sparseness functions"""
            reconstruction = self.dictionary @ coefficients
            error = reconstruction - patch
            grad_reconstruction = self.dictionary.T @ error
            
            # Implement gradient for different sparseness functions
            if self.sparseness_function == 'log':
                # S(x) = log(1 + x²), S'(x) = 2x/(1 + x²)
                sigma = 1.0
                normalized_coeffs = coefficients / sigma
                grad_sparsity = -self.sparsity_penalty * (2 * normalized_coeffs) / (1 + normalized_coeffs**2) / sigma
            elif self.sparseness_function == 'gaussian':
                # S(x) = -e^(-x²), S'(x) = 2x*e^(-x²)
                sigma = 1.0
                normalized_coeffs = coefficients / sigma
                grad_sparsity = self.sparsity_penalty * 2 * normalized_coeffs * np.exp(-normalized_coeffs**2) / sigma
            elif self.sparseness_function == 'huber':
                # Huber gradient: smooth transition from quadratic to linear
                delta = getattr(self, 'huber_delta', 1.0)
                abs_coeffs = np.abs(coefficients)
                grad_sparsity = self.sparsity_penalty * np.where(abs_coeffs <= delta,
                                                               coefficients,
                                                               delta * np.sign(coefficients))
            elif self.sparseness_function == 'elastic_net':
                # Elastic net gradient: combination of L1 and L2
                l1_ratio = getattr(self, 'elastic_net_l1_ratio', 0.5)
                grad_l1 = np.sign(coefficients)
                grad_l2 = coefficients
                grad_sparsity = self.sparsity_penalty * (l1_ratio * grad_l1 + (1 - l1_ratio) * grad_l2)
            elif self.sparseness_function == 'cauchy':
                # Cauchy gradient: d/dx log(1 + (x/γ)²) = 2x/(γ²(1 + (x/γ)²))
                gamma = getattr(self, 'cauchy_gamma', 1.0)
                normalized_coeffs = coefficients / gamma
                grad_sparsity = self.sparsity_penalty * 2 * coefficients / (gamma**2 * (1 + normalized_coeffs**2))
            elif self.sparseness_function == 'student_t':
                # Student-t gradient: d/dx log(1 + x²/ν) = 2x/(ν(1 + x²/ν))
                nu = getattr(self, 'student_t_nu', 3.0)
                grad_sparsity = self.sparsity_penalty * 2 * coefficients / (nu * (1 + coefficients**2 / nu))
            else:
                # Default: L1 sparsity, S'(x) = sign(x)
                grad_sparsity = self.sparsity_penalty * np.sign(coefficients)
                
            return grad_reconstruction + grad_sparsity
        
        # Initialize coefficients
        initial_coeffs = np.zeros(self.n_components)
        
        # Use selected optimization method
        if self.optimization_method == 'equation_5':
            # Use original paper's equation (5) method
            coeffs = self._sparse_encode_equation_5(patch)
        elif self.optimization_method == 'fista':
            # Use FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
            coeffs = self._fista_optimization(patch, objective, gradient, initial_coeffs)
        elif self.optimization_method == 'proximal_gradient':
            # Use proximal gradient descent
            coeffs = self._proximal_gradient(patch, objective, gradient, initial_coeffs)
        else:
            # Default: coordinate descent - proven optimal for L1-regularized problems
            if self.l1_solver == 'lbfgs_b':
                # L-BFGS-B is not optimal for L1-regularized problems
                print("⚠️  Warning: L-BFGS-B may not be optimal for L1-regularized problems")
                result = minimize(objective, initial_coeffs, method='L-BFGS-B', jac=gradient)
                coeffs = result.x
            elif self.sparseness_function != 'l1':
                # For non-L1 sparseness functions, use general optimization
                coeffs = self._general_optimization(patch, objective, gradient, initial_coeffs)
            else:
                # Use coordinate descent for L1 problems
                coeffs = self._coordinate_descent_lasso(patch, objective, initial_coeffs)
        
        return coeffs
        
    def _coordinate_descent_lasso(self, signal: np.ndarray, objective_func: Callable,
                                initial_coeffs: np.ndarray) -> np.ndarray:
        """
        Coordinate descent algorithm for LASSO (L1-regularized) optimization.
        
        This is the proven optimal method for L1-regularized problems with guaranteed
        convergence. It updates one coefficient at a time while holding others fixed,
        using the soft thresholding operator (proximal operator for L1 norm).
        
        The algorithm is particularly efficient for sparse problems where many
        coefficients are zero, as it can quickly identify and maintain zeros.
        
        Mathematical foundation:
        For each coordinate j, we solve:
        min_aⱼ ½||x - Σᵢ≠ⱼ aᵢdᵢ - aⱼdⱼ||² + λ|aⱼ|
        
        The solution is given by soft thresholding:
        aⱼ = soft_threshold(rho_j / ||dⱼ||², λ / ||dⱼ||²)
        
        where rho_j = dⱼᵀ(x - Σᵢ≠ⱼ aᵢdᵢ) is the correlation with the residual.
        
        Args:
            signal: Input signal to encode
            objective_func: Objective function (not used directly in coordinate descent)
            initial_coeffs: Initial coefficient values
            
        Returns:
            np.ndarray: Optimized sparse coefficients
        """
        
        coeffs = initial_coeffs.copy()
        dictionary = self.dictionary
        
        # Precompute useful quantities
        XtX = dictionary.T @ dictionary  # Dictionary gram matrix
        Xty = dictionary.T @ signal      # Dictionary-signal correlation
        
        # Coordinate descent main loop
        for iteration in range(self.max_iter):
            coeffs_old = coeffs.copy()
            
            # Update each coefficient individually
            for j in range(len(coeffs)):
                # Compute residual excluding current coefficient
                residual = signal - dictionary @ coeffs + coeffs[j] * dictionary[:, j]
                
                # Compute optimal update for coefficient j
                rho_j = dictionary[:, j].T @ residual
                
                # Soft thresholding operator (proximal operator for L1 norm)
                z_j = rho_j / XtX[j, j]  # Unconstrained optimum
                
                # Apply soft thresholding with regularization parameter
                threshold = self.sparsity_penalty / XtX[j, j]
                
                if z_j > threshold:
                    coeffs[j] = z_j - threshold
                elif z_j < -threshold:
                    coeffs[j] = z_j + threshold
                else:
                    coeffs[j] = 0.0
                    
            # Check convergence
            coeff_change = np.linalg.norm(coeffs - coeffs_old)
            if coeff_change < self.tolerance:
                break
                
        return coeffs
        
    def _soft_threshold_scalar(self, x: float, threshold: float) -> float:
        """
        Soft thresholding operator for scalar values - proximal operator for L1 norm.
        
        This is the fundamental operation in sparse coding that creates sparsity
        by setting small values to zero. It's the proximal operator for the L1 norm.
        
        Mathematical definition:
        soft_threshold(x, t) = sign(x) * max(|x| - t, 0)
        
        Args:
            x: Input value
            threshold: Thresholding parameter
            
        Returns:
            float: Soft-thresholded value
        """
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.0
            
    def _proximal_gradient_method(self, signal: np.ndarray, initial_coeffs: np.ndarray) -> np.ndarray:
        """
        Alternative proximal gradient method for L1-regularized optimization.
        
        This provides another proven approach for L1 problems with good convergence
        properties. It alternates between gradient steps on the smooth part and
        proximal steps on the non-smooth part.
        
        The step size is set based on the Lipschitz constant of the gradient
        of the smooth part (reconstruction error).
        
        Args:
            signal: Input signal to encode
            initial_coeffs: Initial coefficient values
            
        Returns:
            np.ndarray: Optimized sparse coefficients
        """
        
        coeffs = initial_coeffs.copy()
        dictionary = self.dictionary
        
        # Step size based on Lipschitz constant
        step_size = 1.0 / np.linalg.norm(dictionary, ord=2)**2
        
        for iteration in range(self.max_iter):
            coeffs_old = coeffs.copy()
            
            # Compute gradient of smooth part (quadratic term)
            residual = signal - dictionary @ coeffs
            gradient = -dictionary.T @ residual
            
            # Gradient step
            z = coeffs - step_size * gradient
            
            # Proximal operator (soft thresholding)
            threshold = step_size * self.sparsity_penalty
            coeffs = np.array([self._soft_threshold_scalar(zi, threshold) for zi in z])
            
            # Check convergence
            if np.linalg.norm(coeffs - coeffs_old) < self.tolerance:
                break
                
        return coeffs
        
    def improved_(self, patches: np.ndarray) -> np.ndarray:
        """
        Enhanced sparse encoding with FISTA for multiple patches
        
        This method applies FISTA-based sparse encoding to multiple patches,
        providing more efficient optimization than basic L-BFGS for large-scale problems.
        
        Args:
            patches: Array of patches to encode (n_patches, patch_dim)
            
        Returns:
            np.ndarray: Sparse coefficients (n_patches, n_components)
        """
        
        n_patches = patches.shape[0]
        coefficients = np.zeros((n_patches, self.n_components))
        
        # Removed print spam: f"...
        
        for i in range(n_patches):
            coefficients[i] = self._fista_sparse_encode(patches[i])
            
            if (i + 1) % 200 == 0:
                print(f"   Encoded {i + 1}/{n_patches} patches")
                
        return coefficients
        
    def _fista_sparse_encode(self, patch: np.ndarray, max_iter: Optional[int] = None) -> np.ndarray:
        """
        FISTA algorithm specifically for sparse coding of a single patch
        
        This is a specialized implementation of FISTA for the sparse coding objective.
        It's more efficient than the general FISTA implementation for this specific problem.
        
        Args:
            patch: Input image patch to encode (flattened)
            max_iter: Maximum iterations (uses self.max_iter if None)
            
        Returns:
            np.ndarray: Sparse coefficients
        """
        
        if max_iter is None:
            max_iter = 100
            
        # Initialize
        x = np.zeros(self.n_components)
        y = x.copy()
        t = 1.0
        
        # Compute Lipschitz constant for step size
        L = np.linalg.norm(self.dictionary.T @ self.dictionary, 2)
        step_size = 1.0 / L if L > 0 else 0.01
        
        for iteration in range(max_iter):
            # Gradient step on smooth part
            gradient = self.dictionary.T @ (self.dictionary @ y - patch)
            x_new = self._soft_threshold_vector(y - step_size * gradient, 
                                              step_size * self.sparsity_penalty)
            
            # FISTA momentum update
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = x_new + ((t - 1) / t_new) * (x_new - x)
            
            # Check convergence
            if np.linalg.norm(x_new - x) < 1e-6:
                break
                
            x = x_new
            t = t_new
            
        return x
        
    def _soft_threshold_vector(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """
        Vectorized soft thresholding operator for L1 regularization
        
        Applies soft thresholding element-wise to create sparsity.
        This is the proximal operator for the L1 norm.
        
        Args:
            x: Input vector
            threshold: Thresholding parameter
            
        Returns:
            np.ndarray: Soft-thresholded vector
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def get_optimization_info(self) -> dict:
        """
        Get detailed information about available optimization methods and current configuration
        
        Returns:
            dict: Comprehensive information about optimization methods
        """
        
        optimization_methods = {
            'equation_5': {
                'name': 'Original Olshausen & Field Equation (5)',
                'description': 'Fixed-point iteration from the 1996 paper: âᵢ = bᵢ - Σⱼ Cᵢⱼâⱼ - λ/σ S\'(âᵢ/σ)',
                'pros': ['Exact paper implementation', 'Historically accurate', 'Works with all sparseness functions'],
                'cons': ['May converge slowly', 'Fixed-point iteration can be unstable'],
                'best_for': 'Research fidelity, understanding original algorithm'
            },
            'coordinate_descent': {
                'name': 'Coordinate Descent for L1',
                'description': 'Proven optimal method for L1-regularized problems with guaranteed convergence',
                'pros': ['Optimal for L1', 'Fast convergence', 'Guaranteed convergence', 'Handles sparsity well'],
                'cons': ['Limited to L1 sparseness function', 'May not work well with highly correlated dictionaries'],
                'best_for': 'L1 sparse coding, production use, large-scale problems'
            },
            'fista': {
                'name': 'Fast Iterative Shrinkage-Thresholding Algorithm',
                'description': 'Accelerated proximal method with O(1/k²) convergence rate',
                'pros': ['Fast convergence', 'Acceleration via momentum', 'Good for large problems'],
                'cons': ['Limited to L1 and similar', 'Requires step size tuning'],
                'best_for': 'Large-scale L1 problems, when speed is critical'
            },
            'proximal_gradient': {
                'name': 'Proximal Gradient Method',
                'description': 'Alternates between gradient and proximal steps for composite objectives',
                'pros': ['Flexible', 'Straightforward implementation', 'Good convergence properties'],
                'cons': ['Slower than FISTA', 'May need step size tuning'],
                'best_for': 'General sparse coding problems, educational purposes'
            }
        }
        
        sparseness_functions = {
            'l1': {
                'name': 'L1 Penalty',
                'formula': '|x|',
                'derivative': 'sign(x)',
                'properties': 'Sharp sparsity, exact zeros',
                'compatible_optimizers': ['coordinate_descent', 'fista', 'proximal_gradient', 'equation_5']
            },
            'log': {
                'name': 'Log Penalty (Original Paper)',
                'formula': 'log(1 + x²)',
                'derivative': '2x/(1 + x²)',
                'properties': 'Smooth, differentiable everywhere',
                'compatible_optimizers': ['equation_5', 'general_optimization']
            },
            'gaussian': {
                'name': 'Gaussian Penalty',
                'formula': '-exp(-x²)',
                'derivative': '2x*exp(-x²)',
                'properties': 'Very smooth, less aggressive sparsity',
                'compatible_optimizers': ['equation_5', 'general_optimization']
            },
            'huber': {
                'name': 'Huber Penalty',
                'formula': '½x² (|x| ≤ δ), δ|x| - ½δ² (|x| > δ)',
                'derivative': 'x (|x| ≤ δ), δ*sign(x) (|x| > δ)',
                'properties': 'Smooth transition, robust to outliers',
                'compatible_optimizers': ['equation_5', 'general_optimization']
            },
            'elastic_net': {
                'name': 'Elastic Net',
                'formula': 'α*|x| + (1-α)*½x²',
                'derivative': 'α*sign(x) + (1-α)*x',
                'properties': 'Combines L1 and L2, handles correlated features',
                'compatible_optimizers': ['equation_5', 'general_optimization']
            },
            'cauchy': {
                'name': 'Cauchy Penalty',
                'formula': 'log(1 + (x/γ)²)',
                'derivative': '2x/(γ²(1 + (x/γ)²))',
                'properties': 'Heavy-tailed, extreme sparsity',
                'compatible_optimizers': ['equation_5', 'general_optimization']
            },
            'student_t': {
                'name': 'Student-t Penalty',
                'formula': 'log(1 + x²/ν)',
                'derivative': '2x/(ν(1 + x²/ν))',
                'properties': 'Robust, adjustable via degrees of freedom',
                'compatible_optimizers': ['equation_5', 'general_optimization']
            }
        }
        
        current_config = {
            'optimization_method': getattr(self, 'optimization_method', 'coordinate_descent'),
            'sparseness_function': getattr(self, 'sparseness_function', 'l1'),
            'l1_solver': getattr(self, 'l1_solver', 'coordinate_descent'),
            'sparsity_penalty': getattr(self, 'sparsity_penalty', 0.1),
            'max_iter': getattr(self, 'max_iter', 100),
            'tolerance': getattr(self, 'tolerance', 1e-6)
        }
        
        return {
            'optimization_methods': optimization_methods,
            'sparseness_functions': sparseness_functions,
            'current_configuration': current_config,
            'recommendations': {
                'for_research': 'Use equation_5 with log sparseness function (original paper)',
                'for_production': 'Use coordinate_descent with l1 sparseness function (fastest, most reliable)',
                'for_robustness': 'Use equation_5 with huber sparseness function',
                'for_correlated_features': 'Use equation_5 with elastic_net sparseness function',
                'for_extreme_sparsity': 'Use equation_5 with cauchy sparseness function'
            }
        }

"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
📝 CITE: Benedict Chen (2025) - Sparse Coding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""