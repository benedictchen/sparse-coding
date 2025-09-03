"""
🏗️ Sparse Coding - Optimization Algorithms Module
=================================================


Author: Benedict Chen (benedict@benedictchen.com)
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties"

🎯 MODULE PURPOSE:
=================
Optimization algorithms for sparse coefficient inference including FISTA,
coordinate descent, gradient descent, and sparse coding step implementations.

🔬 RESEARCH FOUNDATION:
======================
Implements optimization methods from:
- Beck & Teboulle (2009): FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
- Wright et al. (2009): Coordinate descent for sparse coding
- Olshausen & Field (1996): Original gradient descent formulation
- Modern sparse optimization: Proximal methods and convergence guarantees

This module contains the optimization components, split from the
1544-line monolith for specialized optimization processing.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings


class OptimizationAlgorithmsMixin:
    """
    🏗️ Optimization Algorithms Mixin for Sparse Coding
    
    Contains optimization methods for sparse coefficient inference
    including FISTA, coordinate descent, and gradient descent algorithms.
    
    Research-accurate implementations with proper convergence guarantees.
    """
    
    def _sparse_coding_step(self, X: np.ndarray) -> np.ndarray:
        """
        Sparse coding step: infer coefficients for given dictionary.
        
        Solves: argmin_A ||X - DA||² + α||A||₁ for all samples in X
        where A is the coefficient matrix [n_components, n_samples]
        
        Based on Olshausen & Field (1996) sparse inference procedure.
        
        Args:
            X: Input data [n_samples, n_features]
            
        Returns:
            Sparse codes [n_samples, n_components]
        """
        
        n_samples = X.shape[0]
        codes = np.zeros((n_samples, self.n_components))
        
        # Process each sample individually (can be parallelized)
        for i in range(n_samples):
            if self.algorithm == 'fista':
                codes[i] = self._fista_optimization(X[i])
            elif self.algorithm == 'coordinate_descent':
                codes[i] = self._coordinate_descent_optimization(X[i])
            elif self.algorithm == 'gradient_descent':
                codes[i] = self._gradient_descent_optimization(X[i])
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        return codes
    
    def _fista_optimization(self, x: np.ndarray) -> np.ndarray:
        """
        Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
        
        Implements Beck & Teboulle (2009) FISTA for solving:
        argmin_a ||x - Da||² + α||a||₁
        
        FISTA provides O(1/k²) convergence rate vs O(1/k) for standard ISTA.
        
        # FIXME: FISTA IMPLEMENTATION NEEDS RESEARCH ACCURACY IMPROVEMENTS
        #    - Missing proper Lipschitz constant computation L = ||D^T D||₂
        #    - Missing backtracking line search for optimal step size
        #    - Missing: proper momentum parameter computation β = (t_{k-1} - 1) / t_k
        #    - CODE REVIEW SUGGESTION - Implement research-accurate FISTA:
        #      ```python
        #      # Compute Lipschitz constant for step size
        #      L = np.linalg.norm(self.dictionary_ @ self.dictionary_.T, ord=2)
        #      eta = 1.0 / L  # Step size
        #      
        #      # FISTA momentum sequence
        #      t_prev, t_curr = 1.0, 1.0
        #      
        #      for iteration in range(max_iterations):
        #          # Compute gradient at y (not at a)
        #          residual = x - self.dictionary_.T @ y
        #          gradient = -self.dictionary_ @ residual
        #          
        #          # Proximal gradient step with soft thresholding
        #          a_new = self._soft_threshold(y - eta * gradient, eta * self.alpha)
        #          
        #          # FISTA momentum update
        #          t_prev = t_curr
        #          t_curr = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
        #          beta = (t_prev - 1) / t_curr
        #          y = a_new + beta * (a_new - a_prev)
        #      ```
        
        Args:
            x: Single data sample [n_features]
            
        Returns:
            Sparse coefficients [n_components]
        """
        
        # Initialize coefficients
        a_curr = np.zeros(self.n_components)
        a_prev = np.zeros(self.n_components)
        y = np.zeros(self.n_components)  # FISTA momentum variable
        
        # Compute Lipschitz constant (upper bound on largest eigenvalue)
        # L = ||D^T D||₂ where D is dictionary
        try:
            L = np.linalg.norm(self.dictionary_ @ self.dictionary_.T, ord=2)
            eta = 0.99 / L  # Step size (slightly conservative)
        except:
            eta = 0.01  # Fallback step size
        
        # FISTA momentum parameters
        t_curr = 1.0
        t_prev = 1.0
        
        # Optimization loop
        for iteration in range(self.max_iter):
            
            # Compute gradient at momentum point y
            residual = x - self.dictionary_.T @ y
            gradient = -self.dictionary_ @ residual
            
            # Proximal gradient step: soft thresholding
            a_new = self._soft_threshold(y - eta * gradient, eta * self.alpha)
            
            # FISTA momentum update
            t_prev = t_curr  
            t_curr = (1.0 + np.sqrt(1.0 + 4.0 * t_prev**2)) / 2.0
            beta = (t_prev - 1.0) / t_curr
            
            # Update momentum variable
            y = a_new + beta * (a_new - a_curr)
            
            # Check convergence
            if iteration > 0:
                change = np.linalg.norm(a_new - a_curr)
                if change < self.tolerance:
                    break
            
            a_prev = a_curr.copy()
            a_curr = a_new.copy()
        
        return a_curr
    
    def _coordinate_descent_optimization(self, x: np.ndarray) -> np.ndarray:
        """
        Coordinate Descent for L1-regularized least squares.
        
        Implements efficient coordinate descent following Wright et al. (2009).
        Updates one coefficient at a time while keeping others fixed.
        
        Args:
            x: Single data sample [n_features] 
            
        Returns:
            Sparse coefficients [n_components]
        """
        
        # Initialize coefficients
        a = np.zeros(self.n_components)
        
        # Precompute dictionary gram matrix for efficiency
        # G[i,j] = φᵢ^T φⱼ (inner products between dictionary atoms)
        G = self.dictionary_ @ self.dictionary_.T
        
        # Precompute dictionary-data inner products
        # d[i] = φᵢ^T x (correlation between atoms and data)
        d = self.dictionary_ @ x
        
        # Coordinate descent iterations
        for iteration in range(self.max_iter):
            a_old = a.copy()
            
            # Update each coefficient sequentially
            for j in range(self.n_components):
                
                # Compute residual excluding current coefficient
                # r_j = d_j - Σ_{i≠j} G_{ji} * a_i
                residual_j = d[j] - np.dot(G[j], a) + G[j, j] * a[j]
                
                # Soft thresholding update
                # a_j = soft_threshold(r_j / G_{jj}, α / G_{jj})
                if G[j, j] > 1e-12:  # Avoid division by zero
                    threshold = self.alpha / G[j, j]
                    a[j] = self._soft_threshold_scalar(residual_j / G[j, j], threshold)
                else:
                    a[j] = 0.0
            
            # Check convergence
            change = np.linalg.norm(a - a_old)
            if change < self.tolerance:
                break
        
        return a
    
    def _gradient_descent_optimization(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient descent with soft thresholding (ISTA).
        
        Implements Iterative Shrinkage-Thresholding Algorithm (ISTA)
        following Olshausen & Field (1996) original formulation.
        
        Args:
            x: Single data sample [n_features]
            
        Returns:
            Sparse coefficients [n_components]
        """
        
        # Initialize coefficients
        a = np.zeros(self.n_components)
        
        # Adaptive step size (conservative)
        eta = self.learning_rate
        
        # Gradient descent iterations
        for iteration in range(self.max_iter):
            
            # Compute gradient of reconstruction term: ∇_a ||x - Da||²
            residual = x - self.dictionary_.T @ a
            gradient = -self.dictionary_ @ residual
            
            # Gradient step
            a_grad = a - eta * gradient
            
            # Proximal step: soft thresholding for L1 penalty
            a_new = self._soft_threshold(a_grad, eta * self.alpha)
            
            # Check convergence
            if iteration > 0:
                change = np.linalg.norm(a_new - a)
                if change < self.tolerance:
                    break
            
            a = a_new
        
        return a
    
    def _compute_objective_function(self, X: np.ndarray, codes: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute Olshausen & Field (1996) objective function.
        
        E = ||X - D @ codes||² + α * S(codes)
        
        where S(codes) is the sparsity penalty function.
        
        Args:
            X: Data samples [n_samples, n_features]
            codes: Sparse coefficients [n_samples, n_components]
            
        Returns:
            tuple: (total_objective, reconstruction_error, sparsity_penalty)
        """
        
        # Reconstruction error: ||X - D @ codes||²
        reconstruction = X.T - self.dictionary_.T @ codes.T
        reconstruction_error = np.sum(reconstruction ** 2)
        
        # Sparsity penalty based on specified function
        if self.sparsity_func == 'l1':
            sparsity_penalty = np.sum(np.abs(codes))
        elif self.sparsity_func == 'l2':
            sparsity_penalty = np.sum(codes ** 2)
        elif self.sparsity_func == 'log':
            # Log penalty: Σlog(1 + aᵢ²) (Olshausen & Field 1996)
            sparsity_penalty = np.sum(np.log(1 + codes ** 2))
        elif self.sparsity_func == 'student_t':
            # Student-t penalty: Σlog(1 + aᵢ²/2)
            sparsity_penalty = np.sum(np.log(1 + codes ** 2 / 2))
        else:
            raise ValueError(f"Unknown sparsity function: {self.sparsity_func}")
        
        # Total objective
        total_objective = reconstruction_error + self.alpha * sparsity_penalty
        
        return total_objective, reconstruction_error, sparsity_penalty
    
    def _adaptive_lambda_schedule(self, iteration: int, method: str = 'exponential') -> float:
        """
        Adaptive sparsity parameter scheduling.
        
        Implements various schedules for the sparsity parameter α(t)
        to balance reconstruction vs sparsity during learning.
        
        Args:
            iteration: Current iteration number
            method: Schedule type ('constant', 'exponential', 'polynomial', 'cosine')
            
        Returns:
            Adapted sparsity parameter α(t)
        """
        
        if method == 'constant':
            return self.alpha
        elif method == 'exponential':
            # Exponential decay: α(t) = α₀ * exp(-βt)
            decay_rate = 0.01
            return self.alpha * np.exp(-decay_rate * iteration)
        elif method == 'polynomial':
            # Polynomial decay: α(t) = α₀ * (1 + t)^(-β)
            decay_power = 0.5
            return self.alpha * (1 + iteration) ** (-decay_power)
        elif method == 'cosine':
            # Cosine annealing: α(t) = α_min + (α₀ - α_min)(1 + cos(πt/T))/2
            alpha_min = 0.01 * self.alpha
            return alpha_min + (self.alpha - alpha_min) * (1 + np.cos(np.pi * iteration / self.max_iter)) / 2
        else:
            return self.alpha


# Export the mixin class
__all__ = ['OptimizationAlgorithmsMixin']


if __name__ == "__main__":
    print("🏗️ Sparse Coding - Optimization Algorithms Module")
    print("=" * 50)
    print("📊 MODULE CONTENTS:")
    print("  • OptimizationAlgorithmsMixin - Optimization methods")
    print("  • FISTA (Fast Iterative Shrinkage-Thresholding)")
    print("  • Coordinate Descent for L1-regularized problems")
    print("  • Gradient Descent with soft thresholding (ISTA)")
    print("  • Adaptive sparsity parameter scheduling")
    print("  • Research-accurate objective function computation")
    print("")
    print("✅ Optimization algorithms module loaded successfully!")
    print("🔬 Advanced sparse coding optimization methods!")