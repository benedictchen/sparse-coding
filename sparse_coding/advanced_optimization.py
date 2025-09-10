"""
Advanced Optimization Methods for Sparse Coding

Implements multiple optimization algorithms beyond basic FISTA:
- Proximal gradient methods
- Accelerated proximal gradient (FISTA variants)
- Coordinate descent
- Iterative soft thresholding (ISTA)
- Adaptive step size methods
"""

import numpy as np
from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod


class ProximalOperator(ABC):
    """Abstract base class for proximal operators"""
    
    @abstractmethod
    def prox(self, x: np.ndarray, step_size: float) -> np.ndarray:
        """Apply proximal operator"""
        pass
    
    @abstractmethod
    def value(self, x: np.ndarray) -> float:
        """Evaluate penalty function"""
        pass


class L1Proximal(ProximalOperator):
    """L1 proximal operator (soft thresholding)"""
    
    def __init__(self, lam: float):
        self.lam = lam
    
    def prox(self, x: np.ndarray, step_size: float) -> np.ndarray:
        """Soft thresholding"""
        threshold = step_size * self.lam
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)
    
    def value(self, x: np.ndarray) -> float:
        """L1 penalty"""
        return self.lam * np.sum(np.abs(x))


class ElasticNetProximal(ProximalOperator):
    """Elastic Net proximal operator"""
    
    def __init__(self, l1: float, l2: float):
        self.l1 = l1
        self.l2 = l2
    
    def prox(self, x: np.ndarray, step_size: float) -> np.ndarray:
        """Elastic net proximal operator"""
        threshold = step_size * self.l1
        scaling = 1.0 / (1.0 + step_size * self.l2)
        return scaling * np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)
    
    def value(self, x: np.ndarray) -> float:
        """Elastic net penalty"""
        return self.l1 * np.sum(np.abs(x)) + 0.5 * self.l2 * np.sum(x**2)


class NonNegativeL1Proximal(ProximalOperator):
    """Non-negative L1 proximal operator"""
    
    def __init__(self, lam: float):
        self.lam = lam
    
    def prox(self, x: np.ndarray, step_size: float) -> np.ndarray:
        """Non-negative soft thresholding"""
        threshold = step_size * self.lam
        return np.maximum(x - threshold, 0.0)
    
    def value(self, x: np.ndarray) -> float:
        """Non-negative L1 penalty"""
        if np.all(x >= 0):
            return self.lam * np.sum(x)
        else:
            return np.inf


class AdvancedOptimizer:
    """Advanced optimization algorithms for sparse coding"""
    
    def __init__(self, 
                 dictionary: np.ndarray,
                 proximal_op: ProximalOperator,
                 max_iter: int = 1000,
                 tolerance: float = 1e-6):
        """
        Initialize advanced optimizer
        
        Args:
            dictionary: Dictionary matrix
            proximal_op: Proximal operator for regularization
            max_iter: Maximum iterations
            tolerance: Convergence tolerance
        """
        self.dictionary = dictionary
        self.proximal_op = proximal_op
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # Precompute for efficiency
        self.DtD = dictionary.T @ dictionary
        self.lipschitz_constant = np.linalg.norm(self.DtD, ord=2)
    
    def ista(self, signal: np.ndarray, x0: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Iterative Soft Thresholding Algorithm (ISTA)
        
        Args:
            signal: Input signal to encode
            x0: Initial coefficients
            
        Returns:
            Dict containing solution and convergence info
        """
        
        if x0 is None:
            x = np.zeros(self.dictionary.shape[1])
        else:
            x = x0.copy()
        
        Dt_signal = self.dictionary.T @ signal
        step_size = 1.0 / self.lipschitz_constant
        
        history = {'objectives': [], 'residuals': []}
        
        for iteration in range(self.max_iter):
            x_old = x.copy()
            
            # Gradient step
            gradient = self.DtD @ x - Dt_signal
            z = x - step_size * gradient
            
            # Proximal step
            x = self.proximal_op.prox(z, step_size)
            
            # Compute objective and residual
            data_term = 0.5 * np.linalg.norm(signal - self.dictionary @ x)**2
            reg_term = self.proximal_op.value(x)
            objective = data_term + reg_term
            residual = np.linalg.norm(x - x_old)
            
            history['objectives'].append(objective)
            history['residuals'].append(residual)
            
            # Check convergence
            if residual < self.tolerance:
                break
        
        return {
            'solution': x,
            'iterations': iteration + 1,
            'converged': residual < self.tolerance,
            'final_objective': objective,
            'history': history
        }
    
    def fista(self, signal: np.ndarray, x0: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Fast Iterative Soft Thresholding Algorithm (FISTA)
        
        Args:
            signal: Input signal to encode
            x0: Initial coefficients
            
        Returns:
            Dict containing solution and convergence info
        """
        
        if x0 is None:
            x = np.zeros(self.dictionary.shape[1])
        else:
            x = x0.copy()
        
        y = x.copy()
        t = 1.0
        
        Dt_signal = self.dictionary.T @ signal
        step_size = 1.0 / self.lipschitz_constant
        
        history = {'objectives': [], 'residuals': []}
        
        for iteration in range(self.max_iter):
            x_old = x.copy()
            
            # Gradient step
            gradient = self.DtD @ y - Dt_signal
            z = y - step_size * gradient
            
            # Proximal step
            x = self.proximal_op.prox(z, step_size)
            
            # Update momentum
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            beta = (t - 1) / t_new
            y = x + beta * (x - x_old)
            t = t_new
            
            # Compute objective and residual
            data_term = 0.5 * np.linalg.norm(signal - self.dictionary @ x)**2
            reg_term = self.proximal_op.value(x)
            objective = data_term + reg_term
            residual = np.linalg.norm(x - x_old)
            
            history['objectives'].append(objective)
            history['residuals'].append(residual)
            
            # Check convergence
            if residual < self.tolerance:
                break
        
        return {
            'solution': x,
            'iterations': iteration + 1,
            'converged': residual < self.tolerance,
            'final_objective': objective,
            'history': history
        }
    
    def coordinate_descent(self, signal: np.ndarray, x0: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Coordinate Descent for L1-regularized problems
        
        Args:
            signal: Input signal to encode
            x0: Initial coefficients
            
        Returns:
            Dict containing solution and convergence info
        """
        
        if x0 is None:
            x = np.zeros(self.dictionary.shape[1])
        else:
            x = x0.copy()
        
        n_features = len(x)
        Dt_signal = self.dictionary.T @ signal
        
        history = {'objectives': [], 'residuals': []}
        
        for iteration in range(self.max_iter):
            x_old = x.copy()
            
            # Cycle through coordinates
            for j in range(n_features):
                # Compute partial residual
                partial_residual = Dt_signal[j] - np.sum(self.DtD[j, :] * x) + self.DtD[j, j] * x[j]
                
                # Soft thresholding update
                if isinstance(self.proximal_op, L1Proximal):
                    threshold = self.proximal_op.lam / self.DtD[j, j]
                    if partial_residual > threshold:
                        x[j] = (partial_residual - threshold)
                    elif partial_residual < -threshold:
                        x[j] = (partial_residual + threshold)
                    else:
                        x[j] = 0.0
                else:
                    # Use proximal operator for other penalties
                    z = partial_residual / self.DtD[j, j]
                    step_size = 1.0 / self.DtD[j, j]
                    x[j] = self.proximal_op.prox(np.array([z]), step_size)[0]
            
            # Compute objective and residual
            data_term = 0.5 * np.linalg.norm(signal - self.dictionary @ x)**2
            reg_term = self.proximal_op.value(x)
            objective = data_term + reg_term
            residual = np.linalg.norm(x - x_old)
            
            history['objectives'].append(objective)
            history['residuals'].append(residual)
            
            # Check convergence
            if residual < self.tolerance:
                break
        
        return {
            'solution': x,
            'iterations': iteration + 1,
            'converged': residual < self.tolerance,
            'final_objective': objective,
            'history': history
        }
    
    def adaptive_fista(self, signal: np.ndarray, x0: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Adaptive FISTA with backtracking line search
        
        Args:
            signal: Input signal to encode
            x0: Initial coefficients
            
        Returns:
            Dict containing solution and convergence info
        """
        
        if x0 is None:
            x = np.zeros(self.dictionary.shape[1])
        else:
            x = x0.copy()
        
        y = x.copy()
        t = 1.0
        step_size = 1.0 / self.lipschitz_constant
        
        Dt_signal = self.dictionary.T @ signal
        history = {'objectives': [], 'residuals': [], 'step_sizes': []}
        
        def objective_function(coeffs):
            data_term = 0.5 * np.linalg.norm(signal - self.dictionary @ coeffs)**2
            reg_term = self.proximal_op.value(coeffs)
            return data_term + reg_term
        
        for iteration in range(self.max_iter):
            x_old = x.copy()
            
            # Backtracking line search
            gradient = self.DtD @ y - Dt_signal
            
            # Try different step sizes
            for _ in range(20):  # Max backtracking steps
                z = y - step_size * gradient
                x_candidate = self.proximal_op.prox(z, step_size)
                
                # Check sufficient decrease condition
                q_val = (objective_function(y) + 
                        np.dot(gradient, x_candidate - y) + 
                        0.5 / step_size * np.linalg.norm(x_candidate - y)**2)
                
                if objective_function(x_candidate) <= q_val:
                    x = x_candidate
                    break
                else:
                    step_size *= 0.5
            
            # Update momentum
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            beta = (t - 1) / t_new
            y = x + beta * (x - x_old)
            t = t_new
            
            # Compute metrics
            objective = objective_function(x)
            residual = np.linalg.norm(x - x_old)
            
            history['objectives'].append(objective)
            history['residuals'].append(residual)
            history['step_sizes'].append(step_size)
            
            # Adapt step size for next iteration
            step_size = min(step_size * 1.1, 1.0 / self.lipschitz_constant)
            
            # Check convergence
            if residual < self.tolerance:
                break
        
        return {
            'solution': x,
            'iterations': iteration + 1,
            'converged': residual < self.tolerance,
            'final_objective': objective,
            'history': history
        }


def create_advanced_sparse_coder(dictionary: np.ndarray, 
                               penalty_type: str = 'l1',
                               penalty_params: Dict[str, float] = None,
                               **kwargs) -> AdvancedOptimizer:
    """
    Factory function for creating advanced sparse coder
    
    Args:
        dictionary: Dictionary matrix
        penalty_type: Type of penalty ('l1', 'elastic_net', 'non_negative_l1')
        penalty_params: Parameters for penalty function
        **kwargs: Additional arguments for optimizer
        
    Returns:
        AdvancedOptimizer instance
    """
    
    if penalty_params is None:
        penalty_params = {}
    
    if penalty_type == 'l1':
        proximal_op = L1Proximal(penalty_params.get('lam', 0.1))
    elif penalty_type == 'elastic_net':
        proximal_op = ElasticNetProximal(
            penalty_params.get('l1', 0.1),
            penalty_params.get('l2', 0.01)
        )
    elif penalty_type == 'non_negative_l1':
        proximal_op = NonNegativeL1Proximal(penalty_params.get('lam', 0.1))
    else:
        raise ValueError(f"Unknown penalty type: {penalty_type}")
    
    return AdvancedOptimizer(dictionary, proximal_op, **kwargs)