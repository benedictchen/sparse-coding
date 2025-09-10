from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Penalty:
    """
    Abstract base class for penalty functions in sparse coding.
    
    All penalty functions must implement the proximal operator (prox)
    and value function for optimization algorithms.
    """
    
    def prox(self, z: np.ndarray, t: float) -> np.ndarray: 
        """
        Proximal operator: prox_t(z) = argmin_x (0.5 * ||x - z||^2 + t * penalty(x))
        
        Parameters
        ----------
        z : np.ndarray
            Input point
        t : float
            Step size parameter
            
        Returns
        -------
        x : np.ndarray
            Solution of proximal operator
            
        Examples
        --------
        L1 soft thresholding: np.sign(z) * np.maximum(np.abs(z) - t*lam, 0.0)
        """
        raise NotImplementedError("Subclasses must implement prox()")
    
    def value(self, a: np.ndarray) -> float: 
        """
        Evaluate penalty function at point a.
        
        Parameters
        ----------
        a : np.ndarray
            Point to evaluate
            
        Returns
        -------
        penalty_value : float
            Value of penalty function
            
        Examples
        --------
        L1 penalty: lam * np.sum(np.abs(a))
        """
        raise NotImplementedError("Subclasses must implement value()")

@dataclass
class L1(Penalty):
    lam: float
    def prox(self, z, t): return np.sign(z) * np.maximum(np.abs(z) - t*self.lam, 0.0)
    def value(self, a): return self.lam * float(np.sum(np.abs(a)))

@dataclass
class ElasticNet(Penalty):
    l1: float; l2: float
    def prox(self, z, t): return (np.sign(z) * np.maximum(np.abs(z) - t*self.l1, 0.0)) / (1.0 + t*self.l2)
    def value(self, a): return self.l1 * float(np.sum(np.abs(a))) + 0.5*self.l2 * float(np.sum(a*a))

@dataclass
class NonNegL1(Penalty):
    lam: float
    def prox(self, z, t): return np.maximum(z - t*self.lam, 0.0)
    def value(self, a): return self.lam * float(np.sum(a)) if np.all(a>=0) else np.inf
