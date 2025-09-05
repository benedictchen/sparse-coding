"""
Minimal, research-faithful SparseCoder API with both L1 and paper-exact modes.
"""

import numpy as np
from .paper_exact import nonlinear_cg, energy_paper
from .fista import fista, lipschitz_const
from .homeostasis import equalize_variance, apply_gain, lambda_from_sigma

class SparseCoder:
    """
    Research-faithful sparse coder with atoms-as-columns convention.
    
    Supports both modern L1/FISTA and paper-exact Olshausen & Field modes.
    """
    
    def __init__(self, n_atoms=128, lam=None, ratio_lambda_over_sigma=0.14, 
                 max_iter=200, tol=1e-6, seed=0, mode='l1', 
                 do_gain_equalization=True, stop_rule='default', rel_tol=0.01):
        self.n_atoms = n_atoms
        self.lam = lam
        self.ratio_lambda_over_sigma = ratio_lambda_over_sigma
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.do_gain_equalization = do_gain_equalization
        self.stop_rule = stop_rule
        self.rel_tol = rel_tol
        self.D = None
        self.L = None
        self.g = None  # homeostatic gains

    def _init_dict(self, p):
        """Initialize random dictionary with unit-norm atoms."""
        D = self.rng.normal(size=(p, self.n_atoms))
        D /= (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
        return D

    def fit(self, X, n_steps=50, lr=0.1):
        """
        Learn dictionary from patches X via alternating optimization.
        
        Args:
            X: Patches (p, N) - already whitened/preprocessed
            n_steps: Number of dictionary update steps
            lr: Learning rate for dictionary updates
            
        Returns:
            self: Fitted SparseCoder
        """
        # Input validation
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        
        p, N = X.shape
        if p <= 0 or N <= 0:
            raise ValueError(f"Invalid data dimensions: {X.shape}")
        
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")
            
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if self.D is None:
            self.D = self._init_dict(p)
        
        if self.do_gain_equalization:
            self.g = np.ones(self.n_atoms)
        
        for step in range(n_steps):
            # Sparse encode all patches
            A = self.encode(X)
            
            # Update dictionary using simple gradient descent
            R = X - self.D @ A  # residuals
            grad = R @ A.T  # gradient w.r.t D
            self.D += lr * grad
            
            # Normalize atoms
            norms = np.linalg.norm(self.D, axis=0) + 1e-12
            self.D /= norms
            
            # Homeostatic gain control
            if self.do_gain_equalization and step % 5 == 0:
                g_new = equalize_variance(A)
                self.D, self.g = apply_gain(self.D, g_new)
                
        return self

    def encode(self, X):
        """
        Sparse encode patches using FISTA (L1) or nonlinear CG (paper).
        
        Args:
            X: Patches (p, N)
            
        Returns:
            A: Sparse codes (K, N)
        """
        if self.D is None:
            raise ValueError("Dictionary not initialized. Call fit() first.")
        
        # Input validation    
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
            
        p, N = X.shape
        if p != self.D.shape[0]:
            raise ValueError(f"X shape {X.shape} incompatible with dictionary shape {self.D.shape}")
        
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")
        K = self.D.shape[1]
        A = np.zeros((K, N))
        
        # Compute Lipschitz constant for FISTA if needed
        if self.mode == 'l1' and self.L is None:
            self.L = lipschitz_const(self.D)
        
        # Determine effective lambda
        if self.lam is None:
            sigma_x = float(np.std(X))
            lam_eff = self.ratio_lambda_over_sigma * sigma_x
        else:
            lam_eff = self.lam
            
        # Encode each patch
        if self.mode == 'l1':
            for n in range(N):
                a, _ = fista(self.D, X[:, n], lam_eff, L=self.L, 
                           max_iter=self.max_iter, tol=self.tol)
                A[:, n] = a
        elif self.mode == 'paper':
            # Paper's log penalty with nonlinear CG
            sigma = float(np.std(X))
            for n in range(N):
                a0 = np.zeros(K)
                a, _ = nonlinear_cg(self.D, X[:, n], a0, lam=lam_eff, sigma=sigma,
                                 max_iter=self.max_iter, tol=self.tol, rel_tol=self.rel_tol)
                A[:, n] = a
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        return A

    def decode(self, A):
        """Reconstruct patches from sparse codes."""
        if self.D is None:
            raise ValueError("Dictionary not initialized.")
        
        # Input validation
        A = np.asarray(A, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"A must be 2D array, got shape {A.shape}")
        
        K, N = A.shape
        if K != self.D.shape[1]:
            raise ValueError(f"A shape {A.shape} incompatible with dictionary shape {self.D.shape}")
        
        if not np.isfinite(A).all():
            raise ValueError("A contains non-finite values")
            
        return self.D @ A

    def transform(self, X):
        """Alias for encode() - sklearn compatibility."""
        return self.encode(X).T  # Return (N, K) for sklearn compatibility

    def fit_transform(self, X, **fit_params):
        """Fit dictionary and encode - sklearn compatibility."""
        return self.fit(X, **fit_params).transform(X)