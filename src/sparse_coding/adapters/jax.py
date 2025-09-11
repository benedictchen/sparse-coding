"""
JAX integration for sparse coding.

Provides pure functional implementations compatible with jit compilation,
vmap, and other JAX transformations.
"""

import warnings
from typing import Dict, Any, Tuple, Optional, Callable
from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from ..fista_batch import power_iter_L


def _l1_prox(z: jnp.ndarray, threshold: float) -> jnp.ndarray:
    """JAX implementation of L1 proximal operator (soft thresholding)."""
    return jnp.sign(z) * jnp.maximum(jnp.abs(z) - threshold, 0.0)

def _fista_step(params: Dict[str, jnp.ndarray], 
                D: jnp.ndarray, 
                x: jnp.ndarray, 
                lam: float,
                L: Optional[float] = None) -> Dict[str, jnp.ndarray]:
    """Single FISTA iteration step."""
    if L is None:
        # Estimate Lipschitz constant using safe cross-backend method
        D_numpy = np.asarray(D)
        L = float(power_iter_L(D_numpy))
    
    t_old = params['t']
    a_old = params['a']
    z_old = params['z']
    
    # Gradient step
    grad = D.T @ (D @ z_old - x)
    z_new = z_old - grad / L
    
    # Proximal step (L1)
    a_new = _l1_prox(z_new, lam / L)
    
    # Momentum step
    t_new = (1.0 + jnp.sqrt(1.0 + 4.0 * t_old**2)) / 2.0
    beta = (t_old - 1.0) / t_new
    z_new = a_new + beta * (a_new - a_old)
    
    return {
        't': t_new,
        'a': a_new,
        'z': z_new
    }

@partial(jit, static_argnames=['max_iter', 'tol'])
def sparse_encode_jit(D: jnp.ndarray, 
                      x: jnp.ndarray, 
                      lam: float,
                      max_iter: int = 100,
                      tol: float = 1e-6) -> jnp.ndarray:
    """
    JIT-compiled sparse coding inference using FISTA.
    
    Parameters
    ----------
    D : jnp.ndarray of shape (n_features, n_atoms)
        Dictionary matrix
    x : jnp.ndarray of shape (n_features,)
        Signal to encode
    lam : float
        L1 regularization parameter
    max_iter : int, default=100
        Maximum iterations
    tol : float, default=1e-6
        Convergence tolerance
        
    Returns
    -------
    codes : jnp.ndarray of shape (n_atoms,)
        Sparse codes
        
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from sparse_coding.adapters.jax import sparse_encode_jit
    >>> 
    >>> # Create synthetic data
    >>> key = jax.random.PRNGKey(42)
    >>> D = jax.random.normal(key, (64, 32))
    >>> x = jax.random.normal(key, (64,))
    >>> 
    >>> # Encode
    >>> codes = sparse_encode_jit(D, x, lam=0.1)
    >>> print(f"Sparsity: {jnp.mean(jnp.abs(codes) < 1e-6):.1%}")
    """
    n_atoms = D.shape[1]
    
    # Initialize parameters
    params = {
        't': 1.0,
        'a': jnp.zeros(n_atoms),
        'z': jnp.zeros(n_atoms)
    }
    
    # FISTA loop
    for i in range(max_iter):
        params_old = params
        params = _fista_step(params, D, x, lam)
        
        # Check convergence
        diff = jnp.linalg.norm(params['a'] - params_old['a'])
        if diff < tol:
            break
    
    return params['a']

# Vectorized version for batch processing
sparse_encode_batch_jit = vmap(sparse_encode_jit, in_axes=(None, 1, None, None, None))

def _mod_step(D: jnp.ndarray, A: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """Method of Optimal Directions dictionary update step."""
    # Update each atom
    def update_atom(i):
        # Get residual excluding atom i
        R_i = X - D @ A + jnp.outer(D[:, i], A[i, :])
        
        # Update atom i
        numerator = R_i @ A[i, :]
        denominator = jnp.sum(A[i, :] ** 2) + 1e-8
        
        return numerator / denominator
    
    # Update all atoms
    atom_indices = jnp.arange(D.shape[1])
    new_atoms = vmap(update_atom)(atom_indices)
    
    # Normalize atoms
    norms = jnp.linalg.norm(new_atoms, axis=0, keepdims=True)
    new_atoms = new_atoms / jnp.maximum(norms, 1e-8)
    
    return new_atoms

@partial(jit, static_argnames=['max_iter'])
def dictionary_update_jit(D_init: jnp.ndarray,
                          A: jnp.ndarray, 
                          X: jnp.ndarray,
                          max_iter: int = 1) -> jnp.ndarray:
    """
    JIT-compiled dictionary update using MOD.
    
    Parameters
    ----------
    D_init : jnp.ndarray of shape (n_features, n_atoms)
        Initial dictionary
    A : jnp.ndarray of shape (n_atoms, n_samples)
        Sparse codes
    X : jnp.ndarray of shape (n_features, n_samples)
        Training data
    max_iter : int, default=1
        Number of MOD iterations
        
    Returns
    -------
    D : jnp.ndarray of shape (n_features, n_atoms)
        Updated dictionary
    """
    D = D_init
    
    for _ in range(max_iter):
        D = _mod_step(D, A, X)
    
    return D

def sparse_coding_loss(D: jnp.ndarray, 
                       A: jnp.ndarray, 
                       X: jnp.ndarray,
                       lam: float) -> float:
    """
    Compute sparse coding objective function.
    
    Parameters
    ----------
    D : jnp.ndarray of shape (n_features, n_atoms)
        Dictionary matrix
    A : jnp.ndarray of shape (n_atoms, n_samples)
        Sparse codes
    X : jnp.ndarray of shape (n_features, n_samples)
        Data matrix
    lam : float
        L1 regularization parameter
        
    Returns
    -------
    loss : float
        Sparse coding loss (reconstruction + L1 penalty)
    """
    reconstruction_error = jnp.mean((X - D @ A) ** 2)
    sparsity_penalty = lam * jnp.mean(jnp.abs(A))
    return reconstruction_error + sparsity_penalty

# Create gradient functions
grad_D = jit(grad(sparse_coding_loss, argnums=0))
grad_A = jit(grad(sparse_coding_loss, argnums=1))

class JAXSparseCoder:
    """
    JAX-based sparse coder with automatic differentiation.
    
    Provides functional interface compatible with JAX transformations
    while maintaining similar API to classical sparse coding.
    
    Parameters
    ----------
    n_atoms : int
        Number of dictionary atoms
    lam : float, default=0.1
        L1 regularization parameter
    max_iter : int, default=100
        Maximum iterations for inference
    tol : float, default=1e-6
        Convergence tolerance
    """
    
    def __init__(self, n_atoms: int, lam: float = 0.1, 
                 max_iter: int = 100, tol: float = 1e-6):
        self.n_atoms = n_atoms
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.D = None
    
    def fit(self, X: jnp.ndarray, n_iter: int = 10, 
            init_dict: Optional[jnp.ndarray] = None) -> 'JAXSparseCoder':
        """
        Fit dictionary using alternating optimization.
        
        Parameters
        ----------
        X : jnp.ndarray of shape (n_features, n_samples)
            Training data
        n_iter : int, default=10
            Number of alternating optimization iterations
        init_dict : jnp.ndarray, optional
            Initial dictionary. If None, random initialization.
            
        Returns
        -------
        self : JAXSparseCoder
            Fitted instance
        """
        n_features, n_samples = X.shape
        
        # Initialize dictionary
        if init_dict is not None:
            self.D = init_dict
        else:
            key = jax.random.PRNGKey(42)
            self.D = jax.random.normal(key, (n_features, self.n_atoms))
            self.D = self.D / jnp.linalg.norm(self.D, axis=0, keepdims=True)
        
        # Alternating optimization
        for iteration in range(n_iter):
            # Sparse coding step
            A = sparse_encode_batch_jit(self.D, X, self.lam, self.max_iter, self.tol)
            
            # Dictionary update step
            self.D = dictionary_update_jit(self.D, A, X)
        
        return self
    
    def encode(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Encode data using fitted dictionary.
        
        Parameters
        ----------
        X : jnp.ndarray of shape (n_features, n_samples)
            Data to encode
            
        Returns
        -------
        A : jnp.ndarray of shape (n_atoms, n_samples)
            Sparse codes
        """
        if self.D is None:
            raise ValueError("Must call fit() before encode()")
        
        if X.ndim == 1:
            return sparse_encode_jit(self.D, X, self.lam, self.max_iter, self.tol)
        else:
            return sparse_encode_batch_jit(self.D, X, self.lam, self.max_iter, self.tol)
    
    def decode(self, A: jnp.ndarray) -> jnp.ndarray:
        """
        Decode sparse codes to reconstruction.
        
        Parameters
        ----------
        A : jnp.ndarray of shape (n_atoms, n_samples)
            Sparse codes
            
        Returns
        -------
        X_hat : jnp.ndarray of shape (n_features, n_samples)
            Reconstructed data
        """
        if self.D is None:
            raise ValueError("Must call fit() before decode()")
        
        return self.D @ A

