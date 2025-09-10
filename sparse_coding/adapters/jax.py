"""
JAX integration for sparse coding.

Provides pure functional implementations compatible with jit compilation,
vmap, and other JAX transformations.
"""

import warnings
from typing import Dict, Any, Tuple, Optional, Callable
from functools import partial

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    warnings.warn("JAX not available. JAX adapters disabled.")
    
    # Provide stubs
    jax = None
    jnp = None
    jit = lambda f: f
    vmap = lambda f, **kwargs: f
    grad = lambda f: f

import numpy as np


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
        # Estimate Lipschitz constant
        L = jnp.linalg.norm(D, ord=2) ** 2
    
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
    
    >>> D = jnp.array(np.random.randn(64, 32))  # dictionary
    >>> x = jnp.array(np.random.randn(64))      # signal
    >>> codes = sparse_encode_jit(D, x, lam=0.1)
    """
    if not HAS_JAX:
        raise ImportError("JAX required for sparse_encode_jit")
    
    n_atoms = D.shape[1]
    
    # Initialize FISTA parameters
    params = {
        't': jnp.array(1.0),
        'a': jnp.zeros(n_atoms),
        'z': jnp.zeros(n_atoms)
    }
    
    # Estimate Lipschitz constant once
    L = jnp.linalg.norm(D, ord=2) ** 2
    
    def body_fun(carry):
        i, params_old = carry
        params_new = _fista_step(params_old, D, x, lam, L)
        
        # Check convergence (simple difference check)
        diff = jnp.linalg.norm(params_new['a'] - params_old['a'])
        converged = diff < tol
        
        return (i + 1, params_new)
    
    def cond_fun(carry):
        i, params = carry
        return i < max_iter
    
    # Run FISTA loop
    _, final_params = jax.lax.while_loop(cond_fun, body_fun, (0, params))
    
    return final_params['a']


# Vectorized version for batch processing
sparse_encode_batch_jit = vmap(sparse_encode_jit, 
                              in_axes=(None, 1, None, None, None), 
                              out_axes=1)


@partial(jit, static_argnames=['normalize'])
def dictionary_update_jit(D: jnp.ndarray,
                         X: jnp.ndarray, 
                         A: jnp.ndarray,
                         eps: float = 1e-6,
                         normalize: bool = True) -> jnp.ndarray:
    """
    JIT-compiled dictionary update using Method of Optimal Directions (MOD).
    
    Parameters
    ----------
    D : jnp.ndarray of shape (n_features, n_atoms)
        Current dictionary
    X : jnp.ndarray of shape (n_features, n_samples)
        Training data
    A : jnp.ndarray of shape (n_atoms, n_samples)  
        Sparse codes
    eps : float, default=1e-6
        Regularization for numerical stability
    normalize : bool, default=True
        Whether to normalize dictionary columns
        
    Returns
    -------
    D_new : jnp.ndarray of shape (n_features, n_atoms)
        Updated dictionary
        
    Examples
    --------
    >>> D_new = dictionary_update_jit(D, X, A)
    """
    if not HAS_JAX:
        raise ImportError("JAX required for dictionary_update_jit")
    
    # MOD update: D = X A^T (A A^T + eps I)^-1
    At = A.T
    G = A @ At
    G = G + eps * jnp.eye(G.shape[0])
    
    # Use solve instead of inv for numerical stability
    D_new = jnp.linalg.solve(G, (X @ At).T).T
    
    if normalize:
        # Normalize columns to unit norm
        norms = jnp.linalg.norm(D_new, axis=0, keepdims=True)
        norms = jnp.where(norms < 1e-12, 1.0, norms)
        D_new = D_new / norms
    
    return D_new


@partial(jit, static_argnames=['n_iter'])
def dictionary_learning_step_jit(D: jnp.ndarray,
                                X: jnp.ndarray,
                                lam: float,
                                n_iter: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Single dictionary learning step (sparse coding + dictionary update).
    
    Parameters
    ----------
    D : jnp.ndarray
        Current dictionary  
    X : jnp.ndarray
        Training data
    lam : float
        Sparsity parameter
    n_iter : int, default=1
        Number of alternating steps
        
    Returns
    -------
    D_new : jnp.ndarray
        Updated dictionary
    A : jnp.ndarray
        Final sparse codes
    """
    if not HAS_JAX:
        raise ImportError("JAX required for dictionary_learning_step_jit")
    
    def step(carry, _):
        D_curr, _ = carry
        
        # Sparse coding step
        A = sparse_encode_batch_jit(D_curr, X, lam)
        
        # Dictionary update step  
        D_new = dictionary_update_jit(D_curr, X, A)
        
        return (D_new, A), None
    
    (D_final, A_final), _ = jax.lax.scan(step, (D, None), jnp.arange(n_iter))
    
    return D_final, A_final


def create_dictionary_learner_jax(n_features: int, 
                                 n_atoms: int,
                                 lam: float = 0.1,
                                 max_iter: int = 30,
                                 solver_steps: int = 100) -> Dict[str, Callable]:
    """
    Create JAX-based dictionary learner with state management.
    
    Returns a dictionary of functions for stateful dictionary learning
    compatible with JAX transformations.
    
    Parameters
    ----------
    n_features : int
        Signal dimension
    n_atoms : int
        Number of dictionary atoms
    lam : float, default=0.1
        Sparsity parameter
    max_iter : int, default=30
        Dictionary learning iterations
    solver_steps : int, default=100
        Sparse coding solver iterations
        
    Returns
    -------
    learner : dict
        Dictionary with 'init', 'step', 'encode', 'decode' functions
        
    Examples
    --------
    >>> learner = create_dictionary_learner_jax(64, 32)
    >>> 
    >>> # Initialize
    >>> state = learner['init'](key=jax.random.PRNGKey(0))
    >>> 
    >>> # Training loop
    >>> for epoch in range(10):
    ...     state = learner['step'](state, X_batch)
    >>> 
    >>> # Encoding
    >>> codes = learner['encode'](state, X_test)
    """
    if not HAS_JAX:
        raise ImportError("JAX required for create_dictionary_learner_jax")
    
    @jit
    def init_fn(key):
        """Initialize learner state."""
        D = jax.random.normal(key, (n_features, n_atoms))
        D = D / jnp.linalg.norm(D, axis=0, keepdims=True)
        return {'dictionary': D, 'step_count': 0}
    
    @jit  
    def step_fn(state, X_batch):
        """Single training step."""
        D = state['dictionary']
        
        # Sparse coding
        A = sparse_encode_batch_jit(D, X_batch, lam, solver_steps)
        
        # Dictionary update
        D_new = dictionary_update_jit(D, X_batch, A)
        
        return {
            'dictionary': D_new,
            'step_count': state['step_count'] + 1
        }
    
    @jit
    def encode_fn(state, X):
        """Encode data using current dictionary."""
        return sparse_encode_batch_jit(state['dictionary'], X, lam, solver_steps)
    
    @jit 
    def decode_fn(state, A):
        """Decode sparse codes."""
        return state['dictionary'] @ A
    
    return {
        'init': init_fn,
        'step': step_fn, 
        'encode': encode_fn,
        'decode': decode_fn
    }


@partial(jit, static_argnames=['penalty_type'])
def general_sparse_encode_jit(D: jnp.ndarray,
                             x: jnp.ndarray,
                             penalty_params: Dict[str, float],
                             penalty_type: str = 'l1',
                             max_iter: int = 100) -> jnp.ndarray:
    """
    General sparse encoding with different penalty types.
    
    Parameters
    ----------
    D : jnp.ndarray
        Dictionary matrix
    x : jnp.ndarray 
        Signal to encode
    penalty_params : dict
        Penalty-specific parameters
    penalty_type : str, default='l1'
        Penalty type ('l1', 'l2', 'elastic_net')
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    codes : jnp.ndarray
        Sparse codes
    """
    if not HAS_JAX:
        raise ImportError("JAX required for general_sparse_encode_jit")
    
    if penalty_type == 'l1':
        return sparse_encode_jit(D, x, penalty_params.get('lam', 0.1), max_iter)
    
    elif penalty_type == 'l2':
        # Ridge regression solution: (D^T D + lam I)^-1 D^T x
        lam = penalty_params.get('lam', 0.1)
        DtD = D.T @ D
        Dtx = D.T @ x
        return jnp.linalg.solve(DtD + lam * jnp.eye(DtD.shape[0]), Dtx)
    
    elif penalty_type == 'elastic_net':
        # Simplified elastic net (would need iterative solver for full version)
        l1_ratio = penalty_params.get('l1_ratio', 0.5)
        lam = penalty_params.get('lam', 0.1)
        
        # Mix of L1 and L2 (approximate)
        l1_weight = lam * l1_ratio
        l2_weight = lam * (1 - l1_ratio)
        
        # Start with L2 solution, then apply L1 prox
        l2_solution = general_sparse_encode_jit(D, x, {'lam': l2_weight}, 'l2')
        return _l1_prox(l2_solution, l1_weight)
    
    else:
        raise ValueError(f"Unknown penalty type: {penalty_type}")


# Create PRNG key utilities for reproducibility
def create_prng_keys(master_key, n_keys: int):
    """Create multiple PRNG keys from master key."""
    if not HAS_JAX:
        raise ImportError("JAX required for create_prng_keys")
    return jax.random.split(master_key, n_keys)


def sparse_coding_loss_jit(params: Dict[str, jnp.ndarray],
                          X: jnp.ndarray,
                          lam: float) -> float:
    """
    JIT-compiled loss function for end-to-end learning.
    
    Parameters
    ----------
    params : dict
        Parameters dict with 'dictionary' key
    X : jnp.ndarray
        Training data
    lam : float
        Sparsity weight
        
    Returns
    -------
    loss : float
        Total loss (reconstruction + sparsity)
    """
    if not HAS_JAX:
        raise ImportError("JAX required for sparse_coding_loss_jit")
    
    D = params['dictionary']
    A = sparse_encode_batch_jit(D, X, lam)
    
    # Reconstruction loss
    reconstruction = D @ A
    recon_loss = jnp.mean((X - reconstruction) ** 2)
    
    # Sparsity loss  
    sparse_loss = jnp.mean(jnp.abs(A))
    
    return recon_loss + lam * sparse_loss


# Gradient function for optimization
sparse_coding_grad_jit = jit(grad(sparse_coding_loss_jit))


# Example training function using JAX optimizers
def train_with_optax(X: jnp.ndarray, 
                    n_atoms: int,
                    n_steps: int = 100,
                    learning_rate: float = 0.01,
                    lam: float = 0.1):
    """
    Example training function using Optax optimizers.
    
    Requires: pip install optax
    
    Parameters
    ---------- 
    X : jnp.ndarray
        Training data
    n_atoms : int
        Number of dictionary atoms
    n_steps : int
        Training steps
    learning_rate : float
        Learning rate
    lam : float
        Sparsity parameter
        
    Returns
    -------
    final_params : dict
        Final learned parameters
    history : list
        Training loss history
    """
    try:
        import optax
    except ImportError:
        raise ImportError("optax required for train_with_optax. Install with: pip install optax")
    
    if not HAS_JAX:
        raise ImportError("JAX required for train_with_optax")
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    n_features = X.shape[0]
    D_init = jax.random.normal(key, (n_features, n_atoms))
    D_init = D_init / jnp.linalg.norm(D_init, axis=0, keepdims=True)
    
    params = {'dictionary': D_init}
    
    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    history = []
    
    # Training loop
    for step in range(n_steps):
        loss_val = sparse_coding_loss_jit(params, X, lam)
        grads = sparse_coding_grad_jit(params, X, lam)
        
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Normalize dictionary
        D = params['dictionary']
        D_norm = D / jnp.linalg.norm(D, axis=0, keepdims=True)
        params['dictionary'] = D_norm
        
        history.append(float(loss_val))
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss_val:.6f}")
    
    return params, history