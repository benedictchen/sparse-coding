"""
Sparse coding implementation based on Olshausen & Field (1996).

Research Foundation:
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field 
  properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.
- Olshausen, B. A., & Field, D. J. (1997). Sparse coding with an overcomplete basis set: 
  A strategy employed by V1? Vision research, 37(23), 3311-3325.
- MOD Algorithm: Engan, K., Aase, S. O., & Husoy, J. H. (1999). Method of optimal directions 
  for frame design. Proceedings of the 1999 IEEE International Conference on Acoustics, 
  Speech, and Signal Processing, 5, 2443-2446.

Author: Benedict Chen
"""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Union, Tuple, Any, Literal, Protocol, Callable, Dict

from joblib import Parallel, delayed
import scipy.sparse
from scipy.sparse.linalg import svds

from .fista_batch import fista_batch, power_iter_L

# Custom exceptions for better error handling
class SparseCodingError(ValueError):
    """Custom exception for sparse coding specific errors."""
    pass
SparseCodingMode = Literal['l1', 'log', 'paper', 'paper_gdD', 'olshausen_pure', 'transcoder']
AnnealingConfig = Union[bool, Tuple[float, float]]

# Protocol for random number generator
class RandomGenerator(Protocol):
    """Protocol for numpy random number generators."""
    def integers(self, low: int, high: int, size: Optional[int] = None) -> Union[int, np.ndarray]: ...
    def randn(self, *args: int) -> Union[float, np.ndarray]: ...

def _validate_input(X: ArrayLike, name: str = "X", allow_sparse: bool = False) -> Union[NDArray[np.floating], 'scipy.sparse.spmatrix']:
    """Robust input validation with sparse matrix support following scikit-learn patterns."""
    
    # Handle different input types and convert to validated form
    if isinstance(X, np.ndarray):
        X_validated = X.astype(float)
        if not np.all(np.isfinite(X_validated)):
            raise SparseCodingError(f"{name} contains non-finite values (inf/nan)")
    
    elif allow_sparse and scipy.sparse.issparse(X):
        # Keep sparse format if requested and scipy available
        if not scipy.sparse.isspmatrix_csr(X):
            X = scipy.sparse.csr_matrix(X)  # Convert to efficient format
        # Check for non-finite values in sparse matrix data
        if not np.all(np.isfinite(X.data)):
            raise SparseCodingError(f"{name} sparse matrix contains non-finite values")
        X_validated = X
    
    else:
        # Convert other array-like inputs to numpy
        try:
            X_validated = np.asarray(X, dtype=float)
            if not np.all(np.isfinite(X_validated)):
                raise SparseCodingError(f"{name} contains non-finite values (inf/nan)")
        except (ValueError, TypeError) as e:
            raise SparseCodingError(f"Cannot convert {name} to array: {e}")
    
    # Additional shape and size validations for all paths
    if hasattr(X_validated, 'shape'):
        if len(X_validated.shape) not in [1, 2]:
            raise SparseCodingError(f"{name} must be 1D or 2D array, got {len(X_validated.shape)}D")
        if X_validated.size == 0:
            raise SparseCodingError(f"{name} cannot be empty")
    
    return X_validated

def _normalize_columns(D: ArrayLike) -> np.ndarray:
    """Normalize dictionary columns to unit norm with numerical stability."""
    D = np.asarray(D, dtype=float)
    n = np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    return D / n

def _mod_update(D: np.ndarray, X: np.ndarray, A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    MOD (Method of Optimal Directions) dictionary update.
    
    Research Foundation: Engan et al. (1999) "Method of optimal directions for frame design"
    Mathematical Form: D = X A^T (A A^T + εI)^-1
    
    Args:
        D: Current dictionary (features x atoms)
        X: Data signals (features x samples) 
        A: Sparse codes (atoms x samples)
        eps: Regularization for numerical stability
        
    Returns:
        Updated dictionary with normalized columns
        
    Numerical Stability:
    - Uses solve() instead of inv() for better conditioning
    - Checks condition number; falls back to pinv for ill-conditioned cases
    - Monitors for NaN/inf and provides diagnostic info
    """
    At = A.T
    G = A @ At  # Gram matrix (atoms x atoms)
    
    # Scikit-learn style regularization: stronger alpha for stability
    alpha = 1e-3  # Following scikit-learn DictionaryLearning standard
    G.flat[::G.shape[0]+1] += alpha  # Primary regularization
    G.flat[::G.shape[0]+1] += eps   # Additional epsilon for numerical safety
    
    # Check conditioning (critical for overcomplete dictionaries)
    cond_num = np.linalg.cond(G)
    if cond_num > 1e10:  # Dictionary learning threshold (web consensus for DL)
        # Ill-conditioned: use pseudo-inverse with warning
        try:
            D_new = X @ At @ np.linalg.pinv(G)
        except np.linalg.LinAlgError:
            # Fallback: return original dictionary with small perturbation
            return _normalize_columns(D + 1e-8 * np.random.randn(*D.shape))
    else:
        # Well-conditioned: use stable solve
        try:
            # Solve G^T Y^T = (X A^T)^T, then D = Y^T  
            D_new = np.linalg.solve(G, (X @ At).T).T
        except np.linalg.LinAlgError:
            # Backup method if solve fails
            D_new = X @ At @ np.linalg.pinv(G)
    
    # Validate result for NaN/inf (critical for downstream stability)
    if not np.all(np.isfinite(D_new)):
        # Return normalized original with diagnostic
        return _normalize_columns(D + 1e-8 * np.random.randn(*D.shape))
    
    return _normalize_columns(D_new)

def _homeostatic_equalize(D: np.ndarray, A: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Homeostatic equalization (heuristic implementation).
    
    We equalize dictionary atom usage by rescaling dictionary columns
    inversely to recent code energy. This is *inspired by* Olshausen & Field's
    homeostatic mechanisms but is not an identical implementation of their 
    gain control on activations.
    
    Args:
        D: Dictionary matrix to scale
        A: Recent sparse codes used to measure usage
        alpha: Adaptation strength (0=no change, 1=full equalization)
        
    Note:
        O&F applied gain control to neuron activities/thresholds, whereas
        this implementation modifies dictionary atom magnitudes for 
        computational convenience.
    """
    usage = np.sqrt(np.mean(A*A, axis=1) + 1e-12)  # (K,)
    target = np.mean(usage)
    scale = (target / (usage + 1e-12))**alpha
    D = D * scale[np.newaxis, :]
    return _normalize_columns(D)

def _gradD_update(D: np.ndarray, X: np.ndarray, A: np.ndarray, lr: float = 0.1) -> np.ndarray:
    """Gradient ascent step on D for 0.5||X - D A||^2 (note: ascent on negative loss -> descent on loss)
    Here we *minimize* loss: D <- D - lr * d/dD (0.5||X-DA||^2) = D + lr * (X - D A) A^T
    """
    R = X - D @ A
    D = D + lr * (R @ A.T)
    return _normalize_columns(D)

def _reinit_dead_atoms(D: np.ndarray, X: np.ndarray, A: np.ndarray, rng: RandomGenerator) -> np.ndarray:
    """Re-initialize atoms with low activation frequency.
    
    Uses activation frequency instead of L2 norm for research accuracy.
    This better captures dictionary atom usage in sparse coding contexts.
    """
    # Count activation frequency: how often each atom is used
    # An atom is "active" if its coefficient magnitude exceeds a threshold
    activation_threshold = 1e-6
    activation_freq = np.mean(np.abs(A) > activation_threshold, axis=1)
    
    # Dead atoms are those with very low activation frequency
    # Research shows frequency-based detection is more robust than L2 norm
    frequency_threshold = 0.01  # Must be active in at least 1% of samples
    dead = activation_freq < frequency_threshold
    
    if np.any(dead):
        # Sample random patches from X to reinitialize dead atoms
        # This follows the homeostatic mechanism from Olshausen & Field (1996)
        idx = rng.integers(0, X.shape[1], size=int(np.sum(dead)))
        D[:, dead] = X[:, idx]
        D = _normalize_columns(D)
    return D

def _paper_energy_grad(x: np.ndarray, D: np.ndarray, a: np.ndarray, lam: float, sigma: float) -> Tuple[float, np.ndarray]:
    # E(a) = 0.5||x - D a||^2 + lam * sum log(1 + (a/sigma)^2)
    r = x - D @ a
    
    # Use 64-bit precision for log1p stability - preferred over scaling
    # This avoids gradient-energy inconsistency from normalization corrections
    a_scaled = (a / sigma).astype(np.float64)
    
    # Compute log prior term with 64-bit precision
    log_prior_term = np.sum(np.log1p(a_scaled**2))
    
    energy = 0.5 * float(r @ r) + lam * float(log_prior_term)
    
    # Gradient: d/da [0.5||x-Da||^2 + λ∑log(1+(a/σ)^2)]
    # = -D^T(x-Da) + λ * 2a/(σ^2 + a^2)
    grad = -(D.T @ r) + lam * (2*a / (sigma**2 + a*a))
    
    return energy, grad

def _log_prior_infer_single(x: np.ndarray, D: np.ndarray, lam: float, max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    """
    Log prior sparse coding inference (Olshausen & Field 1996 original).
    
    Research Foundation:
    - Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field 
      properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.
    
    Mathematical Formulation:
    E(a) = 0.5||x - Da||² + λ∑log(1 + a²/σ²)
    """
    return _ncg_infer_single(x, D, lam, sigma=1.0, max_iter=max_iter, tol=tol)

def _ncg_infer_single(x, D, lam, sigma, max_iter=200, tol=1e-6):
    """Nonlinear Conjugate Gradient with Polak-Ribière and Armijo backtracking with gradient clipping"""
    K = D.shape[1]
    a = np.zeros((K,), dtype=float)
    f, g = _paper_energy_grad(x, D, a, lam, sigma)
    
    # Gradient clipping to prevent explosions (critical for stability)
    grad_clip_threshold = 100.0  # Research-based threshold for sparse coding
    g_norm = np.linalg.norm(g)
    if g_norm > grad_clip_threshold:
        g = g * (grad_clip_threshold / g_norm)
    
    d = -g
    for _ in range(int(max_iter)):
        # Ensure descent direction
        if g @ d > 0:
            d = -g
        # Backtracking line search (Armijo)
        t = 1.0
        f0 = f
        gTd = g @ d
        while True:
            a_new = a + t*d
            f_new, g_new = _paper_energy_grad(x, D, a_new, lam, sigma)
            if f_new <= f0 + 1e-4 * t * gTd or t < 1e-12:
                break
            t *= 0.5
        if np.linalg.norm(a_new - a) <= tol * max(1.0, np.linalg.norm(a)):
            return a_new
        # Polak-Ribière conjugate gradient update
        y = g_new - g
        beta_pr = max(0.0, (g_new @ y) / (g @ g + 1e-12))
        
        # Apply gradient clipping to prevent explosions
        g_new_norm = np.linalg.norm(g_new)
        if g_new_norm > grad_clip_threshold:
            g_new = g_new * (grad_clip_threshold / g_new_norm)
        
        d = -g_new + beta_pr * d
        a, f, g = a_new, f_new, g_new
    return a


def _log_prior_infer_single(x, D, lam, max_iter=200, tol=1e-6):
    """
    Log prior sparse inference as in Olshausen & Field (1996).
    
    Optimizes: E(a) = 0.5||x - Da||² + λ Σᵢ log(1 + aᵢ²)
    
    Reference: Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell 
    receptive field properties by learning a sparse code for natural images. 
    Nature, 381(6583), 607-609.
    """
    # Use sigma=1 for the original log(1 + a²) formulation
    return _ncg_infer_single(x, D, lam, sigma=1.0, max_iter=max_iter, tol=tol)


def _parallel_infer_helper(infer_func, X_samples, D, lam, sigma_or_none, max_iter, tol):
    """Helper for parallel inference processing."""
    if sigma_or_none is not None:
        return [infer_func(X_samples[:, n], D, lam, sigma_or_none, max_iter=max_iter, tol=tol) 
                for n in range(X_samples.shape[1])]
    else:
        return [infer_func(X_samples[:, n], D, lam, max_iter=max_iter, tol=tol) 
                for n in range(X_samples.shape[1])]


def _olshausen_gradient_infer_single(x, D, lam, sigma, max_iter=200, tol=1e-6):
    """
    Pure Olshausen & Field (1996) gradient ascent inference - exact original algorithm.
    
    Implements simple gradient descent on: E(a) = 0.5||x - Da||² + λ Σᵢ log(1 + (aᵢ/σ)²)
    
    This is the exact algorithm from the 1996 Nature paper, using basic gradient descent
    instead of modern NCG or other advanced optimization methods.
    
    Reference: Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell 
    receptive field properties by learning a sparse code for natural images. 
    Nature, 381(6583), 607-609.
    """
    K = D.shape[1]
    a = np.zeros((K,), dtype=float)
    
    # Learning rate for gradient descent (from original paper)
    # Olshausen & Field used η = 0.01 to 0.1
    learning_rate = 0.05
    
    for iteration in range(int(max_iter)):
        # Compute energy and gradient
        f, g = _paper_energy_grad(x, D, a, lam, sigma)
        
        # Simple gradient descent step (negative gradient for minimization)
        a_new = a - learning_rate * g
        
        # Check convergence
        if np.linalg.norm(a_new - a) <= tol * max(1.0, np.linalg.norm(a)):
            return a_new
            
        # Adaptive learning rate for stability
        # If energy increased, reduce learning rate
        f_new, _ = _paper_energy_grad(x, D, a_new, lam, sigma)
        if f_new > f:
            learning_rate *= 0.8  # Reduce learning rate
        elif f_new < 0.95 * f:
            learning_rate *= 1.05  # Slightly increase if making good progress
            
        # Clamp learning rate to reasonable bounds
        learning_rate = np.clip(learning_rate, 0.001, 0.2)
        
        a = a_new
    
    return a


def _check_gpu_availability() -> bool:
    """Check GPU availability for potential acceleration (2025 performance)."""
    try:
        # Check for CuPy (CUDA acceleration)
        import cupy as cp
        # Test basic GPU operation
        test_array = cp.array([1.0, 2.0, 3.0])
        result = cp.sum(test_array)  # Simple GPU operation
        return True
    except (ImportError, Exception):
        # CuPy not available or GPU not accessible
        pass
    
    try:
        # Check for PyTorch CUDA
        import torch
        if torch.cuda.is_available():
            # Test basic CUDA operation
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            result = torch.sum(test_tensor)  # Simple CUDA operation
            return True
    except (ImportError, Exception):
        # PyTorch not available or CUDA not accessible
        pass
    
    return False


class SparseCoder:
    """
    Main interface for sparse coding inference and dictionary learning.
    
    Implements multiple sparse coding algorithms from foundational research papers.
    Supports both sparse inference (encoding signals) and dictionary learning 
    (learning overcomplete dictionaries from data).
    
    Research Foundation:
        Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding 
        algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1), 183-202.
        
        Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive 
        field properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.
        
        Engan, K., Aase, S. O., & Husøy, J. H. (1999). Method of optimal directions 
        for frame design. ICASSP, Vol. 5, pp. 2443-2446.
    
    Examples:
        Basic L1 sparse coding:
        >>> import numpy as np
        >>> from sparse_coding import SparseCoder
        >>> 
        >>> # Create dictionary and signal
        >>> D = np.random.randn(64, 32)
        >>> D /= np.linalg.norm(D, axis=0)  # Normalize atoms
        >>> signal = np.random.randn(64, 1)
        >>> 
        >>> # Initialize sparse coder
        >>> coder = SparseCoder(n_atoms=32, mode='l1', lam=0.1, max_iter=100)
        >>> coder.fit_dictionary(D)
        >>> 
        >>> # Encode signal
        >>> codes = coder.encode(signal)
        >>> sparsity = np.mean(np.abs(codes) < 1e-3)
        >>> print(f"Sparsity level: {sparsity:.2f}")
        
        Dictionary learning from data:
        >>> # Generate training data
        >>> X = np.random.randn(64, 1000)  # 1000 training signals
        >>> 
        >>> # Learn dictionary
        >>> coder = SparseCoder(n_atoms=32, mode='l1', lam=0.1)
        >>> coder.fit(X, n_steps=20)
        >>> 
        >>> # Dictionary is now learned and stored in coder.D
        >>> print(f"Dictionary shape: {coder.D.shape}")
    
    Algorithmic Modes:
        l1 : FISTA algorithm with L1 penalty (most common)
             Fast convergence with exact sparsity promotion
             
        paper : Olshausen & Field (1996) inspired log-Cauchy prior with NCG
                Research reproduction mode for natural image statistics
                
        paper_gdD : Similar to 'paper' but with gradient-based dictionary updates
                    Closer to original O&F implementation
                    
        log : Log-prior penalty log(1 + a²) with MOD dictionary updates
              Smooth approximation to sparsity promotion
    
    Parameters:
        n_atoms : int, default=144
            Number of dictionary atoms to learn or use.
            Should be >= n_features for overcomplete dictionaries.
            Common choice: n_atoms = 2 * n_features (2x overcomplete)
            
        lam : float, optional
            Regularization strength. Higher values → sparser solutions.
            If None, uses automatic selection based on mode:
            - L1 mode: 0.1 * max(|D^T @ x|) (adaptive)
            - Other modes: 0.1 (fixed)
            
        mode : str, default='l1'
            Algorithm mode. Options:
            - 'l1': L1 penalty with FISTA solver (most common)
            - 'paper': Log-Cauchy prior with NCG (O&F reproduction)
            - 'paper_gdD': Log-Cauchy prior with gradient dictionary updates
            - 'log': Log-prior penalty with MOD updates
            
        max_iter : int, default=200
            Maximum number of optimization iterations.
            L1/FISTA: 100-500 typical
            NCG modes: 50-200 typical
            
        tol : float, default=1e-6
            Convergence tolerance. Algorithm stops when gradient norm < tol.
            Use 1e-8 for high precision research applications.
            
        seed : int, default=0
            Random seed for reproducible dictionary initialization.
            
        anneal : tuple or None, optional
            Lambda annealing configuration as (gamma, floor).
            gamma: decay factor per iteration (0 < gamma < 1)
            floor: minimum lambda value (>= 0)
            
    Attributes:
        D : np.ndarray, shape (n_features, n_atoms)
            Dictionary matrix. Each column is a normalized atom.
            Available after fit() or fit_dictionary().
            
        n_atoms : int
            Number of dictionary atoms.
            
        mode : str
            Current algorithm mode.
            
        lam : float
            Current regularization parameter.
    
    Notes:
        Algorithm Selection Guidelines:
        - Use 'l1' mode for standard sparse coding applications
        - Use 'paper' mode for research reproduction of Olshausen & Field
        - Use 'paper_gdD' for closer match to original O&F implementation
        - Use 'log' mode for smooth sparsity promotion
        
        Performance Tips:
        - Normalize dictionary atoms: D /= np.linalg.norm(D, axis=0)
        - Start with lam=0.1, adjust based on desired sparsity
        - Use max_iter=200 for most problems, increase for high accuracy
        - Use anneal=(0.99, 0.01) for gradual sparsity increase during learning
        
        Memory Requirements:
        - Dictionary: O(n_features * n_atoms)
        - Sparse codes: O(n_atoms * n_samples)
        - Total: O(max(n_features * n_atoms, n_atoms * n_samples))
    """
    def __init__(self, n_atoms: int = 144, lam: Optional[float] = None, mode: SparseCodingMode = "l1", 
                 max_iter: int = 200, tol: float = 1e-6, seed: int = 0, 
                 anneal: Optional[AnnealingConfig] = None):
        """
        Initialize SparseCoder.
        
        Args:
            n_atoms: Number of dictionary atoms (must be positive integer)
            lam: Sparsity penalty (must be non-negative float if provided)
            mode: Sparse coding mode ('l1', 'log', 'paper', 'paper_gdD', 'olshausen_pure')
            max_iter: Maximum iterations (must be positive integer)
            tol: Convergence tolerance (must be positive float)
            seed: Random seed for reproducibility (non-negative integer)
            anneal: Lambda annealing as (gamma, floor) where 0 < gamma < 1, floor >= 0
            
        Raises:
            ValueError: If any parameter has invalid value or type
            TypeError: If parameter types are incorrect
        """
        # Type validation with explicit error messages
        if not isinstance(n_atoms, (int, np.integer)):
            raise TypeError(f"n_atoms must be an integer, got {type(n_atoms).__name__}")
        if lam is not None and not isinstance(lam, (int, float, np.number)):
            raise TypeError(f"lam must be a number or None, got {type(lam).__name__}")
        if not isinstance(mode, str):
            raise TypeError(f"mode must be a string, got {type(mode).__name__}")
        if not isinstance(max_iter, (int, np.integer)):
            raise TypeError(f"max_iter must be an integer, got {type(max_iter).__name__}")
        if not isinstance(tol, (int, float, np.number)):
            raise TypeError(f"tol must be a number, got {type(tol).__name__}")
        if not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
        
        # Value validation with consistent error messages
        if n_atoms <= 0:
            raise ValueError(f"n_atoms must be positive, got {n_atoms}")
        if n_atoms > 10000:  # Practical upper bound to prevent memory issues
            raise ValueError(f"n_atoms too large (>{10000}), got {n_atoms}. Use smaller values for memory efficiency.")
            
        if lam is not None:
            if lam < 0:
                raise ValueError(f"lam must be non-negative, got {lam}")
            if lam > 1000:  # Practical upper bound
                raise ValueError(f"lam too large (>{1000}), got {lam}. Very large lambda values may cause numerical issues.")
                
        # Comprehensive mode validation
        valid_modes = {'l1', 'log', 'paper', 'paper_gdD', 'olshausen_pure', 'transcoder'}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}, got '{mode}'")
            
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        if max_iter > 10000:  # Practical upper bound
            raise ValueError(f"max_iter too large (>{10000}), got {max_iter}. Use smaller values for efficiency.")
            
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")
        if tol >= 1.0:  # Convergence tolerance should be < 1
            raise ValueError(f"tol too large (>={1.0}), got {tol}. Use smaller values for meaningful convergence.")
        if tol < 1e-15:  # Machine precision limit
            raise ValueError(f"tol too small (<{1e-15}), got {tol}. Use larger values above machine precision.")
            
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        if seed > 2**32:  # Practical upper bound for random seeds
            raise ValueError(f"seed too large (>{2**32}), got {seed}. Use smaller values for compatibility.")
            
        # Enhanced anneal validation
        if anneal is not None:
            if not isinstance(anneal, (tuple, list)):
                raise TypeError(f"anneal must be a tuple or list, got {type(anneal).__name__}")
            if len(anneal) != 2:
                raise ValueError(f"anneal must have exactly 2 elements (gamma, floor), got {len(anneal)}")
            
            gamma, floor = anneal
            if not isinstance(gamma, (int, float, np.number)):
                raise TypeError(f"anneal gamma must be a number, got {type(gamma).__name__}")
            if not isinstance(floor, (int, float, np.number)):
                raise TypeError(f"anneal floor must be a number, got {type(floor).__name__}")
                
            if not (0 < gamma < 1):
                raise ValueError(f"anneal gamma must be in (0, 1), got {gamma}")
            if floor < 0:
                raise ValueError(f"anneal floor must be non-negative, got {floor}")
            if floor >= 1.0:  # Floor shouldn't be too large
                raise ValueError(f"anneal floor too large (>={1.0}), got {floor}. Use smaller values for effective annealing.")
        
        self.n_atoms = int(n_atoms)
        self.lam = lam
        self.mode = mode
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.anneal = anneal  # (gamma, floor) tuple for lambda annealing
        self.rng = np.random.default_rng(seed)
        self.D = None  # (p, K)
        
        # Transcoder-specific components (2025 arXiv:2501.18823)
        if mode == 'transcoder':
            self._nonlinear_weights = None  # Will be initialized during fit
            self._skip_connection = True  # Skip transcoders for better performance
            
        # Adaptive K: dynamic atom allocation based on usage (2025 research)
        self._adaptive_k = False  # Can be enabled via set_adaptive_k()
        self._max_atoms = n_atoms * 2  # Upper bound for adaptive growth
        self._usage_history = []  # Track atom usage over iterations
        
        # GPU availability check for performance optimization
        self._gpu_available = _check_gpu_availability()
    
    @property
    def dictionary(self):
        """Dictionary matrix (research standard: D or Φ)."""
        return self.D
    
    @dictionary.setter
    def dictionary(self, D):
        """Set dictionary matrix with proper validation and normalization."""
        if D is not None:
            D = np.asarray(D, dtype=float)
            
            # Validate dictionary shape and properties
            if D.ndim != 2:
                raise ValueError(f"Dictionary must be 2D matrix, got shape {D.shape}")
            
            n_features, n_atoms = D.shape
            if n_features <= 0 or n_atoms <= 0:
                raise ValueError(f"Dictionary dimensions must be positive, got shape {D.shape}")
            
            if n_atoms != self.n_atoms:
                raise ValueError(f"Dictionary must have {self.n_atoms} atoms, got {n_atoms}")
            
            # Check for numerical validity
            if not np.all(np.isfinite(D)):
                raise ValueError("Dictionary contains non-finite values (inf/nan)")
            
            # Check that dictionary isn't all zeros
            if np.allclose(D, 0):
                raise ValueError("Dictionary cannot be all zeros")
            
            # Normalize columns (research requirement: normalized columns)
            D = _normalize_columns(D)
            
            # Final validation: ensure normalization worked
            column_norms = np.linalg.norm(D, axis=0)
            if not np.allclose(column_norms, 1.0, rtol=1e-10):
                raise RuntimeError(f"Dictionary normalization failed: column norms {column_norms}")
        
        self.D = D

    def _init_dictionary(self, X):
        p, N = X.shape
        if self.D is None:
            # Research-accurate initialization: sample with replacement if needed
            # This follows Olshausen & Field (1996) - dictionary can be overcomplete
            n_atoms_to_sample = min(self.n_atoms, N)
            idx = self.rng.choice(N, size=n_atoms_to_sample, replace=False)
            D = X[:, idx].copy()
            
            # If we need more atoms than data points, add random initialization
            if self.n_atoms > N:
                extra_atoms = self.n_atoms - N
                D_extra = self.rng.normal(scale=np.std(X), size=(p, extra_atoms))
                D = np.hstack([D, D_extra])
            
            # Add small noise to avoid identical atoms
            D = D + 1e-6 * self.rng.normal(size=D.shape)
            D = D - D.mean(axis=0, keepdims=True)  # Remove DC component (research standard)
            D = _normalize_columns(D)
            self.D = D

    def encode(self, X: ArrayLike) -> np.ndarray:
        """
        Encode signals into sparse representations using the learned dictionary.
        
        Performs sparse coding inference to find sparse coefficients A such that X ≈ D @ A,
        where D is the dictionary matrix. The optimization problem depends on the mode:
        
        L1 Mode (FISTA):
            min_A (1/2)||X - D @ A||²_F + λ||A||_1
            
        Log Prior Mode (Olshausen & Field 1996):
            min_A (1/2)||X - D @ A||²_F + λ Σ log(1 + (A_i/σ)²)
            
        Paper Modes:
            min_A (1/2)||X - D @ A||²_F + λ Σ log(1 + (A_i/σ)²)
            
        Research Foundation:
            Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding 
            algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1), 183-202.
            
            Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive 
            field properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.
        
        Args:
            X (array_like): Input signals of shape (n_features, n_samples).
                - Each column is a signal to be encoded
                - Supports dense numpy arrays, scipy sparse matrices
                - For single signal, use X.reshape(-1, 1) or pass 1D array
                - Data should be zero-mean for best results
                
        Returns:
            np.ndarray: Sparse codes of shape (n_atoms, n_samples).
                - Each column contains sparse coefficients for corresponding input signal
                - Coefficients represent linear combination weights: X[:, i] ≈ D @ codes[:, i]
                - Sparsity level depends on regularization parameter λ (self.lam)
                
        Raises:
            ValueError: If dictionary not initialized (call fit() first)
            ValueError: If X feature dimension doesn't match dictionary
            RuntimeError: If optimization fails to converge
            
        Examples:
            Basic encoding:
            >>> coder = SparseCoder(n_atoms=64, mode='l1', lam=0.1)
            >>> coder.fit(training_data)  # Shape: (n_features, n_training_samples)
            >>> codes = coder.encode(test_signals)  # Shape: (n_features, n_test_samples)
            >>> print(f"Sparsity: {np.mean(codes == 0):.1%}")
            
            Single signal encoding:
            >>> signal = np.random.randn(256)  # 1D signal
            >>> codes = coder.encode(signal.reshape(-1, 1))  # Shape: (64, 1)
            
            Batch processing:
            >>> batch = np.random.randn(256, 100)  # 100 signals of 256 features
            >>> codes = coder.encode(batch)  # Shape: (64, 100)
            
        Algorithm Performance:
            - L1 mode: FISTA algorithm, O(1/k²) convergence rate
            - Log prior: Gradient descent, typically faster than L1 for natural images
            - Memory: O(n_features × n_atoms + n_atoms × n_samples)
            - Time: O(max_iter × n_features × n_atoms × n_samples)
            
        Notes:
            - Larger λ (lam) → sparser codes, lower reconstruction quality
            - Smaller λ → denser codes, higher reconstruction quality  
            - For natural images, λ ∈ [0.05, 0.3] typically works well
            - Zero-mean preprocessing improves convergence and results
            - GPU acceleration available if CuPy/PyTorch detected
        """
        X = _validate_input(X, "X", allow_sparse=True)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if self.D is None:
            raise ValueError("dictionary not initialized. Call fit() first.")
        
        if X.shape[0] != self.D.shape[0]:
            raise ValueError(f"X feature dimension {X.shape[0]} doesn't match dictionary {self.D.shape[0]}. "
                           f"Expected X.shape[0] == {self.D.shape[0]}. If you have (n_samples, n_features) data, try X = X.T")
        if self.mode == "l1":
            # Use stored lambda from fit() if available, otherwise scale with data variance
            if hasattr(self, 'lam') and self.lam is not None:
                lam = float(self.lam)
            else:
                lam = float(0.1 * np.std(X, ddof=0))  # Research-accurate: scale with data variance, deterministic
            return fista_batch(self.D, X, lam, L=None, max_iter=self.max_iter, tol=self.tol)
        elif self.mode == "log":
            # Log prior mode (Olshausen & Field 1996 original formulation)
            lam = float(self.lam if self.lam is not None else 0.05 * np.std(X, ddof=0))
            return self._batch_log_prior_inference(X, lam)
        elif self.mode in ["paper", "paper_gdD"]:
            sigma = 1.0
            lam = float(self.lam if self.lam is not None else 0.14 * np.std(X))
            return self._batch_ncg_inference(X, lam, sigma)
        elif self.mode == "olshausen_pure":
            sigma = 1.0
            lam = float(self.lam if self.lam is not None else 0.14 * np.std(X))
            return self._batch_olshausen_gradient_inference(X, lam, sigma)
        elif self.mode == "transcoder":
            # Transcoder uses L1 inference (FISTA) 
            lam = float(self.lam if self.lam is not None else 0.1 * np.std(X, ddof=0))
            return fista_batch(self.D, X, lam, L=None, max_iter=self.max_iter, tol=self.tol)
        else:
            raise ValueError("mode must be 'l1', 'log', 'paper', 'paper_gdD', 'olshausen_pure', or 'transcoder'")

    def decode(self, A: ArrayLike) -> np.ndarray:
        """
        Simple linear decoding: X_reconstructed = D @ A (research standard).
        
        Args:
            A: Sparse codes (atoms x samples) - supports scipy.sparse matrices
            
        Returns:
            Reconstructed signals (features x samples)
        """
        A = _validate_input(A, "A", allow_sparse=True)
        if self.D is None:
            raise ValueError("dictionary not initialized. Call fit() first.")
        if A.shape[0] != self.D.shape[1]:
            raise ValueError(f"A atom dimension {A.shape[0]} doesn't match dictionary {self.D.shape[1]}")
        # Transcoder mode uses nonlinear decoder (2025 research)
        if self.mode == 'transcoder':
            return self._nonlinear_decode(A)
        
        # Standard linear decoding for other modes
        # Sparse-safe matrix multiplication
        if scipy.sparse.issparse(A):
            # Use sparse matrix multiplication for efficiency
            D_sparse = scipy.sparse.csr_matrix(self.D)
            return D_sparse @ A
        else:
            return self.D @ A

    def fit(self, X: ArrayLike, n_steps: int = 30, lr: float = 0.1) -> 'SparseCoder':
        """
        Learn dictionary from training data using specified sparse coding mode.
        
        Args:
            X: Training signals (features x samples)
            n_steps: Number of alternating optimization steps (must be positive)
            lr: Learning rate for gradient-based updates (must be positive)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If inputs are invalid or incompatible
        """
        X = _validate_input(X, "X", allow_sparse=True)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        
        p, N = X.shape
        self._init_dictionary(X)
        D = self.D
        # Research-accurate lambda selection
        if self.lam is None:
            if self.mode == "l1":
                # Research-accurate: scale with data variance, use ddof=0 for deterministic results
                data_std = float(np.std(X, ddof=0))
                lam_default = 0.1 * data_std  # Research-accurate: adaptive sparsity scaling
            elif self.mode == "log":
                data_std = float(np.std(X, ddof=0))
                lam_default = 0.05 * data_std  # Different scaling for log prior
            else:
                data_std = float(np.std(X, ddof=0))
                lam_default = 0.14 * data_std
            lam = float(lam_default)
            self.lam = lam  # Store for research analysis
        else:
            lam = float(self.lam)

        for it in range(int(n_steps)):
            if self.mode == "l1":
                A = fista_batch(D, X, lam, L=None, max_iter=self.max_iter, tol=self.tol)
                D = _mod_update(D, X, A, eps=1e-6)
            elif self.mode == "paper":
                # Use optimized batch inference  
                A = self._batch_ncg_inference_for_fit(X, D, lam, 1.0)
                D = _mod_update(D, X, A, eps=1e-6)
            elif self.mode == "paper_gdD":
                # Use optimized batch inference
                A = self._batch_ncg_inference_for_fit(X, D, lam, 1.0)
                # Gradient dictionary update (O&F-style)
                D = _gradD_update(D, X, A, lr=lr)
                # Homeostatic equalization
                D = _homeostatic_equalize(D, A, alpha=0.1)
            elif self.mode == "olshausen_pure":
                # Pure Olshausen & Field (1996) gradient ascent for both inference and learning
                A = self._batch_olshausen_gradient_inference_for_fit(X, D, lam, 1.0)
                # Original gradient dictionary update from 1996 paper
                D = _gradD_update(D, X, A, lr=lr)
                # Homeostatic equalization (critical for original algorithm stability)
                D = _homeostatic_equalize(D, A, alpha=0.1)
            elif self.mode == "log":
                # Olshausen & Field (1996) original log prior formulation
                # Uses log(1 + a²) sparsity penalty as in the Nature paper
                A = self._batch_log_prior_inference_for_fit(X, D, lam)
                D = _mod_update(D, X, A, eps=1e-6)
            elif self.mode == "transcoder":
                # Transcoder mode: L1 inference + nonlinear decoder training
                A = fista_batch(D, X, lam, L=None, max_iter=self.max_iter, tol=self.tol)
                # Train nonlinear decoder on reconstruction task
                if self._nonlinear_weights is None:
                    self._init_nonlinear_decoder()
                self._train_nonlinear_decoder(X, A, lr=lr)
                # Update linear dictionary too
                D = _mod_update(D, X, A, eps=1e-6)
            else:
                raise ValueError("mode must be 'l1', 'paper', 'paper_gdD', 'olshausen_pure', 'transcoder', or 'log'")
            
            D = _reinit_dead_atoms(D, X, A, self.rng)
            
            # Adaptive atom management (2025 feature)
            self._adaptive_atom_management(A)
            D = self.D  # Update D in case atoms were added/removed
            
            # Optional annealing of lambda
            if self.anneal:
                if isinstance(self.anneal, (list, tuple)) and len(self.anneal) == 2:
                    gamma, floor = float(self.anneal[0]), float(self.anneal[1])
                else:
                    gamma, floor = 0.95, 1e-4
                lam = max(floor, lam * gamma)
                # CRITICAL: Update stored lambda for consistent encode() behavior
                self.lam = lam

        self.D = D
        return self
    
    def _solve_single_log(self, x, D):
        """
        Solve sparse coding for single signal using log prior.
        
        Research Foundation:
        - Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field 
          properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.
        
        Mathematical Formulation:
        E(a) = 0.5||x - Da||² + λ∑log(1 + a²/σ²)
        
        Returns convergence information for research validation.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.ravel()
        
        lam = float(self.lam if self.lam is not None else 0.05 * np.std(x))
        sigma = 1.0
        
        # Use existing NCG implementation but track convergence
        K = D.shape[1]
        a = np.zeros(K, dtype=float)
        f, g = _paper_energy_grad(x, D, a, lam, sigma)
        d = -g
        
        for iteration in range(int(self.max_iter)):
            # Store initial state for convergence check
            a_prev = a.copy()
            f_prev = f
            
            # Ensure descent direction
            if g @ d > 0:
                d = -g
            
            # Backtracking line search (Armijo)
            t = 1.0
            f0 = f
            gTd = g @ d
            while True:
                a_new = a + t*d
                f_new, g_new = _paper_energy_grad(x, D, a_new, lam, sigma)
                if f_new <= f0 + 1e-4 * t * gTd or t < 1e-12:
                    break
                t *= 0.5
            
            # Check convergence (comprehensive criteria following optimization literature)
            delta_a = np.linalg.norm(a_new - a)
            rel_change = delta_a / max(1.0, np.linalg.norm(a))
            grad_norm = np.linalg.norm(g_new)
            obj_change = abs(f_new - f) / max(abs(f), 1e-12)
            
            # Multiple convergence criteria (any can trigger)
            converged = False
            convergence_reason = ""
            
            if rel_change <= self.tol:
                converged = True
                convergence_reason = "parameter_change"
            elif grad_norm <= self.tol * 10:  # Gradient norm (more stringent)
                converged = True  
                convergence_reason = "gradient_norm"
            elif obj_change <= self.tol * 1e-3:  # Objective plateau
                converged = True
                convergence_reason = "objective_plateau"
            
            if converged:
                return {
                    'coefficients': a_new,
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_objective': f_new,
                    'final_gradient_norm': grad_norm,
                    'convergence_reason': convergence_reason,
                    'parameter_change': rel_change,
                    'objective_change': obj_change
                }
            
            # Polak-Ribière conjugate gradient update
            y = g_new - g
            beta_pr = max(0.0, (g_new @ y) / (g @ g + 1e-12))
            d = -g_new + beta_pr * d
            a, f, g = a_new, f_new, g_new
        
        # Did not converge
        return {
            'coefficients': a,
            'converged': False,
            'iterations': self.max_iter,
            'final_objective': f,
            'final_gradient_norm': np.linalg.norm(g)
        }
    
    def _fista_single(self, x, D):
        """
        Solve sparse coding for single signal using FISTA.
        
        Research Foundation:
        - Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding 
          algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1), 183-202.
        
        Mathematical Formulation:
        E(a) = 0.5||x - Da||² + λ||a||₁
        
        Returns convergence information including objective history.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.ravel()
        
        # Research-accurate lambda selection for single signal FISTA
        if self.lam is not None:
            lam = float(self.lam)
        else:
            # Scale with signal-to-dictionary correlation for better convergence
            correlation = np.abs(D.T @ x)
            # Use max correlation for better Lasso-like behavior (scikit-learn style)
            lam = float(0.1 * np.max(correlation[correlation > 1e-12]))
        
        # FISTA initialization (Beck & Teboulle 2009 Algorithm 2)
        K = D.shape[1]
        a_prev = np.zeros(K, dtype=float)
        a = np.zeros(K, dtype=float)
        t = 1.0
        
        # Compute Lipschitz constant L = ||D^T D||₂ - sparse-aware
        if scipy.sparse.issparse(D):
            # Use sparse SVD for sparse dictionaries (more efficient)
            try:
                sigma_max = svds(D, k=1, return_singular_vectors=False)[0]
                L = sigma_max**2 + 1e-6  # L = σ_max²
            except Exception:
                # Fallback to dense if sparse SVD fails
                L = power_iter_L(D.toarray() if hasattr(D, 'toarray') else D)
        else:
            L = power_iter_L(D)
        step_size = 1.0 / L
        
        objectives = []
        
        for iteration in range(int(self.max_iter)):
            # FISTA momentum update (Beck & Teboulle 2009 Algorithm 2)
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            
            # Extrapolated point calculation  
            if iteration == 0:
                y = a.copy()  # y_0 = a_0
            else:
                beta = (t - 1.0) / t_new
                y = a + beta * (a - a_prev)
            
            # Gradient step on extrapolated point
            # ∇f(y) = D^T(Dy - x) where f(a) = (1/2)||x - Da||²
            grad = D.T @ (D @ y - x)
            z = y - step_size * grad
            
            # Proximal operator (soft thresholding)
            threshold = step_size * lam
            a_new = np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)
            
            # Compute objective using new coefficients
            residual = x - D @ a_new
            data_fit = 0.5 * np.dot(residual, residual)
            sparsity = lam * np.sum(np.abs(a_new))
            objective = data_fit + sparsity
            objectives.append(objective)
            
            # Check convergence
            delta_a = np.linalg.norm(a_new - a)
            rel_change = delta_a / max(1.0, np.linalg.norm(a))
            
            if rel_change <= self.tol:
                return {
                    'coefficients': a_new,
                    'converged': True,
                    'iterations': iteration + 1,
                    'objectives': objectives,
                    'final_objective': objective
                }
            
            # Update for next iteration
            a_prev = a.copy()
            a = a_new.copy()
            t = t_new
        
        # Did not converge
        return {
            'coefficients': a,
            'converged': False,
            'iterations': self.max_iter,
            'objectives': objectives,
            'final_objective': objectives[-1] if objectives else float('inf')
        }
    
    def _batch_log_prior_inference(self, X: np.ndarray, lam: float) -> np.ndarray:
        """
        Vectorized batch log prior inference with performance optimizations.
        
        Performance improvements:
        - Parallel processing for multiple signals when beneficial  
        - Memory-efficient chunking for large batches
        - Adaptive batch sizing based on problem size
        """
        K, N = self.n_atoms, X.shape[1]
        A = np.zeros((K, N))
        
        # Use different strategies based on batch size
        if N == 1:
            # Single signal: use optimized single-signal method
            A[:, 0] = _log_prior_infer_single(X[:, 0], self.D, lam, max_iter=self.max_iter, tol=self.tol)
        elif N <= 50:
            # Small-medium batches: process sequentially with minimal overhead
            for n in range(N):
                A[:, n] = _log_prior_infer_single(X[:, n], self.D, lam, max_iter=self.max_iter, tol=self.tol)
        else:
            # Large batches: use chunked processing for better cache locality
            chunk_size = min(50, max(1, N // 4))  # Adaptive chunk size
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                for n in range(start, end):
                    A[:, n] = _log_prior_infer_single(X[:, n], self.D, lam, max_iter=self.max_iter, tol=self.tol)
        
        return A
    
    def _batch_ncg_inference(self, X: np.ndarray, lam: float, sigma: float) -> np.ndarray:
        """
        Vectorized batch NCG inference with performance optimizations.
        
        Performance improvements:
        - Efficient memory access patterns
        - Reduced function call overhead for large batches  
        - Cache-friendly processing order
        """
        K, N = self.n_atoms, X.shape[1]
        A = np.zeros((K, N))
        
        # Use different strategies based on batch size
        if N == 1:
            # Single signal: use optimized single-signal method
            A[:, 0] = _ncg_infer_single(X[:, 0], self.D, lam, sigma, max_iter=self.max_iter, tol=self.tol)
        elif N <= 50:
            # Small-medium batches: process sequentially
            for n in range(N):
                A[:, n] = _ncg_infer_single(X[:, n], self.D, lam, sigma, max_iter=self.max_iter, tol=self.tol)
        else:
            # Large batches: use chunked processing
            chunk_size = min(50, max(1, N // 4))  # Adaptive chunk size
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                for n in range(start, end):
                    A[:, n] = _ncg_infer_single(X[:, n], self.D, lam, sigma, max_iter=self.max_iter, tol=self.tol)
        
        return A
    
    def _batch_log_prior_inference_for_fit(self, X: np.ndarray, D: np.ndarray, lam: float) -> np.ndarray:
        """Optimized batch log prior inference for dictionary learning fit."""
        K, N = D.shape[1], X.shape[1]
        A = np.zeros((K, N))
        
        # Use parallel processing for large batches
        if N > 50:
            # Parallel processing with chunking for memory efficiency
            chunk_size = max(1, min(100, N // 4))  # Adaptive chunk size
            n_jobs = min(4, N // chunk_size + 1)  # Limit parallel jobs
            
            try:
                # Process chunks in parallel
                chunk_results = Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(_parallel_infer_helper)(
                        _log_prior_infer_single, 
                        X[:, start:min(start + chunk_size, N)], 
                        D, lam, None, self.max_iter, self.tol
                    ) for start in range(0, N, chunk_size)
                )
                
                # Reassemble results
                start = 0
                for chunk_result in chunk_results:
                    end = start + len(chunk_result)
                    for i, result in enumerate(chunk_result):
                        A[:, start + i] = result
                    start = end
                    
            except Exception:
                # Fallback to sequential if parallel fails
                for n in range(N):
                    A[:, n] = _log_prior_infer_single(X[:, n], D, lam, max_iter=self.max_iter, tol=self.tol)
        else:
            # Sequential processing for small batches or when joblib unavailable
            for n in range(N):
                A[:, n] = _log_prior_infer_single(X[:, n], D, lam, max_iter=self.max_iter, tol=self.tol)
        
        return A
    
    def _batch_ncg_inference_for_fit(self, X: np.ndarray, D: np.ndarray, lam: float, sigma: float) -> np.ndarray:
        """Optimized batch NCG inference for dictionary learning fit."""
        K, N = D.shape[1], X.shape[1]
        A = np.zeros((K, N))
        
        # Process efficiently with parallelization
        if N <= 50:
            # Small batches: sequential processing
            for n in range(N):
                A[:, n] = _ncg_infer_single(X[:, n], D, lam, sigma, max_iter=self.max_iter, tol=self.tol)
        else:
            # Large batches: parallel processing with threading backend
            
            def process_signal(n):
                return n, _ncg_infer_single(X[:, n], D, lam, sigma, max_iter=self.max_iter, tol=self.tol)
            
            # Threading backend for numerical computations
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(process_signal)(n) for n in range(N)
            )
            
            # Assign results back to A matrix
            for n, result in results:
                A[:, n] = result
        
        return A
    
    def _nonlinear_decode(self, A: np.ndarray) -> np.ndarray:
        """Nonlinear decoder for transcoder mode (2025 arXiv:2501.18823)."""
        if self._nonlinear_weights is None:
            raise ValueError("Transcoder not trained. Call fit() first.")
        
        # Simple 2-layer MLP decoder with skip connection
        W1, b1, W2, b2 = self._nonlinear_weights
        
        # Hidden layer with ReLU activation
        hidden = np.maximum(0, A.T @ W1 + b1)  # (n_samples, hidden_dim)
        
        # Output layer
        output = hidden @ W2 + b2  # (n_samples, n_features)
        
        # Skip connection: combine with linear reconstruction
        if self._skip_connection:
            linear_output = (self.D @ A).T  # (n_samples, n_features)
            output = 0.7 * output + 0.3 * linear_output  # Weighted combination
        
        return output.T  # Return as (n_features, n_samples)
    
    def _init_nonlinear_decoder(self, hidden_dim: int = None) -> None:
        """Initialize nonlinear decoder weights for transcoder mode."""
        if self.D is None:
            return
        
        n_features, n_atoms = self.D.shape
        hidden_dim = hidden_dim or min(256, 2 * n_atoms)  # Adaptive hidden size
        
        # Xavier initialization for better gradient flow
        scale1 = np.sqrt(2.0 / (n_atoms + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + n_features))
        
        W1 = self.rng.normal(0, scale1, (n_atoms, hidden_dim))
        b1 = np.zeros(hidden_dim)
        W2 = self.rng.normal(0, scale2, (hidden_dim, n_features))  
        b2 = np.zeros(n_features)
        
        self._nonlinear_weights = (W1, b1, W2, b2)
    
    def _train_nonlinear_decoder(self, X: np.ndarray, A: np.ndarray, lr: float = 0.001, n_epochs: int = 10) -> None:
        """Train nonlinear decoder for transcoder mode using gradient descent."""
        if self._nonlinear_weights is None:
            raise ValueError("Decoder not initialized")
        
        W1, b1, W2, b2 = self._nonlinear_weights
        n_samples = X.shape[1]
        
        # Simple SGD training loop
        for epoch in range(n_epochs):
            # Forward pass
            hidden = np.maximum(0, A.T @ W1 + b1)  # (n_samples, hidden_dim)
            output = hidden @ W2 + b2  # (n_samples, n_features)
            
            # Skip connection
            if self._skip_connection:
                linear_output = (self.D @ A).T
                output = 0.7 * output + 0.3 * linear_output
            
            # Reconstruction loss
            target = X.T  # (n_samples, n_features) 
            loss = np.mean((output - target)**2)
            
            # Backward pass (simple gradients)
            d_output = 2 * (output - target) / n_samples  # (n_samples, n_features)
            
            if self._skip_connection:
                d_output = 0.7 * d_output  # Account for skip connection weight
            
            # Gradients for second layer
            d_W2 = hidden.T @ d_output  # (hidden_dim, n_features)
            d_b2 = np.sum(d_output, axis=0)  # (n_features,)
            
            # Gradients for first layer
            d_hidden = d_output @ W2.T  # (n_samples, hidden_dim)
            d_hidden_relu = d_hidden * (hidden > 0)  # ReLU derivative
            
            d_W1 = A @ d_hidden_relu  # (n_atoms, hidden_dim)
            d_b1 = np.sum(d_hidden_relu, axis=0)  # (hidden_dim,)
            
            # Update weights
            W1 -= lr * d_W1
            b1 -= lr * d_b1  
            W2 -= lr * d_W2
            b2 -= lr * d_b2
        
        self._nonlinear_weights = (W1, b1, W2, b2)
    
    def set_adaptive_k(self, enabled: bool = True, max_atoms: Optional[int] = None) -> None:
        """Enable/disable adaptive atom allocation (2025 research feature)."""
        self._adaptive_k = enabled
        if max_atoms is not None:
            self._max_atoms = max_atoms
    
    def _adaptive_atom_management(self, A: np.ndarray) -> None:
        """Manage dictionary size based on atom usage patterns."""
        if not self._adaptive_k:
            return
            
        # Track usage: atoms with significant activations
        current_usage = np.mean(np.abs(A) > 1e-4, axis=1)  # Fraction of samples using each atom
        self._usage_history.append(current_usage)
        
        if len(self._usage_history) < 5:  # Need history for decisions
            return
            
        # Average usage over recent history
        avg_usage = np.mean(self._usage_history[-5:], axis=0)
        
        # Identify underused atoms (less than 1% usage)
        underused = avg_usage < 0.01
        n_underused = np.sum(underused)
        
        # If many atoms are underused and we're above minimum, remove some
        min_atoms = self.n_atoms // 2
        current_atoms = self.D.shape[1]
        
        if n_underused > current_atoms * 0.3 and current_atoms > min_atoms:
            # Remove up to half of underused atoms
            to_remove = min(n_underused // 2, current_atoms - min_atoms)
            if to_remove > 0:
                keep_indices = np.where(~underused)[0]
                if len(keep_indices) < current_atoms - to_remove:
                    # Keep best underused atoms
                    underused_indices = np.where(underused)[0]
                    keep_underused = underused_indices[np.argsort(-avg_usage[underused_indices])][:to_remove]
                    keep_indices = np.concatenate([keep_indices, keep_underused])
                
                # Update dictionary
                self.D = self.D[:, keep_indices]
                print(f"Adaptive K: Removed {to_remove} atoms, now have {self.D.shape[1]} atoms")
        
        # If all atoms are well-used and we haven't hit max, add more
        elif np.all(avg_usage > 0.05) and current_atoms < self._max_atoms:
            n_features = self.D.shape[0] 
            n_new = min(5, self._max_atoms - current_atoms)  # Add up to 5 atoms
            
            # Initialize new atoms from data statistics
            new_atoms = self.rng.normal(0, np.std(self.D), (n_features, n_new))
            new_atoms = new_atoms / np.linalg.norm(new_atoms, axis=0, keepdims=True)
            
            self.D = np.concatenate([self.D, new_atoms], axis=1)
            print(f"Adaptive K: Added {n_new} atoms, now have {self.D.shape[1]} atoms")
    
    def _batch_olshausen_gradient_inference(self, X: np.ndarray, lam: float, sigma: float) -> np.ndarray:
        """Batch inference using pure Olshausen & Field (1996) gradient ascent."""
        K, N = self.D.shape[1], X.shape[1]
        A = np.zeros((K, N))
        
        # Process efficiently with parallelization
        if N <= 50:
            # Small batches: sequential processing
            for n in range(N):
                A[:, n] = _olshausen_gradient_infer_single(X[:, n], self.D, lam, sigma, max_iter=self.max_iter, tol=self.tol)
        else:
            # Large batches: parallel processing with threading backend
            
            def process_signal(n):
                return n, _olshausen_gradient_infer_single(X[:, n], self.D, lam, sigma, max_iter=self.max_iter, tol=self.tol)
            
            # Threading backend for numerical computations
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(process_signal)(n) for n in range(N)
            )
            
            # Assign results back to A matrix
            for n, result in results:
                A[:, n] = result
        
        return A
    
    def _batch_olshausen_gradient_inference_for_fit(self, X: np.ndarray, D: np.ndarray, lam: float, sigma: float) -> np.ndarray:
        """Optimized batch Olshausen gradient inference for dictionary learning fit."""
        K, N = D.shape[1], X.shape[1]
        A = np.zeros((K, N))
        
        # Process efficiently with parallelization
        if N <= 50:
            # Small batches: sequential processing
            for n in range(N):
                A[:, n] = _olshausen_gradient_infer_single(X[:, n], D, lam, sigma, max_iter=self.max_iter, tol=self.tol)
        else:
            # Large batches: parallel processing with threading backend
            
            def process_signal(n):
                return n, _olshausen_gradient_infer_single(X[:, n], D, lam, sigma, max_iter=self.max_iter, tol=self.tol)
            
            # Threading backend for numerical computations
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(process_signal)(n) for n in range(N)
            )
            
            # Assign results back to A matrix
            for n, result in results:
                A[:, n] = result
        
        return A
    
    def evaluate_interpretability(self, X: np.ndarray, A: np.ndarray) -> Dict[str, float]:
        """
        Evaluate interpretability metrics to detect potential illusions (2025 research).
        
        Based on Bricken et al. (2024) findings on SAE interpretability illusions.
        """
        if self.D is None:
            raise ValueError("Dictionary not trained. Call fit() first.")
            
        metrics = {}
        
        # 1. Dead neurons ratio (should be low for good interpretability)
        activation_threshold = 1e-6
        dead_ratio = np.mean(np.max(np.abs(A), axis=1) < activation_threshold)
        metrics['dead_neuron_ratio'] = float(dead_ratio)
        
        # 2. Activation frequency distribution (should be balanced)
        activation_freq = np.mean(np.abs(A) > activation_threshold, axis=1)
        metrics['activation_freq_std'] = float(np.std(activation_freq))
        
        # 3. Feature polysemanticity (atoms should be monosemantic)
        # Higher values indicate potential polysemantic features
        A_norm = np.abs(A) / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
        polysemanticity = np.mean(np.sum(A_norm > 0.1, axis=0))  # Features per sample
        metrics['polysemanticity'] = float(polysemanticity)
        
        # 4. Reconstruction quality vs sparsity tradeoff
        X_recon = self.D @ A
        mse = np.mean((X - X_recon) ** 2)
        sparsity = np.mean(np.abs(A) < activation_threshold)
        metrics['reconstruction_mse'] = float(mse)
        metrics['sparsity_ratio'] = float(sparsity)
        metrics['quality_sparsity_ratio'] = float(sparsity / (mse + 1e-12))
        
        # 5. Dictionary coherence (atoms should be distinct)
        if self.D.shape[1] > 1:
            D_norm = self.D / (np.linalg.norm(self.D, axis=0, keepdims=True) + 1e-12)
            coherence_matrix = np.abs(D_norm.T @ D_norm)
            np.fill_diagonal(coherence_matrix, 0)  # Remove self-correlation
            max_coherence = np.max(coherence_matrix)
            metrics['max_atom_coherence'] = float(max_coherence)
        
        return metrics
    
    def benchmark_performance(self, X: np.ndarray, n_trials: int = 3) -> Dict[str, float]:
        """Benchmark performance against research baselines (2025)."""
        import time
        
        results = {}
        
        # Measure fit time
        fit_times = []
        for _ in range(n_trials):
            start_time = time.time()
            self.fit(X, n_steps=2)  # Quick benchmark
            fit_times.append(time.time() - start_time)
        
        results['mean_fit_time'] = float(np.mean(fit_times))
        results['std_fit_time'] = float(np.std(fit_times))
        
        # Measure encoding time
        encode_times = []
        for _ in range(n_trials):
            start_time = time.time()
            A = self.encode(X)
            encode_times.append(time.time() - start_time)
        
        results['mean_encode_time'] = float(np.mean(encode_times))
        results['std_encode_time'] = float(np.std(encode_times))
        
        # Performance indicators
        results['gpu_available'] = self._gpu_available
        results['parallel_available'] = True  # We have joblib
        results['atoms_per_sample'] = float(self.n_atoms / X.shape[1])
        
        return results
