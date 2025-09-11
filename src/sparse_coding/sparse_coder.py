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
from .fista_batch import fista_batch, power_iter_L

def _normalize_columns(D):
    n = np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    return D / n

def _mod_update(D, X, A, eps=1e-6):
    # MOD: D = X A^T (A A^T + eps I)^-1, then renormalize columns
    At = A.T
    G = A @ At
    G.flat[::G.shape[0]+1] += eps
    # Use solve instead of inv for numerical stability: D = X A^T G^-1 = (G^-1 (X A^T)^T)^T
    D_new = np.linalg.solve(G, (X @ At).T).T
    return _normalize_columns(D_new)

def _homeostatic_equalize(D, A, alpha=0.1):
    """Equalize atom norms of codes to balance usage (simple homeostasis).
    Scale dictionary columns inversely to recent code energy.
    """
    usage = np.sqrt(np.mean(A*A, axis=1) + 1e-12)  # (K,)
    target = np.mean(usage)
    scale = (target / (usage + 1e-12))**alpha
    D = D * scale[np.newaxis, :]
    return _normalize_columns(D)

def _gradD_update(D, X, A, lr=0.1):
    """Gradient ascent step on D for 0.5||X - D A||^2 (note: ascent on negative loss -> descent on loss)
    Here we *minimize* loss: D <- D - lr * d/dD (0.5||X-DA||^2) = D + lr * (X - D A) A^T
    """
    R = X - D @ A
    D = D + lr * (R @ A.T)
    return _normalize_columns(D)

def _reinit_dead_atoms(D, X, A, rng):
    """Re-initialize atoms with near-zero norm usage"""
    usage = np.linalg.norm(A, axis=1)  # Use L2 norm like alternate implementation
    dead = usage < 1e-8  # More sensitive threshold
    if np.any(dead):
        m = D.shape[0]
        # sample random patches from X
        idx = rng.integers(0, X.shape[1], size=int(np.sum(dead)))
        D[:, dead] = X[:, idx]
        D = _normalize_columns(D)
    return D

def _paper_energy_grad(x, D, a, lam, sigma):
    # E(a) = 0.5||x - D a||^2 + lam * sum log(1 + (a/sigma)^2)
    r = x - D @ a
    energy = 0.5 * float(r @ r) + lam * float(np.sum(np.log1p((a / sigma)**2)))
    grad = -(D.T @ r) + lam * (2*a / (sigma**2 + a*a))
    return energy, grad

def _log_prior_infer_single(x, D, lam, max_iter=200, tol=1e-6):
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
    """Nonlinear Conjugate Gradient with Polak-Ribière and Armijo backtracking"""
    K = D.shape[1]
    a = np.zeros((K,), dtype=float)
    f, g = _paper_energy_grad(x, D, a, lam, sigma)
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


class SparseCoder:
    """
    Dictionary learning + sparse inference.
    - mode='l1': FISTA on L1 objective (batch) + MOD update
    - mode='paper': per-sample NCG with log prior (Cauchy-like) + MOD update
    - mode='paper_gdD': per-sample NCG with log prior + gradient dictionary update (O&F-style) + homeostasis
    - mode='log': Olshausen & Field (1996) original log prior formulation + MOD update
    
    Features:
    - Lambda annealing: anneal=(gamma, floor) for geometric decay
    - Homeostatic equalization: prevents dead atoms
    - NCG: Polak-Ribière conjugate gradient
    - Dead atom reinitialization with detection and handling
    """
    def __init__(self, n_atoms=144, lam=None, mode="l1", max_iter=200, tol=1e-6, seed=0, anneal=None):
        self.n_atoms = int(n_atoms)
        self.lam = lam
        self.mode = mode
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.anneal = anneal  # (gamma, floor) tuple for lambda annealing
        self.rng = np.random.default_rng(seed)
        self.D = None  # (p, K)
    
    @property
    def dictionary(self):
        """Dictionary matrix (research standard: D or Φ)."""
        return self.D
    
    @dictionary.setter
    def dictionary(self, D):
        """Set dictionary matrix with proper normalization."""
        if D is not None:
            D = np.asarray(D, dtype=float)
            D = _normalize_columns(D)  # Research requirement: normalized columns
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

    def encode(self, X):
        X = np.asarray(X, float)
        if self.D is None:
            raise ValueError("dictionary not initialized. Call fit() first.")
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
            K, N = self.n_atoms, X.shape[1]
            A = np.zeros((K, N))
            for n in range(N):
                A[:, n] = _log_prior_infer_single(X[:, n], self.D, lam, max_iter=self.max_iter, tol=self.tol)
            return A
        elif self.mode == "paper" or self.mode == "paper_gdD":
            sigma = 1.0
            lam = float(self.lam if self.lam is not None else 0.14 * np.std(X))
            K, N = self.n_atoms, X.shape[1]
            A = np.zeros((K, N))
            for n in range(N):
                A[:, n] = _ncg_infer_single(X[:, n], self.D, lam, sigma, max_iter=self.max_iter, tol=self.tol)
            return A
        else:
            raise ValueError("mode must be 'l1', 'log', 'paper', or 'paper_gdD'")

    def decode(self, A):
        assert self.D is not None, "Dictionary not initialized."
        return self.D @ A

    def fit(self, X, n_steps=30, lr=0.1):
        X = np.asarray(X, float)
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
                K = D.shape[1]; A = np.zeros((K, N))
                for n in range(N):
                    A[:, n] = _ncg_infer_single(X[:, n], D, lam, 1.0, max_iter=self.max_iter, tol=self.tol)
                D = _mod_update(D, X, A, eps=1e-6)
            elif self.mode == "paper_gdD":
                K = D.shape[1]; A = np.zeros((K, N))
                for n in range(N):
                    A[:, n] = _ncg_infer_single(X[:, n], D, lam, 1.0, max_iter=self.max_iter, tol=self.tol)
                # Gradient dictionary update (O&F-style)
                D = _gradD_update(D, X, A, lr=lr)
                # Homeostatic equalization
                D = _homeostatic_equalize(D, A, alpha=0.1)
            elif self.mode == "log":
                # Olshausen & Field (1996) original log prior formulation
                # Uses log(1 + a²) sparsity penalty as in the Nature paper
                K = D.shape[1]; A = np.zeros((K, N))
                for n in range(N):
                    A[:, n] = _log_prior_infer_single(X[:, n], D, lam, max_iter=self.max_iter, tol=self.tol)
                D = _mod_update(D, X, A, eps=1e-6)
            else:
                raise ValueError("mode must be 'l1', 'paper', 'paper_gdD', or 'log'")
            
            D = _reinit_dead_atoms(D, X, A, self.rng)
            
            # Optional annealing of lambda
            if self.anneal:
                if isinstance(self.anneal, (list, tuple)) and len(self.anneal) == 2:
                    gamma, floor = float(self.anneal[0]), float(self.anneal[1])
                else:
                    gamma, floor = 0.95, 1e-4
                lam = max(floor, lam * gamma)

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
            
            # Check convergence
            delta_a = np.linalg.norm(a_new - a)
            rel_change = delta_a / max(1.0, np.linalg.norm(a))
            
            if rel_change <= self.tol:
                return {
                    'coefficients': a_new,
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_objective': f_new,
                    'final_gradient_norm': np.linalg.norm(g_new)
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
        
        lam = float(self.lam if self.lam is not None else 0.1 * np.median(np.abs(D.T @ x)))
        
        # FISTA implementation with objective tracking
        K = D.shape[1]
        a = np.zeros(K, dtype=float)
        y = a.copy()
        t = 1.0
        
        # Compute Lipschitz constant
        L = np.linalg.norm(D, ord=2)**2
        step_size = 1.0 / L
        
        objectives = []
        
        for iteration in range(int(self.max_iter)):
            a_prev = a.copy()
            
            # Gradient step
            grad = D.T @ (D @ y - x)
            z = y - step_size * grad
            
            # Proximal operator (soft thresholding)
            threshold = step_size * lam
            a = np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)
            
            # Compute objective
            residual = x - D @ a
            data_fit = 0.5 * np.dot(residual, residual)
            sparsity = lam * np.sum(np.abs(a))
            objective = data_fit + sparsity
            objectives.append(objective)
            
            # Check convergence
            delta_a = np.linalg.norm(a - a_prev)
            rel_change = delta_a / max(1.0, np.linalg.norm(a_prev))
            
            if rel_change <= self.tol:
                return {
                    'coefficients': a,
                    'converged': True,
                    'iterations': iteration + 1,
                    'objectives': objectives,
                    'final_objective': objective
                }
            
            # FISTA momentum update
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            y = a + ((t - 1.0) / t_new) * (a - a_prev)
            t = t_new
        
        # Did not converge
        return {
            'coefficients': a,
            'converged': False,
            'iterations': self.max_iter,
            'objectives': objectives,
            'final_objective': objectives[-1] if objectives else float('inf')
        }
