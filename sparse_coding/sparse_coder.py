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
    D_new = (X @ At) @ np.linalg.inv(G)
    return _normalize_columns(D_new)

def _reinit_dead_atoms(D, X, A, rng):
    usage = np.sum(np.abs(A) > 1e-12, axis=1)
    dead = np.where(usage == 0)[0]
    if dead.size == 0:
        return D
    N = X.shape[1]
    for k in dead:
        i = rng.integers(0, N)
        D[:, k] = X[:, i]
    return _normalize_columns(D)

def _paper_energy_grad(x, D, a, lam, sigma):
    # E(a) = 0.5||x - D a||^2 - lam * sum log(1 + (a/sigma)^2)
    r = x - D @ a
    energy = 0.5 * float(r @ r) - lam * float(np.sum(np.log1p((a / sigma)**2)))
    grad = -(D.T @ r) - lam * (2*a / (sigma**2 + a*a))
    return energy, grad

def _ncg_infer_single(x, D, lam, sigma, max_iter=200, tol=1e-6):
    # Nonlinear Conjugate Gradient with Armijo backtracking
    a = np.zeros(D.shape[1])
    E, g = _paper_energy_grad(x, D, a, lam, sigma)
    d = -g
    for _ in range(max_iter):
        # line search
        t = 1.0
        gd = float(g @ d)
        if gd > 0: d = -g; gd = float(g @ d)
        while t > 1e-8:
            a_new = a + t * d
            E_new, g_new = _paper_energy_grad(x, D, a_new, lam, sigma)
            if E_new <= E + 1e-4 * t * gd:
                break
            t *= 0.5
        a, E, g = a_new, E_new, g_new
        if np.linalg.norm(g) <= tol * max(1.0, np.linalg.norm(a)):
            break
        beta = float((g @ (g - g_new)) / (gd + 1e-12)) if 'g_new' in locals() else 0.0
        beta = max(beta, 0.0)
        d = -g + beta * d
    return a

class SparseCoder:
    """
    Dictionary learning + sparse inference.
    - mode='l1': FISTA on L1 objective (batch)
    - mode='paper': log-penalty with NLCG per-sample inference
    Dictionary update uses MOD with column renorm + dead-atom refresh.
    """
    def __init__(self, n_atoms=144, lam=None, mode="paper", max_iter=200, tol=1e-6, seed=0):
        self.n_atoms = int(n_atoms)
        self.lam = lam
        self.mode = mode
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.D = None  # (p, K)

    def _init_dictionary(self, X):
        p, N = X.shape
        if self.D is None:
            idx = self.rng.choice(N, size=self.n_atoms, replace=False)
            D = X[:, idx].copy()
            D = D + 1e-6 * self.rng.normal(size=D.shape)
            D = D - D.mean(axis=0, keepdims=True)
            D = _normalize_columns(D)
            self.D = D

    def encode(self, X):
        X = np.asarray(X, float)
        assert self.D is not None, "Dictionary not initialized. Fit first."
        if self.mode == "l1":
            lam = float(self.lam if self.lam is not None else 0.1 * np.median(np.abs(self.D.T @ X)))
            return fista_batch(self.D, X, lam, L=None, max_iter=self.max_iter, tol=self.tol)
        elif self.mode == "paper":
            sigma = 1.0
            lam = float(self.lam if self.lam is not None else 0.14 * np.std(X))
            K, N = self.n_atoms, X.shape[1]
            A = np.zeros((K, N))
            for n in range(N):
                A[:, n] = _ncg_infer_single(X[:, n], self.D, lam, sigma, max_iter=self.max_iter, tol=self.tol)
            return A
        else:
            raise ValueError("mode must be 'l1' or 'paper'")

    def decode(self, A):
        assert self.D is not None, "Dictionary not initialized."
        return self.D @ A

    def fit(self, X, n_steps=30, lr=0.1):
        X = np.asarray(X, float)
        p, N = X.shape
        self._init_dictionary(X)
        D = self.D
        lam_default = 0.1 * np.median(np.abs(D.T @ X)) if self.mode == "l1" else 0.14 * np.std(X)
        lam = float(self.lam if self.lam is not None else lam_default)

        for _ in range(int(n_steps)):
            if self.mode == "l1":
                A = fista_batch(D, X, lam, L=None, max_iter=self.max_iter, tol=self.tol)
            else:  # paper
                K = D.shape[1]; A = np.zeros((K, N))
                for n in range(N):
                    A[:, n] = _ncg_infer_single(X[:, n], D, lam, 1.0, max_iter=self.max_iter, tol=self.tol)

            D = _mod_update(D, X, A, eps=1e-6)
            D = _reinit_dead_atoms(D, X, A, self.rng)

        self.D = D
        return self
