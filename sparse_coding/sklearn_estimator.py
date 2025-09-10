import numpy as np
from .sparse_coder import SparseCoder

try:
    from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
except Exception:  # pragma: no cover
    class BaseEstimator: ...
    class TransformerMixin: ...

class SparseCoderEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, n_atoms=144, mode="l1", max_iter=200, tol=1e-6, seed=0, lam=None):
        self.n_atoms = n_atoms
        self.mode = mode
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.lam = lam
        self._coder = None

    def fit(self, X, y=None, n_steps=20, lr=0.1):
        Xc = np.asarray(X, dtype=float).T
        self._coder = SparseCoder(n_atoms=self.n_atoms, mode=self.mode, max_iter=self.max_iter, tol=self.tol, seed=self.seed, lam=self.lam)
        self._coder.fit(Xc, n_steps=n_steps, lr=lr)
        return self

    def transform(self, X):
        Xc = np.asarray(X, dtype=float).T
        A = self._coder.encode(Xc)
        return A.T

    def inverse_transform(self, A):
        A = np.asarray(A, dtype=float).T
        X_hat = self._coder.decode(A)
        return X_hat.T
