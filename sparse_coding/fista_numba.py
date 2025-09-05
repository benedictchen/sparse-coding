import numpy as np
try:
    from numba import njit
    NUMBA_OK = True
except Exception:  # pragma: no cover
    NUMBA_OK = False
    def njit(*a, **k):
        def wrap(f): return f
        return wrap

@njit(cache=True)
def _soft_thresh(x, t):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        xi = x[i]; s = abs(xi) - t
        out[i] = (1.0 if xi >= 0 else -1.0) * s if s > 0.0 else 0.0
    return out

@njit(cache=True)
def _fista_batch_numba(DtD, DtX, lam, L, K, N, max_iter):
    A = np.zeros((K, N)); Y = np.zeros_like(A); t = 1.0
    for _ in range(max_iter):
        G = DtD @ Y - DtX
        for n in range(N):
            Yn = Y[:, n] - G[:, n] / L
            A[:, n] = _soft_thresh(Yn, lam / L)
        t_next = (1.0 + (1.0 + 4.0*t*t)**0.5) / 2.0
        Y = A + ((t - 1.0) / t_next) * (A - Y)
        t = t_next
    return A

def fista_batch_numba(D, X, lam, L=None, max_iter=200):
    from .fista_batch import fista_batch, power_iter_L
    p, K = D.shape; N = X.shape[1]
    DtD = D.T @ D; DtX = D.T @ X
    if L is None:
        L = power_iter_L(D)
    if not NUMBA_OK:
        return fista_batch(D, X, lam, L=L, max_iter=max_iter)
    return _fista_batch_numba(DtD, DtX, lam, L, K, N, max_iter)
