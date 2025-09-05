import numpy as np

def soft_thresh(X, t):
    return np.sign(X) * np.maximum(np.abs(X) - t, 0.0)

def power_iter_L(D, n_iter=50, tol=1e-7, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    K = D.shape[1]
    v = rng.normal(size=(K,))
    v /= (np.linalg.norm(v) + 1e-12)
    last = 0.0
    DtD = D.T @ D
    for _ in range(n_iter):
        v = DtD @ v
        nrm = np.linalg.norm(v) + 1e-12
        v = v / nrm
        lam = float(v @ (DtD @ v))
        if abs(lam - last) < tol * max(1.0, last):
            break
        last = lam
    return max(lam, 1e-12)

def fista_batch(D, X, lam, L=None, max_iter=200, tol=1e-6):
    """
    Vectorized FISTA for: min_A 0.5||X - D A||_F^2 + lam ||A||_1
    D: (p,K), X: (p,N). Returns A: (K,N).
    """
    D = np.asarray(D, float); X = np.asarray(X, float)
    p, K = D.shape; N = X.shape[1]
    DtD = D.T @ D
    DtX = D.T @ X
    if L is None:
        L = power_iter_L(D)
    A = np.zeros((K, N)); Y = np.zeros_like(A); t = 1.0
    for _ in range(max_iter):
        G = DtD @ Y - DtX
        A_next = soft_thresh(Y - G / L, lam / L)
        t_next = (1 + np.sqrt(1 + 4*t*t)) / 2.0
        Y = A_next + ((t - 1) / t_next) * (A_next - A)
        if np.linalg.norm(A_next - A) <= tol * max(1.0, np.linalg.norm(A)):
            return A_next
        A, t = A_next, t_next
    return A
