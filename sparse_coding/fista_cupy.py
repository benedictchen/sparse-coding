from .xp import xp, GPU
def soft_thresh(X, t):
    return xp.sign(X) * xp.maximum(xp.abs(X) - t, 0)
def fista_batch_cupy(D, X, lam, L=None, max_iter=200, tol=1e-6):
    if not GPU:  # pragma: no cover
        raise RuntimeError("CuPy not available")
    Dt = D.T; DtD = Dt @ D; DtX = Dt @ X
    if L is None:
        v = xp.random.normal(size=(D.shape[1],))
        v /= xp.linalg.norm(v) + 1e-12
        last = 0.0
        for _ in range(50):
            v = DtD @ v; nrm = xp.linalg.norm(v) + 1e-12; v = v / nrm
            lamL = float(v @ (DtD @ v))
            if abs(lamL - last) < 1e-7 * max(1.0, last): break
            last = lamL
        L = max(lamL, 1e-12)
    K, N = DtX.shape; A = xp.zeros((K, N), dtype=D.dtype); Y = xp.zeros_like(A); t = 1.0
    for _ in range(max_iter):
        G = DtD @ Y - DtX
        A2 = soft_thresh(Y - G / L, lam / L)
        t2 = (1 + (1 + 4*t*t)**0.5) / 2.0
        Y = A2 + ((t - 1)/t2) * (A2 - A)
        if xp.linalg.norm(A2 - A) <= tol * max(1.0, float(xp.linalg.norm(A))): break
        A, t = A2, t2
    return A
