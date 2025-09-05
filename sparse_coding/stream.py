from __future__ import annotations
import os, numpy as np
from typing import Iterator

def stream_columns(X_path: str, batch: int = 10000) -> Iterator[np.ndarray]:
    X = np.load(X_path, mmap_mode='r'); N = X.shape[1]
    for i in range(0, N, batch):
        yield np.asarray(X[:, i:i+batch])

def encode_stream(D: np.ndarray, X_path: str, lam: float, batch: int = 10000, out_path: str | None = None) -> str:
    from .fista_batch import fista_batch, power_iter_L
    L = power_iter_L(D)
    X = np.load(X_path, mmap_mode='r')
    K, N = D.shape[1], X.shape[1]
    out = out_path or (os.path.splitext(X_path)[0] + ".codes.npy")
    A = np.memmap(out, dtype=float, mode='w+', shape=(K, N))
    off = 0
    for chunk in stream_columns(X_path, batch=batch):
        Ac = fista_batch(D, chunk, lam, L=L)
        m = Ac.shape[1]
        A[:, off:off+m] = Ac; off += m
    del A
    return out
