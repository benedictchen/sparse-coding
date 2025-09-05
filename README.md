# sparse-coding (v2)

A compact, research-faithful sparse coding library with a clean CLI and an sklearn-style wrapper.
- **Modes:** `paper` (Olshausen & Field-style log penalty) and `l1` (FISTA).
- **Ready-to-ship:** minimal footprint, validated configs, deterministic mode, streaming encode, optional CPU JIT & GPU.
- **No placeholders.** Everything here is implemented and usable.

## Install
```bash
pip install .
# or with extras
pip install '.[gpu]'        # CuPy GPU acceleration (CUDA)
pip install '.[dev]'        # dev tools
```

## CLI quickstart
```bash
# Train from images in a folder (whitening + random patches)
sparse-coding train --images ./images --out out --mode paper --seed 0 --deterministic

# Train directly from a (p x N) NumPy patch matrix
sparse-coding train-patches --patches X.npy --out out --mode l1

# Encode and reconstruct
sparse-coding encode --dictionary out/D.npy --patches X.npy --out A.npy
sparse-coding reconstruct --dictionary out/D.npy --codes A.npy --out X_hat.npy

# Streaming encode for giant matrices (chunked, memmap output)
sparse-coding encode-stream --dictionary out/D.npy --patches X.npy --lam 0.1 --batch 20000
```

## Python quickstart
```python
import numpy as np
from sparse_coding import SparseCoder

X = np.load("X.npy")  # shape (p, N), p = patch_size**2
coder = SparseCoder(n_atoms=256, mode="l1", max_iter=200, tol=1e-6, seed=0)
coder.fit(X, n_steps=30, lr=0.1)
A = coder.encode(X)
X_hat = coder.decode(A)
```

## Notes
- `paper` mode uses a log penalty with a non-linear CG solver (with Armijo line search).
- Dictionary updates use the **MOD** step (`D = X A^T (A A^T + εI)^{-1}`) with column renormalization and dead-atom refresh.
- Whitening follows a zero-phase filter `R(f)=|f| e^{-(f/f0)^4}` applied at the **image** level.
