# Quick Start Guide

## Basic Dictionary Learning

```python
import numpy as np
from sparse_coding import SparseCoder

# Create synthetic data
np.random.seed(42)
X = np.random.randn(64, 1000)  # 64 features, 1000 samples

# Initialize sparse coder
sc = SparseCoder(
    n_atoms=128,        # Dictionary size
    mode='l1',          # L1 regularization
    max_iter=500,       # Maximum iterations
    tol=1e-6           # Convergence tolerance
)

# Learn dictionary
sc.fit(X, n_steps=30)
print(f"Learned dictionary: {sc.dictionary.shape}")
```

## Sparse Encoding

```python
# Encode new signals
test_X = np.random.randn(64, 50)
codes = sc.encode(test_X)

# Check sparsity
sparsity = np.mean(codes == 0) * 100
print(f"Sparsity: {sparsity:.1f}% zeros")

# Reconstruct signals
reconstructed = sc.decode(codes)
mse = np.mean((test_X - reconstructed)**2)
print(f"Reconstruction MSE: {mse:.6f}")
```

## Algorithm Modes

### L1 Regularization (Production)
```python
# FISTA optimization - fastest convergence
sc = SparseCoder(n_atoms=128, mode='l1', lam=0.1)
```

### Research-Accurate Olshausen & Field
```python
# Exact 1996 Nature paper algorithm
sc = SparseCoder(n_atoms=128, mode='olshausen_pure', lam=0.05)
```

### Nonlinear Conjugate Gradient
```python
# NCG with log-prior - robust optimization
sc = SparseCoder(n_atoms=128, mode='paper', lam=0.1)
```

## Common Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `n_atoms` | Dictionary size | 64, 128, 256 |
| `lam` | Sparsity penalty | 0.01 - 0.5 |
| `max_iter` | Max iterations | 200 - 1000 |
| `tol` | Convergence tolerance | 1e-6 - 1e-4 |

## Next Steps

- [API Reference](../api/sparse-coder.md)
- [Algorithm Details](../user-guide/sparse-coding.md)
- [Performance Tuning](../user-guide/performance.md)