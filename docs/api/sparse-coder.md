# SparseCoder API Reference

::: sparse_coding.sparse_coder.SparseCoder
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## Class Overview

The `SparseCoder` class is the main interface for dictionary learning and sparse coding. It implements multiple algorithms with research accuracy and production stability.

### Supported Modes

| Mode | Algorithm | Research Source | Best For |
|------|-----------|----------------|----------|
| `'l1'` | FISTA L1 | Beck & Teboulle (2009) | Production use |
| `'paper'` | NCG log-prior + MOD | Olshausen & Field (1996) + Engan et al. (1999) | Research accuracy |
| `'olshausen_pure'` | Pure gradient ascent | Olshausen & Field (1996) exact | Historical reference |
| `'log'` | Log-prior + MOD | Olshausen & Field (1996) + Engan et al. (1999) | Balanced approach |

### Mathematical Stability Features

- **Condition number checking**: Automatic detection at 1e10 threshold
- **Gradient clipping**: Prevents optimization explosions (100.0 threshold)
- **Input validation**: Comprehensive NaN/inf checking with sparse support
- **Numerical stability**: Uses `solve()` instead of `inv()` for better conditioning

### Performance Optimizations

- **Parallel processing**: Joblib-based parallelization for large batches
- **Memory efficiency**: Chunked processing for large datasets  
- **Sparse matrix support**: Native scipy.sparse integration
- **Adaptive algorithms**: Problem-size aware optimizations

## Constructor Parameters

### Core Parameters

- **n_atoms** (int): Number of dictionary atoms. Typical: 64, 128, 256
- **lam** (float, optional): Sparsity penalty. Auto-scaled if None. Range: 0.01-0.5
- **mode** (str): Algorithm mode. Options: 'l1', 'paper', 'olshausen_pure', 'log'

### Optimization Parameters

- **max_iter** (int): Maximum iterations. Range: 200-1000
- **tol** (float): Convergence tolerance. Range: 1e-6 to 1e-4
- **seed** (int): Random seed for reproducibility

### Advanced Parameters

- **anneal** (tuple, optional): Lambda annealing as (gamma, floor). Example: (0.95, 1e-4)

## Usage Examples

### Basic Dictionary Learning
```python
import numpy as np
from sparse_coding import SparseCoder

# Create data
X = np.random.randn(64, 1000)

# Learn dictionary
sc = SparseCoder(n_atoms=128, mode='l1')
sc.fit(X, n_steps=30)

# Encode/decode
codes = sc.encode(X)
reconstructed = sc.decode(codes)
```

### Research-Accurate Mode
```python
# Exact Olshausen & Field (1996) algorithm
sc = SparseCoder(
    n_atoms=144,
    mode='olshausen_pure',
    lam=0.05,
    max_iter=500
)
```

### Production Optimization
```python
# Fast L1 regularization
sc = SparseCoder(
    n_atoms=256,
    mode='l1',
    lam=0.1,
    max_iter=1000,
    tol=1e-6
)
```