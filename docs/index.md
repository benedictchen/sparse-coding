# Sparse Coding Documentation

!!! tip "Quick Start"
    ```python
    pip install sparse-coding
    ```

## Overview

Production-ready sparse coding and dictionary learning library with mathematical correctness, research citations, and industry-grade stability.

### Key Features

- **Research Accurate**: Implements Olshausen & Field (1996), Beck & Teboulle (2009)
- **Production Ready**: Mathematical stability fixes, gradient clipping
- **High Performance**: Parallel processing, memory optimization
- **Multiple Algorithms**: L1, log-prior, NCG, pure Olshausen modes

## Quick Example

```python
import numpy as np
from sparse_coding import SparseCoder

# Generate sample data
X = np.random.randn(64, 1000)  # 64 features, 1000 samples

# Create and train sparse coder
sc = SparseCoder(n_atoms=128, mode='l1')
sc.fit(X, n_steps=30)

# Encode signals to sparse representation
codes = sc.encode(X)
print(f"Sparsity: {np.mean(codes == 0) * 100:.1f}% zeros")

# Reconstruct signals
reconstructed = sc.decode(codes)
mse = np.mean((X - reconstructed)**2)
print(f"Reconstruction MSE: {mse:.6f}")
```

## Algorithm Modes

| Mode | Algorithm | Best For |
|------|-----------|----------|
| `l1` | FISTA L1 | Production use, fast convergence |
| `paper` | NCG log-prior | Research accuracy with MOD |
| `olshausen_pure` | Exact 1996 | Historical/reference implementation |
| `log` | Log-prior + MOD | Balanced accuracy and speed |

## Applications

### Computer Vision
- Feature learning from natural images
- Image denoising and restoration
- Object detection and recognition

### Signal Processing  
- Audio source separation
- Compressed sensing reconstruction
- Speech feature extraction

### Medical Imaging
- MRI reconstruction
- CT denoising
- Biomarker discovery

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quick-start.md)
- [API Reference](api/sparse-coder.md)
- [Mathematical Foundations](research/foundations.md)