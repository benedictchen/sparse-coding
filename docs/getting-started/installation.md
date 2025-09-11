# Installation

## Requirements

- Python 3.9+
- NumPy 1.21+
- SciPy 1.7+
- scikit-learn 1.0+

## Install from PyPI

```bash
pip install sparse-coding
```

## Development Installation

```bash
git clone https://github.com/benedictchen/sparse-coding.git
cd sparse-coding
pip install -e .[dev,test]
```

## Optional Dependencies

For enhanced performance:

```bash
# For parallel processing
pip install joblib

# For GPU acceleration  
pip install cupy-cuda11x  # or appropriate CUDA version

# For visualization
pip install matplotlib seaborn
```

## Verify Installation

```python
import sparse_coding
print(f"Sparse Coding {sparse_coding.__version__} installed successfully")

# Quick test
import numpy as np
from sparse_coding import SparseCoder

X = np.random.randn(32, 100)
sc = SparseCoder(n_atoms=16)
sc.fit(X, n_steps=5)
print("✅ Installation verified")
```