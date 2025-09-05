# Sparse Coding - Research-Faithful Implementation

**Paper-exact Olshausen & Field (1996) sparse coding with modern L1/FISTA optimization.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Paper-Exact Quickstart

```bash
# Install dependencies
python -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install pillow matplotlib pyyaml

# Run paper-exact demo (requires natural images folder)
sparse-coding train --images ./images --out of_out --mode paper --seed 0
```

**Outputs:**
- `of_out/D.npy` — learned dictionary (edge filters)
- `of_out/A.npy` — sparse coefficients
- Reproduces classic Olshausen & Field (1996) results

## ⚡ CLI Quickstart

```bash
# Install package
pip install -e .[dev]

# Train dictionary from image folder
sparse-coding train --images ./images --out results --mode paper --seed 0

# Encode patches with existing dictionary
sparse-coding encode --dictionary results/D.npy --patches X.npy --out A.npy

# Reconstruct from sparse codes
sparse-coding reconstruct --dictionary results/D.npy --codes A.npy --out X_hat.npy
```

## 🔬 Python API

### Modern L1 Sparse Coding (Fast)
```python
import numpy as np
from sparse_coding import SparseCoder

# Create modern L1 sparse coder
coder = SparseCoder(n_atoms=128, mode='l1', seed=42)

# Fit dictionary to patches (p, N)
patches = np.random.randn(256, 10000)  # 16x16 patches
coder.fit(patches, n_steps=30)

# Sparse encode new patches
codes = coder.encode(patches[:, :100])

# Reconstruct 
reconstruction = coder.decode(codes)
```

### Paper-Exact Olshausen & Field Mode
```python
# Research-accurate reproduction
coder_paper = SparseCoder(
    n_atoms=144,           # Overcomplete dictionary
    mode='paper',          # Log sparsity penalty
    ratio_lambda_over_sigma=0.14,  # Paper's λ/σ ratio
    seed=0
)

# Train with paper-exact alternating optimization
coder_paper.fit(whitened_patches, n_steps=50, lr=0.1)

# Results match original 1996 paper
dictionary = coder_paper.D  # Edge-like receptive fields
```

## 📊 Key Features

- **Paper-Exact Mode**: Reproduces Olshausen & Field (1996) exactly
- **Modern L1/FISTA**: Fast optimization with KKT validation
- **Zero-Phase Whitening**: Research-accurate `R(f) = |f| exp(-(f/f₀)⁴)` filter  
- **Homeostatic Gains**: Coefficient variance equalization
- **CLI Interface**: Train/encode/reconstruct from command line
- **YAML Configs**: Reproducible experiments with config files

## 🧪 Validation

The implementation passes rigorous tests:
- **KKT Conditions**: L1 solutions satisfy optimality conditions
- **Sparsity Levels**: 70-95% zero coefficients (typical)
- **Paper Reproduction**: Matches 1996 results on natural images
- **Convergence**: Objective decreases monotonically

## ⚙️ Configuration

Create `config.yaml`:
```yaml
patch_size: 16
n_atoms: 144
n_steps: 50
lr: 0.1
f0: 200.0
samples: 50000
ratio_lambda_over_sigma: 0.14
```

Use with CLI:
```bash
sparse-coding train --images ./images --config config.yaml --out results
```

## 📚 Research Background

Implements the seminal sparse coding algorithm from:

> Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381(6583), 607-609.

**Key insight:** Natural images can be efficiently represented using a sparse set of basis functions that resemble edge detectors found in primary visual cortex.

## 🎯 Mathematical Framework

**Objective:** Learn dictionary `D` and sparse codes `A` such that `X ≈ DA`

**L1 Mode (Modern):**
```
min_{D,A} ½‖X - DA‖²₂ + λ‖A‖₁
```

**Paper Mode (Research-Exact):**
```  
min_{D,A} ½‖X - DA‖²₂ - λ ∑ log(1 + (aᵢ/σ)²)
```

Solved via alternating optimization with homeostatic gain control.

## 🔧 Installation & Development

```bash
# Development install
git clone https://github.com/your-repo/sparse-coding
cd sparse-coding
pip install -e .[dev]

# Run tests
pytest

# Lint code  
ruff check .
mypy sparse_coding
```

## 📖 Citation

If you use this implementation in research:

```bibtex
@software{sparse_coding_2025,
  title={Sparse Coding: Research-Faithful Implementation},  
  year={2025},
  url={https://github.com/your-repo/sparse-coding}
}

@article{olshausen1996emergence,
  title={Emergence of simple-cell receptive field properties by learning a sparse code for natural images},
  author={Olshausen, Bruno A and Field, David J},
  journal={Nature},
  volume={381},
  number={6583}, 
  pages={607--609},
  year={1996}
}
```

## 🏆 Why This Implementation?

- **Research Faithful**: Exact reproduction of seminal 1996 paper
- **Modern Performance**: Fast FISTA optimization for practical use
- **Comprehensive**: CLI, Python API, configs, validation
- **Well-Tested**: Passes KKT conditions and convergence tests
- **Documented**: Clear mathematical formulation and usage examples

Perfect for reproducing classic results or building modern sparse coding applications.