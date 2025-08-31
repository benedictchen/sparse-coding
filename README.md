# 💰 Support This Research - Please Donate!

**🙏 If this library helps your research or project, please consider donating to support continued development:**

**[💳 DONATE VIA PAYPAL - CLICK HERE](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**

[![CI](https://github.com/benedictchen/sparse-coding/workflows/CI/badge.svg)](https://github.com/benedictchen/sparse-coding/actions)
[![PyPI version](https://badge.fury.io/py/sparse-coding.svg)](https://badge.fury.io/py/sparse-coding)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Custom%20Non--Commercial-red.svg)](LICENSE)

---

# Sparse Coding

💰 Biological vision principles for representation learning

**Olshausen, B. A., & Field, D. J. (1996) - "Emergence of simple-cell receptive field properties"**

## 📦 Installation

```bash
pip install sparse-coding
```

## 🚀 Quick Start

```python
import sparse_coding
import numpy as np

# Create sample image patches (8x8 patches)
patches = np.random.randn(1000, 64)

# Initialize sparse coder
coder = sparse_coding.SparseCoder(
    dictionary_size=128,
    sparsity_lambda=0.1
)

# Learn sparse dictionary
coder.fit(patches)

# Encode new patches sparsely
test_patch = np.random.randn(1, 64)
sparse_code = coder.encode(test_patch)
reconstructed = coder.decode(sparse_code)

print(f"✅ Sparse coding: {np.sum(sparse_code != 0)} active out of {len(sparse_code)} atoms")
print(f"✅ Reconstruction error: {np.mean((test_patch - reconstructed)**2):.4f}")
```

## 🎓 About the Implementation

Implemented by **Benedict Chen** - bringing foundational AI research to modern Python.

📧 Contact: benedict@benedictchen.com

## 📖 Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
Olshausen, B. A., & Field, D. J. (1996) - "Emergence of simple-cell receptive field properties"
```

## 📜 License

Custom Non-Commercial License with Donation Requirements - See LICENSE file for details.

---

## 💰 Support This Work - Donation Appreciated!

**This implementation represents hundreds of hours of research and development. If you find it valuable, please consider donating:**

**[💳 DONATE VIA PAYPAL - CLICK HERE](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**

**Your support helps maintain and expand these research implementations! 🙏**