# Sparse Coding v2.5.0

**Research-grade sparse coding implementation with dictionary learning, advanced optimization, and comprehensive visualization tools.**

## 🌟 What's New in v2.5.0

**MAJOR FUNCTIONALITY RESTORATION** - From minimal v2 (549 lines) to full-featured research library (2,468 lines):

### ✨ New Features

- **🧠 Complete Dictionary Learning** - Full Olshausen & Field (1996) implementation
- **⚡ Advanced Optimization** - ISTA, FISTA, Coordinate Descent, Adaptive methods  
- **📊 Professional Visualization** - Dictionary atoms, training progress, sparsity analysis
- **📈 TensorBoard Integration** - Real-time monitoring with professional dashboards
- **🎯 Comprehensive Examples** - Production-ready demonstrations and tutorials

### 🔬 Research Applications

- **Computer Vision**: Feature learning, image denoising, object detection
- **Medical Imaging**: MRI reconstruction, CT denoising, pattern analysis  
- **Audio Processing**: Source separation, compression, recognition
- **Data Science**: Anomaly detection, dimensionality reduction, interpretable features

## 🚀 Quick Start

```python
import numpy as np
from sparse_coding import DictionaryLearner, visualization

# Generate synthetic natural image patches
images = np.random.randn(10, 64, 64)  # 10 images, 64x64 pixels

# Learn sparse dictionary
learner = DictionaryLearner(
    n_components=100,        # 100 dictionary atoms
    patch_size=(8, 8),       # 8x8 patches
    sparsity_penalty=0.05,   # L1 regularization
    max_iterations=500
)

# Train on image patches
history = learner.fit(images, verbose=True)

# Extract sparse features
features = learner.transform(images, pooling='max')
print(f"Feature shape: {features.shape}")

# Visualize learned dictionary
fig = visualization.plot_dictionary_atoms(
    learner.dictionary, 
    learner.patch_size, 
    title="Learned Sparse Dictionary"
)
fig.show()
```

## 📦 Installation

```bash
pip install sparse-coding
```

Or install from source:
```bash
git clone https://github.com/benedictchen/sparse-coding
cd sparse-coding  
pip install -e .
```

## 🔧 Core Components

### DictionaryLearner
Complete dictionary learning with alternating optimization:
```python
from sparse_coding import DictionaryLearner

learner = DictionaryLearner(
    n_components=144,
    patch_size=(12, 12),
    sparsity_penalty=0.03,
    learning_rate=0.01,
    max_iterations=1000
)
```

### Advanced Optimization
Multiple state-of-the-art solvers:
```python
from sparse_coding import create_advanced_sparse_coder

# Create optimizer with different methods
optimizer = create_advanced_sparse_coder(
    dictionary, 
    penalty_type='l1',  # 'l1', 'elastic_net', 'non_negative_l1'
    penalty_params={'lam': 0.1}
)

# Compare optimization methods
methods = ['ista', 'fista', 'coordinate_descent', 'adaptive_fista']
for method in methods:
    result = getattr(optimizer, method)(signal)
    print(f"{method}: {result['iterations']} iterations")
```

### Professional Visualization
Comprehensive analysis tools:
```python
from sparse_coding import visualization

# Create complete analysis report
figures = visualization.create_comprehensive_report(
    dictionary=learner.dictionary,
    codes=sparse_codes,
    history=training_history,
    patch_size=(8, 8),
    save_path="analysis_report"
)
```

### TensorBoard Integration
Real-time monitoring:
```python
from sparse_coding import DashboardLogger

# Setup logging
logger = DashboardLogger(
    tensorboard_dir="logs/sparse_coding",
    csv_path="metrics.csv"
)

# Log training metrics
logger.log_training_metrics({
    'reconstruction_error': 0.001,
    'sparsity_level': 0.05
})

# Visualize dictionary evolution
logger.log_dictionary_atoms(dictionary, patch_size)
```

## 📚 Examples

### Basic Dictionary Learning
```bash
python examples/basic_dictionary_learning.py
```
Reproduces the classic Olshausen & Field natural image experiment.

### Advanced Optimization Comparison  
```bash
python examples/advanced_optimization_comparison.py
```
Compares ISTA, FISTA, Coordinate Descent, and Adaptive FISTA methods.

### Complete Pipeline Demo
```bash
python examples/complete_pipeline_demo.py
```
Full workflow with dictionary learning, optimization, visualization, and TensorBoard logging.

## 🔬 Mathematical Foundation

**Core Optimization Problem:**
```
min_{D,α} Σᵢ [½||xᵢ - Dαᵢ||₂² + λ||αᵢ||₁]
```

Where:
- `xᵢ` = input patch i
- `D` = dictionary matrix (learned features)
- `αᵢ` = sparse coefficients for patch i  
- `λ` = sparsity penalty

**Key Algorithms:**
- **Dictionary Learning**: Alternates between sparse coding and dictionary updates
- **FISTA**: Fast Iterative Soft Thresholding with optimal convergence rate
- **Coordinate Descent**: Efficient for high-dimensional problems
- **Adaptive Methods**: Automatic step size selection with backtracking

## 📊 Performance Benchmarks

| Method | Convergence Rate | Memory Usage | Best For |
|--------|------------------|--------------|----------|
| **FISTA** | O(1/k²) | Low | General purpose |
| **ISTA** | O(1/k) | Low | Simple problems |
| **Coordinate Descent** | Linear | Very Low | High-dimensional |
| **Adaptive FISTA** | O(1/k²) | Medium | Unknown Lipschitz |

## 🤝 Contributing

We welcome contributions! Areas of particular interest:

- **GPU Acceleration**: CUDA/CuPy optimization
- **New Penalties**: Group sparsity, structured sparsity
- **Applications**: Domain-specific examples and tutorials
- **Algorithms**: Modern sparse coding variants

## 📄 License

Custom Non-Commercial License with Donation Requirements

## 🙏 Support This Research

If this library helps your research or project, please consider donating:
- **PayPal**: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
- **GitHub Sponsors**: https://github.com/sponsors/benedictchen

Your support makes advanced AI research accessible to everyone! 🚀

## 📖 Citation

```bibtex
@software{sparse_coding_2024,
  author = {Benedict Chen},
  title = {Sparse Coding: Research-grade Dictionary Learning Library},
  url = {https://github.com/benedictchen/sparse-coding},
  version = {2.5.0},
  year = {2024}
}
```

## 🔗 References

- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381(6583), 607-609.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM journal on imaging sciences*, 2(1), 183-202.
- Mairal, J., Bach, F., Ponce, J., & Sapiro, G. (2010). Online dictionary learning for sparse coding. *ICML*.

---

**v2.5.0** - Complete dictionary learning restoration with 4.5x functionality expansion  
**v2.4.0** - Minimal but mathematically correct implementation  
**Maintained by**: Benedict Chen <benedict@benedictchen.com>