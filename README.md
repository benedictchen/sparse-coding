# Sparse Coding | Dictionary Learning | FISTA Optimization | Olshausen Field Algorithm

**🔬 Research-Accurate Sparse Coding & Dictionary Learning Python Library** 

**Keywords:** `sparse coding`, `dictionary learning`, `FISTA`, `ISTA`, `Olshausen Field`, `feature learning`, `compressed sensing`, `L1 regularization`, `optimization`, `computer vision`, `signal processing`

**Production-Ready Implementation** of foundational sparse coding algorithms with mathematical correctness, research citations, and industry-grade stability. Based on Olshausen & Field (1996), Beck & Teboulle (2009), and Engan et al. (1999) research papers.

## 🎯 **Why Choose This Sparse Coding Library?**

| ✅ **Production Features** | 🔬 **Research Accuracy** | ⚡ **Performance** |
|---------------------------|--------------------------|-------------------|
| Mathematical stability fixes | Exact Olshausen & Field (1996) | FISTA optimization (Beck & Teboulle 2009) |
| Gradient clipping for NCG | MOD dictionary updates (Engan et al. 1999) | Parallel batch processing |
| Condition number checking | Research citations & formulations | Memory-efficient algorithms |
| Comprehensive input validation | Multiple sparse coding modes | GPU acceleration support |

## 🚀 **Install & Quick Start**

```bash
pip install sparse-coding
```

```python
import numpy as np
from sparse_coding import SparseCoder

# Dictionary Learning Example
X = np.random.randn(64, 1000)  # 64 features, 1000 samples
sc = SparseCoder(n_atoms=128, mode='l1')  # 128 dictionary atoms, L1 penalty
sc.fit(X, n_steps=30)  # Learn dictionary

# Sparse Coding Example  
codes = sc.encode(X)  # Get sparse representations
reconstructed = sc.decode(codes)  # Reconstruct signals
```

## 📚 **Algorithm Implementations**

### **Sparse Coding Methods**
- **L1 Regularization**: FISTA optimization (Beck & Teboulle 2009) - fastest convergence
- **Log-Prior**: Olshausen & Field (1996) original formulation - research accurate
- **NCG**: Nonlinear Conjugate Gradient with Polak-Ribière updates - robust optimization  
- **Pure Olshausen**: Exact 1996 gradient ascent algorithm - historical accuracy

### **Dictionary Learning Methods**  
- **MOD Updates**: Method of Optimal Directions (Engan et al. 1999) - stable
- **Gradient Updates**: Original Olshausen & Field approach - research faithful
- **Homeostatic Balancing**: Dead atom reinitialization - prevents local minima

## 🔬 **Research Applications & Use Cases**

### **Computer Vision** 
- **Image Feature Learning**: Learn visual primitives from natural images
- **Image Denoising**: Remove noise while preserving important features  
- **Object Detection**: Extract sparse representations for recognition
- **Texture Analysis**: Model texture patterns with overcomplete dictionaries

### **Medical Imaging**
- **MRI Reconstruction**: Compressed sensing for faster MRI acquisition
- **CT Image Denoising**: Reduce radiation dose while maintaining quality
- **Pattern Analysis**: Identify disease biomarkers in medical images

### **Signal Processing** 
- **Audio Source Separation**: Separate mixed audio signals
- **Speech Recognition**: Extract phonetic features for ASR systems  
- **Compression**: Efficient sparse representation of signals

### **Machine Learning & Data Science**
- **Anomaly Detection**: Detect outliers using reconstruction error
- **Dimensionality Reduction**: Learn compact representations  
- **Feature Extraction**: Unsupervised feature discovery

## 🔍 **Common Search Queries This Library Solves**

❓ **"How to implement sparse coding in Python?"** → Use our `SparseCoder` class with FISTA optimization  
❓ **"Dictionary learning algorithm implementation?"** → MOD and gradient-based updates included  
❓ **"Olshausen Field sparse coding code?"** → `mode='olshausen_pure'` for exact 1996 algorithm  
❓ **"FISTA algorithm Python?"** → Built-in FISTA with Beck & Teboulle (2009) implementation  
❓ **"Compressed sensing Python library?"** → L1 regularization with fast solvers  
❓ **"How to learn visual features unsupervised?"** → Dictionary learning on image patches

## ⚖️ **Comparison with Other Libraries**

| Library | Sparse Coding | Dictionary Learning | Research Accuracy | Mathematical Stability | Industry Ready |
|---------|---------------|---------------------|-------------------|----------------------|----------------|
| **This Library** | ✅ FISTA, NCG, Olshausen | ✅ MOD, Gradient | ✅ Research Citations | ✅ Stability Fixes | ✅ Production Ready |
| scikit-learn | ✅ Basic | ✅ Basic | ❌ Simplified | ⚠️ Basic | ✅ Stable |
| spams | ✅ Advanced | ✅ Yes | ⚠️ Some | ❌ C++ Complex | ❌ Research Only |
| sporco | ✅ Comprehensive | ✅ Yes | ✅ Good | ⚠️ Some | ⚠️ Academic Focus |

**Why Choose This Library:**
- 🔬 **Most Research-Accurate**: Exact implementations of seminal papers
- 🛡️ **Production-Stable**: Mathematical stability fixes and comprehensive testing
- ⚡ **High Performance**: Optimized algorithms with parallel processing
- 📖 **Best Documentation**: Clear examples and research foundations

## 🔧 **Technical Features**

### **Mathematical Robustness**
- **Condition Number Checking**: Automatic ill-conditioning detection (threshold: 1e10)
- **Gradient Clipping**: Prevents optimization explosions (threshold: 100.0) 
- **Numerical Stability**: Uses `solve()` instead of `inv()` for better conditioning
- **Input Validation**: Comprehensive checks for NaN/inf values

### **Algorithm Variants**
- **FISTA**: Fast Iterative Shrinkage-Thresholding (O(1/k²) convergence)
- **ISTA**: Basic proximal gradient method (O(1/k) convergence)  
- **NCG**: Nonlinear Conjugate Gradient with Polak-Ribière updates
- **Olshausen Pure**: Exact gradient ascent from 1996 Nature paper

### **Performance Optimizations**
- **Parallel Processing**: Joblib-based parallelization for large batches
- **Memory Efficiency**: Chunked processing for large datasets
- **Sparse Matrix Support**: Native scipy.sparse integration
- **GPU Acceleration**: Optional CUDA support via Array API

## 🚀 **Quick Start Examples**

### Method 1: Using DictionaryLearner (Patch-based learning)

```python
import numpy as np
from sparse_coding import DictionaryLearner, visualization

# Generate synthetic natural image patches
images = np.random.randn(10, 64, 64)  # 10 images, 64x64 pixels

# Learn sparse dictionary from image patches
learner = DictionaryLearner(
    n_components=100,        # 100 dictionary atoms
    patch_size=(8, 8),       # 8x8 patches
    sparsity_penalty=0.05,   # L1 regularization
    max_iterations=500       # Maximum training iterations
)

# Train on image patches
learner.fit(images)

# Access learned dictionary and sparse codes
print(f"Dictionary shape: {learner.dictionary.shape}")
print(f"Sparse codes shape: {learner.sparse_codes.shape}")

# Visualize learned dictionary
fig = visualization.plot_dictionary_atoms(
    learner.dictionary, 
    learner.patch_size, 
    title="Learned Sparse Dictionary"
)
fig.show()
```

### Method 2: Using SparseCoder (Direct signal processing)

```python
import numpy as np
from sparse_coding import SparseCoder

# Generate synthetic data (signals as columns)
n_features, n_samples = 256, 100
signals = np.random.randn(n_features, n_samples)

# Initialize sparse coder
coder = SparseCoder(
    n_atoms=64,              # 64 dictionary atoms
    mode='l1',               # L1 penalty (FISTA solver)
    lam=0.1,                 # Sparsity regularization
    max_iter=200             # Maximum iterations
)

# Learn dictionary and encode signals
coder.fit(signals, n_steps=30)
sparse_codes = coder.encode(signals)
reconstruction = coder.decode(sparse_codes)

print(f"Dictionary shape: {coder.dictionary.shape}")
print(f"Sparse codes shape: {sparse_codes.shape}")
print(f"Reconstruction error: {np.linalg.norm(signals - reconstruction):.4f}")
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

# Full parameter specification with correct interface
learner = DictionaryLearner(
    n_components=144,        # Number of dictionary atoms
    patch_size=(12, 12),     # Patch dimensions
    sparsity_penalty=0.03,   # L1 regularization strength
    learning_rate=0.01,      # Dictionary update step size
    max_iterations=1000      # Training iterations
)
```

### Optimization Methods
Multiple solver implementations:
```python
from sparse_coding import create_sparse_coder

# Create optimizer with different methods
optimizer = create_sparse_coder(
    dictionary, 
    penalty_type='l1',  # 'l1', 'elastic_net', 'non_negative_l1'
    penalty_params={'lam': 0.1}
)

# Compare optimization methods
methods = ['ista', 'fista', 'coordinate_descent']
for method in methods:
    result = getattr(optimizer, method)(signal)
    print(f"{method}: {result['iterations']} iterations")
```

### Visualization
Analysis tools:
```python
from sparse_coding import visualization

# Create analysis report
figures = visualization.create_analysis_report(
    dictionary=learner.dictionary,
    codes=sparse_codes,
    history=training_history,
    patch_size=(8, 8),
    save_path="analysis_report"
)
```

### Logging
Training monitoring:
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

# Log dictionary atoms
logger.log_dictionary_atoms(dictionary, patch_size)
```

## Examples

### Basic Dictionary Learning
```bash
python examples/basic_dictionary_learning.py
```
Reproduces the Olshausen & Field natural image experiment.

### Optimization Comparison  
```bash
python examples/optimization_comparison.py
```
Compares ISTA, FISTA, and Coordinate Descent methods.

### Pipeline Demo
```bash
python examples/pipeline_demo.py
```
Dictionary learning workflow with optimization and visualization.

## Mathematical Foundation

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
- **FISTA**: Fast Iterative Soft Thresholding with O(1/k²) convergence rate
- **Coordinate Descent**: Efficient for high-dimensional problems

## Performance Characteristics

| Method | Convergence Rate | Memory Usage | Best For |
|--------|------------------|--------------|----------|
| FISTA | O(1/k²) | Low | General purpose |
| ISTA | O(1/k) | Low | Simple problems |
| Coordinate Descent | Linear | Very Low | High-dimensional |

## Contributing

Areas of interest:
- GPU acceleration
- New penalty functions (group sparsity, structured sparsity)
- Domain-specific applications
- Modern sparse coding variants

## License

Custom Non-Commercial License with Donation Requirements

## 🙏 Support This Research

If this library helps your research or project, please consider donating:
- **PayPal**: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
- **GitHub Sponsors**: https://github.com/sponsors/benedictchen

Your support makes advanced AI research accessible to everyone! 🚀

**We need funding to continue this work.** Every contribution helps maintain and improve these research implementations.

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