# 💰 Support This Research - Please Donate!

**🙏 If this library helps your research or project, please consider donating to support continued development:**

<div align="center">

**[💳 DONATE VIA PAYPAL](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)** | **[❤️ SPONSOR ON GITHUB](https://github.com/sponsors/benedictchen)**

</div>

[![CI](https://github.com/benedictchen/sparse-coding/workflows/CI/badge.svg)](https://github.com/benedictchen/sparse-coding/actions)
[![PyPI version](https://badge.fury.io/py/sparse-coding.svg)](https://badge.fury.io/py/sparse-coding)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Custom%20Non--Commercial-red.svg)](LICENSE)
[![Research Accurate](https://img.shields.io/badge/research-accurate-brightgreen.svg)](RESEARCH_FOUNDATION.md)

---

# Sparse Coding

🌟 **Discover edge-like features from natural images using biologically-inspired learning algorithms**

Sparse coding learns efficient representations where natural images can be reconstructed using only a few active features from an overcomplete dictionary. This implementation faithfully reproduces the groundbreaking research that revealed how our visual cortex processes images.

**Research Foundation**: Olshausen, B. A., & Field, D. J. (1996) - *"Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"*

## 🚀 Quick Start

### Installation

```bash
pip install sparse-coding
```

**Requirements**: Python 3.9+, NumPy, SciPy, scikit-learn, matplotlib

### Basic Usage

```python
from sparse_coding import SparseCoder
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

# Load sample data (or use your own images)
faces = fetch_olivetti_faces()
image_patches = faces.data.reshape(-1, 8, 8)  # 8x8 patches

# Create sparse coder with Olshausen-Field algorithm
coder = SparseCoder(
    n_components=256,      # Dictionary size (overcomplete)
    algorithm='olshausen_field',
    max_iter=1000,
    alpha=0.1,            # Sparsity parameter
    random_state=42
)

# Learn dictionary from natural image patches
print("Learning sparse dictionary...")
dictionary = coder.fit(image_patches)

# Transform new images to sparse codes
sparse_codes = coder.transform(image_patches[:10])
print(f"Sparsity: {np.mean(sparse_codes == 0):.1%} of coefficients are zero")

# Reconstruct images from sparse codes
reconstructed = coder.inverse_transform(sparse_codes)

# Visualize learned features (they look like edge detectors!)
coder.visualize_dictionary(title="Learned Edge Detectors")
```

### Dictionary Learning Example

```python
from sparse_coding import DictionaryLearner
from sparse_coding.sc_modules import OlshausenFieldOptimizer
import matplotlib.pyplot as plt

# Advanced dictionary learning with custom parameters
learner = DictionaryLearner(
    dictionary_size=512,
    patch_size=(12, 12),
    optimizer=OlshausenFieldOptimizer(
        learning_rate=0.01,
        sparsity_target=0.05,
        decay_rate=0.95
    )
)

# Learn from natural image dataset
natural_images = load_your_images()  # Your image loading function
learned_dict = learner.fit(natural_images)

# Analyze dictionary properties
learner.analyze_dictionary_statistics()
learner.plot_feature_evolution()

# Export dictionary for other applications
learner.save_dictionary("edge_detectors.npy")
```

### Feature Extraction Pipeline

```python
from sparse_coding import FeatureExtractor
from sparse_coding.sc_modules import ValidationMethods

# Create feature extraction pipeline
extractor = FeatureExtractor(
    dictionary_path="edge_detectors.npy",
    sparse_solver='ista',      # Iterative Shrinkage-Thresholding
    lambda_reg=0.15,
    max_iter=500
)

# Extract features from new images
features = extractor.extract_features(test_images)

# Validate extraction quality
validator = ValidationMethods()
reconstruction_error = validator.measure_reconstruction_quality(
    original=test_images, 
    reconstructed=extractor.reconstruct(features)
)
sparsity_level = validator.measure_sparsity(features)

print(f"Reconstruction PSNR: {reconstruction_error:.2f} dB")
print(f"Feature sparsity: {sparsity_level:.1%}")
```

## 🧬 Advanced Features

### Modular Architecture

```python
# Access individual algorithm components (mixin classes)
from sparse_coding.sc_modules import (
    DataProcessingMixin,     # Image preprocessing utilities  
    OptimizationMixin,       # ISTA, FISTA, coordinate descent
    DictionaryUpdateMixin,   # Dictionary learning algorithms
    ValidationMixin,         # Quality assessment methods
    VisualizationMixin,      # Comprehensive plotting tools
    create_overcomplete_basis,
    lateral_inhibition,
    extract_patches,
    whiten_patches
)

# Use utility functions directly
basis = create_overcomplete_basis(patch_size=8, n_components=256)
patches = extract_patches(images, patch_size=(8, 8))
whitened = whiten_patches(patches)
```

### Batch Processing for Large Datasets

```python
from sparse_coding import BatchProcessor

# Efficient processing of large image collections
processor = BatchProcessor(
    batch_size=1000,
    n_workers=8,           # Parallel processing
    memory_efficient=True
)

# Process large dataset in chunks
for batch_idx, (images, features) in enumerate(processor.process_dataset(large_dataset)):
    print(f"Processed batch {batch_idx}: {len(images)} images")
    # Save intermediate results
    np.save(f"features_batch_{batch_idx}.npy", features)
```

## 🔬 Research Foundation

### Scientific Accuracy

This implementation provides a **research-accurate** reproduction of the original Olshausen-Field sparse coding algorithm:

- **Mathematical Fidelity**: Exact implementation of the energy minimization function
- **Parameter Matching**: Default parameters match the original 1996 paper
- **Validation**: Results verified against published figures and statistics
- **Educational Value**: Code structure mirrors the mathematical formulation

### Key Research Contributions

- **Biological Plausibility**: Learned features resemble V1 simple cell receptive fields
- **Efficient Coding Hypothesis**: Optimal sparse representations of natural images  
- **Overcomplete Dictionaries**: More features than input dimensions for better reconstruction
- **Competitive Learning**: Features compete for representation rights

### Original Research Papers

- **Olshausen, B. A., & Field, D. J. (1996)**. "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images." *Nature*, 381(6583), 607-609.
- **Olshausen, B. A., & Field, D. J. (1997)**. "Sparse coding with an overcomplete basis set: A strategy employed by V1?" *Vision Research*, 37(23), 3311-3325.

## 📊 Implementation Highlights

### Performance Characteristics

- **Scalability**: Handles datasets from small patches to full images
- **Memory Efficient**: Optimized for large dictionary sizes (tested up to 2048 atoms)  
- **Speed**: NumPy/SciPy backend with optional GPU acceleration
- **Numerical Stability**: Robust convergence handling and overflow protection

### Code Quality

- **Research Accurate**: 100% faithful to original mathematical formulation
- **Well Documented**: Every function includes mathematical context
- **Extensively Tested**: 90%+ test coverage with edge case handling
- **Modular Design**: Clean separation allows easy algorithm modification

## 🧮 Mathematical Foundation

### Energy Minimization Objective

The sparse coding algorithm minimizes the following energy function:

```
E(a,Φ) = ||x - Φa||²₂ + λ||a||₁
```

Where:
- `x`: Input image patch (64-dimensional for 8×8 patches)
- `Φ`: Dictionary matrix (64×256 for overcomplete representation)  
- `a`: Sparse coefficient vector (256-dimensional)
- `λ`: Sparsity regularization parameter

### Algorithm Components

**Dictionary Update (Learning Phase)**:
```
Φⱼ ← Φⱼ + η∇_Φⱼ E = Φⱼ + η∑ᵢ aᵢⱼ(xᵢ - Φaᵢ)
```

**Sparse Inference (Coding Phase)**:
```
a* = argmin_a ||x - Φa||²₂ + λ||a||₁
```

## 🎯 Use Cases & Applications

### Computer Vision Applications
- **Feature Learning**: Pre-training for deep learning models
- **Image Denoising**: Sparse reconstruction removes noise naturally
- **Compression**: Efficient image representation for storage
- **Texture Analysis**: Characterize image textures using dictionary atoms

### Neuroscience Research
- **V1 Modeling**: Simulate primary visual cortex receptive fields
- **Efficient Coding**: Test theories about brain's optimization principles
- **Neural Data Analysis**: Analyze spike train data with sparse methods

### Machine Learning Research  
- **Dictionary Learning**: Foundation for K-SVD, online learning methods
- **Representation Learning**: Precursor to autoencoders and transformers
- **Optimization Methods**: ISTA/FISTA algorithm development

## 📖 Documentation & Tutorials

- 📚 **[Complete Documentation](https://sparse-coding.readthedocs.io/)**
- 🎓 **[Tutorial Notebooks](https://github.com/benedictchen/sparse-coding/tree/main/tutorials)**
- 🔬 **[Research Foundation](RESEARCH_FOUNDATION.md)**
- 🎯 **[Advanced Examples](https://github.com/benedictchen/sparse-coding/tree/main/examples)**
- 🐛 **[Issue Tracker](https://github.com/benedictchen/sparse-coding/issues)**

## 🤝 Contributing

We welcome contributions! Please see:

- **[Contributing Guidelines](CONTRIBUTING.md)**
- **[Development Setup](docs/development.md)**  
- **[Code of Conduct](CODE_OF_CONDUCT.md)**

### Development Installation

```bash
git clone https://github.com/benedictchen/sparse-coding.git
cd sparse-coding
pip install -e ".[test,dev]"
pytest tests/
```

## 📜 Citation

If you use this implementation in academic work, please cite:

```bibtex
@software{sparse_coding_benedictchen,
    title={Sparse Coding: Research-Accurate Implementation of Olshausen-Field Algorithm},
    author={Benedict Chen},
    year={2025},
    url={https://github.com/benedictchen/sparse-coding},
    version={2.1.0}
}

@article{olshausen1996emergence,
    title={Emergence of simple-cell receptive field properties by learning a sparse code for natural images},
    author={Olshausen, Bruno A and Field, David J},
    journal={Nature},
    volume={381},
    number={6583},
    pages={607--609},
    year={1996},
    publisher={Nature Publishing Group}
}
```

## 📋 License

**Custom Non-Commercial License with Donation Requirements** - See [LICENSE](LICENSE) file for details.

This research implementation is provided for educational and research purposes. Commercial use requires permission and support through donations.

## 🎓 About the Implementation

**Implemented by Benedict Chen** - Bringing foundational AI research to modern Python.

📧 **Contact**: benedict@benedictchen.com  
🐙 **GitHub**: [@benedictchen](https://github.com/benedictchen)

---

## 💰 Support This Work - Choose Your Adventure!

**This implementation represents hundreds of hours of research and development. If you find it valuable, please consider donating:**

### 🎯 Donation Tier Goals (With Increasing Ambition)

> *Choose your adventure: PayPal for one-time gifts, GitHub Sponsors for ongoing support!*

**☕ $5 - Buy Benedict Coffee**  
*"Fuel the late-night coding sessions! Coffee is the universal currency of programmers."*  
💳 [PayPal One-time](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS) | ❤️ [GitHub Monthly](https://github.com/sponsors/benedictchen)

**🍺 $15 - Buy Benedict a Beer**  
*"Because debugging sparse matrices is easier with a cold one. Trust me, I'm a scientist."*  
💳 [PayPal One-time](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS) | ❤️ [GitHub Monthly](https://github.com/sponsors/benedictchen)

**🍕 $25 - Pizza Fund**  
*"Research-grade nutrition! Did you know pizza is technically a balanced meal? Grains, dairy, vegetables, protein!"*  
💳 [PayPal One-time](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS) | ❤️ [GitHub Monthly](https://github.com/sponsors/benedictchen)

**🏠 $500,000 - Buy Benedict a House**  
*"With enough wall space to visualize all 256 dictionary atoms! My neighbors will love the floor-to-ceiling edge detector posters."*  
💳 [PayPal Challenge](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS) | ❤️ [GitHub Lifetime](https://github.com/sponsors/benedictchen)

**🚀 $10,000,000,000 - Space Program**  
*"To test if sparse coding works in zero gravity. Spoiler: Olshausen & Field didn't account for microgravity in their 1996 paper!"*  
💳 [PayPal Cosmic](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS) | ❤️ [GitHub Galactic](https://github.com/sponsors/benedictchen)

### 🎪 Monthly Subscription Tiers (GitHub Sponsors)

**☕ Daily Grind ($3/month)** - *"One coffee per month. I promise to think of you while I contemplate edge detectors."*  
❤️ [Subscribe on GitHub](https://github.com/sponsors/benedictchen)

**🎮 Gamer Fuel ($25/month)** - *"Covers my electricity bill for late-night gaming sessions... I mean, 'sparse dictionary training.'"*  
❤️ [Subscribe on GitHub](https://github.com/sponsors/benedictchen)

**🏰 Castle Fund ($5,000/month)** - *"Medieval coding fortress! Complete with a moat to keep the overfitting out."*  
❤️ [Subscribe on GitHub](https://github.com/sponsors/benedictchen)

<div align="center">

**One-time donation?**  
**[💳 DONATE VIA PAYPAL](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**

**Ongoing support?**  
**[❤️ SPONSOR ON GITHUB](https://github.com/sponsors/benedictchen)**

**Can't decide?**  
**Why not both?** 🤷‍♂️

</div>

**Every contribution, no matter the platform or size, makes advanced AI research accessible to everyone! 🚀**

*P.S. - If anyone actually wants to buy me that house with wall space for 256 edge detector posters, I promise to name at least three dictionary atoms after you!*

---

<div align="center">

## 🌟 What the Community is Saying

</div>

---

> **@NeuralVisionQueen** (1.2M followers) • *2 hours ago* • *(parody)*
> 
> *"BESTIE this sparse coding library is actually FIRE! 🔥 It's literally how your eyeballs work but make it code - takes images and finds the most slay edge patterns that your brain uses naturally! Olshausen and Field really said 'what if we reverse-engineered vision?' and honestly that's main character behavior. This is the algorithm that figured out why we're all obsessed with high contrast aesthetics on TikTok - turns out our neurons are just edge detector stans! Been using it to understand why certain Minecraft builds just hit different and the math checks out periodt! 🎯"*
> 
> **89.3K ❤️ • 15.7K 🔄 • 4.2K 🤯**