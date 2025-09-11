# Examples

## Basic Dictionary Learning

```python
import numpy as np
from sparse_coding import SparseCoder

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(64, 1000)

# Learn dictionary
sc = SparseCoder(n_atoms=128, mode='l1', lam=0.1)
sc.fit(X, n_steps=20)

# Encode signals
codes = sc.encode(X[:, :100])
print(f"Sparsity: {(codes == 0).mean():.2f}")

# Reconstruct
reconstructed = sc.decode(codes)
error = np.linalg.norm(X[:, :100] - reconstructed) / np.linalg.norm(X[:, :100])
print(f"Reconstruction error: {error:.4f}")
```

## Algorithm Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from sparse_coding import SparseCoder

# Test data
X = np.random.randn(32, 500)

# Compare different modes
modes = ['l1', 'paper', 'log']
results = {}

for mode in modes:
    sc = SparseCoder(n_atoms=64, mode=mode, lam=0.1, max_iter=100)
    sc.fit(X, n_steps=5)
    codes = sc.encode(X[:, :50])
    
    sparsity = (codes == 0).mean()
    reconstruction = sc.decode(codes)
    error = np.linalg.norm(X[:, :50] - reconstruction) / np.linalg.norm(X[:, :50])
    
    results[mode] = {'sparsity': sparsity, 'error': error}
    print(f"{mode}: sparsity={sparsity:.3f}, error={error:.4f}")
```

## Natural Image Processing

```python
import numpy as np
from sparse_coding import SparseCoder

def extract_patches(image, patch_size=8, stride=4):
    """Extract overlapping patches from image."""
    h, w = image.shape
    patches = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size].flatten()
            patches.append(patch)
    return np.array(patches).T

# Example with synthetic image data
# In practice, use real images: image = plt.imread('image.jpg')
image = np.random.randn(64, 64)
patches = extract_patches(image, patch_size=8)

# Learn dictionary on image patches
sc = SparseCoder(n_atoms=144, mode='log', lam=0.05)  # Olshausen & Field settings
sc.fit(patches, n_steps=50)

print(f"Learned dictionary shape: {sc.dictionary.shape}")
print(f"Dictionary covers {sc.dictionary.shape[1]} basis functions")
```

## Monitoring Training Progress

```python
import numpy as np
from sparse_coding import SparseCoder
from sparse_coding.sparse_coding_monitoring import CSVDump

# Setup monitoring
logger = CSVDump("training_log.csv")
X = np.random.randn(64, 1000)

# Train with monitoring
sc = SparseCoder(n_atoms=128, mode='l1', lam=0.1)
sc.fit(X, n_steps=30)

# Analysis after training
codes = sc.encode(X)
sparsity = (codes == 0).mean()
reconstruction_error = np.linalg.norm(X - sc.decode(codes)) / np.linalg.norm(X)

print(f"Final sparsity: {sparsity:.3f}")
print(f"Final reconstruction error: {reconstruction_error:.4f}")
```

## Advanced: Custom Lambda Annealing

```python
import numpy as np
from sparse_coding import SparseCoder

# Lambda annealing for better convergence
X = np.random.randn(64, 1000)

# Start with high lambda, anneal to lower value
sc = SparseCoder(
    n_atoms=128, 
    mode='l1', 
    lam=0.5,  # Start high
    anneal=(0.9, 0.01)  # Decay by 0.9 each step, floor at 0.01
)

sc.fit(X, n_steps=50)
print(f"Final lambda: {sc.lam:.4f}")
```

## Performance Optimization

```python
import numpy as np
import time
from sparse_coding import SparseCoder

# Large dataset processing
X_large = np.random.randn(128, 5000)

# Optimize for speed
sc = SparseCoder(
    n_atoms=256,
    mode='l1',  # Fastest mode
    lam=0.1,
    max_iter=100,  # Reduce iterations for speed
    tol=1e-4       # Relax tolerance
)

start_time = time.time()
sc.fit(X_large, n_steps=10)
fit_time = time.time() - start_time

start_time = time.time()
codes = sc.encode(X_large[:, :1000])
encode_time = time.time() - start_time

print(f"Dictionary learning: {fit_time:.2f}s")
print(f"Encoding 1000 signals: {encode_time:.2f}s")
print(f"Throughput: {1000/encode_time:.1f} signals/second")
```

## Research Applications

### Sparse Feature Learning
```python
# Learn sparse features from data
sc = SparseCoder(n_atoms=256, mode='l1', lam=0.1)
sc.fit(training_data, n_steps=100)

# Extract features for classification
features = sc.encode(test_data)
# Use features in downstream ML pipeline
```

### Signal Denoising
```python
# Denoise signals using sparse representation
noisy_signal = clean_signal + 0.1 * np.random.randn(*clean_signal.shape)

# Learn dictionary on clean signals
sc = SparseCoder(n_atoms=128, mode='l1', lam=0.2)  # Higher lambda for denoising
sc.fit(clean_training_signals, n_steps=50)

# Denoise by sparse reconstruction
sparse_codes = sc.encode(noisy_signal)
denoised_signal = sc.decode(sparse_codes)
```

### Dictionary Visualization
```python
import matplotlib.pyplot as plt

# Visualize learned dictionary atoms
sc = SparseCoder(n_atoms=64, mode='l1')
sc.fit(image_patches, n_steps=30)

# Plot first 16 atoms as images
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    atom = sc.dictionary[:, i].reshape(8, 8)  # Reshape to patch size
    ax.imshow(atom, cmap='gray')
    ax.set_title(f'Atom {i}')
    ax.axis('off')

plt.tight_layout()
plt.show()
```