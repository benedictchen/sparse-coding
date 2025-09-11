# Dictionary Learning

## Overview

Dictionary learning is the process of finding a set of basis functions (atoms) that can efficiently represent data through sparse linear combinations. This is the core training phase in sparse coding.

## Mathematical Foundation

Dictionary learning solves the bi-convex optimization problem:

```
minimize  (1/2)||X - DA||²_F + λΨ(A)
  D,A
```

Where:
- **X**: Data matrix (features × samples)
- **D**: Dictionary matrix (features × atoms)
- **A**: Sparse coefficient matrix (atoms × samples)
- **λ**: Sparsity penalty parameter
- **Ψ(A)**: Sparsity-inducing penalty function

## Algorithm Selection

### MOD (Method of Optimal Directions)

**Research Foundation**: Engan et al. (1999)

- **Update Rule**: `D = XA^T(AA^T)^(-1)`
- **Properties**: Globally optimal given fixed sparse codes
- **Best For**: Fast convergence, stable updates

```python
# MOD is used automatically in most modes
sc = SparseCoder(n_atoms=128, mode='l1')  # Uses MOD updates
sc.fit(X, n_steps=30)
```

### Gradient-Based Updates

**Research Foundation**: Olshausen & Field (1996)

- **Update Rule**: `D ← D + η(X - DA)A^T`
- **Properties**: Simple, biologically plausible
- **Best For**: Research reproduction, biological modeling

```python
# Gradient updates available in specific modes
sc = SparseCoder(n_atoms=128, mode='paper_gdD')  # Uses gradient updates
sc.fit(X, n_steps=50, lr=0.1)  # Learning rate for gradients
```

## Training Parameters

### Number of Steps (n_steps)

Controls the alternating optimization iterations:

```python
# Quick training (development)
sc.fit(X, n_steps=10)

# Standard training (production)
sc.fit(X, n_steps=30)

# High-quality training (research)
sc.fit(X, n_steps=100)
```

**Guidelines**:
- **10-20 steps**: Quick prototyping, small datasets
- **30-50 steps**: Production applications, balanced quality/speed
- **100+ steps**: Research applications, high accuracy needed

### Learning Rate (lr)

Only affects gradient-based dictionary updates:

```python
# Conservative learning (stable)
sc.fit(X, n_steps=30, lr=0.01)

# Standard learning (balanced)
sc.fit(X, n_steps=30, lr=0.1)

# Aggressive learning (fast but risky)
sc.fit(X, n_steps=30, lr=0.5)
```

**Guidelines**:
- **0.01-0.05**: Safe for sensitive applications
- **0.1-0.2**: Standard range for most problems
- **0.5+**: Only for well-conditioned problems

## Dictionary Initialization

### Data-Based Initialization

The default method samples atoms from training data:

```python
# Automatic data-based initialization
sc = SparseCoder(n_atoms=128)
sc.fit(X)  # Dictionary initialized from X columns
```

**Advantages**:
- Atoms start close to data manifold
- Fast initial convergence
- Works well for overcomplete dictionaries

### Custom Initialization

For specialized applications:

```python
# Custom dictionary initialization
import numpy as np

# Create custom initial dictionary
n_features, n_atoms = 64, 128
custom_dict = np.random.randn(n_features, n_atoms)
custom_dict = custom_dict / np.linalg.norm(custom_dict, axis=0)

# Set initial dictionary
sc = SparseCoder(n_atoms=n_atoms)
sc.dictionary = custom_dict
sc.fit(X, n_steps=30)
```

## Convergence and Monitoring

### Convergence Criteria

Dictionary learning uses multiple convergence indicators:

1. **Dictionary Stability**: Changes in dictionary atoms
2. **Sparse Code Consistency**: Changes in coefficient patterns
3. **Objective Function**: Overall energy decrease

### Dead Atom Handling

Atoms that become unused are automatically reinitialized:

```python
# Dead atoms are detected and re-initialized automatically
sc = SparseCoder(n_atoms=256)  # Large dictionary
sc.fit(X, n_steps=50)  # Some atoms may become dead and get re-initialized
```

**Detection Criteria**:
- Low activation frequency (< 1% of samples)
- Based on coefficient magnitude thresholds
- Tracked over multiple iterations

**Reinitialization Strategy**:
- Sample from data columns
- Add small random noise
- Normalize to unit norm

### Homeostatic Balancing

Prevents atom over-specialization:

```python
# Homeostatic balancing in gradient-based modes
sc = SparseCoder(n_atoms=128, mode='paper_gdD')
sc.fit(X, n_steps=50)  # Includes automatic homeostatic balancing
```

**Mechanism**:
- Measures atom usage frequency
- Scales atom magnitudes inversely to usage
- Promotes balanced atom utilization

## Advanced Features

### Lambda Annealing

Gradually reduce sparsity penalty during training:

```python
# Annealing configuration: (decay_factor, floor_value)
sc = SparseCoder(
    n_atoms=128,
    mode='l1',
    lam=0.5,  # Start with high sparsity
    anneal=(0.95, 0.01)  # Decay by 5% each step, floor at 0.01
)
sc.fit(X, n_steps=50)

print(f"Final lambda: {sc.lam:.4f}")
```

**Benefits**:
- Better convergence to sparse solutions
- Avoids local minima in early training
- Improves final solution quality

### Adaptive Atom Management

Dynamically adjust dictionary size during training:

```python
# Enable adaptive atom management
sc = SparseCoder(n_atoms=128)
sc.set_adaptive_k(enabled=True, max_atoms=256)
sc.fit(X, n_steps=50)

print(f"Final dictionary size: {sc.dictionary.shape[1]} atoms")
```

**Features**:
- Removes consistently unused atoms
- Adds atoms when all are heavily used
- Maintains efficiency while adapting to data complexity

## Mode-Specific Guidance

### L1 Mode (Production)

```python
sc = SparseCoder(n_atoms=128, mode='l1', lam=0.1)
sc.fit(X, n_steps=30)
```

**Characteristics**:
- FISTA inference + MOD dictionary updates
- Fast convergence with O(1/k²) rate
- Robust and numerically stable

### Paper Mode (Research)

```python
sc = SparseCoder(n_atoms=128, mode='paper', lam=0.05)
sc.fit(X, n_steps=50)
```

**Characteristics**:
- NCG log-prior inference + MOD updates
- Research-accurate sparse solutions
- Better for natural image statistics

### Olshausen Pure (Historical)

```python
sc = SparseCoder(n_atoms=144, mode='olshausen_pure', lam=0.05)
sc.fit(X, n_steps=100, lr=0.05)
```

**Characteristics**:
- Exact 1996 Nature paper algorithm
- Gradient ascent inference + gradient dictionary updates
- Includes homeostatic balancing

## Performance Optimization

### Memory Efficiency

```python
# For large dictionaries, process in chunks
X_large = np.random.randn(1024, 10000)

# Use smaller batch sizes to manage memory
sc = SparseCoder(n_atoms=512, mode='l1')
# Library automatically chunks large datasets
sc.fit(X_large, n_steps=20)
```

### Speed Optimization

```python
# Optimize for speed over accuracy
sc = SparseCoder(
    n_atoms=128,
    mode='l1',      # Fastest mode
    max_iter=50,    # Reduce inference iterations
    tol=1e-4        # Relax convergence tolerance
)
sc.fit(X, n_steps=20)  # Fewer dictionary update steps
```

### Parallel Processing

```python
# Parallel processing is automatic for large datasets
# No configuration needed - scales with available cores
large_X = np.random.randn(128, 5000)
sc = SparseCoder(n_atoms=256, mode='l1')
sc.fit(large_X, n_steps=30)  # Automatically uses parallel processing
```

## Troubleshooting

### Poor Convergence

**Symptoms**: High reconstruction error, unstable training

**Solutions**:
1. Reduce learning rate: `lr=0.01`
2. Increase training steps: `n_steps=100`
3. Use lambda annealing: `anneal=(0.95, 0.01)`
4. Try different mode: `mode='l1'` for stability

### Slow Training

**Symptoms**: Long training times

**Solutions**:
1. Reduce dictionary size: `n_atoms=64`
2. Use L1 mode: `mode='l1'`
3. Reduce max_iter: `max_iter=100`
4. Fewer training steps: `n_steps=20`

### Memory Issues

**Symptoms**: Out of memory errors

**Solutions**:
1. Smaller dictionaries: `n_atoms=128`
2. Process data in chunks
3. Use sparse matrices for large feature spaces
4. Reduce batch sizes

## Quality Assessment

### Dictionary Quality Metrics

```python
# Evaluate dictionary quality
codes = sc.encode(X)

# 1. Reconstruction quality
reconstruction = sc.decode(codes)
mse = np.mean((X - reconstruction) ** 2)
print(f"MSE: {mse:.6f}")

# 2. Sparsity level
sparsity = (codes == 0).mean()
print(f"Sparsity: {sparsity:.3f}")

# 3. Atom utilization
atom_usage = np.mean(np.abs(codes) > 1e-6, axis=1)
print(f"Dead atoms: {np.sum(atom_usage < 0.01)}")

# 4. Dictionary coherence
D_norm = sc.dictionary / np.linalg.norm(sc.dictionary, axis=0)
coherence = np.max(np.abs(D_norm.T @ D_norm) - np.eye(sc.n_atoms))
print(f"Max coherence: {coherence:.4f}")
```

### Research Validation

```python
# For research applications, validate against known results
if sc.mode == 'olshausen_pure':
    # Should reproduce Olshausen & Field (1996) results
    # Check for oriented edge filters in learned dictionary
    pass
```