# Sparse Coding Methods

## Overview

This library implements multiple sparse coding algorithms, each with different trade-offs between speed, accuracy, and research fidelity.

## Algorithm Selection Guide

### L1 Regularization (Production Recommended)
```python
sc = SparseCoder(n_atoms=128, mode='l1', lam=0.1)
```

**Best for:**
- Production applications
- Fast convergence required
- When you need proven stability

**Algorithm:** FISTA (Beck & Teboulle 2009)
- O(1/k²) convergence rate
- Proximal gradient with momentum
- Optimal for first-order methods

### Log-Prior (Research Accurate)
```python
sc = SparseCoder(n_atoms=128, mode='log', lam=0.05)
```

**Best for:**
- Research reproduction
- Natural image statistics
- Biological plausibility studies

**Algorithm:** NCG + MOD (Olshausen & Field 1996 + Engan et al. 1999)
- Log penalty: λ∑log(1 + a²)
- Conjugate gradient optimization
- MOD dictionary updates

### Pure Olshausen (Historical Reference)
```python
sc = SparseCoder(n_atoms=128, mode='olshausen_pure', lam=0.05)
```

**Best for:**
- Exact reproduction of 1996 Nature paper
- Historical comparison studies
- Understanding original algorithm

**Algorithm:** Pure gradient ascent (Olshausen & Field 1996)
- Simple gradient descent on energy function
- Adaptive learning rate
- Homeostatic balancing

## Parameter Tuning

### Lambda (Sparsity Penalty)
- **Too small**: Dense representations, poor compression
- **Too large**: Over-sparse, poor reconstruction
- **Typical range**: 0.01 - 0.5
- **Auto-scaling**: Set `lam=None` for automatic scaling

### Number of Atoms
- **Undercomplete** (n_atoms < n_features): Dimensionality reduction
- **Complete** (n_atoms = n_features): Basis representation  
- **Overcomplete** (n_atoms > n_features): Sparse representation

### Convergence Parameters
- **max_iter**: 200-1000 (more for higher accuracy)
- **tol**: 1e-6 to 1e-4 (smaller for tighter convergence)

## Performance Considerations

### Memory Usage
- **Dictionary size**: O(n_features × n_atoms)
- **Codes**: O(n_atoms × n_samples)
- **Batch size**: Process in chunks for large datasets

### Computational Complexity
- **L1 mode**: O(n_atoms² × iterations) per sample
- **Log mode**: O(n_atoms² × iterations) per sample
- **Parallelization**: Automatic for large batches

## Common Issues and Solutions

### Poor Convergence
- Increase max_iter
- Decrease tolerance
- Check lambda scaling
- Try different mode

### Memory Problems
- Reduce n_atoms
- Process in smaller batches
- Use sparse matrices for large dictionaries

### Slow Performance
- Use L1 mode for speed
- Reduce max_iter for approximate solutions
- Enable parallel processing