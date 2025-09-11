# Mathematical Foundations

## Research Papers

### Olshausen & Field (1996)
**"Emergence of simple-cell receptive field properties by learning a sparse code for natural images"**
*Nature, 381(6583), 607-609*

- Introduced sparse coding for natural image statistics
- Log-prior penalty: λ∑log(1 + a²/σ²)
- Gradient ascent optimization
- Biological plausibility for V1 neurons

### Beck & Teboulle (2009)
**"A fast iterative shrinkage-thresholding algorithm for linear inverse problems"**
*SIAM Journal on Imaging Sciences, 2(1), 183-202*

- FISTA algorithm with O(1/k²) convergence
- Proximal gradient method
- Momentum acceleration
- Optimal for L1 regularization

### Engan et al. (1999)
**"Method of optimal directions for frame design"**
*IEEE ICASSP, 5, 2443-2446*

- MOD dictionary update method
- Closed-form solution: D = XA^T(AA^T)^(-1)
- Superior to gradient-based updates
- Numerical stability considerations

## Mathematical Formulations

### Sparse Coding Problem
```
minimize  (1/2)||X - DA||²_F + λΨ(A)
   A,D
```

Where:
- X: Data matrix (features × samples)
- D: Dictionary (features × atoms)  
- A: Sparse codes (atoms × samples)
- Ψ(A): Sparsity penalty function

### Penalty Functions

#### L1 Penalty
```
ψ(a) = λ||a||₁ = λ∑|aᵢ|
```
Proximal operator: `prox(z,t) = sign(z) ⊙ max(|z| - tλ, 0)`

#### Log Penalty (Olshausen & Field)
```
ψ(a) = λ∑log(1 + aᵢ²/σ²)
```
Gradient: `∇ψ(a) = λ · 2a/(σ² + a²)`

### Optimization Algorithms

#### FISTA (Beck & Teboulle 2009)
```python
# Momentum parameter
t_{k+1} = (1 + √(1 + 4t_k²))/2

# Extrapolation  
y_k = a_k + ((t_k - 1)/t_{k+1})(a_k - a_{k-1})

# Proximal step
a_{k+1} = prox_{η·ψ}(y_k - η∇f(y_k))
```

#### NCG (Polak-Ribière)
```python
# Search direction
d_0 = -g_0
d_{k+1} = -g_{k+1} + β_k d_k

# Polak-Ribière parameter
β_k = max(0, (g_{k+1} - g_k)^T g_{k+1} / ||g_k||²)
```

## Convergence Theory

### FISTA Convergence Rate
- **Objective**: O(1/k²) for convex problems
- **Optimal**: Achieves lower bound for first-order methods
- **Practical**: Fast convergence in finite precision

### NCG Convergence
- **Superlinear**: For strongly convex problems
- **Robust**: Good performance on ill-conditioned problems
- **Adaptive**: Automatic restart mechanisms