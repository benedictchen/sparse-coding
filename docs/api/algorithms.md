# Algorithm Implementations

## Core Optimization Algorithms

The sparse coding package provides multiple optimization algorithms for solving sparse representation problems.

### FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)

**Research Foundation**: Beck & Teboulle (2009) "A fast iterative shrinkage-thresholding algorithm for linear inverse problems"

- **Convergence Rate**: O(1/k²) for convex problems
- **Method**: Accelerated proximal gradient with Nesterov momentum
- **Best For**: L1 regularization problems
- **Implementation**: `sparse_coding.core.solver_implementations.FistaSolver`

**Mathematical Formulation**:
```
minimize (1/2)||X - DA||²_F + λ||A||₁
   A
```

**Algorithm Steps**:
1. Gradient step: `z_grad = z - (1/L)∇f(z)`
2. Proximal step: `a = prox_{t·ψ}(z_grad)`  
3. Momentum update: `z = a + β(a - a_old)`

### ISTA (Iterative Shrinkage-Thresholding Algorithm)

**Research Foundation**: Daubechies et al. (2004) "An iterative thresholding algorithm"

- **Convergence Rate**: O(1/k) for convex problems
- **Method**: Basic proximal gradient without acceleration
- **Best For**: Simple L1 problems, educational purposes
- **Implementation**: `sparse_coding.core.inference.ista_basic_solver`

### Nonlinear Conjugate Gradient (NCG)

**Research Foundation**: Polak & Ribière (1969) conjugate gradient methods

- **Convergence Rate**: Superlinear for strongly convex problems
- **Method**: Conjugate gradient with Polak-Ribière updates
- **Best For**: Smooth penalties (log-prior, elastic net)
- **Implementation**: `sparse_coding.core.inference.nonlinear_conjugate_gradient`

### Orthogonal Matching Pursuit (OMP)

**Research Foundation**: Pati et al. (1993) "Orthogonal matching pursuit"

- **Method**: Greedy pursuit algorithm
- **Best For**: Exact sparsity constraints
- **Guarantees**: Exact recovery under restricted isometry property
- **Implementation**: `sparse_coding.core.inference.orthogonal_matching_pursuit`

## Dictionary Learning Methods

### Method of Optimal Directions (MOD)

**Research Foundation**: Engan et al. (1999) "Method of optimal directions for frame design"

- **Update Rule**: `D = XA^T(AA^T)^(-1)`
- **Properties**: Globally optimal given fixed sparse codes
- **Advantages**: Closed-form solution, fast convergence
- **Implementation**: `sparse_coding.core.dictionary.method_optimal_directions`

### K-SVD Dictionary Learning

**Research Foundation**: Aharon et al. (2006) "K-SVD: An algorithm for designing overcomplete dictionaries"

- **Method**: SVD-based atom-by-atom updates
- **Properties**: Sparse representation awareness
- **Advantages**: Better sparsity preservation than MOD
- **Implementation**: `sparse_coding.core.dictionary.ksvd_dictionary_learning`

### Online Dictionary Learning

**Research Foundation**: Mairal et al. (2010) "Online learning for matrix factorization"

- **Method**: Stochastic gradient descent on mini-batches
- **Properties**: Scalable to large datasets
- **Advantages**: Memory efficient, streaming data support
- **Implementation**: `sparse_coding.core.dictionary.online_dictionary_learning`

## Penalty Functions

### L1 Penalty (Lasso)

**Mathematical Form**: `ψ(a) = λ||a||₁ = λ∑|aᵢ|`

**Proximal Operator**: Soft thresholding
```python
prox(z, t) = sign(z) ⊙ max(|z| - tλ, 0)
```

**Properties**:
- Convex and non-smooth
- Promotes sparsity
- Efficient proximal operator

### Log Penalty (Olshausen & Field)

**Mathematical Form**: `ψ(a) = λ∑log(1 + aᵢ²/σ²)`

**Gradient**: `∇ψ(a) = λ · 2a/(σ² + a²)`

**Properties**:
- Non-convex but smooth
- Biological motivation
- Better sparse solutions for natural images

### Elastic Net

**Mathematical Form**: `ψ(a) = λ₁||a||₁ + λ₂||a||₂²`

**Properties**:
- Combines L1 and L2 penalties
- Handles correlated features
- Grouped variable selection

## Algorithm Selection Guide

| Problem Type | Recommended Algorithm | Convergence Rate | Memory Usage |
|--------------|----------------------|------------------|--------------|
| L1 regularization | FISTA | O(1/k²) | Low |
| Smooth penalties | NCG | Superlinear | Medium |
| Large datasets | Online methods | - | Low |
| Exact sparsity | OMP | Finite steps | Low |
| Research accuracy | Olshausen methods | Linear | Medium |

## Mathematical Properties

All algorithms implement proper mathematical guarantees:

- **FISTA**: Proven O(1/k²) convergence for convex problems
- **NCG**: Superlinear convergence for strongly convex objectives
- **MOD**: Globally optimal dictionary atoms given fixed sparse codes
- **Homeostatic**: Activity balancing prevents dead or overactive atoms

## Implementation Notes

### Numerical Stability
- All solvers include condition number checking
- Gradient clipping prevents optimization explosions
- Adaptive step sizes ensure robust convergence

### Performance Optimization
- Vectorized operations for batch processing
- Parallel solver execution for large datasets
- Memory-efficient sparse matrix support

### Research Accuracy
- Direct implementations from original papers
- Mathematical formulations verified against literature
- Extensive validation against reference implementations