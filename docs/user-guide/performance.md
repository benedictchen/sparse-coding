# Performance Optimization

## Memory Management

### Dictionary Size Considerations
```python
# Memory usage scales as O(n_features × n_atoms)
n_features = 64
n_atoms = 256
memory_mb = (n_features * n_atoms * 8) / (1024**2)  # ~0.125 MB
print(f"Dictionary memory: {memory_mb:.3f} MB")
```

### Batch Processing for Large Datasets
```python
# Process large datasets in chunks
def process_large_dataset(X_large, chunk_size=1000):
    n_samples = X_large.shape[1]
    all_codes = []
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = X_large[:, start:end]
        codes = sc.encode(chunk)
        all_codes.append(codes)
    
    return np.hstack(all_codes)
```

### Sparse Matrix Support
```python
import scipy.sparse as sp

# Use sparse matrices for memory efficiency
X_sparse = sp.csr_matrix(X_dense)
codes_sparse = sc.encode(X_sparse)  # Automatic sparse handling
reconstructed = sc.decode(codes_sparse)
```

## Algorithm Performance

### Speed Comparison (Relative)
| Algorithm | Speed | Convergence | Memory |
|-----------|-------|-------------|--------|
| L1 (FISTA) | 100% | O(1/k²) | Low |
| Paper (NCG) | 80% | Superlinear | Medium |
| Olshausen Pure | 60% | Linear | Medium |
| Log Prior | 75% | Superlinear | Medium |

### Parallelization
```python
# Automatic parallel processing for large batches
sc = SparseCoder(n_atoms=128, mode='l1')

# Sequential processing (N ≤ 50)
X_small = np.random.randn(64, 30)
codes = sc.encode(X_small)  # Processed sequentially

# Parallel processing (N > 50)  
X_large = np.random.randn(64, 200)
codes = sc.encode(X_large)  # Automatic parallelization
```

## Parameter Tuning for Speed

### Fast Approximation
```python
# Reduce iterations for faster approximate solutions
sc = SparseCoder(
    n_atoms=128,
    mode='l1',
    max_iter=100,  # Reduced from default 200
    tol=1e-4       # Relaxed from default 1e-6
)
```

### Lambda Scaling
```python
# Auto-scaling reduces manual tuning
sc = SparseCoder(n_atoms=128, lam=None)  # Automatic scaling
sc.fit(X)

# Manual scaling for specific datasets
data_std = np.std(X)
optimal_lam = 0.1 * data_std
sc = SparseCoder(n_atoms=128, lam=optimal_lam)
```

## GPU Acceleration

### Array API Support
```python
# Works with CuPy for GPU acceleration
import cupy as cp

X_gpu = cp.asarray(X_cpu)
sc = SparseCoder(n_atoms=128, mode='l1')
sc.fit(X_gpu)  # GPU computation when available
```

### Device Management
```python
# Automatic device detection
from sparse_coding.core.array import to_device, get_array_info

X_cuda = to_device(X, 'cuda:0')  # Move to GPU
info = get_array_info(X_cuda)
print(f"Device: {info['device']}")
```

## Profiling and Monitoring

### Convergence Monitoring
```python
# Enable detailed convergence tracking
sc = SparseCoder(n_atoms=128, mode='log')
result = sc._solve_single_log(X[:, 0], sc.D)
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['iterations']}")
print(f"Final objective: {result['final_objective']:.6f}")
```

### Memory Profiling
```python
import psutil
import os

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # MB

print(f"Before training: {memory_usage():.1f} MB")
sc.fit(X)
print(f"After training: {memory_usage():.1f} MB")
```

## Best Practices

### Production Recommendations
1. **Use L1 mode** for production applications
2. **Set reasonable max_iter** (200-500) 
3. **Enable parallel processing** for large datasets
4. **Use sparse matrices** when appropriate
5. **Profile memory usage** for large dictionaries

### Research Recommendations  
1. **Use olshausen_pure** for exact reproduction
2. **Higher max_iter** (500-1000) for accuracy
3. **Tighter tolerance** (1e-8) for convergence
4. **Multiple runs** with different seeds
5. **Detailed convergence analysis**