# Utility Functions and Helpers

## Array Operations

### Array API Support

The sparse coding package provides utilities for handling different array backends and device management:

**Device Management**:
- `sparse_coding.core.array.to_device()` - Move arrays to specified devices (CPU/GPU)
- `sparse_coding.core.array.get_array_info()` - Get device and array metadata
- `sparse_coding.core.array.ensure_array()` - Ensure consistent array format

**Array Backend Support**:
- NumPy arrays (default)
- CuPy arrays (GPU acceleration)  
- Sparse matrices (scipy.sparse)
- Array API standard compatibility

### Data Validation

**Input Validation**:
- `sparse_coding.core.array.validate_input()` - Comprehensive input checking
- NaN/inf detection and handling
- Sparse matrix format validation
- Shape and dtype consistency checks

## Configuration Management

### Solver Configuration

**SolverConfig**: Type-safe configuration for optimization algorithms

```python
from sparse_coding.core.unified_config_system import SolverConfig

config = SolverConfig(
    solver_type='fista',
    max_iter=1000,
    tol=1e-6,
    adaptive_restart=True
)
```

**Features**:
- Type validation and checking
- Default parameter values
- Algorithm-specific configurations
- Validation on initialization

### Sparse Coding Configuration

**SparseCodingConfig**: Main configuration class for sparse coding problems

```python
from sparse_coding.sparse_coding_configuration import SparseCodingConfig

config = SparseCodingConfig(
    mode='l1',
    n_atoms=128,
    lam=0.1,
    max_iter=1000
)
```

## Monitoring and Logging

### TensorBoard Integration

**TB**: Real-time monitoring with TensorBoard

```python
from sparse_coding.sparse_coding_monitoring import TB

monitor = TB(log_dir='./logs')
monitor.log_scalar('loss', loss_value, step)
monitor.log_image('dictionary', D, step)
```

**Features**:
- Scalar logging (loss, sparsity, reconstruction error)
- Image logging (dictionaries, reconstructions)
- Histogram logging (sparse codes, gradients)
- Real-time visualization

### CSV Export

**CSVDump**: Export training metrics to CSV files

```python
from sparse_coding.sparse_coding_monitoring import CSVDump

logger = CSVDump(filename='training_metrics.csv')
logger.log_scalar('iteration', iteration)
logger.log_scalar('objective', objective_value)
```

**Features**:
- Structured data export
- Compatible with pandas/plotting tools
- Automatic timestamping
- Configurable column names

### Dashboard Logger

**DashboardLogger**: Multi-format logging with unified interface

```python
from sparse_coding.sparse_coding_monitoring import DashboardLogger

logger = DashboardLogger(
    monitors=[TB('./logs'), CSVDump('metrics.csv')]
)
logger.log_training_step(loss, codes, dictionary)
```

## Mathematical Utilities

### Convergence Analysis

**Convergence Checking**: Robust convergence detection with multiple criteria

```python
from sparse_coding.core.math_utils import check_convergence

converged = check_convergence(
    current_objective, 
    previous_objective, 
    tolerance=1e-6
)
```

**Features**:
- Relative and absolute tolerance checking
- Gradient norm convergence
- Objective function stagnation detection
- Numerical stability considerations

### Norm Calculations

**Safe Norm**: Numerically stable norm calculations

```python
from sparse_coding.core.math_utils import safe_norm

norm_value = safe_norm(data, ord=2, overflow_threshold=1e10)
```

**Features**:
- Overflow protection
- Multiple norm types (L1, L2, Frobenius)
- Sparse matrix support
- Automatic scaling for large values

## Data Processing

### Batch Processing

**Memory-Efficient Processing**: Handle large datasets with automatic chunking

```python
from sparse_coding.core.array import batch_process

results = batch_process(
    large_data, 
    processing_function, 
    chunk_size=1000
)
```

**Features**:
- Automatic memory management
- Progress tracking
- Error recovery
- Configurable chunk sizes

### Sparse Matrix Support

**Sparse Handling**: Transparent support for scipy.sparse matrices

```python
import scipy.sparse as sp
from sparse_coding.core.array import handle_sparse

# Automatic format detection and conversion
result = handle_sparse(sp.csr_matrix(data))
```

**Supported Formats**:
- CSR (Compressed Sparse Row)
- CSC (Compressed Sparse Column)  
- COO (Coordinate)
- Dense array fallback

## Performance Utilities

### Parallel Processing

**Parallel Encoding**: Joblib-based parallelization for sparse coding

```python
from sparse_coding.core.parallel import parallel_encode

codes = parallel_encode(
    dictionary, 
    data, 
    penalty, 
    n_jobs=-1  # Use all available cores
)
```

**Features**:
- Automatic load balancing
- Memory sharing for large dictionaries
- Progress monitoring
- Fault tolerance

### Memory Management

**Memory Estimation**: Predict memory usage for large problems

```python
from sparse_coding.core.memory import estimate_memory_usage

memory_mb = estimate_memory_usage(
    n_features=1024, 
    n_atoms=512, 
    n_samples=10000
)
print(f"Estimated memory: {memory_mb:.1f} MB")
```

## Usage Examples

### Basic Utility Usage
```python
import numpy as np
from sparse_coding.core.array import to_device, get_array_info
from sparse_coding.core.validation import validate_input

# Array device management
X_gpu = to_device(X_cpu, 'cuda:0')
info = get_array_info(X_gpu)
print(f"Device: {info['device']}, Shape: {info['shape']}")

# Input validation
try:
    validate_input(X, allow_sparse=True)
    print("✅ Input validation passed")
except ValueError as e:
    print(f"❌ Validation failed: {e}")
```

### Monitoring Setup
```python
from sparse_coding.sparse_coding_monitoring import TB, CSVDump

# Multi-format monitoring
monitors = [
    TB(log_dir='./logs'),
    CSVDump(filename='training_log.csv')
]

# Use in training loop
for epoch in range(n_epochs):
    # ... training code ...
    for monitor in monitors:
        monitor.log_scalar('loss', loss_value, epoch)
```

### Factory Pattern Usage
```python
from sparse_coding.factories.algorithm_factory import create_solver, create_penalty

# Type-safe algorithm creation
solver = create_solver('fista', max_iter=1000, tol=1e-6)
penalty = create_penalty('l1', lam=0.1)

# Automatic error handling and validation
try:
    codes = solver.solve(D, X, penalty)
except SparseCodingError as e:
    print(f"Algorithm error: {e}")
```