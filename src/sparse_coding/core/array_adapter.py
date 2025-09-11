"""
Real backend-agnostic array operations with unified API.

Fixes the broken backend abstraction by providing a proper adapter layer
that translates between different array API conventions.
"""

from typing import Any, Union, Optional, Tuple
import warnings
import numpy as np

ArrayLike = Any


class BackendAdapter:
    """Unified array operations that work across NumPy/PyTorch/JAX/CuPy."""
    
    def __init__(self, backend_name: str, backend_module: Any):
        self.name = backend_name
        self.module = backend_module
        
    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Matrix multiplication: a @ b"""
        if hasattr(self.module, 'matmul'):
            return self.module.matmul(a, b)
        return a @ b
    
    def sum(self, x: ArrayLike, axis: Optional[Union[int, Tuple[int, ...]]] = None, 
            keepdims: bool = False) -> ArrayLike:
        """Sum with unified axis/dim parameter handling."""
        if self.name == 'torch':
            # PyTorch uses 'dim' instead of 'axis'
            if axis is None:
                return self.module.sum(x)
            else:
                return self.module.sum(x, dim=axis, keepdim=keepdims)
        else:
            # NumPy/JAX/CuPy use 'axis'
            return self.module.sum(x, axis=axis, keepdims=keepdims)
    
    def norm(self, x: ArrayLike, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdims: bool = False, ord: Optional[Union[int, float, str]] = None) -> ArrayLike:
        """Vector/matrix norm with backend-specific handling."""
        if self.name == 'torch':
            # PyTorch: torch.linalg.norm uses 'dim' and 'keepdim'
            if hasattr(self.module, 'linalg') and hasattr(self.module.linalg, 'norm'):
                kwargs = {}
                if axis is not None:
                    kwargs['dim'] = axis
                if ord is not None:
                    kwargs['ord'] = ord
                kwargs['keepdim'] = keepdims
                return self.module.linalg.norm(x, **kwargs)
            else:
                # Fallback for older PyTorch
                if axis is None:
                    return self.module.sqrt(self.module.sum(x * x))
                else:
                    return self.module.sqrt(self.module.sum(x * x, dim=axis, keepdim=keepdims))
        else:
            # NumPy/JAX/CuPy
            if hasattr(self.module, 'linalg') and hasattr(self.module.linalg, 'norm'):
                kwargs = {}
                if axis is not None:
                    kwargs['axis'] = axis
                if ord is not None:
                    kwargs['ord'] = ord
                kwargs['keepdims'] = keepdims
                return self.module.linalg.norm(x, **kwargs)
            else:
                # Fallback L2 norm
                return self.module.sqrt(self.module.sum(x * x, axis=axis, keepdims=keepdims))
    
    def spectral_norm(self, x: ArrayLike) -> ArrayLike:
        """Matrix spectral norm (largest singular value)."""
        if self.name == 'torch':
            # PyTorch: use matrix_norm for spectral norm
            if hasattr(self.module, 'linalg') and hasattr(self.module.linalg, 'matrix_norm'):
                return self.module.linalg.matrix_norm(x, ord=2)
            else:
                # Fallback: SVD
                _, s, _ = self.svd(x, full_matrices=False)
                return s[0]
        else:
            # NumPy/JAX/CuPy: use ord=2 for spectral norm
            if hasattr(self.module, 'linalg') and hasattr(self.module.linalg, 'norm'):
                return self.module.linalg.norm(x, ord=2)
            else:
                # Fallback: SVD
                _, s, _ = self.svd(x, full_matrices=False)
                return s[0]
    
    def solve(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Linear system solve: a @ x = b."""
        if hasattr(self.module, 'linalg') and hasattr(self.module.linalg, 'solve'):
            return self.module.linalg.solve(a, b)
        else:
            # Convert to NumPy, solve, convert back
            a_np = self._to_numpy(a)
            b_np = self._to_numpy(b)
            result_np = np.linalg.solve(a_np, b_np)
            return self._from_numpy(result_np, like=a)
    
    def solve_triangular(self, a: ArrayLike, b: ArrayLike, lower: bool = False) -> ArrayLike:
        """Triangular system solve with backend-specific implementations."""
        if self.name == 'torch':
            # PyTorch: use triangular_solve
            if hasattr(self.module, 'triangular_solve'):
                # Older PyTorch API
                solution, _ = self.module.triangular_solve(b, a, upper=not lower)
                return solution
            elif hasattr(self.module, 'linalg') and hasattr(self.module.linalg, 'solve_triangular'):
                # Newer PyTorch API
                return self.module.linalg.solve_triangular(a, b, upper=not lower)
            else:
                # Fallback to regular solve
                return self.solve(a, b)
        elif self.name in ['jax', 'jax.numpy']:
            # JAX: use jax.scipy.linalg.solve_triangular
            try:
                import jax.scipy.linalg
                return jax.scipy.linalg.solve_triangular(a, b, lower=lower)
            except ImportError:
                return self.solve(a, b)
        else:
            # NumPy/CuPy: use scipy if available, fallback to regular solve
            try:
                import scipy.linalg
                a_np = self._to_numpy(a)
                b_np = self._to_numpy(b)
                result_np = scipy.linalg.solve_triangular(a_np, b_np, lower=lower)
                return self._from_numpy(result_np, like=a)
            except ImportError:
                return self.solve(a, b)
    
    def svd(self, x: ArrayLike, full_matrices: bool = True) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """SVD decomposition."""
        if hasattr(self.module, 'linalg') and hasattr(self.module.linalg, 'svd'):
            return self.module.linalg.svd(x, full_matrices=full_matrices)
        else:
            # Convert to NumPy, compute SVD, convert back
            x_np = self._to_numpy(x)
            u_np, s_np, vh_np = np.linalg.svd(x_np, full_matrices=full_matrices)
            return (self._from_numpy(u_np, like=x),
                    self._from_numpy(s_np, like=x),
                    self._from_numpy(vh_np, like=x))
    
    def eye(self, n: int, dtype: Optional[Any] = None) -> ArrayLike:
        """Identity matrix."""
        if dtype is None:
            return self.module.eye(n)
        else:
            return self.module.eye(n, dtype=dtype)
    
    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> ArrayLike:
        """Zero array."""
        if dtype is None:
            return self.module.zeros(shape)
        else:
            return self.module.zeros(shape, dtype=dtype)
    
    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> ArrayLike:
        """Ones array.""" 
        if dtype is None:
            return self.module.ones(shape)
        else:
            return self.module.ones(shape, dtype=dtype)
    
    def _to_numpy(self, x: ArrayLike) -> np.ndarray:
        """Convert array to NumPy."""
        if hasattr(x, 'numpy'):  # PyTorch, JAX
            return x.numpy()
        elif hasattr(x, 'get'):  # CuPy
            return x.get()
        else:
            return np.asarray(x)
    
    def _from_numpy(self, x: np.ndarray, like: ArrayLike) -> ArrayLike:
        """Convert NumPy array to same backend as 'like'."""
        if self.name == 'torch':
            return self.module.from_numpy(x)
        elif self.name == 'cupy':
            return self.module.asarray(x)
        elif self.name in ['jax', 'jax.numpy']:
            return self.module.asarray(x)
        else:
            return x


def get_backend_adapter(arr: ArrayLike) -> BackendAdapter:
    """Get backend adapter for array."""
    if arr is None:
        return BackendAdapter('numpy', np)
    
    # Try Array API standard first
    if hasattr(arr, "__array_namespace__"):
        try:
            ns = arr.__array_namespace__()
            backend_name = getattr(ns, '__name__', str(type(ns)))
            return BackendAdapter(backend_name, ns)
        except Exception:
            pass
    
    # Backend detection
    arr_type = str(type(arr).__module__)
    
    if "torch" in arr_type:
        try:
            import torch
            return BackendAdapter('torch', torch)
        except ImportError:
            raise ImportError("PyTorch not available")
    elif "cupy" in arr_type:
        try:
            import cupy
            return BackendAdapter('cupy', cupy)
        except ImportError:
            raise ImportError("CuPy not available")
    elif "jax" in arr_type:
        try:
            import jax.numpy as jnp
            return BackendAdapter('jax.numpy', jnp)
        except ImportError:
            raise ImportError("JAX not available")
    else:
        return BackendAdapter('numpy', np)


def convert_array(x: ArrayLike, like: ArrayLike) -> ArrayLike:
    """Convert array x to same backend as 'like' with proper error handling."""
    if x is None or like is None:
        return x
    
    if type(x) is type(like):
        return x
    
    source_adapter = get_backend_adapter(x)
    target_adapter = get_backend_adapter(like)
    
    if source_adapter.name == target_adapter.name:
        return x
    
    # Use adapters for conversion
    if target_adapter.name == 'torch':
        x_np = source_adapter._to_numpy(x)
        return target_adapter.module.from_numpy(x_np)
    elif target_adapter.name == 'cupy':
        x_np = source_adapter._to_numpy(x)
        return target_adapter.module.asarray(x_np)
    elif target_adapter.name in ['jax', 'jax.numpy']:
        x_np = source_adapter._to_numpy(x)
        return target_adapter.module.asarray(x_np)
    else:  # numpy
        return source_adapter._to_numpy(x)


# Unified API functions
def matmul(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Backend-agnostic matrix multiplication."""
    adapter = get_backend_adapter(a)
    return adapter.matmul(a, b)


def sum_array(x: ArrayLike, axis: Optional[Union[int, Tuple[int, ...]]] = None,
              keepdims: bool = False) -> ArrayLike:
    """Backend-agnostic sum."""
    adapter = get_backend_adapter(x)
    return adapter.sum(x, axis=axis, keepdims=keepdims)


def norm(x: ArrayLike, axis: Optional[Union[int, Tuple[int, ...]]] = None,
         keepdims: bool = False, ord: Optional[Union[int, float, str]] = None) -> ArrayLike:
    """Backend-agnostic norm."""
    adapter = get_backend_adapter(x)
    return adapter.norm(x, axis=axis, keepdims=keepdims, ord=ord)


def spectral_norm(x: ArrayLike) -> ArrayLike:
    """Backend-agnostic spectral norm."""
    adapter = get_backend_adapter(x)
    return adapter.spectral_norm(x)


def solve(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Backend-agnostic linear solve."""
    adapter = get_backend_adapter(a)
    return adapter.solve(a, b)


def solve_triangular(a: ArrayLike, b: ArrayLike, lower: bool = False) -> ArrayLike:
    """Backend-agnostic triangular solve."""
    adapter = get_backend_adapter(a)
    return adapter.solve_triangular(a, b, lower=lower)


def svd(x: ArrayLike, full_matrices: bool = True) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Backend-agnostic SVD."""
    adapter = get_backend_adapter(x)
    return adapter.svd(x, full_matrices=full_matrices)


def eye(n: int, dtype: Optional[Any] = None, like: Optional[ArrayLike] = None) -> ArrayLike:
    """Backend-agnostic identity matrix."""
    if like is not None:
        adapter = get_backend_adapter(like)
        return adapter.eye(n, dtype=dtype)
    else:
        return np.eye(n, dtype=dtype)


def zeros(shape: Tuple[int, ...], dtype: Optional[Any] = None, 
          like: Optional[ArrayLike] = None) -> ArrayLike:
    """Backend-agnostic zeros."""
    if like is not None:
        adapter = get_backend_adapter(like)
        return adapter.zeros(shape, dtype=dtype)
    else:
        return np.zeros(shape, dtype=dtype)