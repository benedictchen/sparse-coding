"""
Backend-agnostic array operations using Array API standard.

Supports NumPy, CuPy, PyTorch, JAX with zero-copy bridges via DLPack.
Gracefully falls back to NumPy for unsupported backends.
"""

from typing import Any, Union, Optional
import warnings

ArrayLike = Any  # Union[np.ndarray, torch.Tensor, jax.Array, cupy.ndarray, ...]


def xp(arr: ArrayLike):
    """
    Get array namespace following Array API standard.
    
    Prefers __array_namespace__ (NumPy 2+, PyTorch 2.3+, JAX) over backend detection.
    Falls back to NumPy for compatibility.
    
    Args:
        arr: Array-like object
        
    Returns:
        Array namespace (numpy, torch, jax.numpy, cupy, etc.)
    """
    import numpy as np
    
    # Handle None case
    if arr is None:
        return np
    
    # Try Array API standard first (future-proof)
    if hasattr(arr, "__array_namespace__"):
        try:
            ns_func = getattr(arr, "__array_namespace__")
            if callable(ns_func):
                return ns_func()
        except Exception:
            pass
    
    # Backend detection for current ecosystem
    arr_type = str(type(arr).__module__)
    
    if "torch" in arr_type:
        try:
            import torch
            return torch
        except ImportError:
            pass
    elif "cupy" in arr_type:
        try:
            import cupy
            return cupy
        except ImportError:
            pass
    elif "jax" in arr_type:
        try:
            import jax.numpy
            return jax.numpy
        except ImportError:
            pass
    
    # Default fallback
    return np


def as_same(x: ArrayLike, like: ArrayLike) -> ArrayLike:
    """
    Convert array x to same backend as 'like' using DLPack when possible.
    
    Enables zero-copy conversion between compatible backends (PyTorch/CuPy/JAX).
    Falls back to explicit conversion for unsupported cases.
    
    Args:
        x: Source array
        like: Target array (determines output backend)
        
    Returns:
        Array x converted to same backend as 'like'
    """
    if type(x) is type(like):
        return x
    
    # Try DLPack for zero-copy conversion
    try:
        if hasattr(like, "__dlpack__") and hasattr(x, "__dlpack__"):
            # Check device compatibility
            x_device = getattr(x, "__dlpack_device__", lambda: None)()
            like_device = getattr(like, "__dlpack_device__", lambda: None)()
            
            if x_device and like_device and x_device == like_device:
                # Same device - can use DLPack
                if hasattr(like, "from_dlpack"):
                    return like.from_dlpack(x.__dlpack__())
    except Exception:
        # DLPack failed, fall back to explicit conversion
        pass
    
    # Explicit backend conversion
    try:
        like_type = type(like)
        
        # PyTorch conversions
        if "torch" in like_type.__module__:
            import torch
            if hasattr(x, "numpy"):  # CuPy, JAX arrays
                return torch.from_numpy(x.numpy())
            elif hasattr(x, "detach"):  # Already torch
                return x
            else:  # NumPy
                return torch.from_numpy(x)
        
        # CuPy conversions  
        elif "cupy" in like_type.__module__:
            import cupy as cp
            if hasattr(x, "get"):  # Already CuPy
                return x
            else:
                return cp.asarray(x)
        
        # JAX conversions
        elif "jax" in like_type.__module__:
            import jax.numpy as jnp
            return jnp.asarray(x)
        
        # NumPy (default)
        else:
            import numpy as np
            if hasattr(x, "numpy"):  # PyTorch, JAX
                return x.numpy()
            elif hasattr(x, "get"):  # CuPy
                return x.get()
            else:
                return np.asarray(x)
                
    except Exception as e:
        warnings.warn(f"Backend conversion failed: {e}. Returning original array.")
        return x


def ensure_array(x: Any, dtype: Optional[Any] = None) -> ArrayLike:
    """
    Ensure input is an array, converting if necessary.
    
    Args:
        x: Input data (scalar, list, array, etc.)
        dtype: Optional dtype for conversion
        
    Returns:
        Array representation of x
    """
    # If already array-like with required interface
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        if dtype is None or x.dtype == dtype:
            return x
    
    # Get appropriate backend
    if hasattr(x, "__array_namespace__") or any(
        backend in str(type(x).__module__) 
        for backend in ["torch", "cupy", "jax"]
    ):
        backend_ns = xp(x)
        return backend_ns.asarray(x, dtype=dtype) if dtype else backend_ns.asarray(x)
    
    # Default to NumPy
    import numpy as np
    return np.asarray(x, dtype=dtype) if dtype else np.asarray(x)


def to_device(arr: ArrayLike, device: Optional[str] = None) -> ArrayLike:
    """
    Move array to specified device (GPU/CPU).
    
    Args:
        arr: Input array
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
        
    Returns:
        Array on target device
    """
    if device is None:
        return arr
    
    arr_type = type(arr).__module__
    
    try:
        if "torch" in arr_type:
            return arr.to(device)
        elif "cupy" in arr_type and device.startswith("cuda"):
            # CuPy arrays are already on GPU
            return arr
        elif "cupy" in arr_type and device == "cpu":
            return arr.get()  # CuPy -> NumPy
        # JAX device placement would require more complex handling
        else:
            if device != "cpu":
                warnings.warn(f"Device '{device}' not supported for {type(arr)}. Using CPU.")
            return arr
    except Exception as e:
        warnings.warn(f"Device transfer failed: {e}. Using original array.")
        return arr


def get_array_info(arr: ArrayLike) -> dict:
    """
    Get diagnostic information about an array.
    
    Args:
        arr: Input array
        
    Returns:
        Dict with backend, device, shape, dtype info
    """
    info = {
        "backend": type(arr).__module__.split('.')[0],
        "type": type(arr).__name__,
        "shape": getattr(arr, "shape", None),
        "dtype": getattr(arr, "dtype", None),
        "device": "unknown"
    }
    
    # Device detection
    try:
        if hasattr(arr, "device"):
            info["device"] = str(arr.device)
        elif "cupy" in info["backend"]:
            info["device"] = f"cuda:{arr.device.id}"
        else:
            info["device"] = "cpu"
    except Exception:
        pass
    
    return info


# Common array operations that work across backends
def matmul(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Backend-agnostic matrix multiplication."""
    backend = xp(a)
    return backend.matmul(a, b) if hasattr(backend, 'matmul') else a @ b


def norm(x: ArrayLike, axis: Optional[Union[int, tuple]] = None, keepdims: bool = False) -> ArrayLike:
    """Backend-agnostic vector/matrix norm."""
    backend = xp(x)
    if hasattr(backend, 'linalg') and hasattr(backend.linalg, 'norm'):
        return backend.linalg.norm(x, axis=axis, keepdims=keepdims)
    else:
        # Fallback L2 norm
        return backend.sqrt(backend.sum(x*x, axis=axis, keepdims=keepdims))


def solve(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Backend-agnostic linear system solve."""
    backend = xp(a)
    if hasattr(backend, 'linalg') and hasattr(backend.linalg, 'solve'):
        return backend.linalg.solve(a, b)
    else:
        # Should not happen with major backends, but safety fallback
        import numpy as np
        return np.linalg.solve(as_same(a, np.array([])), as_same(b, np.array([])))


def svd(x: ArrayLike, full_matrices: bool = True) -> tuple:
    """Backend-agnostic SVD decomposition."""
    backend = xp(x)
    if hasattr(backend, 'linalg') and hasattr(backend.linalg, 'svd'):
        return backend.linalg.svd(x, full_matrices=full_matrices)
    else:
        import numpy as np
        x_np = as_same(x, np.array([]))
        u, s, vh = np.linalg.svd(x_np, full_matrices=full_matrices)
        return as_same(u, x), as_same(s, x), as_same(vh, x)