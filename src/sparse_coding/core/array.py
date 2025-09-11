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
        import torch  # Will fail loudly if torch array passed but torch not installed
        return torch
    elif "cupy" in arr_type:
        import cupy  # Will fail loudly if cupy array passed but cupy not installed
        return cupy
    elif "jax" in arr_type:
        import jax.numpy  # Will fail loudly if jax array passed but jax not installed
        return jax.numpy
    
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
        
        # PyTorch conversions with device preservation
        if "torch" in like_type.__module__:
            import torch
            target_device = getattr(like, 'device', torch.device('cpu'))
            
            if hasattr(x, "detach"):  # Already torch
                return x.to(target_device)
            elif hasattr(x, "get"):  # CuPy array
                # CuPy → PyTorch preserving device
                if target_device.type == 'cuda' and hasattr(x, 'device'):
                    x_numpy = x.get()  # CPU copy for conversion
                    torch_tensor = torch.from_numpy(x_numpy)
                    return torch_tensor.to(target_device)  # Move to target GPU
                else:
                    return torch.from_numpy(x.get())
            elif "jax" in type(x).__module__:  # JAX array
                # JAX → PyTorch preserving device
                x_numpy = np.asarray(x)  # JAX to NumPy
                torch_tensor = torch.from_numpy(x_numpy)
                return torch_tensor.to(target_device)
            else:  # NumPy
                torch_tensor = torch.from_numpy(x)
                return torch_tensor.to(target_device)
        
        # CuPy conversions with device preservation
        elif "cupy" in like_type.__module__:
            import cupy as cp
            if hasattr(x, "get"):  # Already CuPy
                return x
            elif hasattr(x, "detach"):  # PyTorch tensor
                # PyTorch → CuPy preserving GPU device
                if x.device.type == 'cuda':
                    x_numpy = x.detach().cpu().numpy()  # Move to CPU first
                    return cp.asarray(x_numpy)  # CuPy will put on current GPU
                else:
                    return cp.asarray(x.detach().numpy())
            else:  # NumPy or JAX
                x_numpy = np.asarray(x)  # Ensure NumPy array
                return cp.asarray(x_numpy)
        
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
        elif "cupy" in arr_type:
            if device.startswith("cuda"):
                # CuPy arrays are already on GPU - check if we need to move between devices
                return arr
            elif device == "cpu":
                return arr.get()  # CuPy -> NumPy
            else:
                raise ValueError(f"CuPy arrays can only be moved to 'cpu' or 'cuda' devices, got '{device}'")
        elif "jax" in arr_type:
            # JAX device placement
            try:
                import jax
                if device == "cpu":
                    return jax.device_put(arr, jax.devices("cpu")[0])
                elif device.startswith("cuda") or device.startswith("gpu"):
                    gpu_devices = jax.devices("gpu")
                    if gpu_devices:
                        device_idx = 0
                        if ":" in device:
                            try:
                                device_idx = int(device.split(":")[-1])
                            except (ValueError, IndexError) as e:
                                raise ValueError(f"Invalid device specification '{device}'. Use format 'cuda:N' where N is device index")
                        if device_idx < len(gpu_devices):
                            return jax.device_put(arr, gpu_devices[device_idx])
                        else:
                            raise ValueError(f"JAX GPU device {device_idx} not available. Available devices: 0-{len(gpu_devices)-1}")
                    else:
                        raise ValueError("No JAX GPU devices available")
                else:
                    raise ValueError(f"JAX device '{device}' not supported. Use 'cpu', 'cuda', or 'gpu'")
            except ImportError:
                raise ImportError("JAX not available for device operations")
        elif "numpy" in arr_type or arr_type == "builtins":
            # NumPy arrays - CPU to GPU conversion
            if device == "cpu":
                return arr  # Already on CPU
            elif device.startswith("cuda") or device.startswith("gpu"):
                # Try to move to GPU using available backend
                try:
                    import cupy as cp
                    return cp.asarray(arr)
                except ImportError:
                    raise ImportError("CuPy not available for GPU operations")
            else:
                raise ValueError(f"Device '{device}' not supported for NumPy arrays")
        else:
            raise ValueError(f"Device operations not supported for {type(arr)} (module: {arr_type})")
    except Exception as e:
        # Fail fast instead of silent fallback for better debugging
        raise RuntimeError(f"Device transfer failed: {e}. Cannot move {type(arr)} to '{device}'")


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
    """
    Backend-agnostic linear system solve with safety checks.
    
    Solves the linear system ax = b for x.
    
    Args:
        a: Coefficient matrix (n, n)
        b: Right-hand side vector/matrix (n,) or (n, k)
        
    Returns:
        Solution x with same backend as input 'a'
        
    Raises:
        LinAlgError: If matrix is singular or poorly conditioned
        ValueError: If dimensions are incompatible
    """
    import numpy as np
    backend = xp(a)
    
    # Input validation
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if a_arr.ndim != 2 or a_arr.shape[0] != a_arr.shape[1]:
        raise ValueError(f"Matrix 'a' must be square, got shape {a_arr.shape}")
    if b_arr.shape[0] != a_arr.shape[0]:
        raise ValueError(f"Incompatible dimensions: a={a_arr.shape}, b={b_arr.shape}")
    
    # Check for non-finite values
    if not np.all(np.isfinite(a_arr)):
        raise ValueError("Matrix 'a' contains non-finite values (inf/nan)")
    if not np.all(np.isfinite(b_arr)):
        raise ValueError("Vector/matrix 'b' contains non-finite values (inf/nan)")
    
    # Check for degenerate cases
    if a_arr.shape[0] == 0:
        raise ValueError("Cannot solve system with empty matrix")
    
    if hasattr(backend, 'linalg') and hasattr(backend.linalg, 'solve'):
        try:
            return backend.linalg.solve(a, b)
        except Exception as e:
            # Fallback to NumPy if backend solve fails
            result = np.linalg.solve(a_arr, b_arr)
            # Try to convert back to original backend
            try:
                return as_same(result, a)
            except Exception:
                return result
    else:
        # Direct NumPy fallback with proper error handling
        result = np.linalg.solve(a_arr, b_arr)
        # Try to convert back to original backend 
        try:
            return as_same(result, a)
        except Exception:
            return result


def svd(x: ArrayLike, full_matrices: bool = True) -> tuple:
    """Backend-agnostic SVD decomposition."""
    backend = xp(x)
    if hasattr(backend, 'linalg') and hasattr(backend.linalg, 'svd'):
        return backend.linalg.svd(x, full_matrices=full_matrices)
    else:
        import numpy as np
        # Safe conversion: use np.asarray instead of empty array reference
        x_np = np.asarray(x)
        u, s, vh = np.linalg.svd(x_np, full_matrices=full_matrices)
        # Convert back to original backend if possible
        try:
            return as_same(u, x), as_same(s, x), as_same(vh, x)
        except Exception:
            # Fallback: return as NumPy arrays if conversion fails
            return u, s, vh