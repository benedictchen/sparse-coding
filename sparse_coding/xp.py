try:
    import cupy as xp  # type: ignore
    GPU = True
except Exception:  # pragma: no cover
    import numpy as xp  # type: ignore
    GPU = False
