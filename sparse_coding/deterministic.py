import os, random, numpy as np

def set_deterministic(seed: int = 0):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import mkl  # type: ignore
        mkl.set_num_threads(1)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
