import numpy as np

def zero_phase_whiten(image: np.ndarray, f0: float = 200.0) -> np.ndarray:
    """
    Zero-phase filter R(f) = |f| * exp(-(f/f0)^4) applied to a single image.
    Works on float images; returns zero-mean output.
    """
    x = np.asarray(image, float)
    H, W = x.shape
    fy = np.fft.fftfreq(H) * H
    fx = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)
    filt = R * np.exp(-(R / (f0 + 1e-12))**4)
    X = np.fft.fft2(x)
    Y = X * filt
    y = np.fft.ifft2(Y).real
    y = y - y.mean()
    return y / (y.std() + 1e-12)
