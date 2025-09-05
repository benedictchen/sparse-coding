"""
Zero-phase whitening filter implementation following Olshausen & Field (1996).
"""

import numpy as np

def zero_phase_whiten(image, f0=200.0, eps=1e-12):
    """
    Apply zero-phase whitening filter R(f) = |f| * exp(-(f/f0)^4) in frequency domain.
    
    This is the exact filter used in Olshausen & Field (1996) for natural image
    preprocessing before sparse coding.
    
    Args:
        image: 2D array (float) - input image
        f0: Frequency cutoff in cycles/picture (default: 200)
        eps: Numerical stability epsilon
        
    Returns:
        Whitened image (same shape as input)
    """
    img = np.asarray(image, dtype=float)
    H, W = img.shape
    
    # Forward FFT
    F = np.fft.fft2(img)
    
    # Create frequency grids normalized to cycles/picture
    fy = np.fft.fftfreq(H) * H
    fx = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy)
    
    # Radial frequency magnitude
    R = np.sqrt(FX**2 + FY**2)
    
    # Avoid division by zero at DC component  
    R_safe = np.maximum(R, eps)
    
    # Olshausen & Field whitening filter: R(f) = |f| * exp(-(f/f0)^4)
    filt = R_safe * np.exp(-(R_safe / f0)**4)
    filt[0, 0] = 0.0  # Zero DC component
    
    # Apply filter (magnitude-only, preserves phase)
    Fw = F * filt
    
    # Inverse FFT (take real part)
    out = np.fft.ifft2(Fw).real
    
    # Normalize to preserve input variance
    if np.std(out) > 0:
        out = out / (np.std(out) + eps) * (np.std(img) + eps)
    
    return out