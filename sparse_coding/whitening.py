"""
Zero-phase whitening for natural image preprocessing in sparse coding.

Implements zero-phase whitening filters as used in Olshausen & Field (1996) sparse coding.
Essential preprocessing step for natural image statistics and receptive field learning.

Whitening transforms are crucial for sparse coding of natural images because:
- Natural images have 1/f power spectrum (energy concentrated in low frequencies)
- Raw images violate independence assumptions needed for sparse coding
- Whitening approximates retinal/LGN preprocessing in biological vision

The filter R(f) = |f| * exp(-(f/f0)^4) combines:
1. High-pass filtering |f| (removes DC and low frequencies) 
2. Gaussian envelope exp(-(f/f0)^4) (prevents noise amplification at high frequencies)

Mathematical Framework:
For image I(x,y), the whitened output W(x,y) is computed as:
1. Forward FFT: Î(fx,fy) = FFT2D[I(x,y)]
2. Frequency filtering: Ŵ(fx,fy) = Î(fx,fy) * R(fx,fy)  
3. Inverse FFT: W(x,y) = IFFT2D[Ŵ(fx,fy)]
4. Normalization: W = (W - mean(W)) / std(W)

Where R(fx,fy) = √(fx² + fy²) * exp(-((√(fx² + fy²))/f0)^4)

Biological Context:
Approximates preprocessing in mammalian visual system:
- Retinal ganglion cells: Center-surround receptive fields (bandpass filtering)
- LGN (Lateral Geniculate Nucleus): Additional spatial-temporal filtering
- Result: Approximately whitened input to primary visual cortex (V1)

References:
    Olshausen & Field (1996). Emergence of simple-cell receptive field properties.
    Field (1987). Relations between the statistics of natural images and response properties.
    Simoncelli & Olshausen (2001). Natural image statistics and neural representation.
    Bell & Sejnowski (1997). The independent components of natural scenes are edge filters.
"""

import numpy as np
from typing import Optional, Union


def zero_phase_whiten(
    image: np.ndarray, 
    f0: float = 200.0,
    normalize_output: bool = True,
    preserve_mean: bool = False,
    numerical_stability_eps: float = 1e-12
) -> np.ndarray:
    """
    Apply zero-phase whitening filter to natural image.
    
    Implements the standard preprocessing used in Olshausen & Field (1996) sparse coding
    experiments. Combines high-pass filtering with Gaussian envelope to approximate
    retinal preprocessing while preventing high-frequency noise amplification.
    
    Args:
        image: Input image as 2D numpy array (grayscale)
        f0: Cutoff frequency parameter controlling filter bandwidth (default: 200.0)
        normalize_output: Whether to normalize output to zero mean, unit variance
        preserve_mean: Whether to preserve original image mean (default: False)  
        numerical_stability_eps: Small constant to prevent division by zero
        
    Returns:
        Whitened image with same spatial dimensions as input
        
    Mathematical Details:
        Filter frequency response: R(f) = |f| * exp(-(f/f0)^4)
        
        Where:
        - |f| = √(fx² + fy²) provides high-pass characteristic 
        - exp(-(f/f0)^4) provides Gaussian envelope to limit high-frequency noise
        - f0 controls transition between high-pass and roll-off regions
        
    Frequency Response Properties:
        - DC (f=0): R(0) = 0 (perfect DC rejection)
        - Peak: Near f ≈ 0.5*f0 (approximate spatial frequency emphasis)
        - High-freq: Exponential decay prevents noise amplification
        - Phase: Zero phase (real-valued filter, symmetric processing)
        
    Example:
        >>> import numpy as np
        >>> # Load natural image (e.g., from PIL, cv2, etc.)
        >>> raw_image = load_image("natural_scene.jpg")
        >>> # Apply standard Olshausen & Field preprocessing
        >>> whitened = zero_phase_whiten(raw_image, f0=200.0)
        >>> # Now ready for sparse coding dictionary learning
        >>> coder = SparseCoder(n_atoms=100)
        >>> coder.fit(extract_patches(whitened))
        
    Research Usage:
        Standard parameters from sparse coding literature:
        - f0=200.0: Used in Olshausen & Field (1996) 
        - f0=150.0: Alternative used in some ICA studies
        - f0=100.0: More aggressive high-pass filtering
        
    Biological Interpretation:
        Approximates processing in retinal ganglion cells and LGN:
        - Center-surround receptive fields ≈ high-pass filtering
        - Limited high-frequency response ≈ neural bandwidth limits
        - Zero-mean output ≈ adaptation to local mean luminance
        
    References:
        - Olshausen & Field (1996): Original sparse coding parameters
        - Field (1987): Natural image statistics and whitening theory
        - Simoncelli & Olshausen (2001): Comprehensive whitening analysis
    """
    # Convert to float for numerical stability
    x = np.asarray(image, dtype=np.float64)
    
    if x.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {x.shape}")
    
    H, W = x.shape
    
    # 📊 Create frequency coordinate arrays
    # fftfreq generates frequencies in cycles per sample, scaled by image size
    fy = np.fft.fftfreq(H) * H  # Vertical frequencies
    fx = np.fft.fftfreq(W) * W  # Horizontal frequencies
    
    # Create 2D frequency grid
    FX, FY = np.meshgrid(fx, fy)
    
    # 🎯 Compute radial frequency |f| = √(fx² + fy²)
    R = np.sqrt(FX**2 + FY**2)
    
    # 🎨 Design zero-phase whitening filter
    # R(f) = |f| * exp(-(f/f0)^4)
    # - |f| term: High-pass characteristic (amplifies high frequencies)
    # - exp() term: Gaussian envelope (prevents excessive noise amplification)
    filt = R * np.exp(-(R / (f0 + numerical_stability_eps))**4)
    
    # 🔄 Apply filter in frequency domain
    X = np.fft.fft2(x)  # Forward FFT
    Y = X * filt        # Frequency-domain multiplication
    y = np.fft.ifft2(Y).real  # Inverse FFT (take real part for numerical stability)
    
    # 📐 Normalization options
    if not preserve_mean:
        y = y - y.mean()  # Remove DC component (zero mean)
    
    if normalize_output:
        y_std = y.std()
        if y_std > numerical_stability_eps:
            y = y / y_std  # Unit variance normalization
        else:
            # Handle edge case of constant image (after whitening)
            y = np.zeros_like(y)
    
    return y


def adaptive_whiten(
    image: np.ndarray,
    percentile_cutoff: float = 95.0,
    min_f0: float = 50.0,
    max_f0: float = 400.0
) -> np.ndarray:
    """
    🔄 Adaptive whitening that adjusts f0 based on image content.
    
    Automatically determines optimal cutoff frequency based on image statistics.
    Useful for processing diverse natural image datasets with varying characteristics.
    
    Args:
        image: Input image as 2D numpy array
        percentile_cutoff: Percentile of power spectrum to use for f0 estimation
        min_f0: Minimum allowed cutoff frequency
        max_f0: Maximum allowed cutoff frequency
        
    Returns:
        Whitened image with adaptive frequency response
        
    Research Context:
        While Olshausen & Field (1996) used fixed f0=200, modern approaches
        often benefit from adaptive preprocessing that accounts for:
        - Varying image resolution and sampling rates
        - Different scene statistics (urban vs. natural scenes)
        - Camera-specific noise characteristics
    """
    # Compute power spectrum to estimate dominant frequencies
    X = np.fft.fft2(image)
    power_spectrum = np.abs(X)**2
    
    # Create radial frequency array
    H, W = image.shape
    fy = np.fft.fftfreq(H) * H
    fx = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)
    
    # Estimate characteristic frequency from power spectrum percentile
    freq_weighted_power = power_spectrum * R
    cumulative_power = np.cumsum(np.sort(freq_weighted_power.flatten()))
    total_power = cumulative_power[-1]
    
    percentile_idx = int((percentile_cutoff / 100.0) * len(cumulative_power))
    characteristic_freq = np.sort(freq_weighted_power.flatten())[percentile_idx]
    
    # Determine adaptive f0 with reasonable bounds
    f0_adaptive = np.clip(characteristic_freq * 0.5, min_f0, max_f0)
    
    # Apply standard whitening with adaptive f0
    return zero_phase_whiten(image, f0=f0_adaptive)


def get_filter_response(
    image_shape: tuple,
    f0: float = 200.0,
    return_frequencies: bool = False
) -> Union[np.ndarray, tuple]:
    """
    📊 Compute whitening filter frequency response for analysis.
    
    Useful for visualizing and analyzing the frequency characteristics
    of the zero-phase whitening filter before applying to images.
    
    Args:
        image_shape: (height, width) of target image
        f0: Cutoff frequency parameter
        return_frequencies: Whether to also return frequency coordinate arrays
        
    Returns:
        Filter response array, optionally with (fx, fy) coordinate arrays
        
    Example:
        >>> # Analyze filter characteristics
        >>> H, W = (256, 256)
        >>> response, (fx, fy) = get_filter_response((H, W), f0=200, return_frequencies=True)
        >>> # Plot filter response
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(10, 4))
        >>> plt.subplot(1, 2, 1)
        >>> plt.imshow(response, extent=[fx.min(), fx.max(), fy.min(), fy.max()])
        >>> plt.title("Whitening Filter Response")
        >>> plt.colorbar()
    """
    H, W = image_shape
    
    # Create frequency coordinates
    fy = np.fft.fftfreq(H) * H
    fx = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy)
    
    # Compute filter response
    R = np.sqrt(FX**2 + FY**2)
    filt = R * np.exp(-(R / (f0 + 1e-12))**4)
    
    if return_frequencies:
        return filt, (fx, fy)
    else:
        return filt
