"""
Research presets for reproducing paper results.
"""

from dataclasses import dataclass

@dataclass
class OFigure4Preset:
    """
    Preset matching Olshausen & Field (1996) Figure 4 experimental setup.
    
    These parameters reproduce the classic sparse coding results on natural images.
    """
    patch_size: int = 16                      # 16x16 patches
    f0: float = 200.0                        # Whitening cutoff (cycles/picture)
    ratio_lambda_over_sigma: float = 0.14    # Sparsity penalty ratio λ/σ
    n_atoms: int = 144                       # Dictionary size (overcomplete)
    max_iter_inner: int = 80                 # Sparse coding iterations
    tol_inner: float = 1e-4                  # Convergence tolerance
    lr_dict: float = 0.1                     # Dictionary learning rate
    n_steps_outer: int = 50                  # Dictionary update steps
    
    def __post_init__(self):
        """Validate preset parameters."""
        if self.n_atoms <= self.patch_size**2:
            print(f"Warning: Dictionary not overcomplete ({self.n_atoms} ≤ {self.patch_size**2})")

@dataclass  
class ModernPreset:
    """
    Modern L1 sparse coding preset with FISTA optimization.
    
    Faster convergence than paper method, suitable for practical applications.
    """
    patch_size: int = 16
    f0: float = 200.0
    ratio_lambda_over_sigma: float = 0.1     # Lower for L1 penalty
    n_atoms: int = 256                       # Larger dictionary
    max_iter_inner: int = 200                # More FISTA iterations
    tol_inner: float = 1e-6                  # Tighter convergence
    lr_dict: float = 0.2                     # Higher learning rate
    n_steps_outer: int = 30                  # Fewer outer steps needed