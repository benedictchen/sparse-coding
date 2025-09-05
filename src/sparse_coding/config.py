"""
Configuration System with Pydantic Validation
=============================================

Type-safe configuration for sparse coding with validation.
Based on ChatGPT's production-quality improvements.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, PositiveInt
from typing import Optional


class SparseCodeConfig(BaseModel):
    """
    Configuration for sparse coding with Pydantic validation.
    
    Ensures type safety and validation of all parameters.
    """
    # Data parameters
    patch_size: PositiveInt = 16
    n_atoms: PositiveInt = 144
    samples: PositiveInt = 50000
    
    # Training parameters  
    steps: PositiveInt = 50
    lr: float = Field(0.1, gt=0.0)
    mode: str = Field("paper", pattern="^(paper|l1)$")
    
    # Whitening parameters
    f0: float = Field(200.0, gt=0.0)  # Frequency cutoff for whitening
    
    # Sparsity parameters
    lam: Optional[float] = Field(None, ge=0.0)  # Explicit lambda
    lam_sigma: Optional[float] = Field(0.14, ge=0.0)  # Lambda/sigma ratio
    
    # Reproducibility
    seed: int = 0
    deterministic: bool = True
    
    # Paper-exact mode parameters
    max_iter: PositiveInt = 200
    tol: float = Field(1e-4, gt=0.0)
    rel_tol: float = Field(0.01, gt=0.0)  # 1% relative change (paper standard)
    
    def model_post_init(self, __context) -> None:
        """Post-initialization validation."""
        # Ensure we have either explicit lambda or lambda/sigma ratio
        if self.lam is None and self.lam_sigma is None:
            raise ValueError("Must specify either 'lam' or 'lam_sigma'")


# Schema version for metadata tracking
SCHEMA_VERSION = 2


def make_metadata(cfg: SparseCodeConfig, D_shape, A_shape, extra=None):
    """
    Create metadata dictionary for results tracking.
    
    Args:
        cfg: Configuration used
        D_shape: Dictionary shape
        A_shape: Coefficients shape  
        extra: Additional metadata
        
    Returns:
        Metadata dictionary
    """
    meta = {
        "schema_version": SCHEMA_VERSION,
        "config": cfg.model_dump(),
        "shapes": {"D": list(D_shape), "A": list(A_shape)},
        "algorithm": "olshausen_field_1996" if cfg.mode == "paper" else "fista_l1"
    }
    
    if extra:
        meta.update(extra)
        
    return meta


class ValidationConfig(BaseModel):
    """Configuration for validation and testing."""
    kkt_tolerance: float = Field(1e-3, gt=0.0)
    coherence_threshold: float = Field(0.99, gt=0.0, le=1.0)
    reconstruction_threshold: float = Field(0.8, gt=0.0)
    sparsity_threshold: float = Field(0.1, ge=0.0, le=1.0)


class PresetConfigs:
    """Preset configurations for common use cases."""
    
    @staticmethod
    def olshausen_field_figure4() -> SparseCodeConfig:
        """
        Configuration to reproduce Olshausen & Field (1996) Figure 4.
        
        16x16 patches, 144 atoms, paper mode with proper parameters.
        """
        return SparseCodeConfig(
            patch_size=16,
            n_atoms=144,
            samples=50000,
            steps=50,
            lr=0.1,
            mode="paper",
            f0=200.0,
            lam_sigma=0.14,
            seed=0,
            deterministic=True,
            max_iter=80,
            tol=1e-4,
            rel_tol=0.01
        )
    
    @staticmethod
    def fast_l1_demo() -> SparseCodeConfig:
        """
        Fast L1 configuration for demonstrations and testing.
        """
        return SparseCodeConfig(
            patch_size=8,
            n_atoms=32,
            samples=1000,
            steps=10,
            lr=0.2,
            mode="l1",
            lam=0.1,
            seed=42,
            deterministic=True,
            max_iter=100,
            tol=1e-6
        )
    
    @staticmethod
    def high_quality_research() -> SparseCodeConfig:
        """
        High-quality configuration for research applications.
        """
        return SparseCodeConfig(
            patch_size=16,
            n_atoms=256,
            samples=100000,
            steps=100,
            lr=0.05,
            mode="paper", 
            f0=200.0,
            lam_sigma=0.14,
            seed=0,
            deterministic=True,
            max_iter=200,
            tol=1e-6,
            rel_tol=0.005
        )