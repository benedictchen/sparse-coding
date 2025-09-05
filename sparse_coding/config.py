from __future__ import annotations
from pydantic import BaseModel, Field, PositiveInt
from typing import Optional

class TrainingConfig(BaseModel):
    patch_size: PositiveInt = 16
    n_atoms: PositiveInt = 144
    steps: PositiveInt = 50
    lr: float = Field(0.1, gt=0.0)
    mode: str = Field("paper", pattern="^(paper|l1)$")
    f0: float = Field(200.0, gt=0.0)
    lam_sigma: Optional[float] = None   # if set, lam = lam_sigma * std(whitened data)
    seed: int = 0
    deterministic: bool = True
    samples: PositiveInt = 50000

SCHEMA_VERSION = 1

def make_metadata(cfg: TrainingConfig, D_shape, A_shape, extra=None):
    meta = {
        "schema_version": SCHEMA_VERSION,
        "patch_size": cfg.patch_size,
        "n_atoms": cfg.n_atoms,
        "steps": cfg.steps,
        "lr": cfg.lr,
        "mode": cfg.mode,
        "f0": cfg.f0,
        "lam_sigma": cfg.lam_sigma,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "shapes": {"D": list(D_shape), "A": list(A_shape)},
    }
    if extra: meta.update(extra)
    return meta
