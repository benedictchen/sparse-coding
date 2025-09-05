from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Penalty:
    def prox(self, z: np.ndarray, t: float) -> np.ndarray: ...
    def value(self, a: np.ndarray) -> float: ...

@dataclass
class L1(Penalty):
    lam: float
    def prox(self, z, t): return np.sign(z) * np.maximum(np.abs(z) - t*self.lam, 0.0)
    def value(self, a): return self.lam * float(np.sum(np.abs(a)))

@dataclass
class ElasticNet(Penalty):
    l1: float; l2: float
    def prox(self, z, t): return (np.sign(z) * np.maximum(np.abs(z) - t*self.l1, 0.0)) / (1.0 + t*self.l2)
    def value(self, a): return self.l1 * float(np.sum(np.abs(a))) + 0.5*self.l2 * float(np.sum(a*a))

@dataclass
class NonNegL1(Penalty):
    lam: float
    def prox(self, z, t): return np.maximum(z - t*self.lam, 0.0)
    def value(self, a): return self.lam * float(np.sum(a)) if np.all(a>=0) else np.inf
