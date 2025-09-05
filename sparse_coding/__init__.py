"""
Sparse Coding - Research-Faithful Implementation
==============================================

Fast, turnkey sparse coding with Olshausen & Field (1996) paper-exact mode
and modern L1/FISTA inference.

Quick Start:
-----------
>>> import numpy as np
>>> from sparse_coding import SparseCoder
>>> 
>>> # Modern L1 sparse coding
>>> coder = SparseCoder(n_atoms=128, mode='l1')
>>> coder.fit(patches)  # patches shape: (p, N)
>>> codes = coder.encode(patches)
>>> 
>>> # Paper-exact Olshausen & Field mode
>>> coder_paper = SparseCoder(n_atoms=144, mode='paper')
>>> coder_paper.fit(patches, n_steps=50)

Command Line Interface:
----------------------
$ sparse-coding train --images ./images --out results --mode paper --seed 0
$ sparse-coding encode --dictionary results/D.npy --patches X.npy --out A.npy
$ sparse-coding reconstruct --dictionary results/D.npy --codes A.npy --out X_hat.npy
"""

from .api import SparseCoder
from .presets import OFigure4Preset
from .whitening import zero_phase_whiten
from .diagnostics import kkt_violation_l1

__version__ = "0.1.0"
__author__ = "Research Implementation"

__all__ = [
    'SparseCoder',
    'OFigure4Preset', 
    'zero_phase_whiten',
    'kkt_violation_l1'
]