"""
Learner implementations for sparse coding.

This package provides configurable learner classes that can be composed
from different penalties, solvers, and dictionary updaters.
"""

from .configurable import ConfigurableLearner

__all__ = ['ConfigurableLearner']