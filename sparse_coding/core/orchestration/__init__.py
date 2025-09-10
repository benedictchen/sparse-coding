"""
High-level orchestration of sparse coding algorithms.

This module implements the main API for sparse coding following Olshausen & Field
(1996) framework with modern algorithmic improvements.
"""

from .olshausen_field_learner import OlshausenFieldLearner

__all__ = ['OlshausenFieldLearner']