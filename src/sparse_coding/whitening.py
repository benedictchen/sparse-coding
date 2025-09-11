"""
Whitening module for sparse coding preprocessing.

This module provides a clean import interface for whitening functionality.
The actual implementation is in data_preprocessing_whitening.py.
"""

from .data_preprocessing_whitening import zero_phase_whiten

__all__ = ['zero_phase_whiten']