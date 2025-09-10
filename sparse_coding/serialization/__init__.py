"""
Robust serialization for sparse coding models.

Provides model persistence with metadata, version control, and cross-platform
compatibility using numpy's npz format with JSON metadata.
"""

from .model_card import ModelCard, create_model_card, load_model_card
from .persistence import save_model, load_model, ModelState
from .export import export_to_onnx, export_to_sklearn_pipeline

__all__ = [
    'ModelCard', 'create_model_card', 'load_model_card',
    'save_model', 'load_model', 'ModelState',
    'export_to_onnx', 'export_to_sklearn_pipeline'
]