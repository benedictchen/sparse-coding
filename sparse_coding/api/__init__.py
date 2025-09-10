"""
Public API and plugin system for sparse coding library.

Provides registry for penalties, solvers, and dictionary updaters.
Enables clean configuration-driven usage and extensibility.
"""

from .registry import register, get_registry, create_from_config, list_registered
from .config import validate_config, load_config, save_config

__all__ = [
    'register', 'get_registry', 'create_from_config', 'list_registered',
    'validate_config', 'load_config', 'save_config'
]