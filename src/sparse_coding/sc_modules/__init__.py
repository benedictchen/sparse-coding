"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀
"""
"""
✨ Sparse Coding Modules (sc_modules)
====================================

Modular components for the Sparse Coding library based on Olshausen & Field (1996).
This package provides utility functions and algorithms extracted from the main
SparseCoder class for better organization and reusability.

Author: Benedict Chen (benedict@benedictchen.com)
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"

Available Modules:
- utilities: Core utility functions including overcomplete basis generation and lateral inhibition
- data_processing: Image patch extraction and whitening functions for preprocessing
- optimization: Optimization algorithms for sparse coding including Olshausen & Field equation (5), FISTA, and coordinate descent
- dictionary_update: Dictionary learning algorithms including equation (6), MOD, and coherence analysis
- validation: Validation and analysis functions for sparse coding parameters and learned dictionaries
- visualization: Comprehensive visualization tools for dictionary analysis, training curves, and sparse coding research
"""

from .utilities import (
    create_overcomplete_basis,
    lateral_inhibition,
    demo_sparse_coding,
    analyze_basis_properties,
    visualize_basis_subset
)

from .data_processing import (
    DataProcessingMixin,
    extract_patches,
    whiten_patches,
    whiten_patches_olshausen_field,
    whiten_patches_zca,
    preprocess_patches,
    get_whitening_methods_info
)

from .optimization import (
    OptimizationMixin
)

from .dictionary_update import (
    DictionaryUpdateMixin
)

from .validation import (
    ValidationMixin,
    validate_sparse_coding_parameters,
    analyze_dictionary_standalone
)

from .visualization import (
    VisualizationMixin
)

__all__ = [
    'create_overcomplete_basis',
    'lateral_inhibition', 
    'demo_sparse_coding',
    'analyze_basis_properties',
    'visualize_basis_subset',
    'DataProcessingMixin',
    'extract_patches',
    'whiten_patches',
    'whiten_patches_olshausen_field',
    'whiten_patches_zca',
    'preprocess_patches',
    'get_whitening_methods_info',
    'OptimizationMixin',
    'DictionaryUpdateMixin',
    'ValidationMixin',
    'validate_sparse_coding_parameters',
    'analyze_dictionary_standalone',
    'VisualizationMixin'
]

__version__ = '1.0.0'


"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
📝 CITE: Benedict Chen (2025) - Sparse Coding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""