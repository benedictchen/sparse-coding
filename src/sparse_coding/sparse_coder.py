"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀

✨ Sparse Coding - Learning the Language of Natural Images
========================================================

Author: Benedict Chen (benedict@benedictchen.com)

This is the main entry point for the modular sparse coding implementation.
The original 1927-line monolithic implementation has been broken down into
focused, maintainable modules while preserving 100% research accuracy.

Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"

🎯 ELI5 Summary:
Imagine you're an artist trying to recreate any picture using the fewest brush strokes possible.
Sparse coding finds the perfect set of "brush strokes" (basis functions) where any natural image
can be recreated using just a few active strokes. Amazingly, these learned strokes look exactly
like what neurons in your visual cortex respond to - edge detectors, line segments, etc!

🏗️ Modular Architecture:
========================
This implementation uses a clean modular architecture:

- core_algorithm.py    : Main SparseCoder class and training loop
- optimization.py      : Sparse coding algorithms (equation 5, FISTA, etc.)
- dictionary_update.py : Dictionary learning methods (equation 6, MOD, etc.)
- data_processing.py   : Patch extraction and whitening
- validation.py        : Parameter validation and analysis
- visualization.py     : Plotting and visualization tools
- olshausen_field.py   : Original 1996 paper implementations
- utilities.py         : Helper functions and basis creation

🔬 Research Accuracy:
====================
All implementations maintain complete research fidelity to the original
Olshausen & Field (1996) algorithms while providing modern software
engineering benefits.

📚 Usage Examples:
==================

Basic Usage:
-----------
>>> from sparse_coding import SparseCoder
>>> import numpy as np
>>> 
>>> # Create sparse coder
>>> sc = SparseCoder(n_components=64, max_iter=100)
>>> 
>>> # Fit to natural images
>>> sc.fit(natural_images)
>>> 
>>> # Transform new images  
>>> sparse_codes = sc.transform(test_images)
>>> 
>>> # Reconstruct from codes
>>> reconstructed = sc.reconstruct(sparse_codes)

Advanced Configuration:
----------------------
>>> # Configure with research-specific options
>>> sc = SparseCoder(
...     n_components=256,
...     sparseness_function='log',      # Olshausen & Field's choice
...     optimization_method='equation_5', # Original paper algorithm
...     dictionary_update='equation_6',   # Original paper update
...     whitening_method='olshausen_field'  # Original whitening
... )

Visualization:
-------------
>>> # Visualize learned dictionary (should show edge detectors!)
>>> sc.visualize_dictionary()
>>> 
>>> # Plot training curves
>>> sc.plot_training_curves()
>>> 
>>> # Comprehensive analysis
>>> sc.create_visualization_report()

🎯 Mathematical Framework:
==========================
Given image patches X, find dictionary D and sparse codes S such that:
X ≈ D × S  where S is sparse (mostly zeros)

Optimization Problem:
min_{D,S} ||X - DS||²₂ + λ∑|S_i|  (L1 penalty for sparsity)

The alternating optimization procedure:
1. Fix D, solve for S: sparse coding step
2. Fix S, solve for D: dictionary update step
3. Repeat until convergence

🔗 References:
==============
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.
- Olshausen, B. A., & Field, D. J. (1997). Sparse coding with an overcomplete basis set: A strategy employed by V1? Vision Research, 37(23), 3311-3325.

For implementation details, see the individual modules in sc_modules/.
"""

# Import the modular SparseCoder implementation
from .sc_modules.core_algorithm import SparseCoder

# Import utilities for direct use
from .sc_modules.utilities import (
    create_overcomplete_basis,
    lateral_inhibition, 
    demo_sparse_coding
)

# Import the original research implementation for reference
from .sc_modules.olshausen_field import OlshausenFieldOriginal

# Backward compatibility aliases
SparseCode = SparseCoder  # Common alias used in research

# Package metadata
__version__ = "2.1.0"  # Incremented for modular architecture
__author__ = "Benedict Chen"
__email__ = "benedict@benedictchen.com"

# Export main classes and functions
__all__ = [
    'SparseCoder',
    'SparseCode', 
    'OlshausenFieldOriginal',
    'create_overcomplete_basis',
    'lateral_inhibition',
    'demo_sparse_coding'
]

# Module docstring for import
def get_module_info():
    """
    Get information about the sparse coding module architecture.
    
    Returns
    -------
    dict
        Dictionary containing module information and architecture details.
    """
    return {
        'version': __version__,
        'architecture': 'modular',
        'modules': [
            'core_algorithm',
            'optimization', 
            'dictionary_update',
            'data_processing',
            'validation',
            'visualization',
            'olshausen_field',
            'utilities'
        ],
        'research_basis': 'Olshausen & Field (1996)',
        'total_lines_modularized': 1927,
        'backward_compatible': True
    }

# Quick functionality check
def _test_import():
    """Quick test to ensure modular components import correctly."""
    try:
        # Test main class import
        sc = SparseCoder(n_components=16, patch_size=(8, 8))
        
        # Test utility functions
        basis = create_overcomplete_basis((8, 8), 2.0, 'gabor')
        
        # Test original implementation
        orig = OlshausenFieldOriginal(n_components=16)
        
        return True
    except Exception as e:
        print(f"Import test failed: {e}")
        return False

# Run import test when module is loaded (development only)
if __name__ == '__main__':
    print("\n" + "="*80)
    print("💰 SUPPORT THIS RESEARCH - PLEASE DONATE!")  
    print("🙏 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS")
    print("="*80 + "\n")
    
    # Only run test when directly executed
    success = _test_import()
    if success:
        print("✅ Sparse coding modular architecture loaded successfully!")
        print("📊 Module info:", get_module_info())
    else:
        print("❌ Module import test failed")
        
    print("\n" + "="*80)
    print("💝 Thank you for using this research software!")
    print("📚 Please donate: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS") 
    print("="*80 + "\n")

"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
📝 CITE: Benedict Chen (2025) - Sparse Coding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""