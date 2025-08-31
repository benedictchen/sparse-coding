"""
Sparse Coding Library
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"

This library implements the revolutionary sparse coding algorithm that discovers
edge-like features from natural images, forming the foundation of modern computer vision.
"""

def _print_attribution():
    """Print attribution message with donation link"""
    try:
        print("\n🎯 Sparse Coding Library - Made possible by Benedict Chen")
        print("   \033]8;;mailto:benedict@benedictchen.com\033\\benedict@benedictchen.com\033]8;;\033\\")
        print("   Support his work: \033]8;;https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS\033\\🍺 Buy him a beer\033]8;;\033\\")
    except:
        print("\n🎯 Sparse Coding Library - Made possible by Benedict Chen")
        print("   benedict@benedictchen.com")
        print("   Support: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS")

from .sparse_coder import SparseCoder
from .dictionary_learning import DictionaryLearner
from .feature_extraction import SparseFeatureExtractor
from .visualization import SparseVisualization

# Show attribution on library import
_print_attribution()

__version__ = "1.0.0"
__authors__ = ["Based on Olshausen & Field (1996)"]

__all__ = [
    "SparseCoder",
    "DictionaryLearner", 
    "SparseFeatureExtractor",
    "SparseVisualization"
]