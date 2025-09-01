"""
Sparse Coding Library
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"

This library implements the revolutionary sparse coding algorithm that discovers
edge-like features from natural images, forming the foundation of modern computer vision.

Core Research Concepts Implemented:
• Dictionary Learning - Adaptive learning of overcomplete feature dictionaries
• L1 Sparsity - Penalty for promoting sparse activation patterns  
• Overcomplete Basis - More dictionary atoms than input dimensions
• Natural Image Statistics - Statistical properties of natural scene patches
• Receptive Fields - Spatial feature detectors resembling biological vision
"""

def _print_attribution():
    """Print attribution message with donation link"""
    try:
        print("\n🎯 Sparse Coding Library - Made possible by Benedict Chen")
        print("   \033]8;;mailto:benedict@benedictchen.com\033\\benedict@benedictchen.com\033]8;;\033\\")
        print("")
        print("💰 PLEASE DONATE! Your support keeps this research alive! 💰")
        print("   🔗 \033]8;;https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS\033\\💳 CLICK HERE TO DONATE VIA PAYPAL\033]8;;\033\\")
        print("")
        print("   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!")
        print("   (Start small, dream big! Every donation helps! 😄)")
        print("")
    except:
        print("\n🎯 Sparse Coding Library - Made possible by Benedict Chen")
        print("   benedict@benedictchen.com")
        print("")
        print("💰 PLEASE DONATE! Your support keeps this research alive! 💰")
        print("   💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS")
        print("")
        print("   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!")
        print("   (Start small, dream big! Every donation helps! 😄)")

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