"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀

Sparse Coding Library
===================

Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"

This library implements the revolutionary sparse coding algorithm that discovers
edge-like features from natural images, forming the foundation of modern computer vision.

🔬 Research Foundation:
- Bruno Olshausen & David Field's sparse coding theory
- Natural image statistics and receptive field emergence
- Efficient coding principles in biological vision
- Dictionary learning and sparse representation

🎯 Key Features:
- Complete Olshausen & Field algorithm implementation
- Dictionary learning with adaptive updates
- Sparse feature extraction and encoding
- Visualization of learned receptive fields
- Research-accurate implementations
"""

def _print_attribution():
    """Print attribution message with donation link"""
    try:
        print("\\n🌟 Sparse Coding Library - Made possible by Benedict Chen")
        print("   \\033]8;;mailto:benedict@benedictchen.com\\033\\\\benedict@benedictchen.com\\033]8;;\\033\\\\")
        print("")
        print("💰 PLEASE DONATE! Your support keeps this research alive! 💰")
        print("   🔗 \\033]8;;https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS\\033\\\\💳 CLICK HERE TO DONATE VIA PAYPAL\\033]8;;\\033\\\\")
        print("")
        print("   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!")
        print("   (Start small, dream big! Every donation helps! 😄)")
        print("")
    except:
        print("\\n🌟 Sparse Coding Library - Made possible by Benedict Chen")
        print("   benedict@benedictchen.com")
        print("")
        print("💰 PLEASE DONATE! Your support keeps this research alive! 💰")
        print("   💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS")
        print("")
        print("   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!")
        print("   (Start small, dream big! Every donation helps! 😄)")

# Import UNIFIED implementations from consolidated files
from .core import (
    SparseCoder
)

# Import additional classes from their specific modules
from .sc_modules.olshausen_field import OlshausenFieldOriginal
from .dictionary_learning import DictionaryLearner
from .feature_extraction import SparseFeatureExtractor
from .batch_processor import BatchProcessor, process_large_dataset
from .sc_modules.utilities import create_overcomplete_basis

from .config import (
    SparseCoderConfig,
    OlshausenFieldConfig,
    DictionaryLearningConfig,
    FeatureExtractionConfig,
    BatchProcessingConfig,
    SparsityFunction,
    Optimizer,
    DictionaryUpdateRule,
    InitializationMethod,
    create_config
)

from .utils import (
    extract_patches_2d,
    extract_patches_from_images,
    normalize_patch_batch,
    whiten_patches,
    soft_threshold,
    hard_threshold,
    validate_sparse_coding_data,
    compute_dictionary_coherence,
    lateral_inhibition_network,
    create_gabor_dictionary,
    create_dct_dictionary
)

from .viz import (
    plot_dictionary,
    plot_training_history,
    plot_sparse_codes,
    plot_reconstruction_comparison,
    plot_sparsity_path,
    setup_publication_style
)

# Show attribution on library import
_print_attribution()

__version__ = "2.1.0"
__authors__ = ["Benedict Chen", "Based on Olshausen & Field (1996)"]

# Define explicit public API - Unified Structure
__all__ = [
    # Core algorithms
    "SparseCoder",
    "OlshausenFieldOriginal", 
    "DictionaryLearner",
    "SparseFeatureExtractor",
    "BatchProcessor",
    
    # Configuration classes
    "SparseCoderConfig",
    "OlshausenFieldConfig", 
    "DictionaryLearningConfig",
    "FeatureExtractionConfig",
    "BatchProcessingConfig",
    
    # Enums
    "SparsityFunction",
    "Optimizer", 
    "DictionaryUpdateRule",
    "InitializationMethod",
    
    # Factory functions
    "create_config",
    "create_overcomplete_basis",
    "process_large_dataset",
    
    # Utility functions
    "extract_patches_2d",
    "extract_patches_from_images", 
    "normalize_patch_batch",
    "whiten_patches",
    "soft_threshold",
    "hard_threshold",
    "validate_sparse_coding_data",
    "compute_dictionary_coherence",
    "lateral_inhibition_network",
    "create_gabor_dictionary",
    "create_dct_dictionary",
    
    # Visualization functions
    "plot_dictionary",
    "plot_training_history", 
    "plot_sparse_codes",
    "plot_reconstruction_comparison",
    "plot_sparsity_path",
    "setup_publication_style",
]

"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
📝 CITE: Benedict Chen (2025) - Sparse Coding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""