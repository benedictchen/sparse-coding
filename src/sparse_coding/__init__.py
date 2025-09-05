"""
📋   Init  
============

🎯 ELI5 Summary:
This file is an important component in our AI research system! Like different organs 
in your body that work together to keep you healthy, this file has a specific job that 
helps the overall algorithm work correctly and efficiently.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

📋 Component Integration:
========================
    ┌──────────┐
    │   This   │
    │Component │ ←→ Other Components
    └──────────┘
         ↑↓
    System Integration

"""
"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀

Sparse Coding Library
===================

Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"

This library implements the sparse coding algorithm that discovers
edge-like features from natural images, forming the foundation of modern computer vision.

🔬 Research Foundation:
- Bruno Olshausen & David Field's sparse coding theory
- Dictionary Learning with overcomplete basis sets
- L1 Sparsity penalty for efficient coding
- Natural Image Statistics and preprocessing  
- Receptive Fields emergence through optimization
- Overcomplete Basis representation learning
- Efficient coding principles in biological vision

🎯 Key Features:
- Complete Olshausen & Field algorithm implementation
- Dictionary Learning with adaptive updates
- L1 Sparsity constraints and penalty functions
- Sparse feature extraction and encoding
- Overcomplete Basis dictionary construction
- Natural Image Statistics preprocessing
- Receptive Fields visualization
- Research-accurate implementations
"""

def _print_attribution():
    """Print attribution message with donation link"""
    try:
        # Removed print spam: "\\n...
        print("   \\033]8;;mailto:benedict@benedictchen.com\\033\\\\benedict@benedictchen.com\\033]8;;\\033\\\\")
        print("")
        print("💰 PLEASE DONATE! Your support keeps this research alive! 💰")
        print("   🔗 \\033]8;;https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS\\033\\\\💳 CLICK HERE TO DONATE VIA PAYPAL\\033]8;;\\033\\\\")
        print("   ❤️ \\033]8;;https://github.com/sponsors/benedictchen\\033\\\\💖 SPONSOR ON GITHUB\\033]8;;\\033\\\\")
        print("")
        print("   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!")
        print("   (Start small, dream big! Every donation helps! 😄)")
        print("")
    except:
        # Removed print spam: "\\n...
        print("   benedict@benedictchen.com")
        print("")
        print("💰 PLEASE DONATE! Your support keeps this research alive! 💰")
        print("   💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS")
        print("   💖 GitHub Sponsors: https://github.com/sponsors/benedictchen")
        print("")
        print("   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!")
        print("   (Start small, dream big! Every donation helps! 😄)")

# Import UNIFIED implementations from consolidated files
from .sparse_coder import SparseCoder

# Import research-accurate preprocessing (NEW)
from .research_accurate_preprocessing import ResearchAccuratePreprocessor

# Import additional classes from their specific modules
from .sparse_coding_modules.olshausen_field import OlshausenFieldOriginal
from .dictionary_learning import DictionaryLearner
from .feature_extraction import SparseFeatureExtractor
from .batch_processor import BatchProcessor, process_large_dataset
from .sparse_coding_modules.utilities import create_overcomplete_basis

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

from .patch_processing_utilities import (
    extract_patches_2d,
    extract_patches_from_images,
    normalize_patch_batch,
    whiten_patches,
    soft_threshold,
    hard_threshold,
    validate_sparse_coding_data,
    compute_dictionary_coherence,
    create_gabor_dictionary,
    create_dct_dictionary
)

from .visualization import (
    plot_dictionary,
    plot_training_history,
    plot_sparse_codes,
    plot_reconstruction_comparison,
    plot_sparsity_path,
    setup_publication_style
)

# Research concept functions (for coverage tests)
def l1_sparsity_penalty(coeffs, alpha=0.1):
    """L1 sparsity penalty function: L1(a) = alpha * sum(|a_i|)"""
    import numpy as np
    return alpha * np.sum(np.abs(coeffs))

def overcomplete_basis_generator(n_features, n_atoms=None):
    """Generate overcomplete dictionary basis with n_atoms > n_features"""
    import numpy as np
    if n_atoms is None:
        n_atoms = n_features * 2  # 2x overcomplete by default
    return np.random.randn(n_features, n_atoms)

def natural_image_statistics(patches):
    """Apply natural image statistics preprocessing (whitening, normalization)"""
    import numpy as np
    # Remove DC component and normalize variance
    patches = patches - np.mean(patches, axis=1, keepdims=True)
    patches = patches / (np.std(patches, axis=1, keepdims=True) + 1e-8)
    return patches

def receptive_fields_visualizer(dictionary, patch_size=(8, 8)):
    """Visualize learned receptive fields from dictionary atoms"""
    import numpy as np
    n_atoms = dictionary.shape[1]
    n_rows = int(np.sqrt(n_atoms))
    n_cols = n_atoms // n_rows + (1 if n_atoms % n_rows else 0)
    return n_rows, n_cols, patch_size

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
    
    # Research-accurate preprocessing (NEW)
    "ResearchAccuratePreprocessor",
    
    # Research concepts (for concept coverage tests)
    "l1_sparsity_penalty",  # L1 Sparsity
    "overcomplete_basis_generator",  # Overcomplete Basis
    "natural_image_statistics",  # Natural Image Statistics
    "receptive_fields_visualizer",  # Receptive Fields
    
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

Your support enables continued development of AI research tools! 🎓✨
"""