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
🏗️ Sparse Coding - Utils Modules Package
========================================

Modular utilities for sparse coding split from monolithic utils.py (994 lines).

Author: Benedict Chen (benedict@benedictchen.com)
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties"

🎯 PACKAGE STRUCTURE:
=======================
This package provides comprehensive sparse coding utilities through
specialized modules, each focused on specific functional domains:

📊 MODULE BREAKDOWN:
===================
• data_processing.py (256 lines) - Patch extraction, normalization, reconstruction
• optimization.py (213 lines) - Thresholding operators, line search, Lipschitz computation  
• validation_metrics.py (244 lines) - Data validation, dictionary coherence, convergence
• advanced_specialized.py (284 lines) - Gabor/DCT dictionaries, lateral inhibition

🎨 USAGE EXAMPLES:
=================

Complete Utils Import:
```python
from sparse_coding.utils_modules import *

# All utility functions available
patches = extract_patches_2d(image, (8, 8))
codes_thresh = soft_threshold(codes, 0.1)
metrics = validate_sparse_coding_data(X, dictionary, codes)
gabor_dict = create_gabor_dictionary((8, 8))
```

Selective Imports (Recommended):
```python
# Import only what you need
from sparse_coding.utils_modules.data_processing import extract_patches_2d, normalize_patch_batch
from sparse_coding.utils_modules.optimization import soft_threshold, compute_lipschitz_constant
from sparse_coding.utils_modules.validation_metrics import compute_dictionary_coherence
from sparse_coding.utils_modules.advanced_specialized import create_gabor_dictionary

# Use specific functionality
patches = extract_patches_2d(image, (8, 8))
normalized = normalize_patch_batch(patches)
thresh_codes = soft_threshold(codes, 0.1)
coherence = compute_dictionary_coherence(dictionary)
```

🔬 RESEARCH FOUNDATION:
======================
Each module maintains research accuracy based on:
- Olshausen & Field (1996): Patch extraction and sparse coding fundamentals
- ISTA/FISTA algorithms: Optimization and thresholding operators
- Dictionary learning theory: Coherence, spark, and quality metrics
- Computer vision: Gabor filters, DCT, and advanced preprocessing

• Utility functions for sparse coding operations
• 4 focused modules with clear responsibilities
• Modular organization for maintainability
• Complete sparse coding utility functions
• Full backward compatibility through integration layer
"""

# Import all modules
from .data_processing import *
from .optimization import *
from .validation_metrics import *
from .advanced_specialized import *

# Export all functions for backward compatibility
__all__ = [
    # Data processing functions
    'extract_patches_2d',
    'extract_patches_from_images', 
    'normalize_patch_batch',
    'whiten_patches',
    'reconstruct_image_from_patches',
    
    # Optimization functions
    'soft_threshold',
    'hard_threshold', 
    'shrinkage_threshold',
    'compute_lipschitz_constant',
    'line_search_backtrack',
    
    # Validation and metrics functions
    'validate_sparse_coding_data',
    'compute_dictionary_coherence',
    'compute_spark',
    'validate_convergence',
    
    # Advanced and specialized functions
    'create_gabor_dictionary',
    'create_dct_dictionary',
    'lateral_inhibition_network',
    'estimate_noise_variance',
    'compute_mutual_coherence_matrix',
    'orthogonalize_dictionary'
]

# Version information
__version__ = "2.0.0"
__author__ = "Benedict Chen"
__email__ = "benedict@benedictchen.com"

# Module information for reporting
MODULE_INFO = {
    'total_modules': 4,
    'original_lines': 994,
    'total_lines': 997,
    'largest_module': 284,
    'average_module_size': 249,
    'organization': "4 focused modules",
    'compliance_status': "✅ All modules under 800-line limit"
}

def print_module_info():
    """Print module information"""
    # print("🏗️ Utils Modules - Information")
    print("=" * 50)
    for key, value in MODULE_INFO.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    # print("🏗️ Sparse Coding - Utils Modules Package")
    print("=" * 50)
    # Removed print spam: "...
    print(f"  Utility functions loaded successfully")
    print(f"  4 modules available with 997 total lines")
    print(f"  All utility modules available")
    print("")
    # Removed print spam: "...
    print(f"  • Data processing utilities: 256 lines")
    print(f"  • Optimization utilities: 213 lines")
    print(f"  • Validation and metrics: 244 lines") 
    print(f"  • Advanced specialized tools: 284 lines")
    print("")
    # # Removed print spam: "...
    # print("🏗️ Sparse coding utilities based on research literature!")
    # # Removed print spam: "...
    print("")
    print_module_info()

print("""
💰 MODULE SUPPORT - Made possible by Benedict Chen
   ]8;;mailto:benedict@benedictchen.com\benedict@benedictchen.com]8;;\

💰 PLEASE DONATE! Your support keeps this research alive! 💰
   🔗 ]8;;https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS\💳 CLICK HERE TO DONATE VIA PAYPAL]8;;\
   ❤️ ]8;;https://github.com/sponsors/benedictchen\💖 SPONSOR ON GITHUB]8;;\

   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!
   (Start small, dream big! Every donation helps! 😄)
""")
