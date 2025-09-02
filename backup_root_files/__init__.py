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

This root __init__.py redirects to the real implementation in src/sparse_coding/
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

# Import everything from the real implementation in src/sparse_coding/
import sys
import os

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import all real implementations from src/sparse_coding/
try:
    from sparse_coding import (
        SparseCoder,
        DictionaryLearner,
        SparseFeatureExtractor,
        BatchProcessor,
        process_large_dataset,
        SparseVisualization,
        sc_modules
    )
    
    # Show attribution on library import
    _print_attribution()
    
    __version__ = "2.1.0"
    __authors__ = ["Benedict Chen", "Based on Olshausen & Field (1996)"]
    
    # Define explicit public API
    __all__ = [
        # Core class (REAL implementation)
        "SparseCoder",
        
        # Dictionary learning
        "DictionaryLearner",
        
        # Feature processing
        "SparseFeatureExtractor",
        "BatchProcessor",
        "process_large_dataset",
        
        # Visualization
        "SparseVisualization",
        
        # Module access
        "sc_modules",
    ]
    
except ImportError as e:
    print(f"⚠️ Error importing from src/sparse_coding/: {e}")
    print("   Please ensure the src/ directory structure is correct")
    
    # Show attribution even if import fails
    _print_attribution()
    
    __version__ = "2.1.0" 
    __authors__ = ["Benedict Chen", "Based on Olshausen & Field (1996)"]

"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
📝 CITE: Benedict Chen (2025) - Sparse Coding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""