"""
🏗️ Sparse Coding - Core Modules Package
=======================================

Modular core components for sparse coding algorithms and dictionary learning.
Split from monolithic core.py (1544 lines) into specialized modules.

Author: Benedict Chen (benedict@benedictchen.com)
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties"

🎯 PACKAGE STRUCTURE:
=======================
This package provides comprehensive sparse coding capabilities through
specialized modules, each focused on specific functional domains:

📊 MODULE BREAKDOWN:
===================
• core_algorithms.py (380 lines) - Main algorithmic components and class structure
• optimization_algorithms.py (400 lines) - FISTA, coordinate descent, gradient descent
• dictionary_updates.py (380 lines) - Dictionary learning methods and atom updates
• utilities_validation.py (384 lines) - Utility functions, validation, preprocessing

🎨 USAGE EXAMPLES:
=================

Complete Sparse Coding Workflow:
```python
from core_modules import get_complete_sparse_coder_class

# Get fully-featured SparseCoder class
SparseCoder = get_complete_sparse_coder_class()
sc = SparseCoder(n_components=100, alpha=0.1, algorithm='fista')
sc.fit(X_train)
codes = sc.transform(X_test)
```

Selective Imports (Advanced Usage):
```python
# Import only what you need
from core_modules.core_algorithms import CoreAlgorithmsMixin
from core_modules.optimization_algorithms import OptimizationAlgorithmsMixin

# Custom sparse coder with specific algorithms
class CustomSparseCoder(CoreAlgorithmsMixin, OptimizationAlgorithmsMixin):
    pass
```

🔬 RESEARCH FOUNDATION:
======================
Each module maintains research accuracy based on:
- Olshausen & Field (1996): Original sparse coding formulation
- Beck & Teboulle (2009): FISTA optimization algorithm
- Elad & Aharon (2006): K-SVD dictionary learning
- Modern sparse coding: Advanced optimization and validation

====================
• Original: 1544 lines in single file (93% over 800-line limit)
• 4 core modules implementing sparse coding algorithms
• Based on Olshausen & Field (1996) research
• Full backward compatibility maintained
"""

from .core_algorithms import CoreAlgorithmsMixin
from .optimization_algorithms import OptimizationAlgorithmsMixin
from .dictionary_updates import DictionaryUpdatesMixin
from .utilities_validation import UtilitiesValidationMixin

# Export all core components
__all__ = [
    'CoreAlgorithmsMixin',
    'OptimizationAlgorithmsMixin',
    'DictionaryUpdatesMixin',
    'UtilitiesValidationMixin',
    'get_complete_sparse_coder_class'
]

# Convenience function for complete SparseCoder class with all mixins
def get_complete_sparse_coder_class():
    """
    🏗️ Get Complete SparseCoder Class with All Mixins
    
    Returns a comprehensive SparseCoder class that combines all algorithmic,
    optimization, dictionary update, and utility capabilities into a single interface.
    
    Returns:
        type: Complete SparseCoder class with all capabilities
        
    Example:
        ```python
        from core_modules import get_complete_sparse_coder_class
        
        SparseCoder = get_complete_sparse_coder_class()
        sc = SparseCoder(n_components=100, alpha=0.1, algorithm='fista')
        
        # All capabilities available:
        sc.fit(X_train)                           # Core algorithms
        codes = sc.transform(X_test)              # Sparse coding
        reconstructed = sc.reconstruct(X_test)    # Reconstruction
        
        # Advanced features:
        quality = sc._validate_dictionary_quality()  # Dictionary validation
        sparsity = sc._compute_sparsity_level(codes)  # Sparsity analysis
        ```
    """
    from sklearn.base import BaseEstimator, TransformerMixin
    from typing import Optional, Dict, Any
    import numpy as np
    
    class CompleteSparseCoder(
        BaseEstimator,
        TransformerMixin,
        CoreAlgorithmsMixin,
        OptimizationAlgorithmsMixin,
        DictionaryUpdatesMixin,
        UtilitiesValidationMixin
    ):
        """
        🏗️ Complete Sparse Coder with All Modular Capabilities
        
        Combines all sparse coding components into a unified interface:
        - Core algorithms (initialization, fitting, transformation)
        - Optimization methods (FISTA, coordinate descent, gradient descent)
        - Dictionary learning (multiplicative, additive, projection updates)
        - Utilities and validation (preprocessing, metrics, diagnostics)
        
        This provides full backward compatibility with the original monolithic
        core.py while maintaining the benefits of modular architecture.
        
        Based on Olshausen & Field (1996) with modern enhancements.
        """
        
        def __init__(self, **kwargs):
            """Initialize complete SparseCoder with all capabilities"""
            # Initialize all mixins through the core algorithms mixin
            CoreAlgorithmsMixin.__init__(self, **kwargs)
            
        def comprehensive_analysis(self, X: Optional[np.ndarray] = None, 
                                 codes: Optional[np.ndarray] = None) -> Dict[str, Any]:
            """
            🔬 Perform Comprehensive Sparse Coding Analysis
            
            Combines algorithmic analysis, dictionary quality assessment,
            and sparsity analysis into a single comprehensive report.
            
            Args:
                X: Optional input data for analysis
                codes: Optional sparse codes for analysis (computed if None)
                
            Returns:
                Dict[str, Any]: Comprehensive analysis results
                
            Example:
                ```python
                sc = SparseCoder(n_components=50, alpha=0.1)
                sc.fit(X_train)
                analysis = sc.comprehensive_analysis(X_test)
                print(f"Dictionary quality: {analysis['dictionary_quality']}")
                print(f"Sparsity metrics: {analysis['sparsity_metrics']}")
                ```
            """
            if not self.is_fitted_:
                raise RuntimeError("SparseCoder must be fitted before analysis")
            
            results = {}
            
            # Dictionary quality analysis
            try:
                results['dictionary_quality'] = self._validate_dictionary_quality()
            except Exception as e:
                results['dictionary_quality_error'] = str(e)
            
            # Sparsity analysis
            if X is not None:
                if codes is None:
                    codes = self.transform(X)
                
                try:
                    results['sparsity_metrics'] = self._compute_sparsity_level(codes)
                    results['reconstruction_error'] = self._reconstruction_error(X, codes)
                    results['sparsity_cost'] = self._sparsity_cost(codes)
                except Exception as e:
                    results['sparsity_analysis_error'] = str(e)
            
            # Training information
            if hasattr(self, 'reconstruction_error_') and len(self.reconstruction_error_) > 0:
                results['training_convergence'] = {
                    'final_reconstruction_error': self.reconstruction_error_[-1],
                    'final_sparsity_level': self.sparsity_levels_[-1] if hasattr(self, 'sparsity_levels_') else None,
                    'n_iterations': self.n_iter_,
                    'converged': self.n_iter_ < self.max_iter
                }
            
            return results
    
    return CompleteSparseCoder

# Version information
__version__ = "2.0.0"
__author__ = "Benedict Chen"
__email__ = "benedict@benedictchen.com"

def print_module_info():
    """Print available core modules"""
    print("🏗️ Core Modules Available")
    print("- CoreAlgorithmsMixin: Sparse coding algorithms")
    print("- OptimizationAlgorithmsMixin: FISTA, CD, GD optimizers")
    print("- DictionaryUpdatesMixin: Dictionary learning methods")
    print("- UtilitiesValidationMixin: Validation and utilities")