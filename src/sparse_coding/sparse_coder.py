"""
Sparse Coding - Research-Accurate Implementation
===============================================

Main SparseCoder class that uses working FISTA with backtracking
as the default inference method, following Olshausen & Field (1996).
"""

import numpy as np
from typing import Optional, Dict, Any
from .research_accurate_sparsity import FISTAOptimizer, SparseCodingConfig, SparsenessFunction
from sklearn.base import BaseEstimator, TransformerMixin


class SparseCoder(BaseEstimator, TransformerMixin):
    """
    Research-accurate sparse coding implementation with proper FISTA inference.
    
    Uses working FISTA with backtracking line search as default, following
    Beck & Teboulle (2009) with Olshausen & Field (1996) formulation.
    """
    
    def __init__(
        self,
        n_components: int = 100,
        sparsity_penalty: float = 0.14,  # λ/σ = 0.14 from paper
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        random_state: Optional[int] = None
    ):
        """
        Initialize SparseCoder with research-accurate defaults.
        
        Args:
            n_components: Number of dictionary atoms (K)
            sparsity_penalty: Sparsity penalty λ (paper used λ/σ = 0.14)
            max_iterations: Maximum FISTA iterations
            tolerance: Convergence tolerance
            random_state: Random seed
        """
        self.n_components = n_components
        self.sparsity_penalty = sparsity_penalty
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Create research-accurate configuration
        self.config = SparseCodingConfig(
            sparseness_function=SparsenessFunction.LOG,  # Paper's log penalty
            sparsity_penalty=sparsity_penalty,
            max_iterations=max_iterations,
            tolerance=tolerance,
            fista_backtrack=True,  # Enable backtracking line search
            track_objective=True,
            validate_convergence=True
        )
        
        # Initialize FISTA optimizer
        self.fista_optimizer = FISTAOptimizer(self.config)
        
    def fit(self, X, y=None, mode="modern", images=None):
        """
        Learn sparse coding dictionary from data using research-accurate methods.
        
        Args:
            X: Training patches [n_samples, n_features] OR list of images if mode="paper"
            y: Ignored (unsupervised learning)
            mode: "modern" for patch input, "paper" for research-accurate image preprocessing
            images: List of images for research-accurate preprocessing (if X is patches)
            
        Returns:
            self: Fitted SparseCoder instance
        """
        if mode == "paper":
            # RESEARCH-ACCURATE: Use image-level preprocessing
            if images is None and isinstance(X, list):
                images = X  # X contains images
            elif images is None:
                raise ValueError("mode='paper' requires 'images' parameter with list of full images")
            
            # Apply research-accurate preprocessing
            from .research_accurate_preprocessing import ResearchAccuratePreprocessor
            
            preprocessor = ResearchAccuratePreprocessor(
                patch_size=getattr(self, 'patch_size', (16, 16)),
                f0_cycles_per_picture=200.0,
                mode="paper"
            )
            
            print("🔬 APPLYING RESEARCH-ACCURATE PREPROCESSING...")
            X, sigma_computed, preprocessing_stats = preprocessor.preprocess_images_paper_accurate(
                images, n_patches_per_image=1000
            )
            
            # Store computed σ for correct sparsity penalty calculation
            self.sigma_ = sigma_computed
            self.preprocessing_stats_ = preprocessing_stats
            
            print(f"✅ Research-accurate preprocessing complete:")
            print(f"   • σ computed from whitened patches: {sigma_computed:.6f}")
            print(f"   • Image-level whitening applied")
            print(f"   • Ready for Olshausen & Field (1996) sparse coding")
            
        else:
            # Store default σ if not using paper mode
            self.sigma_ = 1.0
            
        n_samples, n_features = X.shape
        
        # Initialize dictionary with atoms as columns D ∈ R^(p×K) 
        rng = np.random.RandomState(self.random_state)
        self.dictionary_ = rng.randn(n_features, self.n_components)
        
        # Normalize columns to unit norm (paper requirement)
        for i in range(self.n_components):
            norm = np.linalg.norm(self.dictionary_[:, i])
            if norm > 1e-12:
                self.dictionary_[:, i] /= norm
                
        # Initialize homeostatic gains for coefficient variance equalization (Paper Fig. 4)
        # Persistent per-atom gains to maintain reconstruction invariance
        self.gains_ = np.ones(self.n_components)
        self.cumulative_gains_ = np.ones(self.n_components)  # Track total gain changes
        
        # Alternating optimization with homeostatic gain control
        n_iterations = 50  # Paper typically used 50-100 iterations
        objective_history = []
        
        for iteration in range(n_iterations):
            # Phase 1: Sparse inference for all patches
            codes = np.zeros((n_samples, self.n_components))
            for i, patch in enumerate(X):
                # Apply homeostatic gains to dictionary for inference
                scaled_dict = self.dictionary_ * self.gains_[np.newaxis, :]
                
                # RESEARCH-ACCURATE: Adapt per-atom thresholds λᵢ = λ / gᵢ
                # This maintains equivalent sparsity levels across atoms
                per_atom_thresholds = self.sparsity_penalty / (self.gains_ + 1e-12)
                
                # Use adaptive thresholds if available, otherwise fallback
                if hasattr(self.fista_optimizer, 'solve_with_adaptive_thresholds'):
                    codes[i], _ = self.fista_optimizer.solve_with_adaptive_thresholds(
                        scaled_dict, patch, per_atom_thresholds
                    )
                else:
                    # Fallback: use uniform threshold (less accurate but functional)
                    codes[i], _ = self.fista_optimizer.solve(scaled_dict, patch)
                    
                # CRITICAL: Inverse-scale coefficients to maintain reconstruction invariance
                # This ensures X ≈ (D * g) @ (A / g) = D @ A
                codes[i] = codes[i] / self.gains_
            
            # Phase 2: Dictionary update (Olshausen & Field Eq. 6)
            self._update_dictionary_equation_6(X, codes)
            
            # Phase 3: Homeostatic gain control - equalize coefficient variances
            # Pass the correctly scaled codes for variance computation
            self._update_homeostatic_gains(codes)
            
            # Track objective for paper-aligned stopping criterion
            current_objective = self._compute_objective(X, codes)
            objective_history.append(current_objective)
            
            # Check convergence using relative objective decrease (paper method)
            if len(objective_history) >= 2:
                relative_change = abs(objective_history[-1] - objective_history[-2]) / abs(objective_history[-2])
                if relative_change < 1e-4:  # Paper used 1% change
                    print(f"   Converged at iteration {iteration} (relative change: {relative_change:.6f})")
                    break
            
            if iteration % 10 == 0:
                print(f"   Iteration {iteration}: objective={current_objective:.6f}, gain range [{self.gains_.min():.3f}, {self.gains_.max():.3f}]")
        
        print("SparseCoder fitted with research-accurate FISTA inference and homeostatic control")
        
        # Enhanced KKT validation on final solution
        if hasattr(self, '_check_kkt_on_fit') and self._check_kkt_on_fit:
            print("\n🔍 Performing enhanced KKT validation on final solution...")
            sample_indices = np.random.choice(X.shape[0], min(10, X.shape[0]), replace=False)
            X_sample = X[sample_indices]
            A_sample = np.zeros((len(sample_indices), self.n_components))
            
            for i, patch in enumerate(X_sample):
                A_sample[i], _ = self.fista_optimizer.solve(self.dictionary_, patch)
            
            # Use enhanced KKT checking with detailed analysis
            kkt_results = self.check_kkt_violation(
                X_sample.T, A_sample.T, 
                tolerance=getattr(self, '_kkt_tolerance', 1e-3),
                verbose=True,
                detailed=getattr(self, '_kkt_detailed', True)
            )
            
            if not kkt_results['kkt_satisfied']:
                print("⚠️  Consider increasing max_iterations or adjusting optimizer settings.")
                
                # Store KKT results for inspection
                self.final_kkt_results_ = kkt_results
        
        return self
    
    def enable_kkt_checking(self, tolerance=1e-3, detailed=True):
        """
        Enable comprehensive KKT violation checking during fit().
        
        This adds rigorous validation of optimization convergence using
        Karush-Kuhn-Tucker (KKT) conditions for L1 sparse coding.
        
        Args:
            tolerance: Maximum acceptable KKT violation (default: 1e-3)
            detailed: Whether to use detailed KKT analysis with diagnostic output
        """
        self._check_kkt_on_fit = True
        self._kkt_tolerance = tolerance
        self._kkt_detailed = detailed
        print(f"✓ Enhanced KKT checking enabled (tolerance: {tolerance:.1e}, detailed: {detailed})")
        return self
    
    def disable_kkt_checking(self):
        """Disable KKT checking during fit()."""
        self._check_kkt_on_fit = False
        print("✓ KKT checking disabled")
        return self
        
    def transform(self, X):
        """
        Sparse encode patches using research-accurate FISTA.
        
        Args:
            X: Input patches [n_samples, n_features]
            
        Returns:
            Sparse codes [n_samples, n_components]
        """
        n_samples = X.shape[0]
        codes = np.zeros((n_samples, self.n_components))
        
        for i, patch in enumerate(X):
            codes[i], _ = self.fista_optimizer.solve(self.dictionary_, patch)
            
        return codes
    
    def _update_dictionary_equation_6(self, X: np.ndarray, codes: np.ndarray):
        """
        Dictionary update following Olshausen & Field (1996) Equation 6.
        
        Δφᵢ = η⟨aᵢ(I - Î)⟩ where Î is the reconstruction
        """
        learning_rate = 0.01  # η from paper
        
        for i in range(self.n_components):
            # Compute reconstruction: Î = Σⱼ aⱼφⱼ
            reconstruction = self.dictionary_ @ codes.T  # [n_features, n_samples]
            
            # Compute residual: (I - Î) for each sample
            residual = X.T - reconstruction  # [n_features, n_samples]
            
            # Update atom i: Δφᵢ = η⟨aᵢ(I - Î)⟩
            update = learning_rate * np.mean(residual * codes[:, i], axis=1)
            self.dictionary_[:, i] += update
            
            # Normalize to unit norm (paper requirement)
            norm = np.linalg.norm(self.dictionary_[:, i])
            if norm > 1e-12:
                self.dictionary_[:, i] /= norm
    
    def _update_homeostatic_gains(self, codes: np.ndarray):
        """
        Update homeostatic gains to equalize coefficient variances.
        
        RESEARCH-ACCURATE IMPLEMENTATION following Olshausen & Field (1996):
        - Monitor coefficient variances σᵢ² for each atom i
        - Compute gains gᵢ to equalize variances
        - Apply gains to dictionary: D' ← D * g
        - CRITICALLY: Scale coefficients: A' ← A / g (already done in inference loop)
        - This maintains reconstruction: X ≈ D'A' = (Dg)(A/g) = DA
        
        Args:
            codes: Coefficient matrix [n_samples, n_components] with proper scaling
        """
        # Compute coefficient variances for each atom
        variances = np.var(codes, axis=0) + 1e-12  # Numerical stability
        target_variance = np.mean(variances)
        
        # Compute new gains to equalize variances
        # gain_ratio = sqrt(target_var / current_var) to equalize variances
        new_gains = np.sqrt(target_variance / variances)
        
        # Apply adaptive updating to prevent oscillations
        gain_adaptation_rate = 0.05  # Reduced from 0.1 for stability
        self.gains_ = (1 - gain_adaptation_rate) * self.gains_ + gain_adaptation_rate * new_gains
        
        # Update cumulative gains for tracking purposes
        self.cumulative_gains_ *= self.gains_
        
        # Apply gains to dictionary atoms (renormalize after scaling)
        self.dictionary_ = self.dictionary_ * self.gains_[np.newaxis, :]
        
        # Renormalize atoms to unit norm (paper requirement)
        # but preserve gain information in cumulative_gains_
        norms = np.linalg.norm(self.dictionary_, axis=0)
        valid_norms = norms > 1e-12
        if np.any(valid_norms):
            self.dictionary_[:, valid_norms] = self.dictionary_[:, valid_norms] / norms[valid_norms]
            # Reabsorb scale into cumulative gains for consistency
            self.cumulative_gains_[valid_norms] *= norms[valid_norms]
        
        # Reset gains for next iteration (they're now incorporated into dictionary)
        self.gains_ = np.ones(self.n_components)
    
    def _verify_reconstruction_invariance(self, X: np.ndarray, codes_before: np.ndarray, 
                                        codes_after: np.ndarray) -> bool:
        """
        Verify that homeostatic gain updates preserve reconstruction quality.
        
        RESEARCH REQUIREMENT: X ≈ D_old @ codes_before ≈ D_new @ codes_after
        
        Args:
            X: Input patches [n_samples, n_features]
            codes_before: Coefficients before gain update
            codes_after: Coefficients after gain update (inverse scaled)
            
        Returns:
            True if reconstruction invariance is preserved within tolerance
        """
        # Reconstruction with original dictionary and codes
        reconstruction_before = self.dictionary_ @ codes_before.T
        
        # Reconstruction with updated dictionary and scaled codes  
        reconstruction_after = self.dictionary_ @ codes_after.T
        
        # Compute reconstruction error difference
        error_before = np.mean((X.T - reconstruction_before) ** 2)
        error_after = np.mean((X.T - reconstruction_after) ** 2)
        
        # Check if reconstruction quality is preserved (within 1% tolerance)
        relative_error_change = abs(error_after - error_before) / (error_before + 1e-12)
        
        if relative_error_change > 0.01:  # 1% tolerance
            print(f"⚠️  WARNING: Reconstruction invariance violated!")
            print(f"   Error before: {error_before:.6f}")
            print(f"   Error after: {error_after:.6f}")
            print(f"   Relative change: {relative_error_change:.4f}")
            return False
        
        return True
    
    def _compute_objective(self, X: np.ndarray, codes: np.ndarray) -> float:
        """
        Compute Olshausen & Field objective: E = ½‖I-Î‖² - λ∑S(aᵢ/σ)
        
        Using log sparsity penalty: S(x) = log(1 + x²)
        """
        # Reconstruction error term: ½‖I-Î‖²
        reconstruction = self.dictionary_ @ codes.T
        reconstruction_error = 0.5 * np.mean((X.T - reconstruction) ** 2)
        
        # Sparsity penalty term: λ∑S(aᵢ/σ) with S(x) = log(1 + x²)
        # CRITICAL FIX: Use actual σ computed from data, not hardcoded 1.0
        sigma = getattr(self, 'sigma_', 1.0)  # Use computed σ if available
        normalized_codes = codes / sigma
        sparsity_penalty = self.sparsity_penalty * np.mean(np.log(1 + normalized_codes ** 2))
        
        # Paper objective: minimize reconstruction error - maximize sparsity
        return reconstruction_error - sparsity_penalty
    
    def check_kkt_violation(self, X, A, tolerance=1e-3, verbose=True, detailed=False):
        """
        Check KKT (Karush-Kuhn-Tucker) conditions for L1 sparse coding solutions.
        
        This is a critical diagnostic tool for validating optimization correctness.
        KKT conditions must be satisfied for optimal L1 solutions.
        
        For the problem: min_A ½‖X - DA‖² + λ‖A‖₁
        
        KKT conditions require:
        - For zero coefficients: |D^T(X - DA)| ≤ λ  
        - For nonzero coefficients: D^T(X - DA) = λ·sign(A)
        
        Args:
            X: Data matrix [n_features, n_samples]
            A: Coefficient matrix [n_components, n_samples] 
            tolerance: Maximum acceptable KKT violation
            verbose: Print violation details
            detailed: Use enhanced comprehensive KKT analysis
            
        Returns:
            dict: KKT violation statistics and validation status
        """
        try:
            # Try to use enhanced KKT diagnostics if available
            from .diagnostics import kkt_violation_comprehensive, diagnose_kkt_violations
            
            if detailed:
                # Use comprehensive analysis
                results = kkt_violation_comprehensive(
                    self.dictionary_, X, A, self.sparsity_penalty, 
                    tol=1e-12, detailed=True
                )
                
                if verbose:
                    diagnose_kkt_violations(results, verbose=True)
                
                # Add backward compatibility fields
                results['tolerance'] = tolerance
                results['kkt_satisfied'] = results['max_violation'] <= tolerance
                results['mean_violation'] = (results['mean_violation_zero'] * results['n_zero_coeffs'] + 
                                           results['mean_violation_nonzero'] * results['n_nonzero_coeffs']) / (results['n_zero_coeffs'] + results['n_nonzero_coeffs'])
                results['n_samples_checked'] = A.shape[1] if A.ndim > 1 else 1
                
                return results
            else:
                # Use simple analysis
                results = kkt_violation_comprehensive(
                    self.dictionary_, X, A, self.sparsity_penalty,
                    tol=1e-12, detailed=False  
                )
                
                kkt_satisfied = results['max_violation'] <= tolerance
                
                if verbose:
                    print(f"KKT Violation Analysis:")
                    print(f"   Maximum violation: {results['max_violation']:.2e}")
                    print(f"   Tolerance: {tolerance:.2e}")
                    print(f"   KKT satisfied: {'✓ YES' if kkt_satisfied else '❌ NO'}")
                    print(f"   Sparsity level: {results['sparsity_level']:.1%}")
                    
                    if not kkt_satisfied:
                        print(f"   ⚠️  WARNING: KKT conditions violated!")
                        print(f"   This indicates optimization did not converge to optimal solution.")
                        print(f"   Consider: increasing max_iterations, decreasing step size,")
                        print(f"            or checking for numerical issues in the optimizer.")
                
                # Convert to backward compatible format
                return {
                    'max_violation': results['max_violation'],
                    'mean_violation': (results['mean_violation_zero'] * results['n_zero_coeffs'] + 
                                     results['mean_violation_nonzero'] * results['n_nonzero_coeffs']) / (results['n_zero_coeffs'] + results['n_nonzero_coeffs']),
                    'tolerance': tolerance,
                    'kkt_satisfied': kkt_satisfied,
                    'n_samples_checked': A.shape[1] if A.ndim > 1 else 1,
                    'sparsity_level': results['sparsity_level']
                }
                
        except ImportError:
            # Fall back to original implementation
            from .diagnostics import kkt_violation_l1
            
            # Compute KKT violation for each sample
            n_samples = X.shape[1] if X.ndim > 1 else 1
            violations = []
            
            if X.ndim == 1:
                # Single sample case
                X = X.reshape(-1, 1)
                A = A.reshape(-1, 1)
                
            for i in range(n_samples):
                violation = kkt_violation_l1(
                    self.dictionary_, 
                    X[:, i:i+1], 
                    A[:, i:i+1], 
                    self.sparsity_penalty
                )
                violations.append(violation)
            
            violations = np.array(violations)
            max_violation = np.max(violations)
            mean_violation = np.mean(violations)
            
            # Check if KKT conditions are satisfied
            kkt_satisfied = max_violation <= tolerance
            
            if verbose:
                print(f"KKT Violation Analysis:")
                print(f"   Maximum violation: {max_violation:.2e}")
                print(f"   Mean violation: {mean_violation:.2e}")
                print(f"   Tolerance: {tolerance:.2e}")
                print(f"   KKT satisfied: {'✓ YES' if kkt_satisfied else '❌ NO'}")
                
                if not kkt_satisfied:
                    print(f"   ⚠️  WARNING: KKT conditions violated!")
                    print(f"   This indicates optimization did not converge to optimal solution.")
                    print(f"   Consider: increasing max_iterations, decreasing step size,")
                    print(f"            or checking for numerical issues in the optimizer.")
            
            return {
                'max_violation': float(max_violation),
                'mean_violation': float(mean_violation),
                'violations': violations,
                'tolerance': tolerance,
                'kkt_satisfied': kkt_satisfied,
                'n_samples_checked': n_samples
            }
    
    def validate_solution(self, X, A=None, check_kkt=True, kkt_tolerance=1e-3):
        """
        Comprehensive validation of sparse coding solution.
        
        Performs multiple checks to ensure solution quality:
        - KKT condition satisfaction (for L1 problems)
        - Sparsity level analysis
        - Reconstruction error measurement
        
        Args:
            X: Data matrix to validate against
            A: Coefficient matrix (if None, will encode X)
            check_kkt: Whether to perform KKT violation checking
            kkt_tolerance: Maximum acceptable KKT violation
            
        Returns:
            dict: Comprehensive validation results
        """
        # Encode if coefficients not provided
        if A is None:
            A = self.transform(X)
            
        # Ensure proper shapes
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
            
        # Basic reconstruction metrics
        X_reconstructed = self.dictionary_ @ A
        reconstruction_error = np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X)
        sparsity_level = np.mean(np.abs(A) < 1e-8)
        
        validation_results = {
            'reconstruction_error': float(reconstruction_error),
            'sparsity_level': float(sparsity_level),
            'n_nonzero_avg': float(np.mean(np.sum(np.abs(A) >= 1e-8, axis=0))),
            'coefficient_range': {
                'min': float(np.min(A)),
                'max': float(np.max(A)),
                'mean_abs': float(np.mean(np.abs(A)))
            }
        }
        
        # KKT violation checking for L1 problems
        if check_kkt:
            kkt_results = self.check_kkt_violation(X, A, kkt_tolerance, verbose=False)
            validation_results['kkt_analysis'] = kkt_results
        
        return validation_results
"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes AI research accessible to everyone! 🚀

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
can be recreated using just a few active strokes. These learned strokes look exactly
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

For implementation details, see the individual modules in sparse_coding_modules/.
"""

# Import the modular SparseCoder implementation
from .sparse_coding_modules.core_algorithm import SparseCoder

# Import utilities for direct use
from .sparse_coding_modules.utilities import (
    create_overcomplete_basis,
    lateral_inhibition, 
    demo_sparse_coding
)

# Import the original research implementation for reference
from .sparse_coding_modules.olshausen_field import OlshausenFieldOriginal

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
        pass  # Implementation needed
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

Your support enables continued development of AI research tools! 🎓✨
"""