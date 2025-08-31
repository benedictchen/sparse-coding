"""
✨ Sparse Coding - Learning the Language of Natural Images
========================================================

Author: Benedict Chen (benedict@benedictchen.com)

💝 Support This Work: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
Developing high-quality, research-backed software takes countless hours of study, implementation, 
testing, and documentation. Your support - whether a little or a LOT - makes this work possible and is 
deeply appreciated. Please consider donating based on how much this module impacts your life or work!

Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"

🎯 ELI5 Summary:
Imagine you're an artist trying to recreate any picture using the fewest brush strokes possible.
Sparse coding finds the perfect set of "brush strokes" (basis functions) where any natural image
can be recreated using just a few active strokes. Amazingly, these learned strokes look exactly
like what neurons in your visual cortex respond to - edge detectors, line segments, etc!

🔬 Research Background:
========================
Bruno Olshausen and David Field's 1996 breakthrough solved a fundamental puzzle in neuroscience:
Why do simple cells in the visual cortex have oriented, localized receptive fields? Their answer
was revolutionary: because this is the optimal sparse representation for natural images!

The Sparse Coding Revolution:
- **Efficiency Principle**: Represent data with minimal active elements
- **Biological Discovery**: V1 simple cells = sparse coding basis functions  
- **Unsupervised Learning**: No labels needed, just natural image statistics
- **Dictionary Learning**: Learn optimal basis functions from data
- **Sparsity Prior**: Few active components explain most variation

This launched entire fields:
- Compressed sensing and signal processing
- Dictionary learning and matrix factorization  
- Understanding biological vision systems
- Modern deep learning (CNNs use similar principles)

🏗️ Mathematical Framework:
==========================
Given image patches X, find dictionary D and sparse codes S such that:
X ≈ D × S  where S is sparse (mostly zeros)

Optimization Problem:
min_{D,S} ||X - DS||²₂ + λ∑|S_i|  (L1 penalty for sparsity)

🎨 ASCII Diagram - Sparse Representation:
=========================================
Original Image Patch:     Dictionary:        Sparse Code:
    ╔════════════╗         ╔═══╗ ╔═══╗        [0.0, 2.1, 0.0, 
    ║████░░░░████║    =    ║▌▌▌║+║ ─ ║×        0.0, 0.0, 1.5,
    ║██░░░░░░░░██║         ║▌▌▌║ ║ ─ ║         0.0, 0.0, 0.0,
    ║░░░░░░░░░░░░║         ║▌▌▌║ ║ ─ ║         0.0, 0.0, 0.0]
    ║░░░░░░░░░░░░║         ╚═══╝ ╚═══╝           ↑
    ╚════════════╝        Edge    Line      Only 2 active!
     (256 pixels)       Detector Detector   (98% sparse)

Learning Process:
Input Patches → [Dictionary Learning] → Basis Functions → [Sparse Inference] → Codes

🚀 Key Innovation: Sparse representation matches biological vision
Revolutionary Impact: Unified machine learning and computational neuroscience

⚡ Configurable Options:
=======================
✨ Sparseness Functions:
  - l1: |x| penalty - standard, convex, efficient [default]
  - log: log(1+x²) - smooth, differentiable
  - gaussian: exp(-x²/2σ²) - probabilistic interpretation

✨ Optimization Methods:
  - coordinate_descent: Iterative single-variable optimization [default]
  - equation_5: Original Olshausen-Field update rule  
  - fista: Fast Iterative Shrinkage-Thresholding Algorithm
  - proximal_gradient: Proximal gradient methods

✨ L1 Solvers:
  - coordinate_descent: Efficient for sparse problems [default]
  - lbfgs_b: Limited-memory BFGS with bounds
  - fista: Accelerated proximal methods

🎨 Core Algorithms:
==================
🔧 Dictionary Update: Learn optimal basis functions from data
🔧 Sparse Inference: Find sparse codes given dictionary
🔧 Alternating Optimization: Iteratively update dictionary and codes
🔧 Feature Extraction: Transform raw data to sparse features

📊 Learning Algorithm:
=====================
1. **Initialize**: Random dictionary D, sparse codes S
2. **Sparse Inference**: Fix D, optimize S (sparse coding step)
   S ← argmin_S ||X - DS||²₂ + λ||S||₁
3. **Dictionary Update**: Fix S, optimize D (dictionary learning step)  
   D ← argmin_D ||X - DS||²₂ subject to ||d_i||₂ = 1
4. **Repeat**: Until convergence

Biological Insight:
Dictionary elements learned from natural images spontaneously develop:
- Oriented edge detectors (like V1 simple cells)
- Different scales and orientations
- Localized receptive fields  
- Frequency-selective responses

🎯 Applications:
===============
- 👁️ Computer Vision: Feature extraction, object recognition
- 🎨 Image Processing: Denoising, inpainting, super-resolution  
- 🔊 Audio Processing: Music transcription, speech enhancement
- 🧠 Neuroscience: Understanding visual cortex organization
- 📊 Data Analysis: Dimensionality reduction, pattern discovery
- 💾 Compression: Efficient image and signal compression
- 🎭 Art/Graphics: Texture synthesis, style transfer

⚡ Key Benefits:
===============
✅ Unsupervised Learning: No labeled data required
✅ Biological Plausibility: Matches actual neural responses  
✅ Interpretability: Learned features are meaningful
✅ Efficiency: Sparse representation reduces storage/computation
✅ Robustness: Tolerant to noise and missing data
✅ Universality: Works across many domains (vision, audio, etc.)

⚠️ Computational Considerations:
===============================
- Dictionary learning is non-convex (local minima possible)
- Sparse inference requires iterative optimization
- Computational cost scales with dictionary size
- Memory requirements for large patch collections
- Initialization affects final solution quality

💝 Please support our work: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
Buy us a coffee, beer, or better! Your support makes advanced AI research accessible to everyone! ☕🍺🚀
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any
from scipy.optimize import minimize
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')


class SparseCoder:
    """
    Sparse Coding implementation following Olshausen & Field's original algorithm
    
    The key insight: Natural images are sparse in the right representation.
    Instead of dense pixel representations, we learn a dictionary of basis
    functions where each image is a sparse combination of few active elements.
    """
    
    def __init__(
        self,
        n_components: int = 256,
        sparsity_penalty: float = 0.1,
        patch_size: Tuple[int, int] = (16, 16),
        max_iter: int = 100,
        tolerance: float = 1e-6,
        dictionary: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
        # Configuration options for FIXME implementations
        sparseness_function: str = 'l1',  # 'log', 'l1', 'gaussian', 'huber', 'elastic_net', 'cauchy', 'student_t'
        optimization_method: str = 'coordinate_descent',  # 'equation_5', 'fista', 'proximal_gradient'
        l1_solver: str = 'coordinate_descent'  # 'lbfgs_b', 'fista', 'coordinate_descent'
    ):
        """
        Initialize Sparse Coder
        
        Args:
            n_components: Number of dictionary elements (basis functions)
            sparsity_penalty: λ parameter controlling sparsity vs reconstruction trade-off
            patch_size: Size of image patches to analyze
            max_iter: Maximum iterations for sparse coding
            tolerance: Convergence tolerance
            dictionary: Pre-trained dictionary (optional, will initialize random if None)
            random_seed: Random seed for reproducibility
        """
        
        self.n_components = n_components
        self.sparsity_penalty = sparsity_penalty
        self.patch_size = patch_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # Configuration options for FIXME implementations
        self.sparseness_function = sparseness_function
        self.optimization_method = optimization_method
        self.l1_solver = l1_solver
        
        # Additional configuration for whitening and dictionary updates
        self.whitening_method = 'olshausen_field'  # 'olshausen_field', 'zca', 'standard'
        self.dictionary_update_method = 'equation_6'  # 'equation_6', 'orthogonal', 'batch'
        self.learning_rate = 0.01
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        if dictionary is not None:
            # Use provided dictionary and infer patch_size from it
            patch_dim = dictionary.shape[0]
            # Try to infer square patch size first
            patch_size_inferred = int(np.sqrt(patch_dim))
            if patch_size_inferred * patch_size_inferred == patch_dim:
                self.patch_size = (patch_size_inferred, patch_size_inferred)
                print(f"✓ Inferred patch size {self.patch_size} from dictionary dimensions")
            else:
                # Keep original patch_size but warn about mismatch
                expected_dim = patch_size[0] * patch_size[1]
                if patch_dim != expected_dim:
                    print(f"Warning: Dictionary has {patch_dim} dimensions, adjusting patch_size to match")
                    # Find best rectangular factorization
                    for h in range(1, int(np.sqrt(patch_dim)) + 1):
                        if patch_dim % h == 0:
                            w = patch_dim // h
                            self.patch_size = (h, w)
                    print(f"✓ Adjusted patch size to {self.patch_size}")
            
            if dictionary.shape[1] != n_components:
                print(f"Warning: Using provided dictionary with {dictionary.shape[1]} components instead of {n_components}")
                self.n_components = dictionary.shape[1]
            self.dictionary = dictionary.copy()
        else:
            # Initialize random dictionary (will be learned)
            patch_dim = patch_size[0] * patch_size[1]
            self.dictionary = np.random.randn(patch_dim, n_components)
            
        # Normalize dictionary columns
        self.dictionary = normalize(self.dictionary, axis=0)
        
        # Training history
        self.training_history = {'reconstruction_error': [], 'sparsity': []}
        
        print(f"✓ Sparse Coder initialized: {n_components} components, {patch_size} patches")
        print(f"  Sparseness function: {sparseness_function}")
        print(f"  Optimization method: {optimization_method}")
        
    def _sparse_encode_single(self, patch: np.ndarray) -> np.ndarray:
        """
        Encode a single patch using sparse coding
        
        Solves: min ||x - Da||₂² + λ||a||₁
        where x is patch, D is dictionary, a is sparse code, λ is sparsity penalty
        
        FIXME: The paper's original sparseness cost function (equation 4) is:
        [sparseness of ai] = -Σ S(ai/σ) where σ is a scaling constant.
        
        Current implementation uses standard L1: λ||a||₁
        Paper presents multiple cost function options:
        
        SOLUTION OPTION 1 - Original S(x) = log(1 + x²) [Primary paper choice]:
        def sparseness_cost_log(coefficients, sigma=1.0):
            # S(ai) = -Σ S(ai/σ) where S(x) = log(1 + x²) 
            normalized_coeffs = coefficients / sigma
            return -np.sum(np.log(1 + normalized_coeffs**2))
            
        SOLUTION OPTION 2 - Alternative S(x) = |x| [Also tested in paper]:
        def sparseness_cost_l1(coefficients, sigma=1.0):
            # S(ai) = -Σ |ai/σ|
            normalized_coeffs = coefficients / sigma
            return -np.sum(np.abs(normalized_coeffs))
            
        SOLUTION OPTION 3 - Alternative S(x) = -e^(-x²) [Mentioned in paper]:
        def sparseness_cost_gaussian(coefficients, sigma=1.0):
            # S(ai) = -Σ -e^(-(ai/σ)²) = Σ e^(-(ai/σ)²)
            normalized_coeffs = coefficients / sigma
            return np.sum(np.exp(-normalized_coeffs**2))
            
        CONFIGURATION: Set sparseness_function='log'|'l1'|'gaussian' in __init__
        """
        
        def objective(coefficients):
            """Objective function: reconstruction error + sparsity penalty"""
            reconstruction = self.dictionary @ coefficients
            reconstruction_error = 0.5 * np.sum((patch - reconstruction) ** 2)
            
            # Implement different sparseness functions as mentioned in FIXME comments
            if self.sparseness_function == 'log':
                # S(x) = log(1 + x²) - Original paper choice
                sigma = 1.0  # Scaling constant
                normalized_coeffs = coefficients / sigma
                sparsity_penalty = -self.sparsity_penalty * np.sum(np.log(1 + normalized_coeffs**2))
            elif self.sparseness_function == 'gaussian':
                # S(x) = -e^(-x²) - Alternative from paper
                sigma = 1.0
                normalized_coeffs = coefficients / sigma
                sparsity_penalty = self.sparsity_penalty * np.sum(np.exp(-normalized_coeffs**2))
            elif self.sparseness_function == 'huber':
                # Huber penalty - smooth approximation to L1 for robustness
                delta = getattr(self, 'huber_delta', 1.0)
                abs_coeffs = np.abs(coefficients)
                huber_penalty = np.where(abs_coeffs <= delta, 
                                        0.5 * coefficients**2, 
                                        delta * abs_coeffs - 0.5 * delta**2)
                sparsity_penalty = self.sparsity_penalty * np.sum(huber_penalty)
            elif self.sparseness_function == 'elastic_net':
                # Elastic net: combination of L1 and L2 penalties
                l1_ratio = getattr(self, 'elastic_net_l1_ratio', 0.5)
                l1_penalty = np.sum(np.abs(coefficients))
                l2_penalty = 0.5 * np.sum(coefficients**2)
                sparsity_penalty = self.sparsity_penalty * (l1_ratio * l1_penalty + (1 - l1_ratio) * l2_penalty)
            elif self.sparseness_function == 'cauchy':
                # Cauchy penalty - heavy-tailed for extreme sparsity
                gamma = getattr(self, 'cauchy_gamma', 1.0)
                sparsity_penalty = self.sparsity_penalty * np.sum(np.log(1 + (coefficients / gamma)**2))
            elif self.sparseness_function == 'student_t':
                # Student-t penalty - robust heavy-tailed distribution
                nu = getattr(self, 'student_t_nu', 3.0)  # degrees of freedom
                sparsity_penalty = self.sparsity_penalty * np.sum(np.log(1 + coefficients**2 / nu))
            else:
                # Default: L1 sparsity (|x|)
                sparsity_penalty = self.sparsity_penalty * np.sum(np.abs(coefficients))
                
            return reconstruction_error + sparsity_penalty
        
        def gradient(coefficients):
            """Gradient of objective function with different sparseness functions"""
            reconstruction = self.dictionary @ coefficients
            error = reconstruction - patch
            grad_reconstruction = self.dictionary.T @ error
            
            # Implement gradient for different sparseness functions
            if self.sparseness_function == 'log':
                # S(x) = log(1 + x²), S'(x) = 2x/(1 + x²)
                sigma = 1.0
                normalized_coeffs = coefficients / sigma
                grad_sparsity = -self.sparsity_penalty * (2 * normalized_coeffs) / (1 + normalized_coeffs**2) / sigma
            elif self.sparseness_function == 'gaussian':
                # S(x) = -e^(-x²), S'(x) = 2x*e^(-x²)
                sigma = 1.0
                normalized_coeffs = coefficients / sigma
                grad_sparsity = self.sparsity_penalty * 2 * normalized_coeffs * np.exp(-normalized_coeffs**2) / sigma
            elif self.sparseness_function == 'huber':
                # Huber gradient: smooth transition from quadratic to linear
                delta = getattr(self, 'huber_delta', 1.0)
                abs_coeffs = np.abs(coefficients)
                grad_sparsity = self.sparsity_penalty * np.where(abs_coeffs <= delta,
                                                               coefficients,
                                                               delta * np.sign(coefficients))
            elif self.sparseness_function == 'elastic_net':
                # Elastic net gradient: combination of L1 and L2
                l1_ratio = getattr(self, 'elastic_net_l1_ratio', 0.5)
                grad_l1 = np.sign(coefficients)
                grad_l2 = coefficients
                grad_sparsity = self.sparsity_penalty * (l1_ratio * grad_l1 + (1 - l1_ratio) * grad_l2)
            elif self.sparseness_function == 'cauchy':
                # Cauchy gradient: d/dx log(1 + (x/γ)²) = 2x/(γ²(1 + (x/γ)²))
                gamma = getattr(self, 'cauchy_gamma', 1.0)
                normalized_coeffs = coefficients / gamma
                grad_sparsity = self.sparsity_penalty * 2 * coefficients / (gamma**2 * (1 + normalized_coeffs**2))
            elif self.sparseness_function == 'student_t':
                # Student-t gradient: d/dx log(1 + x²/ν) = 2x/(ν(1 + x²/ν))
                nu = getattr(self, 'student_t_nu', 3.0)
                grad_sparsity = self.sparsity_penalty * 2 * coefficients / (nu * (1 + coefficients**2 / nu))
            else:
                # Default: L1 sparsity, S'(x) = sign(x)
                grad_sparsity = self.sparsity_penalty * np.sign(coefficients)
                
            return grad_reconstruction + grad_sparsity
        
        # Initialize coefficients
        initial_coeffs = np.zeros(self.n_components)
        
        # Use selected optimization method
        if self.optimization_method == 'equation_5':
            # Use original paper's equation (5) method
            coeffs = self._sparse_encode_equation_5(patch)
        elif self.optimization_method == 'fista':
            # Use FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
            coeffs = self._fista_optimization(patch, objective, gradient, initial_coeffs)
        elif self.optimization_method == 'proximal_gradient':
            # Use proximal gradient descent
            coeffs = self._proximal_gradient(patch, objective, gradient, initial_coeffs)
        else:
            # Default: coordinate descent - proven optimal for L1-regularized problems
            if self.l1_solver == 'lbfgs_b':
                # Address FIXME about L-BFGS-B not being optimal for L1
                print("⚠️  Warning: L-BFGS-B may not be optimal for L1-regularized problems")
                result = minimize(objective, initial_coeffs, method='L-BFGS-B', jac=gradient)
                coeffs = result.x
            else:
                coeffs = self._coordinate_descent_lasso(patch, objective, initial_coeffs)
        
        return coeffs
        
    def _coordinate_descent_lasso(self, signal: np.ndarray, objective_func, initial_coeffs: np.ndarray) -> np.ndarray:
        """
        Coordinate descent algorithm for LASSO (L1-regularized) optimization.
        
        FIXME: Paper uses different optimization approach. Equation (5) shows:
        âᵢ = bᵢ - Σⱼ Cᵢⱼâⱼ - λ/σ S'(âᵢ/σ)
        where bᵢ = Σₓ φᵢ(x,y)I(x,y) and Cᵢⱼ = Σₓ φᵢ(x,y)φⱼ(x,y).
        
        Current implementation uses modern coordinate descent, but paper specifies:
        
        SOLUTION OPTION 1 - Original paper equation (5) fixed-point iteration:
        def sparse_encode_equation_5(self, patch):
            coeffs = np.zeros(self.n_components)
            
            # Precompute bᵢ = Σₓ φᵢ(x,y)I(x,y)
            b = self.dictionary.T @ patch
            
            # Precompute Cᵢⱼ = Σₓ φᵢ(x,y)φⱼ(x,y) (Gram matrix)
            C = self.dictionary.T @ self.dictionary
            
            for iteration in range(self.max_iter):
                coeffs_old = coeffs.copy()
                
                for i in range(len(coeffs)):
                    # Compute âᵢ = bᵢ - Σⱼ≠ᵢ Cᵢⱼâⱼ - λ/σ S'(âᵢ/σ)
                    sum_term = np.sum(C[i, :] * coeffs) - C[i, i] * coeffs[i]
                    
                    # S'(x) derivative depends on sparseness function choice
                    if hasattr(self, 'sparseness_function'):
                        if self.sparseness_function == 'log':
                            # S(x) = log(1 + x²), S'(x) = 2x/(1 + x²)
                            sparseness_deriv = 2 * coeffs[i] / (1 + coeffs[i]**2)
                        elif self.sparseness_function == 'l1':
                            # S(x) = |x|, S'(x) = sign(x)
                            sparseness_deriv = np.sign(coeffs[i])
                    else:
                        sparseness_deriv = np.sign(coeffs[i])  # Default to L1
                    
                    # Update equation (5)
                    coeffs[i] = (b[i] - sum_term - self.sparsity_penalty * sparseness_deriv) / C[i, i]
                    
                # Check convergence
                if np.linalg.norm(coeffs - coeffs_old) < self.tolerance:
                    break
                    
            return coeffs
            
        This is the proven optimal method for L1-regularized problems with guaranteed convergence.
        
        Args:
            signal: Input signal to encode
            objective_func: Objective function (not used directly in coordinate descent)
            initial_coeffs: Initial coefficient values
            
        Returns:
            np.ndarray: Optimized sparse coefficients
        """
        
        coeffs = initial_coeffs.copy()
        dictionary = self.dictionary
        
        # Precompute useful quantities
        XtX = dictionary.T @ dictionary  # Dictionary gram matrix
        Xty = dictionary.T @ signal      # Dictionary-signal correlation
        
        # Coordinate descent main loop
        for iteration in range(self.max_iter):
            coeffs_old = coeffs.copy()
            
            # Update each coefficient individually
            for j in range(len(coeffs)):
                # Compute residual excluding current coefficient
                residual = signal - dictionary @ coeffs + coeffs[j] * dictionary[:, j]
                
                # Compute optimal update for coefficient j
                rho_j = dictionary[:, j].T @ residual
                
                # Soft thresholding operator (proximal operator for L1 norm)
                z_j = rho_j / XtX[j, j]  # Unconstrained optimum
                
                # Apply soft thresholding with regularization parameter
                threshold = self.sparsity_penalty / XtX[j, j]
                
                if z_j > threshold:
                    coeffs[j] = z_j - threshold
                elif z_j < -threshold:
                    coeffs[j] = z_j + threshold
                else:
                    coeffs[j] = 0.0
                    
            # Check convergence
            coeff_change = np.linalg.norm(coeffs - coeffs_old)
            if coeff_change < self.tolerance:
                break
                
        return coeffs
        
    def _soft_threshold(self, x: float, threshold: float) -> float:
        """
        Soft thresholding operator - proximal operator for L1 norm.
        
        Args:
            x: Input value
            threshold: Thresholding parameter
            
        Returns:
            float: Soft-thresholded value
        """
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.0
            
    def _proximal_gradient_method(self, signal: np.ndarray, initial_coeffs: np.ndarray) -> np.ndarray:
        """
        Alternative: Proximal gradient method for L1-regularized optimization.
        
        This provides another proven approach for L1 problems with good convergence properties.
        
        Args:
            signal: Input signal to encode
            initial_coeffs: Initial coefficient values
            
        Returns:
            np.ndarray: Optimized sparse coefficients
        """
        
        coeffs = initial_coeffs.copy()
        dictionary = self.dictionary
        
        # Step size (can be adaptive)
        step_size = 1.0 / np.linalg.norm(dictionary, ord=2)**2
        
        for iteration in range(self.max_iter):
            coeffs_old = coeffs.copy()
            
            # Compute gradient of smooth part (quadratic term)
            residual = signal - dictionary @ coeffs
            gradient = -dictionary.T @ residual
            
            # Gradient step
            z = coeffs - step_size * gradient
            
            # Proximal operator (soft thresholding)
            threshold = step_size * self.sparsity_penalty
            coeffs = np.array([self._soft_threshold(zi, threshold) for zi in z])
            
            # Check convergence
            if np.linalg.norm(coeffs - coeffs_old) < self.tolerance:
                break
                
        return coeffs
        
    def sparse_encode(self, patches: np.ndarray) -> np.ndarray:
        """
        Sparse encode multiple patches
        
        Args:
            patches: Array of patches (n_patches, patch_dim)
            
        Returns:
            Sparse coefficients (n_patches, n_components)
        """
        
        n_patches = patches.shape[0]
        coefficients = np.zeros((n_patches, self.n_components))
        
        print(f"🔍 Sparse encoding {n_patches} patches...")
        
        for i in range(n_patches):
            coefficients[i] = self._sparse_encode_single(patches[i])
            
            if (i + 1) % 100 == 0:
                print(f"   Encoded {i + 1}/{n_patches} patches")
                
        return coefficients
        
    def _update_dictionary(self, patches: np.ndarray, coefficients: np.ndarray):
        """
        Update dictionary using method of optimal directions (MOD)
        
        This is Olshausen & Field's key algorithm for learning the dictionary
        """
        
        for j in range(self.n_components):
            # Find patches that use this dictionary element
            using_indices = np.abs(coefficients[:, j]) > 1e-10
            
            if np.sum(using_indices) == 0:
                continue
                
            # Error when removing this dictionary element
            error = patches[using_indices] - (coefficients[using_indices] @ self.dictionary.T) + np.outer(coefficients[using_indices, j], self.dictionary[:, j])
            
            # Update dictionary element and coefficients via SVD
            if np.sum(using_indices) > 0:
                U, s, Vt = np.linalg.svd(error, full_matrices=False)
                
                # Update dictionary column
                self.dictionary[:, j] = Vt[0, :]
                
                # Update coefficients
                coefficients[using_indices, j] = s[0] * U[:, 0]
                
        # Normalize dictionary columns
        self.dictionary = normalize(self.dictionary, axis=0)
        
    def _enhanced_sparse_encode(self, patches: np.ndarray) -> np.ndarray:
        """
        Enhanced sparse encoding with FISTA (Fast Iterative Shrinkage-Thresholding)
        
        More efficient than basic L-BFGS for large-scale problems
        """
        
        n_patches = patches.shape[0]
        coefficients = np.zeros((n_patches, self.n_components))
        
        print(f"🔍 Enhanced sparse encoding {n_patches} patches using FISTA...")
        
        for i in range(n_patches):
            coefficients[i] = self._fista_sparse_encode(patches[i])
            
            if (i + 1) % 200 == 0:
                print(f"   Encoded {i + 1}/{n_patches} patches")
                
        return coefficients
        
    def _fista_sparse_encode(self, patch: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """
        FISTA algorithm for sparse coding
        
        Faster convergence than basic iterative shrinkage
        """
        
        # Initialize
        x = np.zeros(self.n_components)
        y = x.copy()
        t = 1.0
        
        # Compute Lipschitz constant for step size
        L = np.linalg.norm(self.dictionary.T @ self.dictionary, 2)
        step_size = 1.0 / L if L > 0 else 0.01
        
        for iteration in range(max_iter):
            # Gradient step
            gradient = self.dictionary.T @ (self.dictionary @ y - patch)
            x_new = self._soft_threshold(y - step_size * gradient, step_size * self.sparsity_penalty)
            
            # FISTA momentum update
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = x_new + ((t - 1) / t_new) * (x_new - x)
            
            # Check convergence
            if np.linalg.norm(x_new - x) < 1e-6:
                break
                
            x = x_new
            t = t_new
            
        return x
        
    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Soft thresholding operator for L1 regularization"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def _update_dictionary_equation_6(self, patches: np.ndarray, coefficients: np.ndarray):
        """
        Pure Olshausen & Field equation (6) implementation
        Δφᵢ(xₙ,yₙ) = η⟨aᵢ⟨I(xₙ,yₙ) - Î(xₙ,yₙ)⟩⟩
        """
        for i in range(self.n_components):
            # Find patches using this basis function significantly
            active_mask = np.abs(coefficients[:, i]) > 1e-4
            if not np.any(active_mask):
                continue
                
            # Get active patches and coefficients
            active_patches = patches[active_mask]
            active_coeffs = coefficients[active_mask, i]
            
            # Compute reconstruction error: I - Î
            reconstruction = coefficients[active_mask] @ self.dictionary.T
            error = active_patches - reconstruction
            
            # Apply equation (6): Δφᵢ = η⟨aᵢ⟨I - Î⟩⟩
            gradient = np.mean(active_coeffs[:, np.newaxis] * error, axis=0)
            self.dictionary[:, i] += self.learning_rate * gradient
            
            # Normalize to unit length (paper requirement)
            norm = np.linalg.norm(self.dictionary[:, i])
            if norm > 1e-10:
                self.dictionary[:, i] /= norm
    
    def _update_dictionary_with_orthogonality(self, patches: np.ndarray, coefficients: np.ndarray):
        """
        Equation (6) updates with orthogonality enforcement
        """
        # Apply equation (6) updates
        self._update_dictionary_equation_6(patches, coefficients)
        
        # Gram-Schmidt orthogonalization (paper mentions orthogonal basis preference)
        for i in range(self.n_components):
            for j in range(i):
                projection = np.dot(self.dictionary[:, i], self.dictionary[:, j])
                self.dictionary[:, i] -= projection * self.dictionary[:, j]
            # Renormalize
            norm = np.linalg.norm(self.dictionary[:, i])
            if norm > 1e-10:
                self.dictionary[:, i] /= norm
    
    def _update_dictionary_batch(self, patches: np.ndarray, coefficients: np.ndarray, batch_size: int = 100):
        """
        Batch processing version of equation (6) for efficiency
        """
        n_patches = patches.shape[0]
        for start_idx in range(0, n_patches, batch_size):
            end_idx = min(start_idx + batch_size, n_patches)
            batch_patches = patches[start_idx:end_idx]
            batch_coeffs = coefficients[start_idx:end_idx]
            
            self._update_dictionary_equation_6(batch_patches, batch_coeffs)
        
    def _update_dictionary_olshausen(self, patches: np.ndarray, coefficients: np.ndarray):
        """
        Enhanced dictionary update using Olshausen & Field's method with improvements
        
        Includes momentum and adaptive learning rates
        
        FIXME: Missing the exact learning procedure from paper equation (6):
        Δφᵢ(xₙ,yₙ) = η⟨aᵢ⟨I(xₙ,yₙ) - Î(xₙ,yₙ)⟩⟩ 
        where I is original image, Î is reconstruction, aᵢ is coefficient, η is learning rate.
        
        Current implementation uses modern momentum/SVD methods, but paper specifies:
        IMPLEMENTATION NOTE: Now configurable via dictionary_update_method parameter:
        - 'equation_6': Pure Olshausen & Field equation (6) implementation
        - 'orthogonal': SVD-based orthogonal update (current)
        - 'batch': Batch gradient descent with momentum
        
        SOLUTION OPTION 1 - Pure Olshausen & Field equation (6) implementation:
        def update_dictionary_original(self, patches, coefficients, learning_rate=0.01):
            for i in range(self.n_components):
                # Find patches using this basis function significantly
                active_mask = np.abs(coefficients[:, i]) > 1e-4
                if not np.any(active_mask):
                    continue
                    
                # Get active patches and coefficients
                active_patches = patches[active_mask]
                active_coeffs = coefficients[active_mask, i]
                
                # Compute reconstruction error: I - Î
                reconstruction = coefficients[active_mask] @ self.dictionary.T
                error = active_patches - reconstruction
                
                # Apply equation (6): Δφᵢ = η⟨aᵢ⟨I - Î⟩⟩
                gradient = np.mean(active_coeffs[:, np.newaxis] * error, axis=0)
                self.dictionary[:, i] += learning_rate * gradient
                
                # Normalize to unit length (paper requirement)
                self.dictionary[:, i] /= np.linalg.norm(self.dictionary[:, i])
                
        SOLUTION OPTION 2 - With orthogonality enforcement (mentioned in paper):
        def update_dictionary_with_orthogonality(self, patches, coefficients):
            # Apply equation (6) updates
            self.update_dictionary_original(patches, coefficients)
            
            # Gram-Schmidt orthogonalization (paper mentions orthogonal basis preference)
            for i in range(self.n_components):
                for j in range(i):
                    projection = np.dot(self.dictionary[:, i], self.dictionary[:, j])
                    self.dictionary[:, i] -= projection * self.dictionary[:, j]
                # Renormalize
                self.dictionary[:, i] /= np.linalg.norm(self.dictionary[:, i])
                
        SOLUTION OPTION 3 - Batch processing (more efficient but still equation 6):
        def update_dictionary_batch(self, patches, coefficients, batch_size=100):
            for start in range(0, len(patches), batch_size):
                end = min(start + batch_size, len(patches))
                batch_patches = patches[start:end]
                batch_coeffs = coefficients[start:end]
                self.update_dictionary_original(batch_patches, batch_coeffs)
                
        CONFIGURATION: Set dictionary_update='original'|'orthogonal'|'batch' in __init__
        """
        
        learning_rate = 0.01
        momentum = 0.9
        
        if not hasattr(self, 'dictionary_momentum'):
            self.dictionary_momentum = np.zeros_like(self.dictionary)
            
        for j in range(self.n_components):
            # Find patches that use this dictionary element significantly
            using_indices = np.abs(coefficients[:, j]) > 1e-4
            
            if np.sum(using_indices) == 0:
                continue
                
            # Calculate residual error when removing this dictionary element
            residual = patches[using_indices] - (coefficients[using_indices] @ self.dictionary.T)
            residual += np.outer(coefficients[using_indices, j], self.dictionary[:, j])
            
            # Update dictionary element using gradient descent with momentum
            if np.sum(using_indices) > 0:
                # Compute gradient
                gradient = residual.T @ coefficients[using_indices, j]
                gradient = gradient / (np.linalg.norm(coefficients[using_indices, j])**2 + 1e-8)
                
                # Apply momentum
                self.dictionary_momentum[:, j] = momentum * self.dictionary_momentum[:, j] + learning_rate * gradient
                self.dictionary[:, j] += self.dictionary_momentum[:, j]
                
                # Normalize dictionary column
                norm = np.linalg.norm(self.dictionary[:, j])
                if norm > 0:
                    self.dictionary[:, j] /= norm
                    
    def _calculate_dictionary_coherence(self) -> float:
        """
        Calculate dictionary coherence (mutual coherence)
        
        Measures how well-conditioned the dictionary is
        Lower coherence is better for sparse coding
        """
        
        # Compute Gram matrix
        gram_matrix = self.dictionary.T @ self.dictionary
        
        # Remove diagonal elements
        off_diagonal = gram_matrix - np.eye(self.n_components)
        
        # Maximum off-diagonal element is the coherence
        coherence = np.max(np.abs(off_diagonal))
        
        return coherence
        
    def fit(self, images: np.ndarray, n_patches: int = 10000) -> Dict[str, Any]:
        """
        Learn sparse dictionary from natural images
        
        This is the revolutionary algorithm that discovers edge detectors!
        
        FIXME: Missing exact whitening procedure from paper and equation (6) learning rule.
        Paper describes specific preprocessing: "zero-phase whitening/lowpass filter".
        
        Current implementation has basic whitening, but paper specifies:
        IMPLEMENTATION NOTE: Now configurable via whitening_method parameter:
        - 'olshausen_field': Zero-phase whitening filter with R(f) = fe^(-f/f₀)
        - 'zca': ZCA whitening transformation
        - 'standard': Standard whitening (current)
        
        SOLUTION OPTION 1 - Zero-phase whitening filter (equation mentioned in paper):
        def whiten_patches_olshausen_field(self, patches):
            # Paper: "zero-phase whitening/lowpass filter, R(f) = fe^(-f/f0)"
            # where f₀ = 200 cycles/picture
            
            # Step 1: Remove DC component
            patches_centered = patches - np.mean(patches, axis=1, keepdims=True)
            
            # Step 2: Apply whitening filter in frequency domain
            patch_2d = patches_centered.reshape(-1, *self.patch_size)
            whitened_patches = []
            
            for patch in patch_2d:
                # FFT
                fft_patch = np.fft.fft2(patch)
                
                # Create frequency grid
                freqs_y = np.fft.fftfreq(patch.shape[0])
                freqs_x = np.fft.fftfreq(patch.shape[1])
                fy, fx = np.meshgrid(freqs_y, freqs_x, indexing='ij')
                
                # Frequency magnitude
                f_mag = np.sqrt(fx**2 + fy**2)
                
                # Whitening filter: R(f) = f * exp(-f/f0)
                f0 = 200.0 / max(patch.shape)  # Normalize by patch size
                whitening_filter = f_mag * np.exp(-f_mag / f0)
                whitening_filter[0, 0] = 1e-10  # Avoid division by zero
                
                # Apply filter
                whitened_fft = fft_patch * whitening_filter
                whitened_patch = np.real(np.fft.ifft2(whitened_fft))
                whitened_patches.append(whitened_patch.flatten())
                
            return np.array(whitened_patches)
            
        SOLUTION OPTION 2 - ZCA whitening (more standard approach):
        def whiten_patches_zca(self, patches):
            # Center patches
            patches_centered = patches - np.mean(patches, axis=0)
            
            # Compute covariance
            cov_matrix = np.cov(patches_centered.T)
            
            # ZCA whitening transformation
            U, s, Vt = np.linalg.svd(cov_matrix)
            epsilon = 1e-5
            zca_matrix = U @ np.diag(1.0 / np.sqrt(s + epsilon)) @ U.T
            
            return patches_centered @ zca_matrix
            
        SOLUTION OPTION 3 - Simple variance normalization (fastest):
        def whiten_patches_simple(self, patches):
            # Center and normalize variance
            patches_centered = patches - np.mean(patches, axis=1, keepdims=True)
            patches_normalized = patches_centered / (np.std(patches_centered, axis=1, keepdims=True) + 1e-8)
            return patches_normalized
            
        CONFIGURATION: Set whitening_method='olshausen_field'|'zca'|'simple' in __init__
        
        Args:
            images: Natural images (n_images, height, width) 
            n_patches: Number of patches to extract for training
            
        Returns:
            Training statistics
        """
        
        print(f"🎯 Learning sparse dictionary from {len(images)} images...")
        
        # Extract random patches from images
        patches = self._extract_patches(images, n_patches)
        print(f"   Extracted {len(patches)} patches of size {self.patch_size}")
        
        # Whitening preprocessing with configurable methods
        if self.whitening_method == 'olshausen_field':
            patches_whitened = self._whiten_patches_olshausen_field(patches)
        elif self.whitening_method == 'zca':
            patches_whitened = self._whiten_patches_zca(patches)
        else:
            patches_whitened = self._whiten_patches(patches)
        
        # Enhanced alternating optimization with adaptive batch sizes and convergence checking
        batch_size = min(1000, len(patches_whitened))
        convergence_threshold = 1e-6
        prev_error = float('inf')
        
        for iteration in range(50):  # More iterations for better convergence
            print(f"\n📚 Dictionary learning iteration {iteration + 1}/50")
            
            # Adaptive batch size - start small and increase
            current_batch_size = min(batch_size * (1 + iteration // 10), len(patches_whitened))
            batch_patches = patches_whitened[:current_batch_size]
            
            # 1. Sparse encode patches with current dictionary (enhanced algorithm)
            coefficients = self._enhanced_sparse_encode(batch_patches)
            
            # 2. Update dictionary using configurable method
            if self.dictionary_update_method == 'equation_6':
                self._update_dictionary_equation_6(batch_patches, coefficients)
            elif self.dictionary_update_method == 'orthogonal':
                self._update_dictionary_with_orthogonality(batch_patches, coefficients)
            elif self.dictionary_update_method == 'batch':
                self._update_dictionary_batch(batch_patches, coefficients)
            else:
                # Default to original Olshausen method
                self._update_dictionary_olshausen(batch_patches, coefficients)
            
            # 3. Calculate detailed statistics
            reconstruction = coefficients @ self.dictionary.T
            reconstruction_error = np.mean((batch_patches - reconstruction) ** 2)
            sparsity = np.mean(np.sum(np.abs(coefficients) > 1e-3, axis=1))
            
            # Calculate additional metrics
            dictionary_coherence = self._calculate_dictionary_coherence()
            feature_usage = np.mean(np.sum(np.abs(coefficients) > 1e-3, axis=0))  # How many patches use each feature
            
            self.training_history['reconstruction_error'].append(reconstruction_error)
            self.training_history['sparsity'].append(sparsity)
            
            print(f"   Batch size: {current_batch_size}")
            print(f"   Reconstruction error: {reconstruction_error:.6f}")
            print(f"   Average sparsity: {sparsity:.1f} active elements")
            print(f"   Dictionary coherence: {dictionary_coherence:.3f}")
            print(f"   Feature usage rate: {feature_usage:.1f}%")
            
            # Convergence check
            if abs(prev_error - reconstruction_error) < convergence_threshold:
                print(f"   ✓ Converged after {iteration + 1} iterations")
                break
                
            prev_error = reconstruction_error
            
            # Adaptive learning rate decay
            if iteration > 0 and reconstruction_error > prev_error:
                print("   ↓ Reducing sparsity penalty for better convergence")
                self.sparsity_penalty *= 0.95
            
        print(f"✅ Dictionary learning complete!")
        
        return {
            'final_reconstruction_error': reconstruction_error,
            'final_sparsity': sparsity,
            'n_dictionary_elements': self.n_components,
            'patch_size': self.patch_size
        }
        
    def _extract_patches(self, images: np.ndarray, n_patches: int) -> np.ndarray:
        """Extract random patches from images"""
        
        patches = []
        patch_h, patch_w = self.patch_size
        
        for _ in range(n_patches):
            # Select random image
            img_idx = np.random.randint(0, len(images))
            image = images[img_idx]
            
            # Select random patch location
            max_y = image.shape[0] - patch_h
            max_x = image.shape[1] - patch_w
            
            if max_y <= 0 or max_x <= 0:
                continue
                
            y = np.random.randint(0, max_y)
            x = np.random.randint(0, max_x)
            
            # Extract patch
            patch = image[y:y+patch_h, x:x+patch_w]
            patches.append(patch.flatten())
            
        return np.array(patches)
        
    def _whiten_patches(self, patches: np.ndarray) -> np.ndarray:
        """
        Whiten patches to decorrelate pixels (preprocessing step)
        
        This removes the natural correlation structure of images,
        making the sparse structure more apparent.
        """
        
        # Center patches
        patches_centered = patches - np.mean(patches, axis=1, keepdims=True)
        
        # Compute covariance matrix
        cov = np.cov(patches_centered, rowvar=False)
        
        # Eigendecomposition for whitening
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Whitening transform
        epsilon = 1e-5  # Regularization
        whitening_matrix = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals + epsilon)) @ eigenvecs.T
        
        patches_whitened = patches_centered @ whitening_matrix
        
        return patches_whitened
    
    def _whiten_patches_olshausen_field(self, patches: np.ndarray) -> np.ndarray:
        """
        Zero-phase whitening filter as specified in Olshausen & Field 1996
        Paper: "zero-phase whitening/lowpass filter, R(f) = fe^(-f/f0)"
        where f₀ = 200 cycles/picture
        """
        print("   🔬 Applying Olshausen & Field zero-phase whitening filter...")
        
        # Step 1: Remove DC component
        patches_centered = patches - np.mean(patches, axis=1, keepdims=True)
        
        # Step 2: Apply whitening filter in frequency domain
        patch_2d = patches_centered.reshape(-1, *self.patch_size)
        whitened_patches = []
        
        for patch in patch_2d:
            # FFT
            fft_patch = np.fft.fft2(patch)
            
            # Create frequency grid
            freqs_y = np.fft.fftfreq(patch.shape[0])
            freqs_x = np.fft.fftfreq(patch.shape[1])
            fy, fx = np.meshgrid(freqs_y, freqs_x, indexing='ij')
            
            # Frequency magnitude
            f_mag = np.sqrt(fx**2 + fy**2)
            
            # Whitening filter: R(f) = f * exp(-f/f0)
            f0 = 200.0 / max(patch.shape)  # Normalize by patch size (200 cycles/picture)
            whitening_filter = f_mag * np.exp(-f_mag / f0)
            whitening_filter[0, 0] = 1e-10  # Avoid division by zero at DC
            
            # Apply filter
            whitened_fft = fft_patch * whitening_filter
            whitened_patch = np.real(np.fft.ifft2(whitened_fft))
            whitened_patches.append(whitened_patch.flatten())
            
        return np.array(whitened_patches)
    
    def _whiten_patches_zca(self, patches: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        ZCA (Zero-phase Component Analysis) whitening
        Alternative whitening approach mentioned in later sparse coding literature
        """
        print("   🔬 Applying ZCA whitening...")
        
        # Center patches
        patches_centered = patches - np.mean(patches, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(patches_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Whitening transformation
        whitening_transform = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + epsilon)) @ eigenvectors.T
        
        # Apply whitening
        whitened_patches = patches_centered @ whitening_transform
        
        return whitened_patches
        
    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Transform new images using learned sparse dictionary
        
        Returns sparse coefficients for input images
        """
        
        if self.dictionary is None:
            raise ValueError("Dictionary must be learned before transform!")
            
        # Extract patches
        patches = self._extract_patches(images, len(images) * 100)
        
        # Whiten patches using same method as training
        if self.whitening_method == 'olshausen_field':
            patches_whitened = self._whiten_patches_olshausen_field(patches)
        elif self.whitening_method == 'zca':
            patches_whitened = self._whiten_patches_zca(patches)
        else:
            patches_whitened = self._whiten_patches(patches)
        
        # Sparse encode
        coefficients = self.sparse_encode(patches_whitened)
        
        return coefficients
        
    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Reconstruct patches from sparse coefficients
        """
        
        return coefficients @ self.dictionary.T
        
    def visualize_dictionary(self, figsize: Tuple[int, int] = (16, 16)):
        """
        Visualize learned dictionary elements
        
        This is where we see the magic - the algorithm discovers edge detectors!
        """
        
        n_plot = min(self.n_components, 256)  # Plot up to 256 elements
        grid_size = int(np.sqrt(n_plot))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        fig.suptitle('Learned Sparse Dictionary (Basis Functions)', fontsize=16)
        
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < n_plot:
                    # Reshape dictionary element to patch
                    element = self.dictionary[:, idx].reshape(self.patch_size)
                    
                    # Normalize for visualization
                    element = (element - element.min()) / (element.max() - element.min() + 1e-8)
                    
                    axes[i, j].imshow(element, cmap='gray', interpolation='nearest')
                
                axes[i, j].axis('off')
                
        plt.tight_layout()
        plt.show()
        
        # Analyze dictionary properties
        self._analyze_dictionary()
        
    def _analyze_dictionary(self):
        """Analyze properties of learned dictionary"""
        
        print(f"\n📊 Dictionary Analysis:")
        
        # Calculate orientation preferences
        orientations = []
        for i in range(self.n_components):
            element = self.dictionary[:, i].reshape(self.patch_size)
            
            # Simple edge detection to estimate orientation
            grad_y = np.abs(np.gradient(element, axis=0)).mean()
            grad_x = np.abs(np.gradient(element, axis=1)).mean()
            
            if grad_x + grad_y > 0:
                orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi
                orientations.append(orientation)
                
        if orientations:
            print(f"   • {len(orientations)} oriented elements found")
            print(f"   • Orientation range: {np.min(orientations):.1f}° - {np.max(orientations):.1f}°")
            print(f"   • Mean orientation: {np.mean(orientations):.1f}° ± {np.std(orientations):.1f}°")
            
        # Dictionary statistics
        element_norms = np.linalg.norm(self.dictionary, axis=0)
        print(f"   • Element norms: {element_norms.mean():.3f} ± {element_norms.std():.3f}")
        
        # Similarity analysis
        similarity_matrix = self.dictionary.T @ self.dictionary
        off_diagonal = similarity_matrix - np.eye(self.n_components)
        avg_similarity = np.mean(np.abs(off_diagonal))
        print(f"   • Average element similarity: {avg_similarity:.3f}")
        
    def plot_training_curves(self):
        """Plot training curves"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Reconstruction error
        ax1.plot(self.training_history['reconstruction_error'])
        ax1.set_title('Reconstruction Error')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MSE')
        ax1.grid(True, alpha=0.3)
        
        # Sparsity
        ax2.plot(self.training_history['sparsity'])
        ax2.set_title('Average Sparsity')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Active Elements')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def configure_sparseness_function(self, function_name: str, **kwargs):
        """
        Configure sparseness function and its parameters for maximum user flexibility
        
        Args:
            function_name: One of ['l1', 'log', 'gaussian', 'huber', 'elastic_net', 'cauchy', 'student_t']
            **kwargs: Function-specific parameters:
                - huber_delta: Threshold for Huber penalty (default: 1.0)
                - elastic_net_l1_ratio: L1/L2 mixing ratio for elastic net (default: 0.5)
                - cauchy_gamma: Scale parameter for Cauchy penalty (default: 1.0)
                - student_t_nu: Degrees of freedom for Student-t penalty (default: 3.0)
        """
        valid_functions = ['l1', 'log', 'gaussian', 'huber', 'elastic_net', 'cauchy', 'student_t']
        if function_name not in valid_functions:
            raise ValueError(f"Invalid sparseness function. Choose from: {valid_functions}")
            
        self.sparseness_function = function_name
        
        # Set function-specific parameters
        if function_name == 'huber' and 'huber_delta' in kwargs:
            self.huber_delta = kwargs['huber_delta']
        elif function_name == 'elastic_net' and 'elastic_net_l1_ratio' in kwargs:
            self.elastic_net_l1_ratio = kwargs['elastic_net_l1_ratio']
        elif function_name == 'cauchy' and 'cauchy_gamma' in kwargs:
            self.cauchy_gamma = kwargs['cauchy_gamma']
        elif function_name == 'student_t' and 'student_t_nu' in kwargs:
            self.student_t_nu = kwargs['student_t_nu']
            
        print(f"✓ Configured sparseness function: {function_name}")
        if kwargs:
            print(f"  Parameters: {kwargs}")
    
    def get_sparseness_function_info(self) -> Dict[str, Any]:
        """
        Get information about available sparseness functions and current configuration
        
        Returns:
            Dict with function descriptions, current settings, and parameter ranges
        """
        
        function_info = {
            'l1': {
                'description': 'L1 penalty: |x| - Standard sparse coding penalty from Olshausen & Field 1996',
                'parameters': {},
                'properties': 'Sharp sparsity, good for exact zeros'
            },
            'log': {
                'description': 'Log penalty: log(1 + x²) - Smooth approximation to L1',
                'parameters': {},
                'properties': 'Smooth gradients, differentiable everywhere'
            },
            'gaussian': {
                'description': 'Gaussian penalty: -exp(-x²) - Favors small coefficients',
                'parameters': {},
                'properties': 'Very smooth, less aggressive sparsity'
            },
            'huber': {
                'description': 'Huber penalty: Smooth transition from quadratic to linear',
                'parameters': {'huber_delta': 'Transition threshold (default: 1.0)'},
                'properties': 'Robust to outliers, smooth near zero'
            },
            'elastic_net': {
                'description': 'Elastic net: Combination of L1 and L2 penalties',
                'parameters': {'elastic_net_l1_ratio': 'L1/(L1+L2) mixing ratio (default: 0.5)'},
                'properties': 'Grouped variable selection, handles correlated features'
            },
            'cauchy': {
                'description': 'Cauchy penalty: log(1 + (x/γ)²) - Heavy-tailed for extreme sparsity',
                'parameters': {'cauchy_gamma': 'Scale parameter (default: 1.0)'},
                'properties': 'Very sparse solutions, robust to outliers'
            },
            'student_t': {
                'description': 'Student-t penalty: log(1 + x²/ν) - Robust heavy-tailed distribution',
                'parameters': {'student_t_nu': 'Degrees of freedom (default: 3.0)'},
                'properties': 'Flexible tail behavior, adjustable via degrees of freedom'
            }
        }
        
        current_config = {
            'current_function': getattr(self, 'sparseness_function', 'l1'),
            'current_parameters': {}
        }
        
        # Add current parameter values
        if hasattr(self, 'huber_delta'):
            current_config['current_parameters']['huber_delta'] = self.huber_delta
        if hasattr(self, 'elastic_net_l1_ratio'):
            current_config['current_parameters']['elastic_net_l1_ratio'] = self.elastic_net_l1_ratio
        if hasattr(self, 'cauchy_gamma'):
            current_config['current_parameters']['cauchy_gamma'] = self.cauchy_gamma
        if hasattr(self, 'student_t_nu'):
            current_config['current_parameters']['student_t_nu'] = self.student_t_nu
            
        return {
            'available_functions': function_info,
            'current_configuration': current_config,
            'usage_examples': {
                'l1_standard': "coder.configure_sparseness_function('l1')",
                'huber_robust': "coder.configure_sparseness_function('huber', huber_delta=2.0)",
                'elastic_net_balanced': "coder.configure_sparseness_function('elastic_net', elastic_net_l1_ratio=0.7)",
                'cauchy_very_sparse': "coder.configure_sparseness_function('cauchy', cauchy_gamma=0.5)",
                'student_t_adaptive': "coder.configure_sparseness_function('student_t', student_t_nu=5.0)"
            }
        }


# Example usage and demonstration
if __name__ == "__main__":
    print("🖼️  Sparse Coding Library - Olshausen & Field (1996)")
    print("=" * 55)
    
    # Generate test images (natural-like patterns)
    def generate_test_images(n_images=50, img_size=(64, 64)):
        """Generate test images with edge-like patterns"""
        images = []
        
        for _ in range(n_images):
            img = np.zeros(img_size)
            
            # Add random oriented edges
            for _ in range(5):
                # Random line parameters
                y1, x1 = np.random.randint(0, img_size[0], 2)
                y2, x2 = np.random.randint(0, img_size[0], 2)
                
                # Draw line
                length = max(abs(y2-y1), abs(x2-x1))
                if length > 0:
                    for t in np.linspace(0, 1, length):
                        y = int(y1 + t * (y2 - y1))
                        x = int(x1 + t * (x2 - x1))
                        if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
                            img[y, x] = np.random.uniform(0.5, 1.0)
            
            # Add noise
            img += np.random.normal(0, 0.1, img_size)
            images.append(img)
            
        return np.array(images)
    
    def _sparse_encode_equation_5(self, patch: np.ndarray) -> np.ndarray:
        """
        Original paper equation (5) fixed-point iteration (implementing FIXME suggestion)
        âᵢ = bᵢ - Σⱼ Cᵢⱼâⱼ - λ/σ S'(âᵢ/σ)
        """
        coeffs = np.zeros(self.n_components)
        
        # Precompute bᵢ = Σₓ φᵢ(x,y)I(x,y)
        b = self.dictionary.T @ patch
        
        # Precompute Cᵢⱼ = Σₓ φᵢ(x,y)φⱼ(x,y) (Gram matrix)
        C = self.dictionary.T @ self.dictionary
        
        sigma = 1.0  # Scaling constant from paper
        
        for iteration in range(self.max_iter):
            coeffs_old = coeffs.copy()
            
            for i in range(len(coeffs)):
                # Compute âᵢ = bᵢ - Σⱼ≠ᵢ Cᵢⱼâⱼ - λ/σ S'(âᵢ/σ)
                sum_term = np.sum(C[i, :] * coeffs) - C[i, i] * coeffs[i]
                
                # S'(x) derivative depends on sparseness function choice
                if self.sparseness_function == 'log':
                    # S(x) = log(1 + x²), S'(x) = 2x/(1 + x²)
                    sparseness_deriv = 2 * coeffs[i] / (1 + coeffs[i]**2)
                elif self.sparseness_function == 'gaussian':
                    # S(x) = -e^(-x²), S'(x) = 2x*e^(-x²)
                    sparseness_deriv = 2 * coeffs[i] * np.exp(-coeffs[i]**2)
                else:
                    # S(x) = |x|, S'(x) = sign(x)
                    sparseness_deriv = np.sign(coeffs[i])
                
                # Update equation (5)
                if C[i, i] != 0:
                    coeffs[i] = (b[i] - sum_term - (self.sparsity_penalty / sigma) * sparseness_deriv) / C[i, i]
                
            # Check convergence
            if np.linalg.norm(coeffs - coeffs_old) < self.tolerance:
                break
                
        return coeffs
    
    def _fista_optimization(self, patch: np.ndarray, objective_func, gradient_func, initial_coeffs: np.ndarray) -> np.ndarray:
        """
        FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for L1-regularized problems
        Alternative to L-BFGS-B which is not optimal for L1 (addressing FIXME)
        """
        x = initial_coeffs.copy()
        y = initial_coeffs.copy()
        t = 1
        L = 1.0  # Lipschitz constant estimate
        
        for iteration in range(self.max_iter):
            x_old = x.copy()
            
            # Compute gradient at y
            grad = gradient_func(y)
            
            # Gradient step
            z = y - grad / L
            
            # Proximal operator for L1 (soft thresholding)
            threshold = self.sparsity_penalty / L
            x = np.sign(z) * np.maximum(np.abs(z) - threshold, 0)
            
            # Update momentum term
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = x + (t - 1) / t_new * (x - x_old)
            t = t_new
            
            # Check convergence
            if np.linalg.norm(x - x_old) < self.tolerance:
                break
                
        return x
    
    def _proximal_gradient(self, patch: np.ndarray, objective_func, gradient_func, initial_coeffs: np.ndarray) -> np.ndarray:
        """
        Proximal gradient descent method for L1-regularized problems
        Another alternative to L-BFGS-B (addressing FIXME)
        """
        x = initial_coeffs.copy()
        step_size = 0.1
        
        for iteration in range(self.max_iter):
            x_old = x.copy()
            
            # Compute gradient (without L1 term)
            reconstruction = self.dictionary @ x
            error = reconstruction - patch
            grad_reconstruction = self.dictionary.T @ error
            
            # Gradient step
            z = x - step_size * grad_reconstruction
            
            # Proximal operator for L1 (soft thresholding)
            threshold = self.sparsity_penalty * step_size
            x = np.sign(z) * np.maximum(np.abs(z) - threshold, 0)
            
            # Check convergence
            if np.linalg.norm(x - x_old) < self.tolerance:
                break
                
        return x
        
    # Create test data
    test_images = generate_test_images(20, (32, 32))
    print(f"Generated {len(test_images)} test images")
    
    # Create and train sparse coder
    sparse_coder = SparseCoder(
        n_components=64,
        sparsity_penalty=0.1,
        patch_size=(8, 8),
        random_seed=42
    )
    
    # Learn dictionary
    results = sparse_coder.fit(test_images, n_patches=2000)
    
    # Visualize results
    sparse_coder.visualize_dictionary(figsize=(12, 12))
    sparse_coder.plot_training_curves()
    
    print(f"\n💡 Key Innovation:")
    print(f"   • Natural images are sparse in learned basis")
    print(f"   • Algorithm discovers edge detectors automatically") 
    print(f"   • Foundation of modern convolutional neural networks")
    print(f"   • Matches biological visual cortex structure!")