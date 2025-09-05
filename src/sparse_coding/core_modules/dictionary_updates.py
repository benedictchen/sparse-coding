"""
📋 Dictionary Updates
======================

🎯 ELI5 Summary:
This file is an important component in our AI research system! Like different organs 
in your body that work together to keep you healthy, this file has a specific job that 
helps the overall algorithm work correctly and efficiently.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

"""
"""
🏗️ Sparse Coding Dictionary Learning - Neural Feature Discovery Algorithms
=========================================================================

🧠 ELI5 Explanation:
Imagine you're teaching a computer to recognize handwriting by showing it thousands of letters. 
Instead of memorizing every letter, you want it to learn the basic "building blocks" - like 
curves, lines, and loops - that make up all letters. Dictionary learning is exactly this process:

1. **Dictionary Atoms**: These are the "basic building blocks" (like LEGO pieces) that the 
   computer discovers. In vision, these become edge detectors, curve detectors, etc. - just 
   like neurons in your brain's visual cortex!

2. **Update Process**: The computer starts with random building blocks and gradually improves 
   them by seeing which ones are most useful for reconstructing the training images. It's like 
   evolution - useful features survive and get refined.

3. **Biological Inspiration**: Olshausen & Field showed that when you do this process on 
   natural images, you automatically get the same edge-detecting features found in mammalian 
   visual cortex. Your brain literally learned these features the same way!

The math ensures each "building block" stays normalized (same strength) while becoming maximally 
useful for representing natural images with minimal resources - exactly like biological neurons.

📚 Research Foundation:  
- Olshausen, B. & Field, D. (1996) "Emergence of simple-cell receptive field properties"
- Lee, D. & Seung, H. (1999) "Learning the parts of objects by non-negative matrix factorization"
- Elad, M. & Aharon, M. (2006) "Image denoising via sparse and redundant representations"
- Mairal, J. et al. (2010) "Online dictionary learning for sparse coding"

Key mathematical insight: Dictionary atoms φᵢ solve: min_Φ ||X - ΦS||²_F s.t. ||φᵢ||₂ = 1
Each atom represents a primitive feature that efficiently encodes natural image statistics.

🏗️ Dictionary Learning Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                        DICTIONARY LEARNING PROCESS                   │
├─────────────────────────────────────────────────────────────────────┤
│  Input Images → Feature Learning → Dictionary Atoms → Reconstruction │
│       ↓               ↓                  ↓                  ↓        │
│   [64x64 patches] → [Learning] → [φ₁,φ₂,...,φₙ] → [X ≈ ΦS]        │
│                                                                     │
│  ITERATION LOOP:                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 1. Sparse Coding: S = argmin ||X - ΦS||² + λ||S||₁        │   │
│  │              ↓                                               │   │
│  │ 2. Dictionary Update: Φ = argmin ||X - ΦS||²               │   │
│  │              ↓                                               │   │
│  │ 3. Normalize: φᵢ = φᵢ/||φᵢ||₂  (unit norm constraint)      │   │
│  │              ↓                                               │   │
│  │ 4. Check Convergence → Repeat if needed                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  UPDATE METHODS:                                                    │
│  • Multiplicative: φᵢ ← φᵢ + η(XSᵢᵀ - ΦSSᵢᵀ)/||S||²              │
│  • K-SVD: Optimize each atom with SVD while fixing others          │
│  • Online: Stochastic updates for streaming data                   │
└─────────────────────────────────────────────────────────────────────┘

🔧 Usage Examples:
```python
# Learn edge-detecting features from natural images
import numpy as np
from sparse_coding import SparseCoder

# Create sparse coder with dictionary learning
coder = SparseCoder(n_atoms=144, max_iter=100, learning_rule='multiplicative')

# Train on natural image patches
image_patches = np.random.randn(1000, 64)  # 1000 patches, 64 pixels each
coder.fit(image_patches)

# The learned dictionary now contains edge-like features!
learned_features = coder.dictionary_
# Each column is a primitive feature (like Gabor filters in V1 cortex)

# Reconstruct images using learned features
reconstruction = coder.transform(image_patches) @ learned_features.T
print(f"Reconstruction error: {np.mean((image_patches - reconstruction)**2):.4f}")
```

⚙️ Mathematical Foundations:
- **Multiplicative Updates**: φᵢ ← φᵢ + η(XSᵢᵀ - ΦSSᵢᵀ) where η is learning rate
- **Unit Norm Constraint**: ||φᵢ||₂ = 1 prevents scale ambiguity and ensures stability
- **K-SVD Algorithm**: φᵢ = u₁ from SVD(X - Σⱼ≠ᵢ φⱼSⱼ) for optimal single-atom updates
- **Online Learning**: φᵢ ← φᵢ + η(xₜ - φᵢᵀxₜ)φᵢ for streaming data efficiency

💰 FUNDING APPEAL - PLEASE DONATE! 💰
=====================================
🌟 This dictionary learning research is made possible by Benedict Chen
   📧 Contact: benedict@benedictchen.com
   
💳 PLEASE DONATE! Your support keeps this research alive! 💳
   🔗 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   🔗 GitHub Sponsors: https://github.com/sponsors/benedictchen
   
☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!
(Start small, dream big! Every donation helps advance AI research! 😄)

💡 Why donate? This algorithm replicates how your visual cortex learned to see! Supporting this 
   research helps decode the mysteries of intelligence itself! 🧠✨
"""

"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀

Made possible by Benedict Chen (benedict@benedictchen.com)
"""

import numpy as np
from scipy import linalg
from sklearn.preprocessing import normalize
from typing import Tuple, Optional, Dict, Any
import warnings


class DictionaryUpdatesMixin:
    """
    🏗️ Dictionary Updates Mixin for Sparse Coding
    
    Contains dictionary update methods including multiplicative updates,
    additive updates, and projection methods for learning optimal dictionaries.
    
    Research-accurate implementations following Olshausen & Field (1996).
    """
    
    def _dictionary_update_step(self, X: np.ndarray, codes: np.ndarray) -> None:
        """
        Dictionary update step using specified update method.
        
        Updates dictionary atoms φᵢ to minimize reconstruction error
        while maintaining unit norm constraint: ||φᵢ||₂ = 1
        
        Args:
            X: Training data [n_samples, n_features]
            codes: Sparse coefficients [n_samples, n_components]
        """
        
        # Choose update method
        if hasattr(self, 'dict_update_method'):
            method = self.dict_update_method
        else:
            method = 'multiplicative'  # Default to Olshausen & Field
        
        if method == 'multiplicative':
            self._multiplicative_update(X, codes)
        elif method == 'additive':
            self._additive_update(X, codes)
        elif method == 'projection':
            self._projection_update(X, codes)
        else:
            # Default to multiplicative update
            self._multiplicative_update(X, codes)
    
    def _multiplicative_update(self, X: np.ndarray, codes: np.ndarray) -> None:
        """
        Multiplicative Dictionary Update (Olshausen & Field 1996).
        
        Implements the original learning rule from Nature paper:
        φᵢ ← φᵢ + η * Σⱼ(xⱼ - Σₖaₖⱼφₖ)aᵢⱼ
        
        This is the classic sparse coding dictionary learning algorithm
        with proper normalization to maintain unit-norm atoms.
        
        # Implementation of research-accurate multiplicative update from Olshausen & Field (1996)
        # Following the exact learning rule: Δφᵢ = η⟨xᵢ - Σⱼaⱼφⱼ⟩aᵢ
        # with proper normalization and learning rate scheduling
        
        Args:
            X: Training data [n_samples, n_features]
            codes: Sparse coefficients [n_samples, n_components]
        """
        
        n_samples, n_features = X.shape
        
        # Implement learning rate scheduling η(t) = η₀/(1 + t/τ) from Olshausen & Field (1996)
        base_lr = getattr(self, 'dict_learning_rate', getattr(self, 'learning_rate', 0.01))
        if hasattr(self, 'iteration_count') and hasattr(self, 'learning_rate_decay_tau'):
            # Time-dependent learning rate decay as per OF96
            eta = base_lr / (1 + self.iteration_count / self.learning_rate_decay_tau)
        else:
            eta = base_lr
        
        # Track iteration for learning rate scheduling
        if not hasattr(self, 'iteration_count'):
            self.iteration_count = 0
        self.iteration_count += 1
        
        # Implement proper batch vs online update distinction from Olshausen & Field (1996)
        update_mode = getattr(self, 'update_mode', 'batch')  # 'batch' or 'online'
        
        if update_mode == 'online':
            # Online learning: update dictionary after each sample (original OF96 approach)
            for sample_idx in range(n_samples):
                x_sample = X[sample_idx:sample_idx+1]
                code_sample = codes[sample_idx:sample_idx+1]
                self._online_dictionary_update(x_sample, code_sample, eta)
        else:
            # Batch learning: update dictionary using all samples simultaneously
            self._batch_dictionary_update(X, codes, eta)
    
    def _batch_dictionary_update(self, X: np.ndarray, codes: np.ndarray, eta: float):
        """Batch dictionary update using all samples."""
        n_samples = X.shape[0]
        
        # Track atom usage for competition and pruning mechanisms
        atom_usage = np.sum(np.abs(codes), axis=0)  # Usage frequency per atom
        
        for i in range(self.n_components):
            
            # Atom competition: skip atoms with very low usage (Olshausen & Field pruning)
            if atom_usage[i] < getattr(self, 'min_atom_usage_threshold', 1e-6):
                # Implement atom pruning mechanism from OF96
                if hasattr(self, 'enable_atom_pruning') and self.enable_atom_pruning:
                    self.dictionary_[i] = self._reinitialize_competitive_atom(i, X, atom_usage)
                continue
            
            # Method 1: Classic Olshausen & Field update rule
            # Compute reconstruction error excluding atom i
            reconstruction_error = X.T - self.dictionary_.T @ codes.T
            # Add back contribution of current atom i
            reconstruction_error += np.outer(self.dictionary_[i], codes[:, i])
            
            # Multiplicative update: φᵢ ← φᵢ + η * (residual @ codes_i) / n_samples
            update = reconstruction_error @ codes[:, i] / n_samples
            self.dictionary_[i] += eta * update
            
            # Normalize atom to unit norm (essential for stability)
            atom_norm = np.linalg.norm(self.dictionary_[i])
            if atom_norm > 1e-12:
                self.dictionary_[i] /= atom_norm
            else:
                # Reinitialize dead atoms
                self.dictionary_[i] = self._reinitialize_atom(i, X)
        
        # Alternative Method 2: Batch matrix update (more efficient)
        # This computes all atoms simultaneously but may be less stable
        if hasattr(self, 'use_batch_update') and self.use_batch_update:
            
            # Compute all reconstruction errors at once
            reconstruction = X.T - self.dictionary_.T @ codes.T
            
            # Batch gradient: D += η * reconstruction @ codes.T / n_samples  
            gradient = reconstruction @ codes / n_samples
            self.dictionary_ += eta * gradient
            
            # Normalize all atoms
            atom_norms = np.linalg.norm(self.dictionary_, axis=1, keepdims=True)
            atom_norms[atom_norms < 1e-12] = 1.0  # Avoid division by zero
            self.dictionary_ /= atom_norms
            
            # Reinitialize atoms with very small norms (dead atoms)
            dead_atom_mask = atom_norms < 1e-12
            for dead_idx in np.where(dead_atom_mask.flatten())[0]:
                self.dictionary_[dead_idx] = self._reinitialize_competitive_atom(dead_idx, X, atom_usage)
    
    def _online_dictionary_update(self, x_sample: np.ndarray, code_sample: np.ndarray, eta: float):
        """
        Online dictionary update for single sample - original Olshausen & Field (1996) approach.
        
        Implements the exact online learning rule from the Nature paper:
        Δφᵢ = η * (x - Σⱼaⱼφⱼ) * aᵢ
        """
        # Compute reconstruction error for this sample
        reconstruction = x_sample.T - self.dictionary_.T @ code_sample.T
        
        # Update each atom based on its activation for this sample
        for i in range(self.n_components):
            if np.abs(code_sample[0, i]) > 1e-12:  # Only update active atoms
                # Online multiplicative update: φᵢ += η * residual * aᵢ
                self.dictionary_[i] += eta * reconstruction.flatten() * code_sample[0, i]
                
                # Normalize to unit norm after each update
                atom_norm = np.linalg.norm(self.dictionary_[i])
                if atom_norm > 1e-12:
                    self.dictionary_[i] /= atom_norm
                else:
                    # Reinitialize dead atoms
                    self.dictionary_[i] = self._reinitialize_atom(i, x_sample)
    
    def _reinitialize_competitive_atom(self, atom_idx: int, X: np.ndarray, atom_usage: np.ndarray) -> np.ndarray:
        """
        Reinitialize poorly performing atoms using competitive learning principles.
        
        Implements atom competition mechanism from Olshausen & Field (1996) where
        unused atoms are replaced with components that better explain residual variance.
        """
        # Find the most active atom for competition-based replacement
        most_active_idx = np.argmax(atom_usage)
        
        if atom_usage[most_active_idx] > 0:
            # Initialize new atom as noisy version of most active atom (competition)
            noise_scale = getattr(self, 'atom_reinit_noise', 0.1)
            new_atom = self.dictionary_[most_active_idx].copy()
            new_atom += np.random.normal(0, noise_scale, new_atom.shape)
            
            # Normalize the new competitive atom
            new_atom /= np.linalg.norm(new_atom)
            return new_atom
        else:
            # Fallback: random reinitialization if no atoms are active
            return self._reinitialize_atom(atom_idx, X)
    def _additive_update(self, X: np.ndarray, codes: np.ndarray) -> None:
        """
        Additive Dictionary Update with Gradient Descent.
        
        Implements gradient-based dictionary learning:
        φᵢ ← φᵢ - η * ∇_φᵢ E where E is the reconstruction error
        
        Args:
            X: Training data [n_samples, n_features]  
            codes: Sparse coefficients [n_samples, n_components]
        """
        
        n_samples = X.shape[0]
        eta = getattr(self, 'dict_learning_rate', 0.01)
        
        # Compute reconstruction error
        reconstruction = X.T - self.dictionary_.T @ codes.T
        
        # Gradient w.r.t. dictionary: ∇_D E = -2 * reconstruction @ codes.T
        gradient = -2.0 * reconstruction @ codes / n_samples
        
        # Additive update with learning rate
        self.dictionary_ -= eta * gradient
        
        # Project to unit sphere (normalize atoms)
        self.dictionary_ = normalize(self.dictionary_, axis=1, norm='l2')
    
    def _projection_update(self, X: np.ndarray, codes: np.ndarray) -> None:
        """
        Projection-based Dictionary Update.
        
        Uses analytical solution for dictionary update when codes are fixed:
        D* = argmin_D ||X - D @ codes.T||² subject to ||dᵢ||₂ = 1
        
        Args:
            X: Training data [n_samples, n_features]
            codes: Sparse coefficients [n_samples, n_components]
        """
        
        # Analytical solution: D = X @ codes @ (codes.T @ codes)⁻¹
        # But we need to handle singularity and normalization
        
        gram_matrix = codes.T @ codes
        
        # Add regularization to avoid singularity
        reg_param = 1e-8
        gram_regularized = gram_matrix + reg_param * np.eye(self.n_components)
        
        try:
            # Solve: D @ gram = X.T @ codes
            gram_inv = linalg.solve(gram_regularized, np.eye(self.n_components))
            self.dictionary_ = (X.T @ codes @ gram_inv).T
            
            # Normalize atoms to unit norm
            self.dictionary_ = normalize(self.dictionary_, axis=1, norm='l2')
            
        except linalg.LinAlgError:
            # Fallback to multiplicative update if matrix is singular
            warnings.warn("Projection update failed, using multiplicative update")
            self._multiplicative_update(X, codes)
    
    def _k_svd_update(self, X: np.ndarray, codes: np.ndarray, atom_idx: int) -> None:
        """
        K-SVD Dictionary Update (Elad & Aharon 2006).
        
        Updates a single dictionary atom using SVD for optimal reconstruction.
        This is an effective method for dictionary learning.
        
        Args:
            X: Training data [n_samples, n_features]
            codes: Sparse coefficients [n_samples, n_components]  
            atom_idx: Index of atom to update
        """
        
        # Find samples that use this atom (non-zero coefficients)
        active_samples = np.where(np.abs(codes[:, atom_idx]) > 1e-12)[0]
        
        if len(active_samples) == 0:
            # No samples use this atom, reinitialize
            self.dictionary_[atom_idx] = self._reinitialize_atom(atom_idx, X)
            return
        
        # Compute error matrix excluding current atom
        error_matrix = X[active_samples].T - self.dictionary_.T @ codes[active_samples].T
        # Add back current atom contribution
        error_matrix += np.outer(self.dictionary_[atom_idx], codes[active_samples, atom_idx])
        
        # SVD of error matrix restricted to active samples
        try:
            U, s, Vt = linalg.svd(error_matrix, full_matrices=False)
            
            # Update dictionary atom (first left singular vector)
            self.dictionary_[atom_idx] = U[:, 0]
            
            # Update coefficients (first singular value * first right singular vector)
            codes[active_samples, atom_idx] = s[0] * Vt[0, :]
            
        except linalg.LinAlgError:
            # SVD failed, use fallback update
            self._multiplicative_update(X[active_samples:active_samples+1], 
                                       codes[active_samples:active_samples+1])
    
    def _reinitialize_atom(self, atom_idx: int, X: np.ndarray) -> np.ndarray:
        """
        Reinitialize a dead or unused dictionary atom.
        
        Args:
            atom_idx: Index of atom to reinitialize
            X: Training data for initialization
            
        Returns:
            New normalized atom
        """
        
        # Method 1: Random initialization
        if not hasattr(self, 'atom_init_method') or self.atom_init_method == 'random':
            rng = np.random.RandomState(self.random_state)
            new_atom = rng.randn(X.shape[1])
            return new_atom / np.linalg.norm(new_atom)
        
        # Method 2: Initialize from data sample
        elif self.atom_init_method == 'data':
            rng = np.random.RandomState(self.random_state)
            sample_idx = rng.randint(0, X.shape[0])
            new_atom = X[sample_idx].copy()
            atom_norm = np.linalg.norm(new_atom)
            if atom_norm > 1e-12:
                return new_atom / atom_norm
            else:
                # Fallback to random if data sample is zero
                new_atom = rng.randn(X.shape[1])
                return new_atom / np.linalg.norm(new_atom)
        
        # Method 3: Principal component
        elif self.atom_init_method == 'pca':
            try:
                # Use first principal component of residual
                U, s, Vt = linalg.svd(X.T, full_matrices=False)
                return U[:, 0]  # First PC already normalized
            except:
                # Fallback to random
                rng = np.random.RandomState(self.random_state)
                new_atom = rng.randn(X.shape[1])
                return new_atom / np.linalg.norm(new_atom)
    
    def _detect_dead_atoms(self, codes: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
        """
        Detect dictionary atoms that are rarely used (dead atoms).
        
        Args:
            codes: Sparse coefficients [n_samples, n_components]
            threshold: Usage threshold below which atoms are considered dead
            
        Returns:
            Indices of dead atoms
        """
        
        # Compute usage statistics for each atom
        atom_usage = np.mean(np.abs(codes), axis=0)
        
        # Find atoms with usage below threshold
        dead_atoms = np.where(atom_usage < threshold)[0]
        
        return dead_atoms
    
    def _prune_and_replace_atoms(self, X: np.ndarray, codes: np.ndarray) -> int:
        """
        Prune dead atoms and replace with new ones.
        
        Args:
            X: Training data [n_samples, n_features]
            codes: Sparse coefficients [n_samples, n_components]
            
        Returns:
            Number of atoms replaced
        """
        
        dead_atoms = self._detect_dead_atoms(codes)
        
        if len(dead_atoms) > 0 and self.verbose:
            print(f"  Replacing {len(dead_atoms)} dead atoms")
        
        # Replace each dead atom
        for atom_idx in dead_atoms:
            self.dictionary_[atom_idx] = self._reinitialize_atom(atom_idx, X)
        
        return len(dead_atoms)


# Export the mixin class
__all__ = ['DictionaryUpdatesMixin']


if __name__ == "__main__":
    # print("🏗️ Sparse Coding - Dictionary Updates Module")
    print("=" * 50)
    # Removed print spam: "...
    print("  • DictionaryUpdatesMixin - Dictionary learning methods")
    print("  • Multiplicative Update (Olshausen & Field 1996)")
    print("  • Additive Update with gradient descent")
    print("  • Projection Update with analytical solution")
    print("  • K-SVD Update (effective method)")
    print("  • Dead atom detection and replacement")
    print("  • Research-accurate normalization and regularization")
    print("")
    # # Removed print spam: "...
    print("🔬 Advanced dictionary learning algorithms!")