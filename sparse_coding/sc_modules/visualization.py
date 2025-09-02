"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀

✨ Sparse Coding Visualization Module
=====================================

Comprehensive visualization utilities for sparse coding analysis and research.
This module provides a mixin class with visualization methods that maintain
access to the sparse coder's internal state.

Author: Benedict Chen (benedict@benedictchen.com)

Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"

Key Visualizations:
- Dictionary/basis function visualization
- Training curve analysis
- Sparse code analysis
- Reconstruction quality assessment  
- Feature activation patterns
- Dictionary property analysis

🎯 Core Functionality:
======================
- Dictionary element visualization (learned basis functions)
- Training progress tracking and plotting
- Sparse code distribution analysis
- Reconstruction quality assessment
- Feature usage and activation statistics
- Dictionary coherence and condition analysis
- Comparative visualizations for research

🔬 Research Applications:
========================
- Understanding learned representations
- Analyzing convergence properties
- Comparing different algorithms
- Evaluating dictionary quality
- Studying sparsity patterns
- Biological plausibility assessment
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List, Union
import warnings

# Handle matplotlib import with graceful fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Warning: matplotlib not available. Visualization functions will be limited.")

# Handle seaborn import for enhanced plotting
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class VisualizationMixin:
    """
    Visualization mixin for SparseCoder that provides comprehensive plotting utilities.
    
    This mixin requires the following attributes to be available in the parent class:
    - self.dictionary: Learned dictionary matrix
    - self.n_components: Number of dictionary elements
    - self.patch_size: Size of image patches
    - self.training_history: Dictionary with training metrics
    - self.sparsity_penalty: Sparsity regularization parameter
    """

    def visualize_dictionary(self, figsize: Tuple[int, int] = (16, 16), 
                           max_elements: int = 256, title: str = None,
                           save_path: str = None, colormap: str = 'gray') -> None:
        """
        Visualize learned dictionary elements (basis functions).
        
        This is where we see the magic - the algorithm discovers edge detectors!
        The learned basis functions resemble oriented edge filters similar to
        receptive fields of simple cells in the primary visual cortex.
        
        Args:
            figsize: Figure size in inches
            max_elements: Maximum number of elements to display
            title: Custom title for the plot
            save_path: Path to save the figure (optional)
            colormap: Matplotlib colormap name
        """
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Warning: matplotlib not available for visualization")
            return
        
        if not hasattr(self, 'dictionary') or self.dictionary is None:
            print("⚠️  Warning: No dictionary available for visualization")
            return
            
        n_plot = min(self.n_components, max_elements)
        grid_size = int(np.ceil(np.sqrt(n_plot)))
        
        try:
            fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
            
            # Handle single subplot case
            if grid_size == 1:
                axes = np.array([[axes]])
            elif axes.ndim == 1:
                axes = axes.reshape(1, -1)
                
            # Set title
            if title is None:
                title = f'Learned Sparse Dictionary - {n_plot} Basis Functions'
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Plot each dictionary element
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    ax = axes[i, j] if grid_size > 1 else axes[0, 0]
                    
                    if idx < n_plot:
                        # Reshape dictionary element to patch
                        element = self.dictionary[:, idx].reshape(self.patch_size)
                        
                        # Normalize for visualization
                        element_min, element_max = element.min(), element.max()
                        if element_max > element_min:
                            element = (element - element_min) / (element_max - element_min)
                        else:
                            element = np.zeros_like(element)
                        
                        # Display with specified colormap
                        im = ax.imshow(element, cmap=colormap, interpolation='nearest')
                        ax.set_title(f'#{idx+1}', fontsize=8)
                    else:
                        # Hide unused subplots
                        ax.imshow(np.zeros(self.patch_size), cmap=colormap)
                    
                    ax.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Dictionary visualization saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Dictionary visualization failed: {e}")
            return
        
        # Analyze and print dictionary properties
        self._analyze_dictionary_properties()

    def plot_training_curves(self, figsize: Tuple[int, int] = (14, 5),
                           save_path: str = None, show_statistics: bool = True) -> None:
        """
        Plot comprehensive training curves and learning statistics.
        
        Args:
            figsize: Figure size in inches
            save_path: Path to save the figure (optional)
            show_statistics: Whether to print training statistics
        """
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Warning: matplotlib not available for plotting")
            return
            
        if not hasattr(self, 'training_history') or not self.training_history:
            print("⚠️  Warning: No training history available for plotting")
            return
        
        try:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            fig.suptitle('Sparse Coding Training Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Reconstruction Error
            if 'reconstruction_error' in self.training_history:
                errors = self.training_history['reconstruction_error']
                axes[0].plot(errors, 'b-', linewidth=2, alpha=0.8)
                axes[0].set_title('Reconstruction Error')
                axes[0].set_xlabel('Iteration')
                axes[0].set_ylabel('Mean Squared Error')
                axes[0].grid(True, alpha=0.3)
                axes[0].set_yscale('log')
                
                # Add trend line
                if len(errors) > 5:
                    z = np.polyfit(range(len(errors)), np.log(errors + 1e-10), 1)
                    p = np.poly1d(z)
                    axes[0].plot(range(len(errors)), np.exp(p(range(len(errors)))), 
                               'r--', alpha=0.7, label='Trend')
                    axes[0].legend()
            
            # Plot 2: Sparsity Evolution
            if 'sparsity' in self.training_history:
                sparsity = self.training_history['sparsity']
                axes[1].plot(sparsity, 'g-', linewidth=2, alpha=0.8)
                axes[1].set_title('Average Sparsity')
                axes[1].set_xlabel('Iteration')
                axes[1].set_ylabel('Active Elements')
                axes[1].grid(True, alpha=0.3)
                
                # Add target sparsity line if available
                if hasattr(self, 'target_sparsity'):
                    axes[1].axhline(y=self.target_sparsity, color='r', 
                                  linestyle='--', alpha=0.7, label='Target')
                    axes[1].legend()
            
            # Plot 3: Learning Rate / Convergence
            iterations = range(len(self.training_history.get('reconstruction_error', [])))
            if len(iterations) > 1:
                # Compute convergence rate
                errors = self.training_history.get('reconstruction_error', [])
                if len(errors) > 1:
                    convergence_rate = [abs(errors[i] - errors[i-1]) for i in range(1, len(errors))]
                    axes[2].semilogy(convergence_rate, 'm-', linewidth=2, alpha=0.8)
                    axes[2].set_title('Convergence Rate')
                    axes[2].set_xlabel('Iteration')
                    axes[2].set_ylabel('|Error Change|')
                    axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Training curves saved to: {save_path}")
                
            plt.show()
            
            if show_statistics:
                self._print_training_statistics()
                
        except Exception as e:
            print(f"⚠️  Training curve plotting failed: {e}")

    def visualize_sparse_codes(self, coefficients: np.ndarray, 
                             figsize: Tuple[int, int] = (12, 8),
                             max_samples: int = 100, save_path: str = None) -> None:
        """
        Visualize sparse coefficient patterns and distributions.
        
        Args:
            coefficients: Sparse coefficients matrix (n_samples, n_components)
            figsize: Figure size in inches
            max_samples: Maximum number of samples to display
            save_path: Path to save the figure (optional)
        """
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Warning: matplotlib not available for visualization")
            return
            
        if coefficients is None or coefficients.size == 0:
            print("⚠️  Warning: No coefficients provided for visualization")
            return
        
        try:
            # Limit samples for visualization
            n_samples = min(coefficients.shape[0], max_samples)
            coeff_subset = coefficients[:n_samples]
            
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle('Sparse Code Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Coefficient heatmap
            im1 = axes[0, 0].imshow(coeff_subset.T, cmap='RdBu_r', aspect='auto', 
                                   interpolation='nearest')
            axes[0, 0].set_title(f'Coefficient Matrix ({n_samples} samples)')
            axes[0, 0].set_xlabel('Sample Index')
            axes[0, 0].set_ylabel('Dictionary Element')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Plot 2: Sparsity distribution
            sparsity_levels = np.sum(np.abs(coefficients) > 1e-3, axis=1)
            axes[0, 1].hist(sparsity_levels, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_title('Sparsity Distribution')
            axes[0, 1].set_xlabel('Number of Active Elements')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(np.mean(sparsity_levels), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(sparsity_levels):.1f}')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Feature usage frequency
            feature_usage = np.sum(np.abs(coefficients) > 1e-3, axis=0)
            axes[1, 0].bar(range(len(feature_usage)), feature_usage, alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Feature Usage Frequency')
            axes[1, 0].set_xlabel('Dictionary Element Index')
            axes[1, 0].set_ylabel('Usage Count')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Coefficient magnitude distribution
            nonzero_coeffs = coefficients[np.abs(coefficients) > 1e-6]
            if len(nonzero_coeffs) > 0:
                axes[1, 1].hist(nonzero_coeffs, bins=50, alpha=0.7, color='orange', edgecolor='black')
                axes[1, 1].set_title('Non-zero Coefficient Distribution')
                axes[1, 1].set_xlabel('Coefficient Value')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Sparse codes visualization saved to: {save_path}")
                
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Sparse codes visualization failed: {e}")

    def visualize_reconstruction_quality(self, original_patches: np.ndarray,
                                       coefficients: np.ndarray,
                                       n_examples: int = 10,
                                       figsize: Tuple[int, int] = (15, 6),
                                       save_path: str = None) -> None:
        """
        Visualize reconstruction quality by comparing original and reconstructed patches.
        
        Args:
            original_patches: Original input patches
            coefficients: Sparse coefficients for reconstruction
            n_examples: Number of examples to show
            figsize: Figure size in inches
            save_path: Path to save the figure (optional)
        """
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Warning: matplotlib not available for visualization")
            return
            
        if not hasattr(self, 'dictionary') or self.dictionary is None:
            print("⚠️  Warning: No dictionary available for reconstruction")
            return
        
        try:
            # Reconstruct patches
            reconstructed_patches = coefficients @ self.dictionary.T
            
            # Calculate reconstruction errors
            reconstruction_errors = np.mean((original_patches - reconstructed_patches) ** 2, axis=1)
            
            # Select diverse examples (best, worst, and random)
            n_show = min(n_examples, len(original_patches))
            
            # Get best and worst reconstructions
            best_indices = np.argsort(reconstruction_errors)[:n_show//3]
            worst_indices = np.argsort(reconstruction_errors)[-n_show//3:]
            random_indices = np.random.choice(len(original_patches), 
                                            n_show - len(best_indices) - len(worst_indices), 
                                            replace=False)
            
            selected_indices = np.concatenate([best_indices, random_indices, worst_indices])
            
            fig, axes = plt.subplots(3, len(selected_indices), figsize=figsize)
            fig.suptitle('Reconstruction Quality Analysis', fontsize=16, fontweight='bold')
            
            for i, idx in enumerate(selected_indices):
                # Original patch
                orig_patch = original_patches[idx].reshape(self.patch_size)
                recon_patch = reconstructed_patches[idx].reshape(self.patch_size)
                error_patch = np.abs(orig_patch - recon_patch)
                
                # Normalize for display
                orig_norm = (orig_patch - orig_patch.min()) / (orig_patch.max() - orig_patch.min() + 1e-8)
                recon_norm = (recon_patch - recon_patch.min()) / (recon_patch.max() - recon_patch.min() + 1e-8)
                
                # Plot original
                axes[0, i].imshow(orig_norm, cmap='gray', interpolation='nearest')
                axes[0, i].set_title(f'Original #{idx}', fontsize=8)
                axes[0, i].axis('off')
                
                # Plot reconstruction
                axes[1, i].imshow(recon_norm, cmap='gray', interpolation='nearest')
                axes[1, i].set_title(f'Reconstructed', fontsize=8)
                axes[1, i].axis('off')
                
                # Plot error
                axes[2, i].imshow(error_patch, cmap='hot', interpolation='nearest')
                axes[2, i].set_title(f'Error: {reconstruction_errors[idx]:.4f}', fontsize=8)
                axes[2, i].axis('off')
            
            # Add row labels
            fig.text(0.02, 0.75, 'Original', rotation=90, fontsize=12, fontweight='bold', ha='center')
            fig.text(0.02, 0.5, 'Reconstructed', rotation=90, fontsize=12, fontweight='bold', ha='center')
            fig.text(0.02, 0.25, 'Error', rotation=90, fontsize=12, fontweight='bold', ha='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Reconstruction quality visualization saved to: {save_path}")
                
            plt.show()
            
            # Print reconstruction statistics
            self._print_reconstruction_statistics(reconstruction_errors, coefficients)
            
        except Exception as e:
            print(f"⚠️  Reconstruction quality visualization failed: {e}")

    def plot_dictionary_properties(self, figsize: Tuple[int, int] = (15, 10),
                                 save_path: str = None) -> None:
        """
        Analyze and visualize dictionary properties including coherence, condition number,
        and feature correlations.
        
        Args:
            figsize: Figure size in inches
            save_path: Path to save the figure (optional)
        """
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Warning: matplotlib not available for plotting")
            return
            
        if not hasattr(self, 'dictionary') or self.dictionary is None:
            print("⚠️  Warning: No dictionary available for analysis")
            return
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            fig.suptitle('Dictionary Properties Analysis', fontsize=16, fontweight='bold')
            
            # Calculate Gram matrix
            gram_matrix = self.dictionary.T @ self.dictionary
            
            # Plot 1: Gram matrix heatmap
            im1 = axes[0, 0].imshow(gram_matrix, cmap='RdBu_r', interpolation='nearest')
            axes[0, 0].set_title('Gram Matrix (D^T D)')
            axes[0, 0].set_xlabel('Dictionary Element')
            axes[0, 0].set_ylabel('Dictionary Element')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Plot 2: Dictionary element norms
            element_norms = np.linalg.norm(self.dictionary, axis=0)
            axes[0, 1].bar(range(len(element_norms)), element_norms, alpha=0.7, color='lightblue')
            axes[0, 1].set_title('Dictionary Element Norms')
            axes[0, 1].set_xlabel('Element Index')
            axes[0, 1].set_ylabel('L2 Norm')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Coherence analysis
            off_diagonal = gram_matrix - np.eye(self.n_components)
            coherence_values = np.abs(off_diagonal).flatten()
            coherence_values = coherence_values[coherence_values > 1e-10]  # Remove zeros
            
            if len(coherence_values) > 0:
                axes[0, 2].hist(coherence_values, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
                axes[0, 2].set_title(f'Coherence Distribution\nMax: {np.max(coherence_values):.3f}')
                axes[0, 2].set_xlabel('|<d_i, d_j>|')
                axes[0, 2].set_ylabel('Frequency')
                axes[0, 2].set_yscale('log')
                axes[0, 2].grid(True, alpha=0.3)
            
            # Plot 4: Singular values
            U, s, Vt = np.linalg.svd(self.dictionary, full_matrices=False)
            axes[1, 0].semilogy(s, 'o-', alpha=0.7, color='purple')
            axes[1, 0].set_title(f'Singular Values\nCondition Number: {s[0]/s[-1]:.2e}')
            axes[1, 0].set_xlabel('Index')
            axes[1, 0].set_ylabel('Singular Value')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Dictionary element statistics
            element_stats = {
                'Mean': np.mean(self.dictionary, axis=0),
                'Std': np.std(self.dictionary, axis=0),
                'Max': np.max(self.dictionary, axis=0),
                'Min': np.min(self.dictionary, axis=0)
            }
            
            x_pos = np.arange(min(20, self.n_components))  # Show first 20 elements
            width = 0.2
            
            for i, (stat_name, values) in enumerate(element_stats.items()):
                axes[1, 1].bar(x_pos + i*width, values[:len(x_pos)], width, 
                             label=stat_name, alpha=0.7)
            
            axes[1, 1].set_title('Element Statistics (First 20)')
            axes[1, 1].set_xlabel('Dictionary Element')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: Feature correlation network (simplified)
            # Show most correlated pairs
            correlation_threshold = 0.3
            high_corr_pairs = np.where(np.abs(off_diagonal) > correlation_threshold)
            
            if len(high_corr_pairs[0]) > 0:
                axes[1, 2].scatter(high_corr_pairs[0], high_corr_pairs[1], 
                                 c=off_diagonal[high_corr_pairs], 
                                 cmap='RdBu_r', alpha=0.6, s=50)
                axes[1, 2].set_title(f'High Correlations (>{correlation_threshold})')
                axes[1, 2].set_xlabel('Dictionary Element i')
                axes[1, 2].set_ylabel('Dictionary Element j')
                
                # Add colorbar
                divider = make_axes_locatable(axes[1, 2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(axes[1, 2].collections[0], cax=cax)
            else:
                axes[1, 2].text(0.5, 0.5, f'No correlations\n>{correlation_threshold}', 
                              transform=axes[1, 2].transAxes, ha='center', va='center')
                axes[1, 2].set_title('Feature Correlations')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Dictionary properties analysis saved to: {save_path}")
                
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Dictionary properties analysis failed: {e}")

    def compare_dictionaries(self, other_dictionaries: List[np.ndarray],
                           labels: List[str], figsize: Tuple[int, int] = (16, 12),
                           save_path: str = None) -> None:
        """
        Compare multiple dictionaries for research analysis.
        
        Args:
            other_dictionaries: List of dictionary matrices to compare
            labels: Labels for each dictionary
            figsize: Figure size in inches
            save_path: Path to save the figure (optional)
        """
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Warning: matplotlib not available for comparison")
            return
            
        try:
            n_dicts = len(other_dictionaries) + 1  # Include current dictionary
            all_dicts = [self.dictionary] + other_dictionaries
            all_labels = ['Current'] + labels
            
            fig, axes = plt.subplots(2, n_dicts, figsize=figsize)
            fig.suptitle('Dictionary Comparison Analysis', fontsize=16, fontweight='bold')
            
            # Ensure axes is 2D
            if n_dicts == 1:
                axes = axes.reshape(2, 1)
            
            for i, (dictionary, label) in enumerate(zip(all_dicts, all_labels)):
                # Top row: Sample dictionary elements
                n_sample = min(64, dictionary.shape[1])  # Show up to 64 elements
                grid_size = int(np.sqrt(n_sample))
                
                # Create composite image of dictionary elements
                composite_size = (grid_size * self.patch_size[0], grid_size * self.patch_size[1])
                composite = np.zeros(composite_size)
                
                for row in range(grid_size):
                    for col in range(grid_size):
                        idx = row * grid_size + col
                        if idx < n_sample:
                            element = dictionary[:, idx].reshape(self.patch_size)
                            # Normalize element
                            if element.max() > element.min():
                                element = (element - element.min()) / (element.max() - element.min())
                            
                            start_row = row * self.patch_size[0]
                            end_row = start_row + self.patch_size[0]
                            start_col = col * self.patch_size[1]
                            end_col = start_col + self.patch_size[1]
                            
                            composite[start_row:end_row, start_col:end_col] = element
                
                axes[0, i].imshow(composite, cmap='gray', interpolation='nearest')
                axes[0, i].set_title(f'{label}\n({dictionary.shape[1]} elements)')
                axes[0, i].axis('off')
                
                # Bottom row: Properties comparison
                gram_matrix = dictionary.T @ dictionary
                off_diagonal = gram_matrix - np.eye(dictionary.shape[1])
                coherence = np.max(np.abs(off_diagonal))
                condition_num = np.linalg.cond(gram_matrix)
                
                # Create bar plot of key metrics
                metrics = ['Coherence', 'Condition\n(log10)', 'Sparsity\n(if available)']
                values = [coherence, np.log10(condition_num), 0]  # Placeholder for sparsity
                
                colors = ['red' if coherence > 0.9 else 'orange' if coherence > 0.5 else 'green',
                         'red' if condition_num > 1e10 else 'orange' if condition_num > 1e5 else 'green',
                         'gray']
                
                bars = axes[1, i].bar(metrics, values, color=colors, alpha=0.7)
                axes[1, i].set_title(f'Properties: {label}')
                axes[1, i].set_ylabel('Value')
                axes[1, i].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars[:2], values[:2]):
                    height = bar.get_height()
                    axes[1, i].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.3f}' if value < 10 else f'{value:.1e}',
                                   ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Dictionary comparison saved to: {save_path}")
                
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Dictionary comparison failed: {e}")

    def _analyze_dictionary_properties(self) -> Dict[str, float]:
        """
        Analyze dictionary properties and print comprehensive statistics.
        
        Returns:
            Dictionary of computed properties
        """
        
        if not hasattr(self, 'dictionary') or self.dictionary is None:
            print("⚠️  Warning: No dictionary available for analysis")
            return {}
        
        try:
            print("\n📊 Dictionary Analysis")
            print("=" * 50)
            
            # Basic properties
            n_elements = self.dictionary.shape[1]
            patch_dim = self.dictionary.shape[0]
            overcompleteness = n_elements / patch_dim
            
            print(f"Dictionary dimensions: {patch_dim} × {n_elements}")
            print(f"Patch size: {self.patch_size}")
            print(f"Overcompleteness factor: {overcompleteness:.2f}")
            
            # Compute Gram matrix
            gram_matrix = self.dictionary.T @ self.dictionary
            
            # Dictionary coherence (mutual coherence)
            off_diagonal = gram_matrix - np.eye(n_elements)
            coherence = np.max(np.abs(off_diagonal))
            
            # Condition number
            condition_number = np.linalg.cond(gram_matrix)
            
            # Element norms
            element_norms = np.linalg.norm(self.dictionary, axis=0)
            norm_uniformity = np.std(element_norms) / np.mean(element_norms)
            
            # Spectral properties
            eigenvals = np.linalg.eigvals(gram_matrix)
            spectral_radius = np.max(np.real(eigenvals))
            min_eigenval = np.min(np.real(eigenvals))
            
            print(f"\n🔍 Dictionary Quality Metrics:")
            print(f"  Mutual coherence: {coherence:.4f}")
            print(f"  Condition number: {condition_number:.2e}")
            print(f"  Norm uniformity (CV): {norm_uniformity:.4f}")
            print(f"  Spectral radius: {spectral_radius:.4f}")
            print(f"  Minimum eigenvalue: {min_eigenval:.4f}")
            
            # Quality assessment
            print(f"\n✅ Quality Assessment:")
            if coherence < 0.3:
                print(f"  ✓ Low coherence - excellent for sparse recovery")
            elif coherence < 0.6:
                print(f"  ⚠️ Moderate coherence - good for sparse recovery")
            else:
                print(f"  ❌ High coherence - may impact sparse recovery")
                
            if condition_number < 1e6:
                print(f"  ✓ Well-conditioned dictionary")
            elif condition_number < 1e12:
                print(f"  ⚠️ Moderately conditioned dictionary")
            else:
                print(f"  ❌ Ill-conditioned dictionary")
                
            if norm_uniformity < 0.1:
                print(f"  ✓ Uniform element norms")
            else:
                print(f"  ⚠️ Non-uniform element norms")
            
            # Biological plausibility analysis
            print(f"\n🧠 Biological Plausibility:")
            self._analyze_biological_features()
            
            return {
                'coherence': coherence,
                'condition_number': condition_number,
                'norm_uniformity': norm_uniformity,
                'spectral_radius': spectral_radius,
                'min_eigenvalue': min_eigenval,
                'overcompleteness': overcompleteness
            }
            
        except Exception as e:
            print(f"⚠️  Dictionary analysis failed: {e}")
            return {}

    def _analyze_biological_features(self) -> None:
        """
        Analyze biological plausibility of learned features.
        """
        
        try:
            # Analyze orientation selectivity
            oriented_elements = 0
            edge_like_elements = 0
            
            for i in range(min(self.n_components, 50)):  # Sample first 50 elements
                element = self.dictionary[:, i].reshape(self.patch_size)
                
                # Simple edge detection using gradients
                grad_x = np.gradient(element, axis=1)
                grad_y = np.gradient(element, axis=0)
                
                # Measure orientation consistency
                orientation_strength = np.sqrt(np.mean(grad_x**2) + np.mean(grad_y**2))
                
                if orientation_strength > 0.1:  # Threshold for oriented features
                    oriented_elements += 1
                    
                # Check for edge-like structure
                element_range = element.max() - element.min()
                if element_range > 0.3:  # Threshold for significant contrast
                    edge_like_elements += 1
            
            orientation_ratio = oriented_elements / min(self.n_components, 50)
            edge_ratio = edge_like_elements / min(self.n_components, 50)
            
            print(f"  Oriented features: {orientation_ratio:.1%}")
            print(f"  Edge-like features: {edge_ratio:.1%}")
            
            if orientation_ratio > 0.6:
                print(f"  ✓ High proportion of oriented features (similar to V1 cells)")
            else:
                print(f"  ⚠️ Low proportion of oriented features")
                
        except Exception as e:
            print(f"  ⚠️ Biological analysis failed: {e}")

    def _print_training_statistics(self) -> None:
        """Print detailed training statistics."""
        
        if not hasattr(self, 'training_history') or not self.training_history:
            return
        
        try:
            print(f"\n📈 Training Statistics")
            print("=" * 40)
            
            if 'reconstruction_error' in self.training_history:
                errors = self.training_history['reconstruction_error']
                print(f"Final reconstruction error: {errors[-1]:.6f}")
                print(f"Initial reconstruction error: {errors[0]:.6f}")
                print(f"Error reduction: {(1 - errors[-1]/errors[0])*100:.1f}%")
                print(f"Training iterations: {len(errors)}")
            
            if 'sparsity' in self.training_history:
                sparsity = self.training_history['sparsity']
                print(f"Final average sparsity: {sparsity[-1]:.1f} active elements")
                print(f"Sparsity ratio: {sparsity[-1]/self.n_components:.1%}")
            
            print(f"Sparsity penalty: {self.sparsity_penalty:.4f}")
            
        except Exception as e:
            print(f"⚠️  Statistics printing failed: {e}")

    def _print_reconstruction_statistics(self, reconstruction_errors: np.ndarray,
                                       coefficients: np.ndarray) -> None:
        """Print reconstruction quality statistics."""
        
        try:
            print(f"\n📊 Reconstruction Quality Statistics")
            print("=" * 45)
            
            print(f"Mean reconstruction error: {np.mean(reconstruction_errors):.6f}")
            print(f"Median reconstruction error: {np.median(reconstruction_errors):.6f}")
            print(f"Min reconstruction error: {np.min(reconstruction_errors):.6f}")
            print(f"Max reconstruction error: {np.max(reconstruction_errors):.6f}")
            print(f"Error std deviation: {np.std(reconstruction_errors):.6f}")
            
            # Sparsity statistics
            sparsity_levels = np.sum(np.abs(coefficients) > 1e-3, axis=1)
            print(f"\nSparsity Statistics:")
            print(f"Mean active elements: {np.mean(sparsity_levels):.1f}")
            print(f"Median active elements: {np.median(sparsity_levels):.1f}")
            print(f"Sparsity range: {np.min(sparsity_levels):.0f} - {np.max(sparsity_levels):.0f}")
            
            # Signal-to-noise ratio
            signal_power = np.mean(coefficients**2)
            if signal_power > 0:
                snr_db = 10 * np.log10(signal_power / np.mean(reconstruction_errors))
                print(f"Effective SNR: {snr_db:.1f} dB")
            
        except Exception as e:
            print(f"⚠️  Reconstruction statistics failed: {e}")

    def create_visualization_report(self, save_dir: str = "./visualization_report/") -> None:
        """
        Generate comprehensive visualization report for research documentation.
        
        Args:
            save_dir: Directory to save the report files
        """
        
        import os
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"📋 Generating comprehensive visualization report...")
            print(f"   Report directory: {save_dir}")
            
            # Generate all visualizations
            self.visualize_dictionary(save_path=os.path.join(save_dir, "dictionary.png"))
            self.plot_training_curves(save_path=os.path.join(save_dir, "training_curves.png"))
            self.plot_dictionary_properties(save_path=os.path.join(save_dir, "dictionary_properties.png"))
            
            # Generate analysis report
            report_path = os.path.join(save_dir, "analysis_report.txt")
            with open(report_path, 'w') as f:
                f.write("Sparse Coding Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                
                # Dictionary properties
                properties = self._analyze_dictionary_properties()
                f.write("Dictionary Properties:\n")
                for key, value in properties.items():
                    f.write(f"  {key}: {value}\n")
                
                f.write("\nTraining Configuration:\n")
                f.write(f"  n_components: {self.n_components}\n")
                f.write(f"  patch_size: {self.patch_size}\n")
                f.write(f"  sparsity_penalty: {self.sparsity_penalty}\n")
                
                if hasattr(self, 'training_history') and self.training_history:
                    f.write(f"\nTraining Results:\n")
                    if 'reconstruction_error' in self.training_history:
                        errors = self.training_history['reconstruction_error']
                        f.write(f"  Final error: {errors[-1]:.6f}\n")
                        f.write(f"  Iterations: {len(errors)}\n")
            
            print(f"✅ Visualization report generated in: {save_dir}")
            
        except Exception as e:
            print(f"⚠️  Report generation failed: {e}")

    def plot_feature_evolution(self, feature_indices: List[int],
                             figsize: Tuple[int, int] = (12, 8),
                             save_path: str = None) -> None:
        """
        Plot evolution of specific dictionary features during training.
        
        Note: This requires training history of dictionary states to be saved.
        
        Args:
            feature_indices: Indices of features to track
            figsize: Figure size in inches
            save_path: Path to save the figure (optional)
        """
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Warning: matplotlib not available for feature evolution plotting")
            return
        
        # This would require modification of the training loop to save dictionary states
        print("⚠️  Feature evolution tracking requires dictionary history during training")
        print("   Consider implementing dictionary state saving in the training loop")

    def export_dictionary_for_research(self, save_path: str = "./dictionary_export.npz") -> None:
        """
        Export dictionary and metadata for research use.
        
        Args:
            save_path: Path to save the exported data
        """
        
        try:
            if not hasattr(self, 'dictionary') or self.dictionary is None:
                print("⚠️  Warning: No dictionary available for export")
                return
            
            # Prepare export data
            export_data = {
                'dictionary': self.dictionary,
                'n_components': self.n_components,
                'patch_size': self.patch_size,
                'sparsity_penalty': self.sparsity_penalty,
            }
            
            # Add training history if available
            if hasattr(self, 'training_history'):
                export_data['training_history'] = self.training_history
            
            # Add dictionary analysis
            properties = self._analyze_dictionary_properties()
            export_data['dictionary_properties'] = properties
            
            # Save to numpy archive
            np.savez_compressed(save_path, **export_data)
            
            print(f"✅ Dictionary exported to: {save_path}")
            print(f"   Includes: dictionary, metadata, training history, analysis")
            
        except Exception as e:
            print(f"⚠️  Dictionary export failed: {e}")

"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
📝 CITE: Benedict Chen (2025) - Sparse Coding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""