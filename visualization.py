"""
Sparse Coding Visualization Tools
Based on: Olshausen & Field (1996) visualization techniques

Provides visualization tools for dictionary atoms, sparse codes, and
reconstruction quality analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any
from matplotlib.gridspec import GridSpec


class SparseVisualization:
    """
    Visualization tools for sparse coding results
    
    Implements visualizations shown in Olshausen & Field (1996):
    - Dictionary atoms as receptive fields
    - Sparse activation patterns
    - Reconstruction quality analysis
    """
    
    def __init__(self, colormap: str = 'gray', figsize: Tuple[int, int] = (15, 12)):
        """
        Initialize visualization tools
        
        Args:
            colormap: Default colormap for visualizations
            figsize: Default figure size for plots
        """
        self.colormap = colormap
        self.default_figsize = figsize
        
        # Set up matplotlib parameters for better visualization
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 10,
            'figure.titlesize': 14
        })
        
        # Visualization state tracking
        self._last_plot_info = {}
        
        print(f"✅ SparseVisualization initialized with colormap='{colormap}'")
        
    def visualize_dictionary(self, dictionary: np.ndarray, patch_size: Tuple[int, int],
                           figsize: Tuple[int, int] = (15, 12), max_atoms: int = 100,
                           title: str = "Learned Dictionary Atoms") -> None:
        """
        Visualize dictionary atoms as receptive fields
        
        Args:
            dictionary: Dictionary matrix (patch_dim, n_components)
            patch_size: Size of patches (height, width)
            figsize: Figure size
            max_atoms: Maximum number of atoms to display
            title: Plot title
        """
        
        n_atoms = min(dictionary.shape[1], max_atoms)
        grid_size = int(np.ceil(np.sqrt(n_atoms)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        if n_atoms == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(n_atoms):
            # Reshape atom to image
            atom = dictionary[:, i].reshape(patch_size)
            
            # Normalize for visualization
            atom_norm = (atom - atom.min()) / (atom.max() - atom.min() + 1e-8)
            
            ax = axes[i]
            im = ax.imshow(atom_norm, cmap='gray', aspect='equal')
            ax.set_title(f'Atom {i+1}', fontsize=8)
            ax.axis('off')
            
        # Hide unused subplots
        for i in range(n_atoms, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
        
        print(f"📊 Visualized {n_atoms} dictionary atoms")
        print(f"   Patch size: {patch_size}")
        print(f"   Dictionary shape: {dictionary.shape}")
        
    def visualize_sparse_codes(self, codes: np.ndarray, n_examples: int = 8,
                              figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Visualize sparse activation patterns
        
        Args:
            codes: Sparse codes array (n_samples, n_components)
            n_examples: Number of examples to show
            figsize: Figure size
        """
        
        n_show = min(n_examples, len(codes))
        
        fig, axes = plt.subplots(2, (n_show + 1) // 2, figsize=figsize)
        fig.suptitle('Sparse Code Activation Patterns', fontsize=14)
        
        if n_show == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for i in range(n_show):
            code = codes[i]
            active_indices = np.where(np.abs(code) > 1e-6)[0]
            
            ax = axes[i]
            ax.bar(range(len(code)), code, alpha=0.7)
            ax.set_title(f'Sample {i+1} ({len(active_indices)} active)')
            ax.set_xlabel('Component Index')
            ax.set_ylabel('Activation')
            ax.grid(True, alpha=0.3)
            
        # Hide unused subplots
        for i in range(n_show, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        sparsity_levels = [np.sum(np.abs(code) > 1e-6) for code in codes[:n_show]]
        avg_sparsity = np.mean(sparsity_levels)
        
        print(f"📊 Sparse Code Statistics:")
        print(f"   Average active components: {avg_sparsity:.1f}")
        print(f"   Sparsity ratio: {avg_sparsity/codes.shape[1]:.3f}")
        print(f"   Code dimension: {codes.shape[1]}")
        
    def visualize_reconstruction(self, original_patches: np.ndarray, 
                               reconstructed_patches: np.ndarray,
                               patch_size: Tuple[int, int], n_examples: int = 8,
                               figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Visualize reconstruction quality
        
        Args:
            original_patches: Original patches (n_samples, patch_dim)
            reconstructed_patches: Reconstructed patches (n_samples, patch_dim)
            patch_size: Size of patches (height, width)
            n_examples: Number of examples to show
            figsize: Figure size
        """
        
        n_show = min(n_examples, len(original_patches))
        
        fig, axes = plt.subplots(3, n_show, figsize=figsize)
        fig.suptitle('Reconstruction Quality Analysis', fontsize=14)
        
        errors = []
        
        for i in range(n_show):
            # Original
            orig = original_patches[i].reshape(patch_size)
            recon = reconstructed_patches[i].reshape(patch_size)
            error = orig - recon
            
            # Calculate error metrics
            mse = np.mean(error ** 2)
            errors.append(mse)
            
            # Plot original
            axes[0, i].imshow(orig, cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Plot reconstruction
            axes[1, i].imshow(recon, cmap='gray')
            axes[1, i].set_title(f'Reconstructed')
            axes[1, i].axis('off')
            
            # Plot error
            im = axes[2, i].imshow(error, cmap='RdBu_r')
            axes[2, i].set_title(f'Error (MSE: {mse:.4f})')
            axes[2, i].axis('off')
            
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        avg_error = np.mean(errors)
        print(f"📊 Reconstruction Quality:")
        print(f"   Average MSE: {avg_error:.6f}")
        print(f"   Error range: [{min(errors):.6f}, {max(errors):.6f}]")
        
    def visualize_training_progress(self, training_history: Dict[str, List],
                                   figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Visualize training progress
        
        Args:
            training_history: Dictionary with training metrics
            figsize: Figure size
        """
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Training Progress', fontsize=14)
        
        # Reconstruction error
        if 'reconstruction_errors' in training_history:
            ax = axes[0, 0]
            errors = training_history['reconstruction_errors']
            ax.plot(errors, 'b-', linewidth=2)
            ax.set_title('Reconstruction Error')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('MSE')
            ax.grid(True, alpha=0.3)
            ax.semilogy()  # Log scale for better visualization
            
        # Sparsity levels
        if 'sparsity_levels' in training_history:
            ax = axes[0, 1]
            sparsity = training_history['sparsity_levels']
            ax.plot(sparsity, 'g-', linewidth=2)
            ax.set_title('Sparsity Level')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Active Components')
            ax.grid(True, alpha=0.3)
            
        # Dictionary changes
        if 'dictionary_changes' in training_history:
            ax = axes[1, 0]
            changes = training_history['dictionary_changes']
            ax.plot(changes, 'r-', linewidth=2)
            ax.set_title('Dictionary Changes')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('L2 Norm of Change')
            ax.grid(True, alpha=0.3)
            ax.semilogy()
            
        # Learning rates (if available)
        if 'learning_rates' in training_history:
            ax = axes[1, 1]
            lr = training_history['learning_rates']
            ax.plot(lr, 'purple', linewidth=2)
            ax.set_title('Learning Rate')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Learning Rate')
            ax.grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    def analyze_receptive_fields(self, dictionary: np.ndarray, 
                                patch_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Analyze properties of learned receptive fields
        
        Args:
            dictionary: Dictionary matrix (patch_dim, n_components)
            patch_size: Size of patches (height, width)
            
        Returns:
            Analysis results dictionary
        """
        
        n_atoms = dictionary.shape[1]
        
        # Reshape atoms to images
        atoms = dictionary.T.reshape(n_atoms, *patch_size)
        
        # Analyze orientations (simplified)
        orientations = []
        frequencies = []
        
        for atom in atoms:
            # Simple edge detection to estimate orientation
            grad_x = np.gradient(atom, axis=1)
            grad_y = np.gradient(atom, axis=0)
            
            # Dominant gradient direction
            total_grad_x = np.sum(np.abs(grad_x))
            total_grad_y = np.sum(np.abs(grad_y))
            
            if total_grad_x + total_grad_y > 1e-6:
                orientation = np.arctan2(total_grad_y, total_grad_x)
                orientations.append(orientation)
                
                # Rough frequency estimate (number of zero crossings)
                center_row = atom[patch_size[0]//2, :]
                zero_crossings = np.sum(np.diff(np.sign(center_row)) != 0)
                frequencies.append(zero_crossings)
            else:
                orientations.append(0)
                frequencies.append(0)
                
        results = {
            'orientations': np.array(orientations),
            'frequencies': np.array(frequencies),
            'n_atoms': n_atoms,
            'atom_shapes': atoms.shape
        }
        
        # Create orientation histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Orientation distribution
        ax1.hist(orientations, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_title('Orientation Distribution')
        ax1.set_xlabel('Orientation (radians)')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # Frequency distribution
        ax2.hist(frequencies, bins=range(max(frequencies) + 2), alpha=0.7, edgecolor='black')
        ax2.set_title('Spatial Frequency Distribution')
        ax2.set_xlabel('Zero Crossings')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"📊 Receptive Field Analysis:")
        print(f"   Number of atoms: {n_atoms}")
        print(f"   Orientation range: [{np.min(orientations):.2f}, {np.max(orientations):.2f}] rad")
        print(f"   Average frequency: {np.mean(frequencies):.1f} zero crossings")
        
        return results
        
    def compare_dictionaries(self, dict1: np.ndarray, dict2: np.ndarray,
                           patch_size: Tuple[int, int], labels: List[str] = None,
                           figsize: Tuple[int, int] = (15, 8)) -> None:
        """
        Compare two dictionaries side by side
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary  
            patch_size: Size of patches
            labels: Labels for the dictionaries
            figsize: Figure size
        """
        
        if labels is None:
            labels = ['Dictionary 1', 'Dictionary 2']
            
        n_show = min(dict1.shape[1], dict2.shape[1], 50)  # Show up to 50 atoms
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, n_show, figure=fig)
        
        fig.suptitle(f'Dictionary Comparison: {labels[0]} vs {labels[1]}', fontsize=14)
        
        for i in range(n_show):
            # First dictionary
            atom1 = dict1[:, i].reshape(patch_size)
            ax1 = fig.add_subplot(gs[0, i])
            ax1.imshow(atom1, cmap='gray')
            ax1.axis('off')
            if i == 0:
                ax1.set_ylabel(labels[0], rotation=90, labelpad=20)
                
            # Second dictionary
            atom2 = dict2[:, i].reshape(patch_size)
            ax2 = fig.add_subplot(gs[1, i])
            ax2.imshow(atom2, cmap='gray')
            ax2.axis('off')
            if i == 0:
                ax2.set_ylabel(labels[1], rotation=90, labelpad=20)
                
        plt.tight_layout()
        plt.show()
        
        # Calculate similarity metrics
        similarities = []
        for i in range(min(dict1.shape[1], dict2.shape[1])):
            atom1 = dict1[:, i]
            atom2 = dict2[:, i]
            # Normalize atoms
            atom1_norm = atom1 / (np.linalg.norm(atom1) + 1e-8)
            atom2_norm = atom2 / (np.linalg.norm(atom2) + 1e-8)
            # Compute cosine similarity
            similarity = np.dot(atom1_norm, atom2_norm)
            similarities.append(abs(similarity))  # Take absolute value
            
        avg_similarity = np.mean(similarities)
        
        print(f"📊 Dictionary Comparison:")
        print(f"   Average atom similarity: {avg_similarity:.4f}")
        print(f"   Similarity range: [{min(similarities):.4f}, {max(similarities):.4f}]")
        print(f"   Compared {len(similarities)} atom pairs")