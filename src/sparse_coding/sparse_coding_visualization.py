"""
Visualization utilities for Sparse Coding

Provides comprehensive visualization tools for:
- Dictionary atoms
- Training progress
- Sparse codes
- Reconstruction quality
- Convergence analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.gridspec as gridspec


def plot_dictionary_atoms(dictionary: np.ndarray, 
                         patch_size: Tuple[int, int], 
                         n_show: int = 64,
                         figsize: Tuple[int, int] = (12, 12),
                         title: str = "Dictionary Atoms") -> plt.Figure:
    """
    Plot dictionary atoms as image patches
    
    Args:
        dictionary: Dictionary matrix (patch_dim x n_components)
        patch_size: Size of each patch (height, width)
        n_show: Number of atoms to show
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    
    n_atoms = min(n_show, dictionary.shape[1])
    grid_size = int(np.ceil(np.sqrt(n_atoms)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    atoms = dictionary.T.reshape(-1, *patch_size)
    n_actual_atoms = atoms.shape[0]  # Actual number of atoms available
    
    for i in range(grid_size * grid_size):
        ax = axes[i]
        
        if i < min(n_atoms, n_actual_atoms):
            atom = atoms[i]
            # Normalize for visualization
            atom_norm = (atom - atom.min()) / (atom.max() - atom.min() + 1e-8)
            
            ax.imshow(atom_norm, cmap='RdBu_r', interpolation='nearest')
            ax.set_title(f'Atom {i}', fontsize=8)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig


def plot_training_progress(history: Dict[str, List[float]], 
                          figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot training progress metrics
    
    Args:
        history: Training history dictionary
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Reconstruction error
    if 'reconstruction_errors' in history:
        axes[0].plot(history['reconstruction_errors'])
        axes[0].set_title('Reconstruction Error')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('MSE')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
    
    # Sparsity level
    if 'sparsity_levels' in history:
        axes[1].plot(history['sparsity_levels'])
        axes[1].set_title('Sparsity Level')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Fraction Active')
        axes[1].grid(True, alpha=0.3)
    
    # Dictionary changes
    if 'dictionary_changes' in history:
        axes[2].plot(history['dictionary_changes'])
        axes[2].set_title('Dictionary Changes')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('L2 Norm of Change')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_sparse_codes(codes: np.ndarray, 
                     n_show: int = 8,
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot sparse code vectors
    
    Args:
        codes: Sparse codes matrix (n_components x n_patches)
        n_show: Number of code vectors to show
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    
    n_patches = min(n_show, codes.shape[1])
    
    fig, axes = plt.subplots(2, n_patches//2, figsize=figsize)
    fig.suptitle('Sparse Code Vectors', fontsize=16)
    
    axes = axes.flatten()
    
    for i in range(n_patches):
        code = codes[:, i]
        
        # Plot as bar chart
        axes[i].bar(range(len(code)), code, alpha=0.7)
        axes[i].set_title(f'Code {i} (sparsity: {np.mean(np.abs(code) > 1e-6):.2f})')
        axes[i].set_xlabel('Dictionary Index')
        axes[i].set_ylabel('Coefficient')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_reconstruction_quality(original: np.ndarray, 
                              reconstructed: np.ndarray,
                              n_show: int = 8,
                              figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
    """
    Compare original patches with reconstructions
    
    Args:
        original: Original patches (patch_dim x n_patches)
        reconstructed: Reconstructed patches (patch_dim x n_patches)
        n_show: Number of patches to show
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    
    n_patches = min(n_show, original.shape[1])
    patch_dim = original.shape[0]
    patch_size = (int(np.sqrt(patch_dim)), int(np.sqrt(patch_dim)))
    
    fig, axes = plt.subplots(3, n_patches, figsize=figsize)
    fig.suptitle('Reconstruction Quality', fontsize=16)
    
    orig_patches = original.T.reshape(-1, *patch_size)
    recon_patches = reconstructed.T.reshape(-1, *patch_size)
    
    for i in range(n_patches):
        # Original
        axes[0, i].imshow(orig_patches[i], cmap='gray', interpolation='nearest')
        axes[0, i].set_title(f'Original {i}', fontsize=8)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Reconstruction
        axes[1, i].imshow(recon_patches[i], cmap='gray', interpolation='nearest')
        axes[1, i].set_title(f'Reconstruction {i}', fontsize=8)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        
        # Error
        error = np.abs(orig_patches[i] - recon_patches[i])
        mse = np.mean(error**2)
        axes[2, i].imshow(error, cmap='Reds', interpolation='nearest')
        axes[2, i].set_title(f'Error {i} (MSE: {mse:.3f})', fontsize=8)
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])
    
    plt.tight_layout()
    return fig


def plot_sparsity_statistics(codes: np.ndarray,
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot statistics of sparse codes
    
    Args:
        codes: Sparse codes matrix (n_components x n_patches)
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Sparse Code Statistics', fontsize=16)
    
    # Histogram of coefficient magnitudes
    nonzero_codes = codes[np.abs(codes) > 1e-6]
    axes[0, 0].hist(nonzero_codes, bins=50, alpha=0.7, density=True)
    axes[0, 0].set_title('Non-zero Coefficient Magnitudes')
    axes[0, 0].set_xlabel('Magnitude')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sparsity per patch
    sparsity_per_patch = np.mean(np.abs(codes) > 1e-6, axis=0)
    axes[0, 1].hist(sparsity_per_patch, bins=30, alpha=0.7)
    axes[0, 1].set_title('Sparsity Distribution (per patch)')
    axes[0, 1].set_xlabel('Fraction of Active Coefficients')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Usage frequency per dictionary atom
    usage_frequency = np.mean(np.abs(codes) > 1e-6, axis=1)
    axes[1, 0].bar(range(len(usage_frequency)), usage_frequency, alpha=0.7)
    axes[1, 0].set_title('Dictionary Atom Usage Frequency')
    axes[1, 0].set_xlabel('Dictionary Index')
    axes[1, 0].set_ylabel('Usage Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Coefficient magnitude vs usage
    mean_magnitude = np.mean(np.abs(codes), axis=1)
    axes[1, 1].scatter(usage_frequency, mean_magnitude, alpha=0.6)
    axes[1, 1].set_title('Usage vs Mean Magnitude')
    axes[1, 1].set_xlabel('Usage Frequency')
    axes[1, 1].set_ylabel('Mean Coefficient Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_visualization_report(dictionary: np.ndarray,
                              sparse_codes: Optional[np.ndarray] = None,
                              signals: Optional[np.ndarray] = None,
                              training_history: Optional[Dict[str, List[float]]] = None,
                              patch_size: Optional[Tuple[int, int]] = None,
                              codes: Optional[np.ndarray] = None,
                              history: Optional[Dict[str, List[float]]] = None,
                              original_patches: Optional[np.ndarray] = None,
                              save_path: Optional[str] = None) -> List[plt.Figure]:
    """
    Create visualization report
    
    Args:
        dictionary: Learned dictionary
        sparse_codes: Sparse codes (new parameter name)
        signals: Original signals/patches (alias for original_patches)
        training_history: Training history (alias for history)
        patch_size: Patch dimensions
        codes: Sparse codes (legacy parameter name)
        history: Training history (legacy parameter name)
        original_patches: Original patches for reconstruction comparison
        save_path: Optional path to save figures
        
    Returns:
        List of matplotlib Figure objects if save_path is None,
        otherwise returns the save_path string
    """
    
    # Handle parameter aliases for backwards compatibility
    if sparse_codes is not None:
        codes = sparse_codes
    if training_history is not None:
        history = training_history
    if signals is not None:
        original_patches = signals
    
    # Validate required parameters
    if codes is None:
        raise ValueError("Either 'codes' or 'sparse_codes' parameter is required")
    if patch_size is None:
        raise ValueError("patch_size parameter is required")
    
    figures = []
    
    # Dictionary atoms
    fig1 = plot_dictionary_atoms(dictionary, patch_size)
    figures.append(fig1)
    if save_path:
        fig1.savefig(f"{save_path}_dictionary_atoms.png", dpi=150, bbox_inches='tight')
    
    # Training progress
    if history:
        fig2 = plot_training_progress(history)
        figures.append(fig2)
        if save_path:
            fig2.savefig(f"{save_path}_training_progress.png", dpi=150, bbox_inches='tight')
    
    # Sparse codes
    fig3 = plot_sparse_codes(codes)
    figures.append(fig3)
    if save_path:
        fig3.savefig(f"{save_path}_sparse_codes.png", dpi=150, bbox_inches='tight')
    
    # Sparsity statistics
    fig4 = plot_sparsity_statistics(codes)
    figures.append(fig4)
    if save_path:
        fig4.savefig(f"{save_path}_sparsity_stats.png", dpi=150, bbox_inches='tight')
    
    # Reconstruction quality (if original patches provided)
    if original_patches is not None:
        reconstructed = dictionary @ codes
        fig5 = plot_reconstruction_quality(original_patches, reconstructed)
        figures.append(fig5)
        if save_path:
            fig5.savefig(f"{save_path}_reconstruction.png", dpi=150, bbox_inches='tight')
    
    # Return save_path if files were saved, otherwise return figures
    if save_path:
        return save_path
    else:
        return figures