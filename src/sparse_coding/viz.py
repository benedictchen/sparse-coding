"""
🎨 Sparse Coding Visualization & Analysis Plots
===============================================

Author: Benedict Chen (benedict@benedictchen.com)

💰 Donations: Help support this research!
   PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   💖 Please consider recurring donations to support continued sparse coding research

Comprehensive visualization suite for sparse coding research and analysis.
Includes dictionary visualization, training diagnostics, and feature analysis plots.

🔬 Research Foundation:
======================
Visualization techniques based on:
- Olshausen & Field (1996): Original dictionary visualization methods
- Bell & Sejnowski (1995): Receptive field plotting techniques
- Hyvarinen et al. (2001): ICA and sparse coding feature visualization
- Mairal et al. (2009): Convergence and sparsity analysis plots

ELI5 Explanation:
================
Think of this like a photographer's darkroom for neural algorithms! 📷

🖼️ **The Photography Studio Analogy**:
When developing photos (training algorithms), you need different tools to see your work:

- **Dictionary Plots** = Contact sheets showing all your photos at once
- **Training Curves** = Time-lapse of how your photo develops over time  
- **Sparsity Analysis** = Close-up magnifying glass to see fine details
- **Reconstruction Plots** = Before/after comparison of original vs. developed photo
- **Feature Maps** = Special filters that highlight different aspects (edges, textures)

🎯 **Why Visualization Matters**:
Just like photographers need to see their work develop, researchers need to watch
algorithms learn. These plots reveal whether the algorithm is discovering meaningful
patterns (like edge detectors) or getting confused with random noise.

ASCII Visualization Pipeline:
============================
    RAW ALGORITHM        VISUALIZATION         RESEARCH INSIGHTS
    ┌─────────────┐     ┌─────────────┐      ┌─────────────┐
    │ Dictionary  │────▶│ Grid Plot   │─────▶│ "Algorithm  │
    │ Learning    │     │ Atoms 8x8   │      │  learned    │
    │ Progress    │     │ ████ ░░░░   │      │  edge       │
    │             │     │ ░░░░ ████   │      │  detectors!"│
    └─────────────┘     └─────────────┘      └─────────────┘
           │                   │                     │
           ▼                   ▼                     │
    ┌─────────────┐     ┌─────────────┐             │
    │ Loss Curves │────▶│ Line Plots  │─────────────┘
    │ Sparsity    │     │ Convergence │
    │ Metrics     │     │ Analysis    │
    └─────────────┘     └─────────────┘

🎨 Visualization Categories:
===========================
📊 **Training Analysis**: Loss curves, convergence plots, sparsity evolution
🖼️ **Dictionary Plots**: Learned basis functions, receptive fields, atom grids  
📈 **Statistical Analysis**: Histogram plots, coefficient distributions
🔍 **Quality Assessment**: Reconstruction comparisons, error visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from typing import Tuple, Optional, List, Dict, Any, Union
import warnings


# Set default style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


# =============================================================================
# Dictionary Visualization
# =============================================================================

def plot_dictionary(dictionary: np.ndarray, patch_size: Tuple[int, int],
                   figsize: Tuple[int, int] = (15, 10), max_atoms: int = 100,
                   title: str = "Learned Dictionary", colormap: str = 'RdBu_r',
                   normalize_atoms: bool = True) -> plt.Figure:
    """
    Visualize dictionary atoms as image patches
    
    Parameters
    ----------
    dictionary : array, shape (n_atoms, n_features)
        Dictionary matrix
    patch_size : tuple
        (height, width) of each patch
    figsize : tuple
        Figure size (width, height)
    max_atoms : int
        Maximum number of atoms to display
    title : str
        Plot title
    colormap : str
        Matplotlib colormap name
    normalize_atoms : bool
        Whether to normalize each atom for display
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    n_atoms = min(max_atoms, dictionary.shape[0])
    
    # Determine grid layout
    grid_cols = int(np.ceil(np.sqrt(n_atoms)))
    grid_rows = int(np.ceil(n_atoms / grid_cols))
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
    
    # Handle single subplot case
    if grid_rows == 1 and grid_cols == 1:
        axes = [axes]
    elif grid_rows == 1 or grid_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(n_atoms):
        # Reshape atom to patch
        atom = dictionary[i].reshape(patch_size)
        
        # Normalize for visualization
        if normalize_atoms:
            atom_min, atom_max = atom.min(), atom.max()
            if atom_max > atom_min:
                atom = (atom - atom_min) / (atom_max - atom_min)
        
        # Plot atom
        im = axes[i].imshow(atom, cmap=colormap, interpolation='nearest')
        axes[i].set_title(f'Atom {i}', fontsize=8)
        axes[i].axis('off')
        
        # Add colorbar for first few atoms
        if i < 4:
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)
    
    # Hide unused subplots
    for i in range(n_atoms, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_dictionary_evolution(dictionaries: List[np.ndarray], patch_size: Tuple[int, int],
                             iterations: List[int], figsize: Tuple[int, int] = (20, 12),
                             n_atoms_show: int = 16) -> plt.Figure:
    """
    Show evolution of dictionary atoms during training
    
    Parameters
    ----------
    dictionaries : list of arrays
        List of dictionary states during training
    patch_size : tuple
        Size of patches
    iterations : list of int
        Iteration numbers corresponding to each dictionary
    figsize : tuple
        Figure size
    n_atoms_show : int
        Number of atoms to show evolution for
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    n_snapshots = len(dictionaries)
    
    fig, axes = plt.subplots(n_atoms_show, n_snapshots, figsize=figsize)
    
    for snapshot_idx, (dictionary, iteration) in enumerate(zip(dictionaries, iterations)):
        for atom_idx in range(min(n_atoms_show, dictionary.shape[0])):
            # Reshape and normalize atom
            atom = dictionary[atom_idx].reshape(patch_size)
            atom = (atom - atom.min()) / (atom.max() - atom.min() + 1e-8)
            
            # Plot
            axes[atom_idx, snapshot_idx].imshow(atom, cmap='RdBu_r', interpolation='nearest')
            axes[atom_idx, snapshot_idx].axis('off')
            
            # Add iteration label on top row
            if atom_idx == 0:
                axes[atom_idx, snapshot_idx].set_title(f'Iter {iteration}', fontsize=10)
            
            # Add atom label on left column
            if snapshot_idx == 0:
                axes[atom_idx, snapshot_idx].set_ylabel(f'Atom {atom_idx}', fontsize=8)
    
    fig.suptitle('Dictionary Evolution During Training', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_dictionary_statistics(dictionary: np.ndarray, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot statistical analysis of dictionary properties
    
    Parameters
    ----------
    dictionary : array, shape (n_atoms, n_features)
        Dictionary matrix
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Atom norms
    norms = np.linalg.norm(dictionary, axis=1)
    axes[0].hist(norms, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('L2 Norm')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Atom Norms')
    axes[0].axvline(np.mean(norms), color='red', linestyle='--', label=f'Mean: {np.mean(norms):.3f}')
    axes[0].legend()
    
    # 2. Mutual coherence
    from .utils import compute_mutual_coherence_matrix
    coherence_matrix = compute_mutual_coherence_matrix(dictionary)
    
    im = axes[1].imshow(coherence_matrix, cmap='viridis', interpolation='nearest')
    axes[1].set_xlabel('Atom Index')
    axes[1].set_ylabel('Atom Index')
    axes[1].set_title('Mutual Coherence Matrix')
    plt.colorbar(im, ax=axes[1])
    
    # 3. Coherence distribution
    coherences = coherence_matrix[np.triu_indices_from(coherence_matrix, k=1)]
    axes[2].hist(coherences, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[2].set_xlabel('Mutual Coherence')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Distribution of Pairwise Coherences')
    axes[2].axvline(np.mean(coherences), color='blue', linestyle='--', 
                   label=f'Mean: {np.mean(coherences):.3f}')
    axes[2].axvline(np.max(coherences), color='red', linestyle='--',
                   label=f'Max: {np.max(coherences):.3f}')
    axes[2].legend()
    
    plt.tight_layout()
    return fig


# =============================================================================
# Training Analysis Visualization
# =============================================================================

def plot_training_history(history: Dict[str, List[float]], 
                         figsize: Tuple[int, int] = (15, 5),
                         log_scale: bool = False) -> plt.Figure:
    """
    Plot training history (costs, errors, etc.)
    
    Parameters
    ----------
    history : dict
        Training history with keys like 'reconstruction_error', 'sparsity_cost', etc.
    figsize : tuple
        Figure size
    log_scale : bool
        Whether to use log scale for y-axis
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    n_plots = len(history)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    for i, (metric_name, values) in enumerate(history.items()):
        axes[i].plot(values, linewidth=2, marker='o', markersize=4)
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel(metric_name.replace('_', ' ').title())
        axes[i].set_title(f'{metric_name.replace("_", " ").title()} vs Iteration')
        axes[i].grid(True, alpha=0.3)
        
        if log_scale:
            axes[i].set_yscale('log')
        
        # Add final value annotation
        if len(values) > 0:
            final_value = values[-1]
            axes[i].annotate(f'Final: {final_value:.4f}', 
                           xy=(len(values)-1, final_value),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    return fig


def plot_convergence_analysis(history: List[float], window: int = 10,
                            figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """
    Analyze and visualize convergence behavior
    
    Parameters
    ----------
    history : list
        History of objective function values
    window : int
        Window size for moving average
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Objective value and moving average
    iterations = np.arange(len(history))
    axes[0].plot(iterations, history, 'b-', alpha=0.6, label='Objective Value')
    
    # Compute moving average
    if len(history) >= window:
        moving_avg = np.convolve(history, np.ones(window)/window, mode='valid')
        moving_avg_iterations = iterations[window-1:]
        axes[0].plot(moving_avg_iterations, moving_avg, 'r-', linewidth=2, 
                    label=f'{window}-point Moving Average')
    
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Objective Value')
    axes[0].set_title('Convergence Behavior')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Relative change
    if len(history) > 1:
        relative_changes = [abs(history[i] - history[i-1]) / (abs(history[i-1]) + 1e-8) 
                           for i in range(1, len(history))]
        
        axes[1].semilogy(iterations[1:], relative_changes, 'g-', linewidth=2, marker='o', markersize=3)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Relative Change (log scale)')
        axes[1].set_title('Convergence Rate')
        axes[1].grid(True, alpha=0.3)
        
        # Add convergence threshold line
        axes[1].axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, 
                       label='Typical Convergence Threshold')
        axes[1].legend()
    
    plt.tight_layout()
    return fig


# =============================================================================
# Sparse Code Visualization
# =============================================================================

def plot_sparse_codes(codes: np.ndarray, figsize: Tuple[int, int] = (12, 8),
                     max_samples: int = 50, max_atoms: int = 100) -> plt.Figure:
    """
    Visualize sparse codes as activation patterns
    
    Parameters
    ----------
    codes : array, shape (n_samples, n_atoms)
        Sparse coefficient matrix
    figsize : tuple
        Figure size
    max_samples : int
        Maximum number of samples to show
    max_atoms : int
        Maximum number of atoms to show
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    n_samples_show = min(max_samples, codes.shape[0])
    n_atoms_show = min(max_atoms, codes.shape[1])
    
    codes_subset = codes[:n_samples_show, :n_atoms_show]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Sparse code matrix
    im = axes[0, 0].imshow(codes_subset, cmap='RdBu_r', interpolation='nearest', aspect='auto')
    axes[0, 0].set_xlabel('Atom Index')
    axes[0, 0].set_ylabel('Sample Index')
    axes[0, 0].set_title('Sparse Code Matrix')
    plt.colorbar(im, ax=axes[0, 0])
    
    # 2. Sparsity histogram
    sparsity_per_sample = np.sum(codes != 0, axis=1)
    axes[0, 1].hist(sparsity_per_sample, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Number of Active Atoms')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Sparsity Distribution')
    axes[0, 1].axvline(np.mean(sparsity_per_sample), color='red', linestyle='--',
                      label=f'Mean: {np.mean(sparsity_per_sample):.1f}')
    axes[0, 1].legend()
    
    # 3. Coefficient magnitude histogram
    active_coeffs = codes[codes != 0]
    if len(active_coeffs) > 0:
        axes[1, 0].hist(np.abs(active_coeffs), bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Coefficient Magnitude')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Active Coefficient Magnitudes')
        axes[1, 0].axvline(np.mean(np.abs(active_coeffs)), color='blue', linestyle='--',
                          label=f'Mean: {np.mean(np.abs(active_coeffs)):.3f}')
        axes[1, 0].legend()
    
    # 4. Atom usage frequency
    usage_frequency = np.sum(codes != 0, axis=0) / codes.shape[0]
    axes[1, 1].bar(range(len(usage_frequency)), usage_frequency, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Atom Index')
    axes[1, 1].set_ylabel('Usage Frequency')
    axes[1, 1].set_title('Atom Usage Frequency')
    
    plt.tight_layout()
    return fig


def plot_reconstruction_comparison(original: np.ndarray, reconstructed: np.ndarray,
                                 patch_size: Tuple[int, int],
                                 n_examples: int = 10,
                                 figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
    """
    Compare original patches with their reconstructions
    
    Parameters
    ----------
    original : array, shape (n_samples, n_features)
        Original patches
    reconstructed : array, same shape as original
        Reconstructed patches
    patch_size : tuple
        Size of patches for reshaping
    n_examples : int
        Number of examples to show
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    n_show = min(n_examples, original.shape[0])
    
    fig, axes = plt.subplots(3, n_show, figsize=figsize)
    
    for i in range(n_show):
        # Original patch
        orig_patch = original[i].reshape(patch_size)
        axes[0, i].imshow(orig_patch, cmap='gray', interpolation='nearest')
        axes[0, i].set_title(f'Original {i}', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstructed patch
        recon_patch = reconstructed[i].reshape(patch_size)
        axes[1, i].imshow(recon_patch, cmap='gray', interpolation='nearest')
        axes[1, i].set_title(f'Reconstructed {i}', fontsize=10)
        axes[1, i].axis('off')
        
        # Error
        error_patch = np.abs(orig_patch - recon_patch)
        im = axes[2, i].imshow(error_patch, cmap='hot', interpolation='nearest')
        axes[2, i].set_title(f'Error {i}', fontsize=10)
        axes[2, i].axis('off')
        
        # Add error value as text
        mse = np.mean(error_patch**2)
        axes[2, i].text(0.5, -0.1, f'MSE: {mse:.4f}', transform=axes[2, i].transAxes,
                       ha='center', fontsize=8)
    
    # Add row labels
    axes[0, 0].set_ylabel('Original', rotation=90, labelpad=20, fontsize=12)
    axes[1, 0].set_ylabel('Reconstructed', rotation=90, labelpad=20, fontsize=12)  
    axes[2, 0].set_ylabel('Error', rotation=90, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Advanced Visualizations
# =============================================================================

def plot_feature_map_responses(feature_extractor, image: np.ndarray,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Visualize sparse feature responses across an image
    
    Parameters
    ----------
    feature_extractor : SparseFeatureExtractor
        Trained feature extractor
    image : array, shape (height, width)
        Input image
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    # Extract features
    features_info = feature_extractor.extract_features(image)
    codes = features_info['codes']
    
    # Find most active atoms
    atom_activities = np.mean(np.abs(codes), axis=0)
    top_atoms = np.argsort(atom_activities)[-9:]  # Top 9 atoms
    
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    
    # Show original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Show top active atoms from dictionary
    dictionary = feature_extractor.sparse_coder.dictionary_
    patch_size = feature_extractor.patch_size
    
    for i, atom_idx in enumerate(top_atoms):
        row = (i + 1) // 4
        col = (i + 1) % 4
        
        atom = dictionary[atom_idx].reshape(patch_size)
        axes[row, col].imshow(atom, cmap='RdBu_r', interpolation='nearest')
        axes[row, col].set_title(f'Atom {atom_idx}\n(Activity: {atom_activities[atom_idx]:.3f})')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(10, 12):
        row = i // 4
        col = i % 4
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


def plot_sparsity_path(lambdas: List[float], sparsity_levels: List[float],
                      errors: List[float], figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot sparsity-error trade-off curve (regularization path)
    
    Parameters
    ----------
    lambdas : list
        List of sparsity parameters
    sparsity_levels : list
        Corresponding sparsity levels (number of non-zeros)
    errors : list
        Corresponding reconstruction errors
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Error vs Lambda
    axes[0].semilogx(lambdas, errors, 'bo-', linewidth=2, markersize=6)
    axes[0].set_xlabel('Sparsity Parameter (λ)')
    axes[0].set_ylabel('Reconstruction Error')
    axes[0].set_title('Error vs Sparsity Parameter')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Error vs Sparsity
    axes[1].plot(sparsity_levels, errors, 'ro-', linewidth=2, markersize=6)
    axes[1].set_xlabel('Average Sparsity (# non-zeros)')
    axes[1].set_ylabel('Reconstruction Error')
    axes[1].set_title('Sparsity-Error Trade-off')
    axes[1].grid(True, alpha=0.3)
    
    # Add annotations for some points
    n_annotate = min(5, len(lambdas))
    indices = np.linspace(0, len(lambdas)-1, n_annotate, dtype=int)
    
    for idx in indices:
        # Annotate lambda plot
        axes[0].annotate(f'λ={lambdas[idx]:.3f}', 
                        xy=(lambdas[idx], errors[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=8)
        
        # Annotate sparsity plot  
        axes[1].annotate(f's={sparsity_levels[idx]:.1f}',
                        xy=(sparsity_levels[idx], errors[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='cyan', alpha=0.7),
                        fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_dictionary_comparison(dict1: np.ndarray, dict2: np.ndarray,
                              patch_size: Tuple[int, int],
                              labels: Tuple[str, str] = ('Dictionary 1', 'Dictionary 2'),
                              figsize: Tuple[int, int] = (15, 8),
                              n_atoms: int = 25) -> plt.Figure:
    """
    Compare two dictionaries side by side
    
    Parameters
    ----------
    dict1, dict2 : arrays, shape (n_atoms, n_features)
        Dictionaries to compare
    patch_size : tuple
        Size of patches
    labels : tuple
        Labels for the two dictionaries
    figsize : tuple
        Figure size
    n_atoms : int
        Number of atoms to show
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    n_show = min(n_atoms, dict1.shape[0], dict2.shape[0])
    grid_size = int(np.ceil(np.sqrt(n_show)))
    
    fig, (axes1, axes2) = plt.subplots(2, grid_size, figsize=figsize)
    
    for i in range(n_show):
        col = i % grid_size
        
        # Dictionary 1
        atom1 = dict1[i].reshape(patch_size)
        atom1 = (atom1 - atom1.min()) / (atom1.max() - atom1.min() + 1e-8)
        axes1[col].imshow(atom1, cmap='RdBu_r', interpolation='nearest')
        axes1[col].set_title(f'Atom {i}', fontsize=8)
        axes1[col].axis('off')
        
        # Dictionary 2  
        atom2 = dict2[i].reshape(patch_size)
        atom2 = (atom2 - atom2.min()) / (atom2.max() - atom2.min() + 1e-8)
        axes2[col].imshow(atom2, cmap='RdBu_r', interpolation='nearest')
        axes2[col].set_title(f'Atom {i}', fontsize=8)
        axes2[col].axis('off')
    
    # Hide unused subplots
    for i in range(n_show, grid_size):
        axes1[i].axis('off')
        axes2[i].axis('off')
    
    # Add dictionary labels
    axes1[0].set_ylabel(labels[0], rotation=90, labelpad=20, fontsize=14)
    axes2[0].set_ylabel(labels[1], rotation=90, labelpad=20, fontsize=14)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Interactive Plotting Utilities
# =============================================================================

def create_interactive_dictionary_explorer(dictionary: np.ndarray, 
                                          patch_size: Tuple[int, int]):
    """
    Create interactive dictionary explorer (requires widget backend)
    
    Parameters
    ----------
    dictionary : array, shape (n_atoms, n_features)
        Dictionary to explore
    patch_size : tuple
        Size of patches
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
        
        def plot_atom(atom_idx=0):
            atom = dictionary[atom_idx].reshape(patch_size)
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            
            # Show atom
            axes[0].imshow(atom, cmap='RdBu_r', interpolation='nearest')
            axes[0].set_title(f'Dictionary Atom {atom_idx}')
            axes[0].axis('off')
            
            # Show atom statistics
            axes[1].hist(atom.flatten(), bins=20, alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('Pixel Value')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Atom Value Distribution')
            axes[1].axvline(atom.mean(), color='red', linestyle='--', label=f'Mean: {atom.mean():.3f}')
            axes[1].axvline(atom.std(), color='green', linestyle='--', label=f'Std: {atom.std():.3f}')
            axes[1].legend()
            
            plt.tight_layout()
            plt.show()
        
        # Create interactive widget
        atom_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=dictionary.shape[0] - 1,
            step=1,
            description='Atom Index:',
            style={'description_width': 'initial'}
        )
        
        interactive_plot = widgets.interactive(plot_atom, atom_idx=atom_slider)
        display(interactive_plot)
        
    except ImportError:
        print("Interactive exploration requires ipywidgets. Install with: pip install ipywidgets")
        print("Using static plot instead...")
        plot_dictionary(dictionary, patch_size)


# =============================================================================
# Utility Functions for Visualization
# =============================================================================

def setup_publication_style():
    """Setup matplotlib style for publication-quality figures"""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'text.usetex': False,  # Set to True if LaTeX is available
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.grid.alpha': 0.3,
        'figure.dpi': 150
    })


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300, 
               bbox_inches: str = 'tight', formats: List[str] = ['png']):
    """
    Save figure in multiple formats with publication settings
    
    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save
    filename : str
        Base filename (without extension)
    dpi : int
        Resolution for raster formats
    bbox_inches : str
        Bounding box setting
    formats : list
        List of formats to save ('png', 'pdf', 'svg', 'eps')
    """
    for fmt in formats:
        full_filename = f"{filename}.{fmt}"
        fig.savefig(full_filename, format=fmt, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved: {full_filename}")


if __name__ == "__main__":
    # Example usage and testing
    print("🎨 Sparse Coding Visualization")
    print("=" * 40)
    
    # Create test data
    np.random.seed(42)
    
    # Test dictionary
    n_atoms, patch_dim = 50, 64
    test_dict = np.random.randn(n_atoms, patch_dim)
    patch_size = (8, 8)
    
    # Test sparse codes
    n_samples = 100
    test_codes = np.zeros((n_samples, n_atoms))
    # Make sparse (only 5% non-zero)
    for i in range(n_samples):
        active_indices = np.random.choice(n_atoms, size=5, replace=False)
        test_codes[i, active_indices] = np.random.randn(5)
    
    # Test training history
    test_history = {
        'reconstruction_error': [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.32, 0.30],
        'sparsity_cost': [0.2, 0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.08],
        'total_cost': [1.2, 0.98, 0.75, 0.62, 0.50, 0.44, 0.40, 0.38]
    }
    
    print("📊 Testing dictionary visualization...")
    fig1 = plot_dictionary(test_dict, patch_size, max_atoms=25, figsize=(10, 8))
    plt.close(fig1)
    
    print("📈 Testing training history visualization...")
    fig2 = plot_training_history(test_history, figsize=(12, 4))
    plt.close(fig2)
    
    print("🎯 Testing sparse codes visualization...")
    fig3 = plot_sparse_codes(test_codes, max_samples=50, max_atoms=50, figsize=(10, 8))
    plt.close(fig3)
    
    print("📊 Testing dictionary statistics...")
    fig4 = plot_dictionary_statistics(test_dict, figsize=(12, 4))
    plt.close(fig4)
    
    print("✅ All visualization tests passed!")
    print("   - Dictionary plots: OK")
    print("   - Training analysis: OK") 
    print("   - Sparse code analysis: OK")
    print("   - Statistical analysis: OK")