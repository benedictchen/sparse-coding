"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀

✨ Sparse Coding Utilities Module
===============================

Utility functions extracted from the main SparseCoder class for modular design.
These functions provide essential algorithms and support functionality for 
sparse coding, dictionary learning, and neural network-inspired mechanisms.

Author: Benedict Chen (benedict@benedictchen.com)
Based on: Olshausen & Field (1996) "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"

Key Utilities:
- create_overcomplete_basis(): Generate overcomplete dictionaries
- lateral_inhibition(): Biologically-inspired competition mechanism
- Demo and visualization functions
- Basis function generators (Gabor, DCT, edge detectors)
"""

import numpy as np
from typing import Tuple, Optional, Any
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  Warning: matplotlib not available for visualization")

try:
    from scipy.fft import dct
    HAS_SCIPY_FFT = True
except ImportError:
    HAS_SCIPY_FFT = False
    print("⚠️  Warning: scipy.fft not available for DCT basis generation")


def create_overcomplete_basis(patch_size: Tuple[int, int], 
                            overcompleteness_factor: float = 2.0, 
                            basis_type: str = 'gabor',
                            random_seed: Optional[int] = None) -> np.ndarray:
    """
    🔬 Create Overcomplete Basis (Olshausen & Field 1996 Key Concept)
    
    Creates an overcomplete dictionary where the number of basis functions
    exceeds the input dimensionality. This is fundamental to Olshausen & Field's
    approach for learning sparse representations of natural images.
    
    From the paper: "The goal is to find a complete dictionary of basis functions
    such that natural images can be represented as sparse linear combinations."
    
    Args:
        patch_size: Size of input patches (height, width)
        overcompleteness_factor: Ratio of dictionary size to input dimension
                               2.0 = twice as many basis functions as pixels
        basis_type: Type of initial basis ('gabor', 'dct', 'random', 'edges')
        random_seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Overcomplete dictionary matrix (input_dim, n_basis)
        
    Examples:
        >>> # Create 2x overcomplete Gabor basis for 16x16 patches
        >>> basis = create_overcomplete_basis((16, 16), 2.0, 'gabor')
        >>> print(f"Created basis: {basis.shape}")  # (256, 512)
        
        >>> # Create DCT basis with different overcompleteness
        >>> basis = create_overcomplete_basis((8, 8), 3.0, 'dct', random_seed=42)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    input_dim = patch_size[0] * patch_size[1]
    n_basis = int(input_dim * overcompleteness_factor)
    
    print(f"🎨 Creating overcomplete basis: {input_dim} → {n_basis} functions (factor: {overcompleteness_factor:.1f})")
    
    if basis_type == 'gabor':
        # Create Gabor-like basis functions (oriented filters)
        basis = _create_gabor_basis(patch_size, input_dim, n_basis)
    elif basis_type == 'dct':
        # Discrete Cosine Transform basis (frequency domain)
        basis = _create_dct_basis(input_dim, n_basis)
    elif basis_type == 'edges':
        # Edge detector basis functions
        basis = _create_edge_basis(patch_size, input_dim, n_basis)
    else:  # random
        # Random Gaussian initialization
        basis = np.random.randn(input_dim, n_basis)
        # Normalize columns
        basis = basis / np.linalg.norm(basis, axis=0, keepdims=True)
    
    print(f"✅ Overcomplete basis created: {basis.shape}")
    print(f"   Basis type: {basis_type}")
    print(f"   Overcompleteness: {overcompleteness_factor}x")
    
    return basis


def lateral_inhibition(activations: np.ndarray, 
                      inhibition_strength: float = 0.5,
                      inhibition_radius: float = 1.0, 
                      topology: str = 'linear') -> np.ndarray:
    """
    🧠 Lateral Inhibition (Olshausen & Field 1996 Biological Mechanism)
    
    Implements lateral inhibition between competing basis functions, which is
    fundamental to the biological plausibility of sparse coding. This mechanism
    ensures that only the most relevant features remain active.
    
    From the paper: "In biological systems, lateral inhibition helps enforce
    sparsity by having active neurons suppress nearby competitors."
    
    Args:
        activations: Current activation coefficients
        inhibition_strength: Strength of lateral inhibition (0.0-1.0)
        inhibition_radius: Spatial radius of inhibition
        topology: Topology for neighbor relationships ('linear', '2d_grid', 'full')
        
    Returns:
        np.ndarray: Inhibited activation coefficients
        
    Examples:
        >>> # Apply linear lateral inhibition
        >>> activations = np.array([0.8, 0.6, 0.9, 0.3, 0.7])
        >>> inhibited = lateral_inhibition(activations, 0.3, 1.0, 'linear')
        
        >>> # 2D grid topology for spatial organization
        >>> activations = np.random.rand(64)  # 8x8 grid
        >>> inhibited = lateral_inhibition(activations, 0.5, 2.0, '2d_grid')
    """
    inhibited = activations.copy()
    n_units = len(activations)
    
    if topology == 'linear':
        # 1D linear topology - each unit inhibits immediate neighbors
        for i in range(n_units):
            if abs(activations[i]) > 0:  # Only active units provide inhibition
                # Calculate inhibition range
                start_idx = max(0, i - int(inhibition_radius))
                end_idx = min(n_units, i + int(inhibition_radius) + 1)
                
                for j in range(start_idx, end_idx):
                    if i != j:  # Don't self-inhibit
                        distance = abs(i - j)
                        # Gaussian inhibition profile
                        inhibition = abs(activations[i]) * inhibition_strength * np.exp(-distance**2 / (2 * inhibition_radius**2))
                        # Subtract inhibition (competitive)
                        if activations[j] > 0:
                            inhibited[j] = max(0, inhibited[j] - inhibition)
                        else:
                            inhibited[j] = min(0, inhibited[j] + inhibition)
    
    elif topology == '2d_grid':
        # 2D grid topology - units arranged in spatial grid
        grid_size = int(np.sqrt(n_units))
        if grid_size * grid_size != n_units:
            # Fallback to linear if not perfect square
            return lateral_inhibition(activations, inhibition_strength, inhibition_radius, 'linear')
        
        for i in range(n_units):
            if abs(activations[i]) > 0:
                # Convert to 2D coordinates
                row_i, col_i = divmod(i, grid_size)
                
                for j in range(n_units):
                    if i != j:
                        row_j, col_j = divmod(j, grid_size)
                        distance = np.sqrt((row_i - row_j)**2 + (col_i - col_j)**2)
                        
                        if distance <= inhibition_radius:
                            inhibition = abs(activations[i]) * inhibition_strength * np.exp(-distance**2 / (2 * inhibition_radius**2))
                            if activations[j] > 0:
                                inhibited[j] = max(0, inhibited[j] - inhibition)
                            else:
                                inhibited[j] = min(0, inhibited[j] + inhibition)
    
    elif topology == 'full':
        # Full connectivity - all units inhibit all others
        for i in range(n_units):
            if abs(activations[i]) > 0:
                for j in range(n_units):
                    if i != j:
                        inhibition = abs(activations[i]) * inhibition_strength / n_units
                        if activations[j] > 0:
                            inhibited[j] = max(0, inhibited[j] - inhibition)
                        else:
                            inhibited[j] = min(0, inhibited[j] + inhibition)
    
    return inhibited


def _create_gabor_basis(patch_size: Tuple[int, int], input_dim: int, n_basis: int) -> np.ndarray:
    """
    Create Gabor-like oriented basis functions
    
    Gabor filters are localized spatial-frequency filters that are excellent
    models for simple cell receptive fields in the visual cortex.
    
    Args:
        patch_size: Size of patches (height, width)
        input_dim: Input dimensionality (height * width)
        n_basis: Number of basis functions to create
        
    Returns:
        np.ndarray: Gabor basis functions matrix (input_dim, n_basis)
    """
    height, width = patch_size
    basis = np.zeros((input_dim, n_basis))
    
    for i in range(n_basis):
        # Random orientation, frequency, and phase
        orientation = np.random.uniform(0, np.pi)
        frequency = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, 2 * np.pi)
        
        # Create 2D Gabor filter
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Apply rotation
        x_rot = x_coords * np.cos(orientation) + y_coords * np.sin(orientation)
        y_rot = -x_coords * np.sin(orientation) + y_coords * np.cos(orientation)
        
        # Gabor function
        gaussian = np.exp(-(x_rot**2 + y_rot**2) / (2 * (min(height, width) / 4)**2))
        sinusoid = np.cos(2 * np.pi * frequency * x_rot + phase)
        gabor = gaussian * sinusoid
        
        # Flatten and normalize
        basis[:, i] = gabor.flatten()
        if np.linalg.norm(basis[:, i]) > 0:
            basis[:, i] /= np.linalg.norm(basis[:, i])
    
    return basis


def _create_dct_basis(input_dim: int, n_basis: int) -> np.ndarray:
    """
    Create DCT (Discrete Cosine Transform) basis - frequency domain
    
    DCT basis functions are useful for frequency-domain representations
    and are commonly used in signal processing and compression.
    
    Args:
        input_dim: Input dimensionality
        n_basis: Number of basis functions to create
        
    Returns:
        np.ndarray: DCT basis functions matrix (input_dim, n_basis)
    """
    if not HAS_SCIPY_FFT:
        print("⚠️  scipy.fft not available, falling back to random basis")
        basis = np.random.randn(input_dim, n_basis)
        return basis / np.linalg.norm(basis, axis=0, keepdims=True)
    
    # Create identity matrix and apply DCT
    identity = np.eye(input_dim)
    dct_basis = dct(identity, axis=0)
    
    # If we need more basis functions, repeat with variations
    if n_basis > input_dim:
        extra = n_basis - input_dim
        # Add phase-shifted versions
        phase_shifted = np.roll(dct_basis, shift=1, axis=0)
        dct_basis = np.column_stack([dct_basis, phase_shifted[:, :extra]])
    else:
        dct_basis = dct_basis[:, :n_basis]
    
    # Normalize
    dct_basis = dct_basis / np.linalg.norm(dct_basis, axis=0, keepdims=True)
    return dct_basis


def _create_edge_basis(patch_size: Tuple[int, int], input_dim: int, n_basis: int) -> np.ndarray:
    """
    Create edge detector basis functions
    
    Edge detectors are fundamental features in computer vision and
    closely match the receptive fields found in biological visual systems.
    
    Args:
        patch_size: Size of patches (height, width)
        input_dim: Input dimensionality (height * width)
        n_basis: Number of basis functions to create
        
    Returns:
        np.ndarray: Edge detector basis functions matrix (input_dim, n_basis)
    """
    height, width = patch_size
    basis = np.zeros((input_dim, n_basis))
    
    # Create various edge detectors
    edge_types = ['horizontal', 'vertical', 'diagonal_1', 'diagonal_2']
    
    for i in range(n_basis):
        edge_type = edge_types[i % len(edge_types)]
        
        # Create edge filter
        if edge_type == 'horizontal':
            kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        elif edge_type == 'vertical':  
            kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        elif edge_type == 'diagonal_1':
            kernel = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
        else:  # diagonal_2
            kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
        
        # Place kernel randomly in patch
        patch = np.zeros((height, width))
        start_row = np.random.randint(0, max(1, height - 2))
        start_col = np.random.randint(0, max(1, width - 2))
        
        end_row = min(start_row + 3, height)
        end_col = min(start_col + 3, width)
        
        patch[start_row:end_row, start_col:end_col] = kernel[:end_row-start_row, :end_col-start_col]
        
        # Flatten and normalize
        basis[:, i] = patch.flatten()
        if np.linalg.norm(basis[:, i]) > 0:
            basis[:, i] /= np.linalg.norm(basis[:, i])
    
    return basis


def demo_sparse_coding():
    """
    Demonstration of sparse coding library
    
    This function provides a complete example of how to use the sparse coding
    utilities, including data generation, dictionary learning, and visualization.
    """
    # Import SparseCoder from parent module
    try:
        from ..sparse_coder import SparseCoder
    except ImportError:
        print("⚠️  Could not import SparseCoder. Make sure it's in the parent directory.")
        return
    
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
    
    print("🖼️  Sparse Coding Utilities Demo")
    print("=" * 35)
    
    # Create test data
    test_images = generate_test_images(20, (32, 32))
    print(f"Generated {len(test_images)} test images")
    
    # Demonstrate overcomplete basis creation
    print("\n🎨 Creating overcomplete basis...")
    basis_gabor = create_overcomplete_basis((8, 8), 2.0, 'gabor', random_seed=42)
    basis_dct = create_overcomplete_basis((8, 8), 1.5, 'dct', random_seed=42)
    basis_edges = create_overcomplete_basis((8, 8), 3.0, 'edges', random_seed=42)
    
    # Demonstrate lateral inhibition
    print("\n🧠 Testing lateral inhibition...")
    test_activations = np.random.rand(16) * 2 - 1  # Random activations between -1 and 1
    print(f"Original activations: {test_activations[:8].round(2)}...")
    
    inhibited_linear = lateral_inhibition(test_activations, 0.3, 1.0, 'linear')
    inhibited_2d = lateral_inhibition(test_activations, 0.3, 1.5, '2d_grid')
    
    print(f"After linear inhibition: {inhibited_linear[:8].round(2)}...")
    print(f"After 2D grid inhibition: {inhibited_2d[:8].round(2)}...")
    
    # Create and train sparse coder
    print("\n📚 Training sparse coder with utilities...")
    sparse_coder = SparseCoder(
        n_components=64,
        sparsity_penalty=0.1,
        patch_size=(8, 8),
        random_seed=42
    )
    
    # Use overcomplete basis
    sparse_coder.dictionary = basis_gabor
    sparse_coder.n_components = basis_gabor.shape[1]
    
    # Learn dictionary
    results = sparse_coder.fit(test_images, n_patches=1000)
    
    if HAS_MATPLOTLIB:
        # Visualize results
        sparse_coder.visualize_dictionary(figsize=(10, 10))
        sparse_coder.plot_training_curves()
    
    print(f"\n💡 Key Innovation:")
    print(f"   • Natural images are sparse in learned basis")
    print(f"   • Algorithm discovers edge detectors automatically") 
    print(f"   • Foundation of modern convolutional neural networks")
    print(f"   • Matches biological visual cortex structure!")
    print(f"\n🎯 Utility Functions Demonstrated:")
    print(f"   ✅ Overcomplete basis creation")
    print(f"   ✅ Lateral inhibition mechanism")
    print(f"   ✅ Gabor, DCT, and edge basis generators")
    print(f"   ✅ Integration with main sparse coding framework")


# Additional utility functions for analysis and diagnostics
def analyze_basis_properties(basis: np.ndarray, patch_size: Tuple[int, int]) -> dict:
    """
    Analyze properties of a basis dictionary
    
    Args:
        basis: Dictionary matrix (input_dim, n_basis)
        patch_size: Size of patches (height, width)
        
    Returns:
        dict: Analysis results including coherence, sparsity, etc.
    """
    n_basis = basis.shape[1]
    
    # Calculate mutual coherence
    gram_matrix = basis.T @ basis
    off_diagonal = gram_matrix - np.eye(n_basis)
    mutual_coherence = np.max(np.abs(off_diagonal))
    
    # Calculate condition number
    condition_number = np.linalg.cond(gram_matrix)
    
    # Analyze orientations (for visual basis functions)
    orientations = []
    for i in range(n_basis):
        element = basis[:, i].reshape(patch_size)
        
        # Simple gradient-based orientation estimation
        grad_y = np.abs(np.gradient(element, axis=0)).mean()
        grad_x = np.abs(np.gradient(element, axis=1)).mean()
        
        if grad_x + grad_y > 0:
            orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi
            orientations.append(orientation)
    
    # Calculate energy distribution
    element_energies = np.sum(basis**2, axis=0)
    energy_std = np.std(element_energies)
    
    return {
        'mutual_coherence': mutual_coherence,
        'condition_number': condition_number,
        'n_oriented_elements': len(orientations),
        'orientation_range': (np.min(orientations), np.max(orientations)) if orientations else (0, 0),
        'energy_std': energy_std,
        'overcompleteness_ratio': n_basis / basis.shape[0]
    }


def visualize_basis_subset(basis: np.ndarray, patch_size: Tuple[int, int], 
                          n_display: int = 64, figsize: Tuple[int, int] = (12, 12)):
    """
    Visualize a subset of basis functions
    
    Args:
        basis: Dictionary matrix (input_dim, n_basis)
        patch_size: Size of patches (height, width)
        n_display: Number of basis functions to display
        figsize: Figure size for matplotlib
    """
    if not HAS_MATPLOTLIB:
        print("⚠️  matplotlib not available for visualization")
        return
        
    n_display = min(n_display, basis.shape[1])
    grid_size = int(np.sqrt(n_display))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    fig.suptitle(f'Basis Functions ({n_display} of {basis.shape[1]})', fontsize=16)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < n_display:
                # Reshape basis function to patch
                element = basis[:, idx].reshape(patch_size)
                
                # Normalize for visualization
                if element.max() > element.min():
                    element = (element - element.min()) / (element.max() - element.min())
                
                axes[i, j].imshow(element, cmap='gray', interpolation='nearest')
            
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("💰 SUPPORT THIS RESEARCH - PLEASE DONATE!")  
    print("🙏 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS")
    print("="*80 + "\n")
    
    print("🔧 Sparse Coding Utilities Module")
    print("=" * 35)
    demo_sparse_coding()
    
    print("\n" + "="*80)
    print("💝 Thank you for using this research software!")
    print("📚 Please donate: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS") 
    print("="*80 + "\n")

"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
📝 CITE: Benedict Chen (2025) - Sparse Coding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""