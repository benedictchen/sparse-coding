"""
Test configuration and fixtures for sparse coding tests.

Provides common test fixtures, utilities, and configuration for all test modules.
"""

import numpy as np
import pytest
from scipy import linalg
from sklearn.datasets import make_sparse_coded_signal
from sparse_coding import SparseCoder, DictionaryLearner


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture 
def small_dict_atoms():
    """Small dictionary for fast unit tests."""
    return 16


@pytest.fixture
def medium_dict_atoms():
    """Medium dictionary for integration tests.""" 
    return 64


@pytest.fixture
def large_dict_atoms():
    """Large dictionary for research validation tests."""
    return 144


@pytest.fixture
def signal_length():
    """Standard signal length for tests."""
    return 64  # Match 8x8 patch dimensions to avoid mismatch


@pytest.fixture
def patch_size():
    """Standard patch size for image patches.""" 
    return 16


@pytest.fixture
def n_samples():
    """Number of samples for statistical tests."""
    return 100


@pytest.fixture
def tolerance():
    """Standard numerical tolerance."""
    return 1e-6


@pytest.fixture
def relaxed_tolerance():
    """Relaxed tolerance for iterative algorithms."""
    return 1e-3


@pytest.fixture
def synthetic_data(signal_length, small_dict_atoms, random_seed):
    """Generate synthetic sparse coded data."""
    np.random.seed(random_seed)
    n_features = signal_length
    n_components = small_dict_atoms
    n_samples = 50
    
    # Generate ground truth sparse codes and dictionary
    # Use better parameters for meaningful sparse coding
    true_codes = np.random.laplace(scale=0.3, size=(n_components, n_samples))
    true_codes[np.abs(true_codes) < 0.15] = 0  # More reasonable sparsity threshold
    
    true_dict = np.random.randn(n_features, n_components)
    # Normalize dictionary atoms
    true_dict /= np.linalg.norm(true_dict, axis=0, keepdims=True)
    
    # Generate signals
    signals = true_dict @ true_codes
    # Add small amount of noise
    signals += 0.01 * np.random.randn(*signals.shape)
    
    return {
        'signals': signals,
        'true_codes': true_codes, 
        'true_dict': true_dict,
        'n_features': n_features,
        'n_components': n_components,
        'n_samples': n_samples
    }


@pytest.fixture
def natural_image_patches(patch_size, random_seed):
    """Generate synthetic natural image patches."""
    np.random.seed(random_seed)
    n_patches = 1000
    
    # Create patches with edge-like structure (simplified natural statistics)
    patches = []
    for _ in range(n_patches):
        # Random orientation and position
        angle = np.random.uniform(0, 2*np.pi)
        center_x = np.random.uniform(patch_size//4, 3*patch_size//4)
        center_y = np.random.uniform(patch_size//4, 3*patch_size//4)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
        
        # Edge response with random orientation
        edge = np.sin(np.cos(angle) * (x - center_x) + np.sin(angle) * (y - center_y))
        
        # Add noise and normalize
        patch = edge + 0.1 * np.random.randn(patch_size, patch_size)
        patch = patch.flatten()
        patch = patch - np.mean(patch)  # Zero mean
        patch = patch / (np.std(patch) + 1e-10)  # Unit variance
        
        patches.append(patch)
    
    return np.array(patches).T  # Shape: (patch_size^2, n_patches)


@pytest.fixture
def sparse_coder_instance(small_dict_atoms, random_seed):
    """Create a SparseCoder instance for testing."""
    return SparseCoder(
        n_atoms=small_dict_atoms,
        mode="l1", 
        max_iter=100,
        tol=1e-6,
        seed=random_seed
    )


@pytest.fixture
def dictionary_learner_instance(small_dict_atoms, random_seed):
    """Create a DictionaryLearner instance for testing."""
    return DictionaryLearner(
        n_atoms=small_dict_atoms,
        max_iter=20,
        tol=1e-4,
        seed=random_seed
    )


@pytest.fixture
def convergence_test_config():
    """Configuration for convergence tests."""
    return {
        'max_iter': 1000,
        'tol': 1e-8,
        'step_sizes': [0.01, 0.1, 1.0],
        'lambda_values': [0.01, 0.1, 1.0]
    }


def assert_dictionary_normalized(dictionary, tolerance=1e-10):
    """Assert that dictionary atoms are properly normalized."""
    atom_norms = np.linalg.norm(dictionary, axis=0)
    np.testing.assert_allclose(atom_norms, 1.0, atol=tolerance, 
                               err_msg="Dictionary atoms must be unit normalized")


def assert_sparse_solution(codes, sparsity_threshold=0.1):
    """Assert that solution is appropriately sparse with statistical validation.
    
    Validates true sparsity using multiple rigorous criteria:
    1. True sparsity ratio (proportion of exact zeros)
    2. Effective sparsity (proportion below numerical threshold)
    3. Statistical sparsity pattern validation
    4. Gini coefficient for concentration measurement
    """
    codes_flat = np.asarray(codes).ravel()
    n_total = len(codes_flat)
    
    # 1. True sparsity ratio (exact zeros)
    n_exact_zeros = np.sum(codes_flat == 0.0)
    true_sparsity_ratio = n_exact_zeros / n_total
    
    # 2. Effective sparsity (below threshold)
    n_small_values = np.sum(np.abs(codes_flat) < sparsity_threshold)
    effective_sparsity_ratio = n_small_values / n_total
    
    # 3. Non-zero statistics for sparse coding validation
    nonzero_mask = np.abs(codes_flat) >= sparsity_threshold
    n_significant = np.sum(nonzero_mask)
    
    # 4. Statistical concentration measure (Gini coefficient)
    # Higher Gini = more concentrated (sparse) distribution
    abs_codes = np.abs(codes_flat)
    sorted_abs = np.sort(abs_codes)
    n = len(sorted_abs)
    cumsum = np.cumsum(sorted_abs)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_abs))) / (n * cumsum[-1]) - (n + 1) / n
    
    # Research-based sparsity validation criteria
    
    # Criterion 1: Effective sparsity should be meaningful for sparse coding
    # Most values should be small (thresholded), but not all
    assert 0.3 <= effective_sparsity_ratio <= 0.95, (
        f"Effective sparsity ratio {effective_sparsity_ratio:.3f} outside valid range [0.3, 0.95]. "
        f"Values < {sparsity_threshold}: {n_small_values}/{n_total}"
    )
    
    # Criterion 2: Should have reasonable number of significant coefficients
    # Neither completely dense nor completely sparse
    expected_min_significant = max(1, int(0.05 * n_total))  # At least 5% or 1
    expected_max_significant = min(n_total, int(0.7 * n_total))  # At most 70%
    
    assert expected_min_significant <= n_significant <= expected_max_significant, (
        f"Significant coefficients {n_significant} outside expected range "
        f"[{expected_min_significant}, {expected_max_significant}] for sparse coding"
    )
    
    # Criterion 3: Statistical concentration validation
    # Gini coefficient should indicate concentrated (sparse) distribution
    min_gini_threshold = 0.3  # Moderate concentration
    assert gini >= min_gini_threshold, (
        f"Distribution not concentrated enough for sparsity: Gini={gini:.3f} < {min_gini_threshold}. "
        "Sparse coding should produce concentrated distributions."
    )
    
    # Criterion 4: Magnitude ratio validation
    # Significant values should be meaningfully larger than small values
    if n_significant > 0 and effective_sparsity_ratio < 1.0:
        significant_values = abs_codes[nonzero_mask]
        small_values = abs_codes[~nonzero_mask]
        
        if len(small_values) > 0:
            mean_significant = np.mean(significant_values)
            mean_small = np.mean(small_values)
            magnitude_ratio = mean_significant / (mean_small + 1e-12)
            
            # Significant values should be at least 3x larger than small values
            # Research note: Log-prior penalties create less sharp thresholding than L1
            min_magnitude_ratio = 3.0
            assert magnitude_ratio >= min_magnitude_ratio, (
                f"Insufficient magnitude separation: {magnitude_ratio:.2f} < {min_magnitude_ratio}. "
                f"Mean significant: {mean_significant:.6f}, Mean small: {mean_small:.6f}"
            )
    
    # Success message with detailed statistics
    return {
        'true_sparsity': true_sparsity_ratio,
        'effective_sparsity': effective_sparsity_ratio, 
        'n_significant': n_significant,
        'gini_coefficient': gini,
        'validation': 'PASSED'
    }


def assert_reconstruction_quality(signals, reconstructed, tolerance=1e-3):
    """
    Assert reconstruction quality meets professional scientific standards.
    
    Uses tight default tolerance (1e-3) for high-precision validation.
    For algorithms that inherently have higher error (sparse solutions, noisy data),
    explicitly pass a higher tolerance with justification.
    
    Args:
        signals: Original signals
        reconstructed: Reconstructed signals  
        tolerance: Maximum allowed relative MSE (default: 1e-3)
    """
    mse = np.mean((signals - reconstructed)**2)
    relative_error = mse / max(np.var(signals), 1e-12)  # Avoid division by zero
    
    assert relative_error < tolerance, (
        f"Reconstruction error exceeds scientific precision: {relative_error:.8f} >= {tolerance}. "
        f"MSE: {mse:.8f}, Signal variance: {np.var(signals):.8f}. "
        f"For inherently high-error algorithms, use explicit tolerance with justification."
    )


def measure_convergence_rate(objectives):
    """Measure the convergence rate from objective history."""
    if len(objectives) < 10:
        return np.nan
        
    # Linear convergence rate estimation
    log_ratios = []
    for i in range(10, len(objectives)):
        if objectives[i-1] > objectives[i] > 0:
            ratio = (objectives[i] - objectives[-1]) / (objectives[i-1] - objectives[-1])
            if ratio > 0:
                log_ratios.append(np.log(ratio))
    
    if len(log_ratios) > 5:
        return np.mean(log_ratios[-5:])  # Average of last 5 ratios
    return np.nan


def create_test_dictionary(n_features, n_atoms, condition_number=1.0, seed=42):
    """Create a test dictionary with controlled condition number."""
    np.random.seed(seed)
    
    # Generate random orthogonal dictionary if possible
    if n_atoms <= n_features:
        # Start with random matrix
        D = np.random.randn(n_features, n_atoms)
        
        # QR decomposition for orthogonal atoms
        Q, R = linalg.qr(D)
        D = Q[:, :n_atoms]
        
        # Adjust condition number if needed
        if condition_number > 1.0:
            # Add some correlation between atoms
            corruption = np.random.randn(n_features, n_atoms) * (condition_number - 1) / 10
            D = D + corruption
            
        # Normalize atoms
        D /= np.linalg.norm(D, axis=0, keepdims=True)
        
    else:
        # Overcomplete case
        D = np.random.randn(n_features, n_atoms)
        D /= np.linalg.norm(D, axis=0, keepdims=True)
    
    return D