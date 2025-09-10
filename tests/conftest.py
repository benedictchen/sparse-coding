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
    return 256


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
    true_codes = np.random.laplace(scale=0.1, size=(n_components, n_samples))
    true_codes[np.abs(true_codes) < 0.3] = 0  # Enforce sparsity
    
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
    """Assert that solution is appropriately sparse."""
    sparsity_ratio = np.mean(np.abs(codes) < sparsity_threshold) 
    assert sparsity_ratio > 0.5, f"Solution not sparse enough: {sparsity_ratio:.3f} < 0.5"


def assert_reconstruction_quality(signals, reconstructed, tolerance=0.1):
    """Assert reconstruction quality meets minimum standards."""
    mse = np.mean((signals - reconstructed)**2)
    relative_error = mse / np.var(signals)
    assert relative_error < tolerance, f"Reconstruction error too high: {relative_error:.6f}"


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