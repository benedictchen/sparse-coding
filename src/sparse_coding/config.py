"""
Sparse Coding Configuration
===========================

Configuration classes, enums, and hyperparameter settings
for sparse coding algorithms.

Consolidated from scattered configuration files to provide
a unified configuration interface.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union


class SparsityFunction(Enum):
    """Available sparsity regularization functions"""
    L1 = "l1"                    # L1 norm (LASSO)
    LOG = "log"                  # Log penalty: log(1 + a²)
    GAUSSIAN = "gaussian"        # Gaussian: 1 - exp(-a²/2) 
    HUBER = "huber"             # Huber loss (robust L1)
    STUDENT_T = "student_t"      # Student-t penalty
    ELASTIC_NET = "elastic_net"  # L1 + L2 combination


class Optimizer(Enum):
    """Available optimization algorithms"""
    GRADIENT_DESCENT = "gradient_descent"    # Basic gradient descent
    FISTA = "fista"                         # Fast Iterative Shrinkage-Thresholding
    COORDINATE_DESCENT = "coordinate_descent" # Coordinate descent
    PROXIMAL_GRADIENT = "proximal_gradient"  # Proximal gradient method
    ADMM = "admm"                           # Alternating Direction Method of Multipliers
    ISTA = "ista"                           # Iterative Shrinkage-Thresholding


class DictionaryUpdateRule(Enum):
    """Available dictionary update methods"""
    MULTIPLICATIVE = "multiplicative"   # Multiplicative update rules
    ADDITIVE = "additive"              # Additive gradient-based updates
    PROJECTION = "projection"          # Projection-based updates
    K_SVD = "ksvd"                    # K-SVD algorithm
    MOD = "mod"                       # Method of Optimal Directions
    ONLINE = "online"                 # Online dictionary learning


class InitializationMethod(Enum):
    """Dictionary initialization methods"""
    RANDOM = "random"           # Random Gaussian initialization
    ICA = "ica"                # Independent Component Analysis
    PCA = "pca"                # Principal Component Analysis  
    DCT = "dct"                # Discrete Cosine Transform
    GABOR = "gabor"            # Gabor filter bank
    PATCHES = "patches"        # Random patches from training data


@dataclass
class SparseCoderConfig:
    """
    Configuration for main SparseCoder class
    
    Consolidates all hyperparameters and settings for sparse coding
    algorithms in a single, well-documented configuration class.
    """
    
    # Core algorithm parameters
    n_components: int = 100
    patch_size: Tuple[int, int] = (8, 8)
    max_iter: int = 1000
    tolerance: float = 1e-4
    random_state: Optional[int] = None
    
    # Sparsity control
    lambda_sparsity: float = 0.1
    sparsity_func: SparsityFunction = SparsityFunction.L1
    sparsity_target: Optional[int] = None  # Target number of non-zeros
    
    # Optimization settings
    optimizer: Optimizer = Optimizer.FISTA
    learning_rate: float = 0.01
    momentum: float = 0.9
    line_search: bool = True
    max_inner_iter: int = 100
    
    # Dictionary learning
    dict_update_rule: DictionaryUpdateRule = DictionaryUpdateRule.MULTIPLICATIVE
    dict_learning_rate: float = 0.01
    dict_normalize: bool = True
    
    # Initialization
    initialization: InitializationMethod = InitializationMethod.RANDOM
    initialization_scale: float = 1.0
    
    # Training dynamics
    batch_size: int = 100
    n_epochs: int = 10
    shuffle_data: bool = True
    validation_split: float = 0.0
    
    # Convergence criteria
    early_stopping: bool = False
    patience: int = 10
    min_improvement: float = 1e-6
    
    # Regularization
    l2_penalty: float = 0.0      # L2 regularization on dictionary
    elastic_net_ratio: float = 0.5  # For elastic net sparsity
    
    # Advanced options
    positive_codes: bool = False  # Constrain codes to be non-negative
    positive_dict: bool = False   # Constrain dictionary to be non-negative  
    orthogonal_dict: bool = False # Enforce orthogonal dictionary elements
    
    # Memory and performance
    low_memory: bool = False     # Use memory-efficient algorithms
    parallel: bool = False       # Enable parallel processing
    n_jobs: int = 1             # Number of parallel jobs
    
    # Debugging and monitoring
    verbose: bool = True
    debug: bool = False
    save_history: bool = True
    history_interval: int = 50
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        
        if self.lambda_sparsity < 0:
            raise ValueError("lambda_sparsity must be non-negative")
            
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
            
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
            
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
            
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        config_dict = {}
        for field, value in self.__dict__.items():
            if isinstance(value, Enum):
                config_dict[field] = value.value
            else:
                config_dict[field] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SparseCoderConfig':
        """Create configuration from dictionary"""
        # Convert string enum values back to enums
        if 'sparsity_func' in config_dict:
            config_dict['sparsity_func'] = SparsityFunction(config_dict['sparsity_func'])
        if 'optimizer' in config_dict:
            config_dict['optimizer'] = Optimizer(config_dict['optimizer'])
        if 'dict_update_rule' in config_dict:
            config_dict['dict_update_rule'] = DictionaryUpdateRule(config_dict['dict_update_rule'])
        if 'initialization' in config_dict:
            config_dict['initialization'] = InitializationMethod(config_dict['initialization'])
        
        return cls(**config_dict)


@dataclass  
class OlshausenFieldConfig:
    """
    Configuration for original Olshausen & Field (1996) algorithm
    
    Parameters exactly as described in the original Nature paper
    """
    
    # Core parameters from paper
    M: int = 100                    # Number of basis functions
    patch_size: Tuple[int, int] = (8, 8)  # Image patch size
    lambda_sparsity: float = 0.1    # Sparseness parameter (λ in equation 5)
    eta_phi: float = 0.01          # Learning rate for coefficients (η_φ)
    eta_dict: float = 0.01         # Learning rate for dictionary (η_D)
    tau: float = 1.0               # Time constant for dynamics
    
    # Training parameters
    n_iterations: int = 10000      # Number of training iterations
    convergence_check_interval: int = 1000
    convergence_tolerance: float = 1e-6
    
    # Data preprocessing
    subtract_mean: bool = True     # Subtract patch mean (standard preprocessing)
    normalize_variance: bool = False  # Normalize patch variance
    
    # Basis function constraints
    normalize_basis: bool = True   # Normalize basis functions to unit norm
    
    # Monitoring and output
    verbose: bool = True
    save_intermediate: bool = False
    save_interval: int = 1000
    
    # Reproducibility
    random_state: Optional[int] = None


@dataclass
class DictionaryLearningConfig:
    """Configuration for advanced dictionary learning algorithms"""
    
    # Algorithm selection
    algorithm: str = 'ksvd'        # 'ksvd', 'mod', 'online', 'minibatch'
    
    # Core parameters
    n_components: int = 100
    sparsity_constraint: int = 10  # Maximum non-zeros per signal
    max_iter: int = 100
    tolerance: float = 1e-4
    
    # K-SVD specific
    ksvd_max_inner_iter: int = 10
    ksvd_initialization: str = 'data'  # 'random' or 'data'
    
    # Online learning specific  
    online_batch_size: int = 1
    online_forgetting_factor: float = 0.95
    online_learning_rate: float = 0.01
    
    # MOD specific
    mod_regularization: float = 1e-6
    
    # General settings
    random_state: Optional[int] = None
    verbose: bool = True
    n_jobs: int = 1


@dataclass
class FeatureExtractionConfig:
    """Configuration for sparse feature extraction"""
    
    # Patch extraction
    patch_size: Tuple[int, int] = (8, 8)
    overlap: float = 0.5           # Overlap between patches
    stride: Optional[int] = None   # Explicit stride (overrides overlap)
    
    # Preprocessing
    normalize_patches: bool = True
    subtract_mean: bool = True
    unit_variance: bool = False
    whitening: bool = False
    
    # Feature processing
    pooling_method: str = 'max'    # 'max', 'mean', 'sum', 'none'
    pooling_size: Tuple[int, int] = (2, 2)
    
    # Output format
    flatten_features: bool = True
    return_positions: bool = False  # Return patch positions
    
    # Memory management
    batch_processing: bool = False
    max_patches_per_batch: int = 10000


@dataclass
class BatchProcessingConfig:
    """Configuration for large-scale batch processing"""
    
    # Batch settings
    batch_size: int = 1000
    overlap_batches: bool = False
    batch_overlap: float = 0.1
    
    # Memory management
    max_memory_mb: int = 1000
    use_memory_mapping: bool = False
    temp_dir: Optional[str] = None
    
    # Parallel processing
    n_jobs: int = 1
    backend: str = 'threading'     # 'threading', 'multiprocessing'
    
    # Progress tracking
    verbose: bool = True
    progress_interval: int = 100
    
    # Output management
    save_intermediate: bool = False
    output_format: str = 'numpy'   # 'numpy', 'hdf5', 'zarr'


# =============================================================================
# Preset Configurations
# =============================================================================

def get_olshausen_field_config() -> SparseCoderConfig:
    """Get configuration matching original Olshausen & Field (1996) paper"""
    return SparseCoderConfig(
        n_components=100,
        patch_size=(8, 8),
        lambda_sparsity=0.1,
        sparsity_func=SparsityFunction.L1,
        optimizer=Optimizer.GRADIENT_DESCENT,
        learning_rate=0.01,
        max_iter=10000,
        initialization=InitializationMethod.RANDOM,
        batch_size=1,  # Original used online learning
        dict_update_rule=DictionaryUpdateRule.MULTIPLICATIVE,
        tolerance=1e-6,
        verbose=True
    )


def get_fast_config() -> SparseCoderConfig:
    """Get configuration optimized for speed"""
    return SparseCoderConfig(
        n_components=50,
        patch_size=(6, 6),
        lambda_sparsity=0.05,
        sparsity_func=SparsityFunction.L1,
        optimizer=Optimizer.FISTA,
        learning_rate=0.05,
        max_iter=500,
        batch_size=200,
        tolerance=1e-3,
        early_stopping=True,
        patience=5,
        verbose=True
    )


def get_accurate_config() -> SparseCoderConfig:
    """Get configuration optimized for accuracy"""
    return SparseCoderConfig(
        n_components=200,
        patch_size=(12, 12),
        lambda_sparsity=0.01,
        sparsity_func=SparsityFunction.L1,
        optimizer=Optimizer.FISTA,
        learning_rate=0.001,
        max_iter=2000,
        batch_size=50,
        tolerance=1e-6,
        early_stopping=True,
        patience=20,
        line_search=True,
        verbose=True
    )


def get_research_config() -> SparseCoderConfig:
    """Get configuration for research/experimentation"""
    return SparseCoderConfig(
        n_components=100,
        patch_size=(8, 8),
        lambda_sparsity=0.1,
        sparsity_func=SparsityFunction.L1,
        optimizer=Optimizer.FISTA,
        learning_rate=0.01,
        max_iter=1000,
        batch_size=100,
        tolerance=1e-4,
        save_history=True,
        history_interval=10,
        debug=True,
        verbose=True
    )


# =============================================================================
# Configuration Factory
# =============================================================================

def create_config(preset: str = 'default', **kwargs) -> SparseCoderConfig:
    """
    Create configuration with optional preset and custom parameters
    
    Parameters
    ----------
    preset : str
        Configuration preset: 'default', 'olshausen_field', 'fast', 
        'accurate', 'research'
    **kwargs
        Custom parameters to override preset values
        
    Returns
    -------
    config : SparseCoderConfig
        Configured sparse coder configuration
    """
    if preset == 'olshausen_field':
        config = get_olshausen_field_config()
    elif preset == 'fast':
        config = get_fast_config()
    elif preset == 'accurate':
        config = get_accurate_config()
    elif preset == 'research':
        config = get_research_config()
    else:  # default
        config = SparseCoderConfig()
    
    # Override with custom parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    return config


if __name__ == "__main__":
    # Example usage and testing
    print("🔧 Sparse Coding Configuration")
    print("=" * 40)
    
    # Test default configuration
    default_config = SparseCoderConfig()
    print(f"Default config: {default_config.n_components} components, {default_config.optimizer.value} optimizer")
    
    # Test preset configurations
    presets = ['olshausen_field', 'fast', 'accurate', 'research']
    for preset in presets:
        config = create_config(preset)
        print(f"{preset.title()} config: {config.n_components} components, λ={config.lambda_sparsity}")
    
    # Test custom configuration
    custom_config = create_config('fast', n_components=75, lambda_sparsity=0.08)
    print(f"Custom config: {custom_config.n_components} components, λ={custom_config.lambda_sparsity}")
    
    # Test serialization
    config_dict = default_config.to_dict()
    restored_config = SparseCoderConfig.from_dict(config_dict)
    print(f"Serialization test: {restored_config.sparsity_func.value}")
    
    print("✅ All configuration tests passed!")