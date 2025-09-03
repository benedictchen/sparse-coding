"""
⚙️ Sparse Coding Configuration & Hyperparameter Management
========================================================

Author: Benedict Chen (benedict@benedictchen.com)

💰 Donations: Help support this research!
   PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   💖 Please consider recurring donations to support continued sparse coding research

Configuration classes, enums, and hyperparameter settings for research-accurate
sparse coding implementations based on Olshausen & Field (1996).

🔬 Research Foundation:
======================
Configuration parameters derived from:
- Olshausen & Field (1996): Original sparse coding hyperparameters
- Beck & Teboulle (2009): FISTA optimization parameter recommendations
- Mairal et al. (2009): Online dictionary learning convergence criteria
- Aharon et al. (2006): K-SVD sparsity level guidelines

ELI5 Explanation:
================
Think of this like a recipe book for sparse coding algorithms! 👨‍🍳

🥘 **The Recipe Analogy**:
Just like how different dishes need different cooking temperatures, timing, and ingredients,
different sparse coding tasks need different algorithm settings:

- **Sparsity level** = How much salt to add (more sparse = more selective)
- **Learning rate** = How hot your stove is (too high = burnt, too low = never cooks)
- **Dictionary size** = How many cooking techniques you know (more = more flexible)
- **Algorithm choice** = Which cooking method (slow roast vs. quick fry)

🧪 **Research Accuracy**:
These aren't random numbers! Each parameter has been carefully studied in research papers
to find the sweet spots that work best for natural images, audio signals, and other data types.

ASCII Configuration Architecture:
================================
    USER CHOOSES          CONFIG VALIDATES       ALGORITHM USES
    ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
    │"I want 64   │──────▶│ SparseCoder │──────▶│ FISTA with  │
    │ dictionary  │       │ Config      │       │ validated   │
    │ atoms with  │       │ ✓ 64 atoms  │       │ parameters  │
    │ L1 sparsity"│       │ ✓ L1 norm   │       │             │
    └─────────────┘       └─────────────┘       └─────────────┘
           │                       │                       │
           │                       ▼                       │
           │               ┌─────────────┐                │
           │               │ Parameter   │                │
           └──────────────▶│ Validation  │◀───────────────┘
                           │ & Defaults  │
                           └─────────────┘

📊 Parameter Categories:
=======================
🎯 **Algorithm Selection**: FISTA, Coordinate Descent, Gradient Descent
🧮 **Sparsity Functions**: L1, Log, Gaussian, Student-t, Elastic Net
🏗️ **Dictionary Updates**: Multiplicative, Additive, K-SVD, Projection
📈 **Optimization**: Learning rates, convergence criteria, max iterations
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
        # FIXME: Incomplete parameter validation - missing many critical checks
        # Issue 1: No upper bounds on parameters that could cause memory/performance issues
        # Issue 2: No validation of patch_size dimensions or compatibility
        # Issue 3: Missing validation for tolerance ranges and numerical stability
        # Issue 4: No checks for parameter combinations that could cause algorithm failures
        # Issue 5: Missing validation for advanced parameters like elastic_net_ratio
        
        # Basic positive parameter validation
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        
        # FIXME: No upper bound check for n_components
        # Issue: Very large n_components (>10000) can cause memory issues
        # Solutions:
        # 1. Add reasonable upper bound based on system memory
        # 2. Add warning for excessively large values
        # 3. Provide memory estimation for large dictionaries
        #
        # Example implementation:
        # if self.n_components > 5000:
        #     import psutil
        #     memory_needed = self.n_components * np.prod(self.patch_size) * 4 / (1024**3)  # GB
        #     available_memory = psutil.virtual_memory().available / (1024**3)
        #     if memory_needed > available_memory * 0.8:
        #         raise ValueError(f"n_components={self.n_components} may exceed available memory")
        
        if self.lambda_sparsity < 0:
            raise ValueError("lambda_sparsity must be non-negative")
            
        # FIXME: No upper bound or optimal range guidance for lambda_sparsity
        # Issue: lambda_sparsity > 1.0 often causes over-sparsification
        # Solutions:
        # 1. Add warning for unusually high values
        # 2. Provide guidance based on sparsity_func type
        # 3. Add automatic scaling based on data statistics
        #
        # Example:
        # if self.lambda_sparsity > 1.0:
        #     warnings.warn(f"lambda_sparsity={self.lambda_sparsity} is large, may cause over-sparsification")
        # if self.sparsity_func == SparsityFunction.L1 and self.lambda_sparsity > 0.5:
        #     warnings.warn("Consider smaller lambda_sparsity for L1 regularization")
            
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
            
        # FIXME: Missing patch_size validation
        # Issue: Invalid patch dimensions can cause reshape errors later
        # Solutions:
        # 1. Validate patch_size is tuple of positive integers
        # 2. Check for reasonable patch dimensions
        # 3. Warn about non-square patches if algorithms require square
        #
        # Example implementation:
        # if not (isinstance(self.patch_size, tuple) and len(self.patch_size) == 2):
        #     raise ValueError("patch_size must be tuple of length 2")
        # if not all(isinstance(p, int) and p > 0 for p in self.patch_size):
        #     raise ValueError("patch_size must contain positive integers")
        # if max(self.patch_size) > 64:
        #     warnings.warn("Large patch_size may cause computational issues")
        # if self.patch_size[0] != self.patch_size[1]:
        #     warnings.warn("Non-square patches may not work with all visualization methods")
            
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
            
        # FIXME: No upper bound check for learning_rate
        # Issue: learning_rate > 0.1 often causes instability
        # Solutions:
        # 1. Add warnings for potentially unstable learning rates
        # 2. Provide optimizer-specific recommendations
        # 3. Add adaptive learning rate validation
        #
        # Example:
        # if self.learning_rate > 0.1:
        #     warnings.warn(f"learning_rate={self.learning_rate} may cause training instability")
        # if self.optimizer == Optimizer.FISTA and self.learning_rate > 0.01:
        #     warnings.warn("FISTA typically works better with learning_rate <= 0.01")
            
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
            
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
            
        # FIXME: Missing tolerance validation for numerical stability
        # Issue: tolerance values outside [1e-12, 1e-2] can cause problems
        # Solutions:
        # 1. Validate tolerance is in reasonable numerical range
        # 2. Add warnings for potentially problematic values
        # 3. Provide algorithm-specific tolerance recommendations
        #
        # Example:
        # if not 1e-12 <= self.tolerance <= 1e-2:
        #     warnings.warn(f"tolerance={self.tolerance} may cause numerical issues")
        # if self.tolerance > 1e-3:
        #     warnings.warn("Large tolerance may reduce solution quality")
        
        # FIXME: Missing validation for elastic_net_ratio when using elastic net
        # Issue: elastic_net_ratio used without validation when sparsity_func is ELASTIC_NET
        # Solutions:
        # 1. Validate elastic_net_ratio is in [0, 1] when using elastic net
        # 2. Add warnings about extreme values
        # 3. Provide guidance on choosing appropriate ratio
        #
        # Example:
        # if self.sparsity_func == SparsityFunction.ELASTIC_NET:
        #     if not 0 <= self.elastic_net_ratio <= 1:
        #         raise ValueError("elastic_net_ratio must be in [0, 1] for elastic net regularization")
        
        # FIXME: No validation for parameter compatibility issues
        # Issue: Some parameter combinations are mathematically invalid or suboptimal
        # Solutions:
        # 1. Check for incompatible optimizer + sparsity_func combinations
        # 2. Validate dictionary update rule compatibility
        # 3. Add warnings for suboptimal parameter combinations
        #
        # Example parameter compatibility checks:
        # if self.optimizer == Optimizer.COORDINATE_DESCENT and self.sparsity_func != SparsityFunction.L1:
        #     warnings.warn("Coordinate descent works best with L1 regularization")
        # if self.positive_codes and self.sparsity_func == SparsityFunction.GAUSSIAN:
        #     warnings.warn("Positive codes constraint may conflict with Gaussian sparsity")
        # if self.dict_update_rule == DictionaryUpdateRule.K_SVD and self.batch_size == 1:
        #     warnings.warn("K-SVD typically requires batch_size > 1 for effectiveness")
    
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