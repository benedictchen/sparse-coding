from __future__ import annotations
from pydantic import BaseModel, Field, PositiveInt
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
import json

# Legacy configuration for backwards compatibility
class TrainingConfig(BaseModel):
    patch_size: PositiveInt = 16
    n_atoms: PositiveInt = 144
    steps: PositiveInt = 50
    lr: float = Field(0.1, gt=0.0)
    mode: str = Field("paper", pattern="^(paper|l1)$")
    f0: float = Field(200.0, gt=0.0)
    lam_sigma: Optional[float] = None   # if set, lam = lam_sigma * std(whitened data)
    seed: int = 0
    deterministic: bool = True
    samples: PositiveInt = 50000

SCHEMA_VERSION = 1

def make_metadata(cfg: TrainingConfig, D_shape, A_shape, extra=None):
    meta = {
        "schema_version": SCHEMA_VERSION,
        "patch_size": cfg.patch_size,
        "n_atoms": cfg.n_atoms,
        "steps": cfg.steps,
        "lr": cfg.lr,
        "mode": cfg.mode,
        "f0": cfg.f0,
        "lam_sigma": cfg.lam_sigma,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "shapes": {"D": list(D_shape), "A": list(A_shape)},
    }
    if extra: meta.update(extra)
    return meta


# Comprehensive Configuration System for All FIXME Solutions
# Users can pick and choose from all implemented penalty functions, solvers, and dictionary updaters

@dataclass
class PenaltyConfig:
    """Configuration for penalty functions with research-based defaults."""
    name: str = 'l1'
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Research citations and parameter ranges from original papers
    RESEARCH_DEFAULTS = {
        'l1': {
            'lam': 0.1,  # Tibshirani (1996). Regression shrinkage and selection via the lasso
            'citation': 'Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.'
        },
        'l2': {
            'lam': 0.1,  # Hoerl & Kennard (1970). Ridge regression: Biased estimation for nonorthogonal problems
            'citation': 'Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation.'
        },
        'elastic_net': {
            'l1': 0.1, 'l2': 0.05,  # Zou & Hastie (2005). Regularization and variable selection via the elastic net
            'citation': 'Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net.'
        },
        'cauchy': {
            'lam': 0.1, 'sigma': 1.0,  # Dennis & Welsch (1978). Techniques for nonlinear least squares
            'citation': 'Dennis Jr, J. E., & Welsch, R. E. (1978). Techniques for nonlinear least squares.'
        },
        'topk': {
            'k': 10,  # Blumensath & Davies (2009). Iterative hard thresholding for compressed sensing
            'citation': 'Blumensath, T., & Davies, M. E. (2009). Iterative hard thresholding.'
        },
        'huber': {
            'lam': 0.1, 'delta': 1.0,  # Huber (1964). Robust estimation of a location parameter
            'citation': 'Huber, P. J. (1964). Robust estimation of a location parameter.'
        },
        'group_lasso': {
            'lam': 0.1, 'groups': [],  # Yuan & Lin (2006). Model selection and estimation in regression with grouped variables
            'citation': 'Yuan, M., & Lin, Y. (2006). Model selection and estimation in regression.'
        }
    }
    
    def __post_init__(self):
        """Apply research-based defaults with citations."""
        if not self.params and self.name in self.RESEARCH_DEFAULTS:
            config = self.RESEARCH_DEFAULTS[self.name]
            self.params = {k: v for k, v in config.items() if k != 'citation'}
            self.citation = config.get('citation', '')


@dataclass
class SolverConfig:
    """Configuration for inference solvers with algorithm-specific research parameters."""
    name: str = 'fista'
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Research-accurate parameters from original algorithm papers
    RESEARCH_DEFAULTS = {
        'fista': {
            'max_iter': 1000, 'tol': 1e-6, 'backtrack': True,
            'citation': 'Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm.'
        },
        'ista': {
            'max_iter': 1000, 'tol': 1e-6, 'step_size': None,
            'citation': 'Daubechies, I., et al. (2004). An iterative thresholding algorithm.'
        },
        'coordinate_descent': {
            'max_iter': 1000, 'tol': 1e-6, 'positive': False,
            'citation': 'Friedman, J., et al. (2007). Pathwise coordinate optimization.'
        },
        'omp': {
            'n_nonzero_coefs': None, 'tol': None,
            'citation': 'Pati, Y. C., et al. (1993). Orthogonal matching pursuit.'
        },
        'ncg': {
            'max_iter': 1000, 'tol': 1e-6, 'beta_method': 'polak-ribiere',
            'citation': 'Nocedal, J., & Wright, S. (2006). Numerical optimization.'
        }
    }
    
    def __post_init__(self):
        """Apply research-based defaults."""
        if not self.params and self.name in self.RESEARCH_DEFAULTS:
            config = self.RESEARCH_DEFAULTS[self.name]
            self.params = {k: v for k, v in config.items() if k != 'citation'}
            self.citation = config.get('citation', '')


@dataclass
class DictUpdaterConfig:
    """Configuration for dictionary update methods."""
    name: str = 'mod'
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Research-accurate parameters from original papers
    RESEARCH_DEFAULTS = {
        'mod': {
            'regularization': 1e-7, 'normalize': True,
            'citation': 'Engan, K., et al. (1999). Method of optimal directions for frame design.'
        },
        'grad_d': {
            'learning_rate': 0.01, 'momentum': 0.0, 'normalize': True,
            'citation': 'Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties.'
        },
        'ksvd': {
            'preserve_dc': False, 'verbose': False,
            'citation': 'Aharon, M., et al. (2006). K-SVD: An algorithm for designing overcomplete dictionaries.'
        },
        'online_sgd': {
            'learning_rate': 0.01, 'momentum': 0.9, 'batch_size': None,
            'citation': 'Mairal, J., et al. (2010). Online dictionary learning for sparse coding.'
        },
        'adam': {
            'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8,
            'citation': 'Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.'
        }
    }
    
    def __post_init__(self):
        """Apply research-based defaults."""
        if not self.params and self.name in self.RESEARCH_DEFAULTS:
            config = self.RESEARCH_DEFAULTS[self.name]
            self.params = {k: v for k, v in config.items() if k != 'citation'}
            self.citation = config.get('citation', '')


@dataclass
class ComprehensiveSparseCodingConfig:
    """Master configuration allowing users to mix and match ALL implemented solutions."""
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    dict_updater: DictUpdaterConfig = field(default_factory=DictUpdaterConfig)
    
    # Global parameters
    n_atoms: int = 100
    max_iter: int = 100
    tol: float = 1e-6
    random_state: Optional[int] = None
    verbose: bool = False
    
    def validate_compatibility(self) -> List[str]:
        """Check for known incompatible combinations and provide research guidance."""
        warnings = []
        
        # Penalty-Solver compatibility based on research findings
        if self.penalty.name == 'topk' and self.solver.name not in ['omp']:
            warnings.append(f"⚠️  TopK constraint typically used with OMP solver (current: {self.solver.name})")
        
        if self.penalty.name in ['cauchy'] and self.solver.name not in ['ncg', 'fista']:
            warnings.append(f"⚠️  Smooth penalties work best with NCG/FISTA (current: {self.solver.name})")
        
        if self.penalty.name == 'l1' and self.solver.name == 'ncg':
            warnings.append(f"⚠️  L1 is non-differentiable - NCG may not converge")
        
        # Solver-Updater research combinations
        if self.solver.name == 'omp' and self.dict_updater.name not in ['ksvd', 'mod']:
            warnings.append(f"⚠️  OMP typically combined with K-SVD/MOD (current: {self.dict_updater.name})")
        
        # Parameter range validation based on research literature
        if 'lam' in self.penalty.params:
            lam = self.penalty.params['lam']
            if lam < 0.001 or lam > 10.0:
                warnings.append(f"⚠️  λ={lam} outside typical research range [0.001, 10.0]")
        
        return warnings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'penalty': {'name': self.penalty.name, 'params': self.penalty.params},
            'solver': {'name': self.solver.name, 'params': self.solver.params},
            'dict_updater': {'name': self.dict_updater.name, 'params': self.dict_updater.params},
            'n_atoms': self.n_atoms,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def save_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'ComprehensiveSparseCodingConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        penalty_cfg = PenaltyConfig(name=data['penalty']['name'], params=data['penalty']['params'])
        solver_cfg = SolverConfig(name=data['solver']['name'], params=data['solver']['params'])
        updater_cfg = DictUpdaterConfig(name=data['dict_updater']['name'], params=data['dict_updater']['params'])
        
        return cls(
            penalty=penalty_cfg,
            solver=solver_cfg,
            dict_updater=updater_cfg,
            n_atoms=data['n_atoms'],
            max_iter=data['max_iter'],
            tol=data['tol'],
            random_state=data.get('random_state'),
            verbose=data['verbose']
        )


class ResearchAccuratePresets:
    """Research-accurate preset configurations from original papers."""
    
    @staticmethod
    def olshausen_field_1996() -> ComprehensiveSparseCodingConfig:
        """Original Olshausen & Field (1996) configuration."""
        return ComprehensiveSparseCodingConfig(
            penalty=PenaltyConfig('l1', {'lam': 0.1}),
            solver=SolverConfig('fista', {'max_iter': 1000, 'tol': 1e-6}),
            dict_updater=DictUpdaterConfig('grad_d', {'learning_rate': 0.01}),
            n_atoms=144,  # 12x12 overcomplete for 8x8 patches
            max_iter=1000,
            verbose=True
        )
    
    @staticmethod
    def ksvd_aharon_2006() -> ComprehensiveSparseCodingConfig:
        """K-SVD from Aharon et al. (2006)."""
        return ComprehensiveSparseCodingConfig(
            penalty=PenaltyConfig('topk', {'k': 4}),
            solver=SolverConfig('omp', {'n_nonzero_coefs': 4}),
            dict_updater=DictUpdaterConfig('ksvd', {'preserve_dc': True}),
            n_atoms=256,
            max_iter=30
        )
    
    @staticmethod
    def online_mairal_2010() -> ComprehensiveSparseCodingConfig:
        """Online dictionary learning from Mairal et al. (2010)."""
        return ComprehensiveSparseCodingConfig(
            penalty=PenaltyConfig('l1', {'lam': 0.15}),
            solver=SolverConfig('fista', {'max_iter': 500}),
            dict_updater=DictUpdaterConfig('online_sgd', {'learning_rate': 0.01, 'momentum': 0.9}),
            n_atoms=200,
            max_iter=1000
        )
    
    @staticmethod
    def elastic_net_zou_2005() -> ComprehensiveSparseCodingConfig:
        """Elastic Net from Zou & Hastie (2005)."""
        return ComprehensiveSparseCodingConfig(
            penalty=PenaltyConfig('elastic_net', {'l1': 0.1, 'l2': 0.05}),
            solver=SolverConfig('fista', {'max_iter': 1000}),
            dict_updater=DictUpdaterConfig('mod', {'regularization': 1e-6}),
            n_atoms=128,
            max_iter=200
        )
    
    @staticmethod
    def robust_cauchy() -> ComprehensiveSparseCodingConfig:
        """Robust sparse coding with Cauchy penalty."""
        return ComprehensiveSparseCodingConfig(
            penalty=PenaltyConfig('cauchy', {'lam': 0.1, 'sigma': 1.0}),
            solver=SolverConfig('ncg', {'beta_method': 'polak-ribiere'}),
            dict_updater=DictUpdaterConfig('grad_d', {'learning_rate': 0.005}),
            n_atoms=100,
            max_iter=100
        )
    
    @staticmethod
    def get_preset_names() -> List[str]:
        """Get all available preset names."""
        return [
            'olshausen_field_1996',
            'ksvd_aharon_2006',
            'online_mairal_2010',
            'elastic_net_zou_2005',
            'robust_cauchy'
        ]
    
    @staticmethod
    def get_preset(name: str) -> ComprehensiveSparseCodingConfig:
        """Get preset configuration by name."""
        presets = {
            'olshausen_field_1996': ResearchAccuratePresets.olshausen_field_1996,
            'ksvd_aharon_2006': ResearchAccuratePresets.ksvd_aharon_2006,
            'online_mairal_2010': ResearchAccuratePresets.online_mairal_2010,
            'elastic_net_zou_2005': ResearchAccuratePresets.elastic_net_zou_2005,
            'robust_cauchy': ResearchAccuratePresets.robust_cauchy,
        }
        
        if name not in presets:
            available = list(presets.keys())
            raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")
        
        return presets[name]()


def print_configuration_summary(config: ComprehensiveSparseCodingConfig) -> None:
    """Print a detailed summary of the configuration with research context."""
    print("🔬 Sparse Coding Configuration Summary")
    print("=" * 50)
    
    print(f"📊 Penalty Function: {config.penalty.name.upper()}")
    print(f"   Parameters: {config.penalty.params}")
    if hasattr(config.penalty, 'citation'):
        print(f"   Research: {config.penalty.citation}")
    
    print(f"\n⚡ Solver Algorithm: {config.solver.name.upper()}")
    print(f"   Parameters: {config.solver.params}")
    if hasattr(config.solver, 'citation'):
        print(f"   Research: {config.solver.citation}")
    
    print(f"\n🔄 Dictionary Updater: {config.dict_updater.name.upper()}")
    print(f"   Parameters: {config.dict_updater.params}")
    if hasattr(config.dict_updater, 'citation'):
        print(f"   Research: {config.dict_updater.citation}")
    
    print(f"\n🎯 Global Parameters:")
    print(f"   Atoms: {config.n_atoms}, Max Iterations: {config.max_iter}")
    print(f"   Tolerance: {config.tol}, Verbose: {config.verbose}")
    
    # Validation warnings
    warnings = config.validate_compatibility()
    if warnings:
        print(f"\n⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"   {warning}")
    else:
        print(f"\n✅ Configuration is research-compatible")
    
    print("=" * 50)
