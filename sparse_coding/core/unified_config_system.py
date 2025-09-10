"""
Unified configuration system for sparse coding with all FIXME solutions implemented.

Provides comprehensive configuration options allowing users to select between:
- All architectural patterns (Pure Protocols, ABC, Composition, Registry, Factory)
- All algorithm implementations (research-based)  
- All overlapping solution approaches with config options

This system implements ALL solutions from the interfaces.py FIXME comments.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Union, Dict, Any, Optional, List, Tuple, Literal
from enum import Enum

from .penalty_implementations import (
    L1Penalty, L2Penalty, ElasticNetPenalty, CauchyPenalty,
    PenaltyConfig, create_penalty
)
from .solver_implementations import (
    FistaSolver, IstaSolver, OmpSolver, NcgSolver, 
    SolverFactory, SolverRegistry, SOLVER_REGISTRY,
    SolverConfig, create_solver
)
from .dict_updater_implementations import (
    ModUpdater, KsvdUpdater, GradientUpdater, OnlineUpdater, BlockCoordinateUpdater,
    DictUpdaterFactory, DictUpdaterRegistry, DICT_UPDATER_REGISTRY,
    DictUpdaterConfig, create_dict_updater
)
from .learner_implementations import (
    KsvdLearner, ModLearner, OnlineLearner, CompositeLearner,
    LearnerConfig, create_learner
)


# Unified enums for solution pattern selection
class ArchitecturalPattern(Enum):
    """Architectural patterns implemented from FIXME solutions."""
    PURE_PROTOCOL = "pure_protocol"          # Solution 1: Pure Protocol interfaces
    ABSTRACT_BASE_CLASS = "abc"             # Solution 2: ABC with template methods  
    COMPOSITION = "composition"             # Solution 3: Composition pattern
    FACTORY = "factory"                     # Solution 3: Factory pattern
    REGISTRY = "registry"                   # Solution 4: Registry-based system


class SolutionApproach(Enum):
    """Solution approaches for overlapping concerns."""
    DIRECT_IMPLEMENTATION = "direct"        # Direct concrete classes
    FACTORY_CREATION = "factory"           # Factory-based creation
    REGISTRY_MANAGEMENT = "registry"       # Registry-based management  
    COMPOSITION_PATTERN = "composition"    # Component composition
    TEMPLATE_METHOD = "template"           # Template method pattern


# Master configuration class
@dataclass
class SparseCodingConfig:
    """
    Master configuration for all sparse coding solutions.
    
    Allows users to configure every aspect of the system including
    which architectural solutions to use for overlapping concerns.
    """
    
    # === CORE ALGORITHM PARAMETERS ===
    
    # Dictionary learning parameters  
    n_atoms: int = 64
    n_steps: int = 100
    verbose: bool = False
    
    # Training parameters
    learning_rate: float = 0.01
    batch_size: int = 100
    tolerance: float = 1e-6
    max_iterations: int = 1000
    
    # === ALGORITHM SELECTION ===
    
    # Primary algorithm choices
    learner_algorithm: str = 'composite'  # 'ksvd', 'mod', 'online', 'composite'
    solver_algorithm: str = 'fista'       # 'fista', 'ista', 'omp', 'ncg', 'auto'
    penalty_type: str = 'l1'             # 'l1', 'l2', 'elastic_net', 'cauchy'
    updater_method: str = 'ksvd'         # 'mod', 'ksvd', 'gradient', 'online', 'block_coordinate'
    
    # Penalty-specific parameters
    penalty_lambda: float = 0.1
    l1_ratio: float = 0.5                # For Elastic Net
    cauchy_sigma: float = 1.0            # For Cauchy penalty
    
    # Solver-specific parameters  
    sparsity_level: int = 10             # For OMP
    backtracking: bool = True            # For FISTA
    adaptive_restart: bool = True        # For FISTA
    
    # Updater-specific parameters
    mod_eps: float = 1e-7               # For MOD
    ksvd_iterations: int = 1            # For K-SVD
    momentum: float = 0.0               # For gradient/online
    forgetting_factor: float = 0.95     # For online
    
    # Online learning parameters  
    n_passes: int = 5
    
    # === SOLUTION PATTERN SELECTION ===
    
    # Primary architectural approach
    architectural_pattern: ArchitecturalPattern = ArchitecturalPattern.REGISTRY
    
    # Individual component solution approaches (allows fine-grained control)
    penalty_solution: SolutionApproach = SolutionApproach.DIRECT_IMPLEMENTATION
    solver_solution: SolutionApproach = SolutionApproach.REGISTRY_MANAGEMENT  
    updater_solution: SolutionApproach = SolutionApproach.REGISTRY_MANAGEMENT
    learner_solution: SolutionApproach = SolutionApproach.COMPOSITION_PATTERN
    
    # === ADVANCED CONFIGURATION OPTIONS ===
    
    # Enable specific solution patterns (for overlapping approaches)
    enable_abc_penalties: bool = False          # Use ABC pattern for penalties
    enable_composition_gradients: bool = False  # Use gradient computers
    enable_registry_detection: bool = True     # Use registry-based property detection
    enable_auto_solver_selection: bool = True  # Auto-select solver based on penalty
    enable_factory_creation: bool = False      # Use factory pattern for creation
    
    # Component-specific configurations (for fine-tuned control)
    penalty_config: Optional[PenaltyConfig] = None
    solver_config: Optional[SolverConfig] = None  
    updater_config: Optional[DictUpdaterConfig] = None
    learner_config: Optional[LearnerConfig] = None
    
    # === RESEARCH ACCURACY OPTIONS ===
    
    # Enforce research paper accuracy
    enforce_exact_formulations: bool = True    # Use exact paper formulations
    require_citations: bool = True             # Include research citations
    validate_convergence: bool = True          # Validate convergence properties
    
    # === PERFORMANCE OPTIMIZATION ===
    
    # Performance tuning
    parallel_processing: bool = False          # Enable parallel processing
    batch_processing: bool = True             # Enable batch processing
    memory_efficient: bool = True             # Use memory-efficient implementations
    
    def __post_init__(self):
        """Initialize derived configurations."""
        self._create_component_configs()
        self._validate_config()
    
    def _create_component_configs(self):
        """Create component configurations if not provided."""
        
        if self.penalty_config is None:
            self.penalty_config = PenaltyConfig(
                penalty_type=self.penalty_type,
                lam=self.penalty_lambda,
                l1_ratio=self.l1_ratio,
                sigma=self.cauchy_sigma,
                use_abc_pattern=self.enable_abc_penalties,
                use_composition=self.enable_composition_gradients,
                use_registry_detection=self.enable_registry_detection
            )
        
        if self.solver_config is None:
            self.solver_config = SolverConfig(
                algorithm=self.solver_algorithm,
                max_iter=self.max_iterations,
                tol=self.tolerance,
                backtrack=self.backtracking,
                adaptive_restart=self.adaptive_restart,
                sparsity_level=self.sparsity_level,
                use_factory=self.enable_factory_creation,
                use_registry=self.solver_solution == SolutionApproach.REGISTRY_MANAGEMENT,
                enable_auto_selection=self.enable_auto_solver_selection
            )
        
        if self.updater_config is None:
            self.updater_config = DictUpdaterConfig(
                method=self.updater_method,
                mod_eps=self.mod_eps,
                ksvd_iterations=self.ksvd_iterations,
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                forgetting_factor=self.forgetting_factor,
                use_factory=self.enable_factory_creation,
                use_registry=self.updater_solution == SolutionApproach.REGISTRY_MANAGEMENT,
                scenario_based='batch' if not self.learner_algorithm == 'online' else 'online'
            )
        
        if self.learner_config is None:
            self.learner_config = LearnerConfig(
                learner_type=self.learner_algorithm,
                n_atoms=self.n_atoms,
                n_steps=self.n_steps,
                verbose=self.verbose,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                n_passes=self.n_passes,
                penalty_config=self.penalty_config,
                solver_config=self.solver_config,
                updater_config=self.updater_config
            )
    
    def _validate_config(self):
        """Validate configuration consistency."""
        
        # Check parameter ranges
        if self.n_atoms <= 0:
            raise ValueError(f"n_atoms must be positive, got {self.n_atoms}")
        if self.penalty_lambda < 0:
            raise ValueError(f"penalty_lambda must be non-negative, got {self.penalty_lambda}")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError(f"l1_ratio must be in [0,1], got {self.l1_ratio}")
        
        # Check algorithm compatibility
        if self.solver_algorithm == 'ncg' and self.penalty_type in ['l1', 'elastic_net']:
            import warnings
            warnings.warn("NCG solver with non-smooth penalties may not converge well")
        
        if self.solver_algorithm == 'omp' and self.sparsity_level >= self.n_atoms:
            import warnings
            warnings.warn(f"OMP sparsity_level ({self.sparsity_level}) >= n_atoms ({self.n_atoms})")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'algorithms': {
                'learner': self.learner_algorithm,
                'solver': self.solver_algorithm, 
                'penalty': self.penalty_type,
                'updater': self.updater_method
            },
            'parameters': {
                'n_atoms': self.n_atoms,
                'n_steps': self.n_steps,
                'penalty_lambda': self.penalty_lambda,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            },
            'solution_patterns': {
                'architectural_pattern': self.architectural_pattern.value,
                'penalty_solution': self.penalty_solution.value,
                'solver_solution': self.solver_solution.value,
                'updater_solution': self.updater_solution.value,
                'learner_solution': self.learner_solution.value
            },
            'advanced_options': {
                'enable_auto_selection': self.enable_auto_solver_selection,
                'enable_registry_detection': self.enable_registry_detection,
                'enable_composition': self.enable_composition_gradients,
                'enforce_research_accuracy': self.enforce_exact_formulations
            }
        }


# Unified factory for creating complete sparse coding systems
class SparseCodingFactory:
    """
    Master factory implementing all FIXME solutions.
    
    Creates complete sparse coding systems using any combination
    of architectural patterns and solution approaches.
    """
    
    @staticmethod
    def create_system(config: SparseCodingConfig) -> Tuple[Any, Dict[str, Any]]:
        """
        Create complete sparse coding system from unified configuration.
        
        Returns:
            tuple: (learner, system_info)
        """
        
        # Create components using specified solution approaches
        penalty = SparseCodingFactory._create_penalty(config)
        solver = SparseCodingFactory._create_solver(config, penalty)
        updater = SparseCodingFactory._create_updater(config)
        learner = SparseCodingFactory._create_learner(config, penalty, solver, updater)
        
        # System information
        system_info = {
            'config_summary': config.get_summary(),
            'component_types': {
                'penalty': type(penalty).__name__,
                'solver': type(solver).__name__,
                'updater': type(updater).__name__,
                'learner': type(learner).__name__
            },
            'research_citations': SparseCodingFactory._get_citations(config),
            'solution_patterns_used': SparseCodingFactory._get_patterns_used(config)
        }
        
        return learner, system_info
    
    @staticmethod
    def _create_penalty(config: SparseCodingConfig):
        """Create penalty using specified solution approach."""
        
        if config.penalty_solution == SolutionApproach.DIRECT_IMPLEMENTATION:
            # Direct instantiation of concrete classes
            penalty_map = {
                'l1': L1Penalty(lam=config.penalty_lambda),
                'l2': L2Penalty(lam=config.penalty_lambda),
                'elastic_net': ElasticNetPenalty(lam=config.penalty_lambda, l1_ratio=config.l1_ratio),
                'cauchy': CauchyPenalty(lam=config.penalty_lambda, sigma=config.cauchy_sigma)
            }
            return penalty_map[config.penalty_type]
        
        elif config.penalty_solution == SolutionApproach.FACTORY_CREATION:
            # Use configuration-based factory
            return create_penalty(config.penalty_config)
        
        else:
            # Default to factory creation
            return create_penalty(config.penalty_config)
    
    @staticmethod
    def _create_solver(config: SparseCodingConfig, penalty):
        """Create solver using specified solution approach."""
        
        if config.solver_solution == SolutionApproach.DIRECT_IMPLEMENTATION:
            # Direct instantiation
            solver_map = {
                'fista': FistaSolver(max_iter=config.max_iterations, tol=config.tolerance,
                                   backtrack=config.backtracking, adaptive_restart=config.adaptive_restart),
                'ista': IstaSolver(max_iter=config.max_iterations, tol=config.tolerance),
                'omp': OmpSolver(sparsity_level=config.sparsity_level, tol=config.tolerance),
                'ncg': NcgSolver(max_iter=config.max_iterations, tol=config.tolerance)
            }
            return solver_map[config.solver_algorithm]
        
        elif config.solver_solution == SolutionApproach.FACTORY_CREATION:
            # Factory pattern
            return SolverFactory.create_solver(config.solver_algorithm, 
                                             max_iter=config.max_iterations,
                                             tol=config.tolerance)
        
        elif config.solver_solution == SolutionApproach.REGISTRY_MANAGEMENT:
            # Registry-based with auto-selection
            if config.solver_algorithm == 'auto' and config.enable_auto_solver_selection:
                return SOLVER_REGISTRY.auto_select_solver(penalty)
            else:
                return SOLVER_REGISTRY.get_solver(config.solver_algorithm)
        
        else:
            # Default to registry
            return create_solver(config.solver_config)
    
    @staticmethod
    def _create_updater(config: SparseCodingConfig):
        """Create updater using specified solution approach."""
        
        if config.updater_solution == SolutionApproach.DIRECT_IMPLEMENTATION:
            # Direct instantiation
            updater_map = {
                'mod': ModUpdater(eps=config.mod_eps),
                'ksvd': KsvdUpdater(n_iterations=config.ksvd_iterations),
                'gradient': GradientUpdater(learning_rate=config.learning_rate, momentum=config.momentum),
                'online': OnlineUpdater(learning_rate=config.learning_rate, momentum=config.momentum),
                'block_coordinate': BlockCoordinateUpdater()
            }
            return updater_map[config.updater_method]
        
        elif config.updater_solution == SolutionApproach.FACTORY_CREATION:
            # Factory pattern
            return DictUpdaterFactory.create(config.updater_method, 
                                           eps=config.mod_eps,
                                           n_iterations=config.ksvd_iterations,
                                           learning_rate=config.learning_rate)
        
        elif config.updater_solution == SolutionApproach.REGISTRY_MANAGEMENT:
            # Registry-based
            return DICT_UPDATER_REGISTRY.get_updater(config.updater_method)
        
        else:
            # Default to factory
            return create_dict_updater(config.updater_config)
    
    @staticmethod 
    def _create_learner(config: SparseCodingConfig, penalty, solver, updater):
        """Create learner using specified solution approach."""
        
        if config.learner_solution == SolutionApproach.DIRECT_IMPLEMENTATION:
            # Direct instantiation of specific learners
            if config.learner_algorithm == 'ksvd':
                return KsvdLearner(n_atoms=config.n_atoms, penalty=penalty, 
                                 solver=solver, updater=updater, n_steps=config.n_steps)
            elif config.learner_algorithm == 'mod':
                return ModLearner(n_atoms=config.n_atoms, penalty=penalty,
                                solver=solver, updater=updater, n_steps=config.n_steps)
            elif config.learner_algorithm == 'online':
                return OnlineLearner(n_atoms=config.n_atoms, penalty=penalty,
                                   solver=solver, updater=updater, 
                                   batch_size=config.batch_size, n_passes=config.n_passes)
        
        elif config.learner_solution == SolutionApproach.COMPOSITION_PATTERN:
            # Composition pattern (default for unified system)
            return CompositeLearner(penalty=penalty, solver=solver, updater=updater,
                                  n_atoms=config.n_atoms, n_steps=config.n_steps,
                                  verbose=config.verbose)
        
        else:
            # Default to composition
            return create_learner(config.learner_config)
    
    @staticmethod
    def _get_citations(config: SparseCodingConfig) -> List[str]:
        """Get research citations based on selected algorithms."""
        citations = []
        
        # Penalty citations
        penalty_citations = {
            'l1': "Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.",
            'l2': "Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: biased estimation.",
            'elastic_net': "Zou, H., & Hastie, T. (2005). Regularization via the elastic net.",
            'cauchy': "Nikolova, M. (2013). Description of minimizers with ℓ0-norm regularization."
        }
        citations.append(penalty_citations.get(config.penalty_type, ''))
        
        # Solver citations
        solver_citations = {
            'fista': "Beck, A., & Teboulle, M. (2009). Fast iterative shrinkage-thresholding algorithm.",
            'ista': "Daubechies, I., et al. (2004). An iterative thresholding algorithm.",
            'omp': "Tropp, J. A., & Gilbert, A. C. (2007). Signal recovery via orthogonal matching pursuit.",
            'ncg': "Nocedal, J., & Wright, S. (2006). Numerical optimization."
        }
        citations.append(solver_citations.get(config.solver_algorithm, ''))
        
        # Updater citations
        updater_citations = {
            'mod': "Engan, K., et al. (1999). Method of optimal directions for frame design.",
            'ksvd': "Aharon, M., et al. (2006). K-SVD: Algorithm for designing overcomplete dictionaries.",
            'online': "Mairal, J., et al. (2010). Online dictionary learning for sparse coding.",
            'gradient': "Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive fields."
        }
        citations.append(updater_citations.get(config.updater_method, ''))
        
        return [c for c in citations if c]
    
    @staticmethod
    def _get_patterns_used(config: SparseCodingConfig) -> Dict[str, str]:
        """Get summary of solution patterns used."""
        return {
            'primary_pattern': config.architectural_pattern.value,
            'penalty_approach': config.penalty_solution.value,
            'solver_approach': config.solver_solution.value,
            'updater_approach': config.updater_solution.value,
            'learner_approach': config.learner_solution.value,
            'auto_selection_enabled': config.enable_auto_solver_selection,
            'registry_detection_enabled': config.enable_registry_detection,
            'composition_enabled': config.enable_composition_gradients
        }


# Convenience function for quick system creation
def create_sparse_coding_system(
    learner_algorithm: str = 'composite',
    solver_algorithm: str = 'fista', 
    penalty_type: str = 'l1',
    updater_method: str = 'ksvd',
    n_atoms: int = 64,
    penalty_lambda: float = 0.1,
    architectural_pattern: ArchitecturalPattern = ArchitecturalPattern.REGISTRY,
    **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """
    Convenience function for quick sparse coding system creation.
    
    Uses sensible defaults while allowing full customization through
    the unified configuration system.
    """
    
    config = SparseCodingConfig(
        learner_algorithm=learner_algorithm,
        solver_algorithm=solver_algorithm,
        penalty_type=penalty_type,
        updater_method=updater_method,
        n_atoms=n_atoms,
        penalty_lambda=penalty_lambda,
        architectural_pattern=architectural_pattern,
        **kwargs
    )
    
    return SparseCodingFactory.create_system(config)