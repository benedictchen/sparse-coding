from .__about__ import __version__

# Core implementations (backwards compatible)
from .sparse_coder import SparseCoder
from .dictionary_learner import DictionaryLearner

from .core.penalties import (
    L1Penalty, L2Penalty, ElasticNetPenalty, CauchyPenalty,
    PenaltyProtocol
)

from .core.solver_implementations import (
    FistaSolver, IstaSolver, OmpSolver, NcgSolver,
    SolverFactory, SolverRegistry, SOLVER_REGISTRY,
    SolverConfig, create_solver
)

from .core.dict_updater_implementations import (
    ModUpdater, KsvdUpdater, GradientUpdater, OnlineUpdater, BlockCoordinateUpdater,
    DictUpdaterFactory, DictUpdaterRegistry, DICT_UPDATER_REGISTRY,
    DictUpdaterConfig, create_dict_updater
)

from .core.learner_implementations import (
    KsvdLearner, ModLearner, OnlineLearner, CompositeLearner,
    LearnerConfig, create_learner
)

from .core.unified_config_system import (
    SparseCodingConfig, SparseCodingFactory, ArchitecturalPattern, SolutionApproach,
    create_sparse_coding_system
)

# Advanced optimization and proximal operators
from .proximal_gradient_optimization import (
    L1Proximal, ElasticNetProximal, ProximalGradientOptimizer
)

# Alias for backwards compatibility and test requirements
AdvancedOptimizer = ProximalGradientOptimizer

# Essential functionality - individual imports work fine
# Note: penalties and core are available via their individual components above

__all__ = [
    # Version
    "__version__",
    
    # Core API
    "SparseCoder", 
    "DictionaryLearner",
    
    # Penalty functions
    "L1Penalty", "L2Penalty", "ElasticNetPenalty", "CauchyPenalty",
    "PenaltyABC", "PenaltyConfig", "create_penalty",
    
    # Solvers
    "FistaSolver", "IstaSolver", "OmpSolver", "NcgSolver",
    "SolverFactory", "SolverRegistry", "SOLVER_REGISTRY", 
    "SolverConfig", "create_solver",
    
    # Dictionary updaters
    "ModUpdater", "KsvdUpdater", "GradientUpdater", "OnlineUpdater", "BlockCoordinateUpdater",
    "DictUpdaterFactory", "DictUpdaterRegistry", "DICT_UPDATER_REGISTRY",
    "DictUpdaterConfig", "create_dict_updater",
    
    # Learners
    "KsvdLearner", "ModLearner", "OnlineLearner", "CompositeLearner",
    "LearnerConfig", "create_learner",
    
    # Configuration system
    "SparseCodingConfig", "SparseCodingFactory", 
    "ArchitecturalPattern", "SolutionApproach",
    "create_sparse_coding_system",
    
    # Proximal operators
    "L1Proximal", "ElasticNetProximal", "ProximalGradientOptimizer", "AdvancedOptimizer",
]
