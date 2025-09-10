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

# Essential functionality
try:
    from . import penalties
except ImportError:
    penalties = None

try:
    from . import core
except ImportError:
    core = None

__all__ = [
    # Version
    "__version__",
    
    # Backwards compatible API
    "SparseCoder", 
    "DictionaryLearner",
    
    "L1Penalty", "L2Penalty", "ElasticNetPenalty", "CauchyPenalty",
    "PenaltyABC", "PenaltyConfig", "create_penalty",
    
    # Solver system with all solutions
    "FistaSolver", "IstaSolver", "OmpSolver", "NcgSolver",
    "SolverFactory", "SolverRegistry", "SOLVER_REGISTRY", 
    "SolverConfig", "create_solver",
    
    # Dictionary updater system with all solutions  
    "ModUpdater", "KsvdUpdater", "GradientUpdater", "OnlineUpdater", "BlockCoordinateUpdater",
    "DictUpdaterFactory", "DictUpdaterRegistry", "DICT_UPDATER_REGISTRY",
    "DictUpdaterConfig", "create_dict_updater",
    
    # Learner system with all solutions
    "KsvdLearner", "ModLearner", "OnlineLearner", "CompositeLearner",
    "LearnerConfig", "create_learner",
    
    # Unified configuration system (master solution)
    "SparseCodingConfig", "SparseCodingFactory", 
    "ArchitecturalPattern", "SolutionApproach",
    "create_sparse_coding_system",
    
    # Legacy modules
    "penalties", "core",
]
