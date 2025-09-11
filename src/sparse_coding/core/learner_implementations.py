"""
Research-accurate learner implementations for sparse coding.

Implements sparse coding learner patterns:
1. Pure Protocol Interface - defined in interfaces.py
2. Concrete Learner Implementations
3. Composite Learner Pattern

All implementations follow original research papers with exact mathematical formulations.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, Dict, Any, Optional, List, Tuple
from .array import ArrayLike, ensure_array
from .interfaces import Learner, StreamingLearner, Penalty, InferenceSolver, DictUpdater
from .penalties import L1Penalty, PenaltyProtocol
from .solver_implementations import FistaSolver, create_solver, SolverConfig, SOLVER_REGISTRY
from .dict_updater_implementations import KsvdUpdater, create_dict_updater, DictUpdaterConfig, DICT_UPDATER_REGISTRY


# SOLUTION 2: Concrete Learner Implementations

@dataclass
class KsvdLearner:
    """
    K-SVD dictionary learning algorithm (Aharon et al., 2006).
    
    Alternates between sparse coding (using OMP or FISTA) and dictionary 
    update (using K-SVD) to learn overcomplete dictionaries.
    
    Reference:
    Aharon, M., Elad, M., & Bruckstein, A. (2006). K-SVD: An algorithm 
    for designing overcomplete dictionaries for sparse representation.
    IEEE Transactions on Signal Processing, 54(11), 4311-4322.
    """
    n_atoms: int = 64
    penalty: Optional[Penalty] = None
    solver: Optional[InferenceSolver] = None
    updater: Optional[DictUpdater] = None
    n_steps: int = 100
    verbose: bool = False
    
    # Internal state
    _dictionary: Optional[ArrayLike] = field(default=None, init=False)
    _is_fitted: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Initialize default components."""
        if self.penalty is None:
            self.penalty = L1Penalty(lam=0.1)
        if self.solver is None:
            # K-SVD traditionally uses OMP for sparse coding
            self.solver = create_solver(SolverConfig(algorithm='omp', sparsity_level=10))
        if self.updater is None:
            self.updater = KsvdUpdater(n_iterations=1)
    
    def fit(self, X: ArrayLike, **kwargs) -> 'KsvdLearner':
        """
        Learn dictionary using K-SVD algorithm.
        
        K-SVD alternating minimization (Aharon et al., 2006):
        1. Sparse coding: A^(t) = argmin_A [0.5||X - D^(t-1)A||_F^2 + penalty(A)]
        2. Dictionary update: D^(t) = K-SVD_update(D^(t-1), X, A^(t))
        """
        X = ensure_array(X)
        n_features, n_samples = X.shape
        
        # Override parameters from kwargs
        n_steps = kwargs.get('n_steps', self.n_steps)
        verbose = kwargs.get('verbose', self.verbose)
        
        # Initialize dictionary if not exists
        if self._dictionary is None:
            self._init_dictionary(X, self.n_atoms)
        
        # K-SVD alternating minimization
        for step in range(n_steps):
            if verbose and step % 10 == 0:
                self._print_progress(step, n_steps, "K-SVD")
            
            # Sparse coding step  
            codes = self.solver.solve(self._dictionary, X, self.penalty)
            
            # Dictionary update step using K-SVD
            self._dictionary = self.updater.step(self._dictionary, X, codes)
            
            # Compute and log objective if verbose
            if verbose and step % 10 == 0:
                objective = self._compute_objective(X, codes)
                print(f"  Objective: {objective:.6f}")
        
        self._is_fitted = True
        return self
    
    def encode(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """Encode data using learned dictionary."""
        if not self._is_fitted:
            raise ValueError("Learner not fitted. Call fit() first.")
        
        X = ensure_array(X)
        return self.solver.solve(self._dictionary, X, self.penalty, **kwargs)
    
    def decode(self, A: ArrayLike) -> ArrayLike:
        """Decode sparse codes: X̂ = DA"""
        if not self._is_fitted:
            raise ValueError("Learner not fitted. Call fit() first.")
        
        A = ensure_array(A)
        return self._dictionary @ A
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        """Learned dictionary matrix."""
        return self._dictionary
    
    def get_config(self) -> Dict[str, Any]:
        """Get learner configuration."""
        return {
            'class_name': 'KsvdLearner',
            'n_atoms': self.n_atoms,
            'n_features': self._dictionary.shape[0] if self._dictionary is not None else None,
            'penalty_type': type(self.penalty).__name__,
            'penalty_lam': getattr(self.penalty, 'lam', None),
            'solver_type': type(self.solver).__name__,
            'n_steps': self.n_steps,
            'is_fitted': self._is_fitted
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set learner configuration.""" 
        self.n_atoms = config.get('n_atoms', self.n_atoms)
        self.n_steps = config.get('n_steps', self.n_steps)
        self._is_fitted = config.get('is_fitted', False)
        
        # Recreate penalty if specified
        if 'penalty_type' in config and 'penalty_lam' in config:
            penalty_map = {'L1Penalty': 'l1', 'L2Penalty': 'l2'}
            penalty_type = penalty_map.get(config['penalty_type'], 'l1')
            self.penalty = create_penalty(PenaltyConfig(penalty_type=penalty_type, lam=config['penalty_lam']))
    
    def _init_dictionary(self, X: ArrayLike, n_atoms: int):
        """Initialize dictionary from random data samples."""
        n_features, n_samples = X.shape
        indices = np.random.choice(n_samples, min(n_atoms, n_samples), replace=False)
        self._dictionary = X[:, indices].copy()
        
        # Normalize columns
        norms = np.linalg.norm(self._dictionary, axis=0, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        self._dictionary = self._dictionary / norms
    
    def _compute_objective(self, X: ArrayLike, A: ArrayLike) -> float:
        """Compute objective function value."""
        residual = X - self._dictionary @ A
        data_fidelity = 0.5 * np.sum(residual**2)
        penalty_term = self.penalty.value(A)
        return data_fidelity + penalty_term
    
    def _print_progress(self, step: int, total: int, algorithm: str):
        """Print training progress."""
        print(f"{algorithm} iteration {step}/{total}")


@dataclass  
class ModLearner:
    """
    MOD dictionary learning (Engan et al., 1999).
    
    Uses Method of Optimal Directions for dictionary updates with 
    FISTA or ISTA for sparse coding step.
    
    Reference:
    Engan, K., Aase, S. O., & Husøy, J. H. (1999). Method of optimal 
    directions for frame design. ICASSP, Vol. 5, pp. 2443-2446.
    """
    n_atoms: int = 64
    penalty: Optional[Penalty] = None  
    solver: Optional[InferenceSolver] = None
    updater: Optional[DictUpdater] = None
    n_steps: int = 100
    verbose: bool = False
    
    _dictionary: Optional[ArrayLike] = field(default=None, init=False)
    _is_fitted: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Initialize default components."""
        if self.penalty is None:
            self.penalty = L1Penalty(lam=0.1)
        if self.solver is None:
            # MOD typically uses FISTA for sparse coding
            self.solver = FistaSolver(max_iter=200, tol=1e-6)
        if self.updater is None:
            from .dict_updater_implementations import ModUpdater
            self.updater = ModUpdater(eps=1e-7)
    
    def fit(self, X: ArrayLike, **kwargs) -> 'ModLearner':
        """
        Learn dictionary using MOD algorithm.
        
        MOD alternating optimization (Engan et al., 1999):
        1. Sparse coding: A^(t) = argmin_A [0.5||X - D^(t-1)A||_F^2 + penalty(A)]  
        2. Dictionary update: D^(t) = argmin_D ||X - DA^(t)||_F^2 (closed form)
        """
        X = ensure_array(X)
        n_features, n_samples = X.shape
        
        n_steps = kwargs.get('n_steps', self.n_steps)
        verbose = kwargs.get('verbose', self.verbose)
        
        if self._dictionary is None:
            self._init_dictionary(X, self.n_atoms)
        
        # MOD alternating minimization
        for step in range(n_steps):
            if verbose and step % 10 == 0:
                self._print_progress(step, n_steps, "MOD")
            
            # Sparse coding step
            codes = self.solver.solve(self._dictionary, X, self.penalty)
            
            # MOD dictionary update (closed-form)
            self._dictionary = self.updater.step(self._dictionary, X, codes)
            
            if verbose and step % 10 == 0:
                objective = self._compute_objective(X, codes)  
                print(f"  Objective: {objective:.6f}")
        
        self._is_fitted = True
        return self
    
    def encode(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """Encode data using learned dictionary."""
        if not self._is_fitted:
            raise ValueError("Learner not fitted. Call fit() first.")
        
        X = ensure_array(X)
        return self.solver.solve(self._dictionary, X, self.penalty, **kwargs)
    
    def decode(self, A: ArrayLike) -> ArrayLike:
        """Decode sparse codes."""
        if not self._is_fitted:
            raise ValueError("Learner not fitted. Call fit() first.")
        
        A = ensure_array(A)
        return self._dictionary @ A
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        """Learned dictionary matrix."""
        return self._dictionary
    
    def get_config(self) -> Dict[str, Any]:
        """Get learner configuration."""
        return {
            'class_name': 'ModLearner',
            'n_atoms': self.n_atoms,
            'n_features': self._dictionary.shape[0] if self._dictionary is not None else None,
            'penalty_type': type(self.penalty).__name__,
            'penalty_lam': getattr(self.penalty, 'lam', None),
            'solver_type': type(self.solver).__name__,
            'n_steps': self.n_steps,
            'is_fitted': self._is_fitted
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set learner configuration."""
        self.n_atoms = config.get('n_atoms', self.n_atoms)
        self.n_steps = config.get('n_steps', self.n_steps)
        self._is_fitted = config.get('is_fitted', False)
    
    def _init_dictionary(self, X: ArrayLike, n_atoms: int):
        """Initialize dictionary from random data samples.""" 
        n_features, n_samples = X.shape
        indices = np.random.choice(n_samples, min(n_atoms, n_samples), replace=False)
        self._dictionary = X[:, indices].copy()
        
        norms = np.linalg.norm(self._dictionary, axis=0, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        self._dictionary = self._dictionary / norms
    
    def _compute_objective(self, X: ArrayLike, A: ArrayLike) -> float:
        """Compute objective function value."""
        residual = X - self._dictionary @ A
        data_fidelity = 0.5 * np.sum(residual**2)
        penalty_term = self.penalty.value(A)
        return data_fidelity + penalty_term
    
    def _print_progress(self, step: int, total: int, algorithm: str):
        """Print training progress."""
        print(f"{algorithm} iteration {step}/{total}")


@dataclass
class OnlineLearner:
    """
    Online dictionary learning (Mairal et al., 2010).
    
    Processes data in mini-batches with streaming dictionary updates.
    Suitable for large-scale datasets that don't fit in memory.
    
    Reference:
    Mairal, J., Bach, F., Ponce, J., & Sapiro, G. (2010). Online dictionary 
    learning for sparse coding. ICML, pp. 689-696.
    """
    n_atoms: int = 64
    penalty: Optional[Penalty] = None
    solver: Optional[InferenceSolver] = None
    updater: Optional[DictUpdater] = None
    batch_size: int = 100
    learning_rate: float = 0.01
    n_passes: int = 5
    verbose: bool = False
    
    _dictionary: Optional[ArrayLike] = field(default=None, init=False)
    _is_fitted: bool = field(default=False, init=False)
    _n_samples_seen: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Initialize default components."""
        if self.penalty is None:
            self.penalty = L1Penalty(lam=0.1)
        if self.solver is None:
            self.solver = FistaSolver(max_iter=100, tol=1e-4)
        if self.updater is None:
            from .dict_updater_implementations import OnlineUpdater
            self.updater = OnlineUpdater(learning_rate=self.learning_rate)
    
    def fit(self, X: ArrayLike, **kwargs) -> 'OnlineLearner':
        """
        Online dictionary learning with mini-batch processing.
        
        Processes data in mini-batches for scalable learning.
        """
        X = ensure_array(X)
        n_features, n_samples = X.shape
        
        batch_size = kwargs.get('batch_size', self.batch_size)
        n_passes = kwargs.get('n_passes', self.n_passes)  
        verbose = kwargs.get('verbose', self.verbose)
        
        if self._dictionary is None:
            self._init_dictionary(X, self.n_atoms)
        
        # Online learning passes over data
        for pass_idx in range(n_passes):
            if verbose:
                print(f"Online pass {pass_idx + 1}/{n_passes}")
            
            # Shuffle data for each pass
            indices = np.random.permutation(n_samples)
            X_shuffled = X[:, indices]
            
            # Process mini-batches  
            n_batches = (n_samples + batch_size - 1) // batch_size
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                X_batch = X_shuffled[:, start_idx:end_idx]
                
                # Sparse coding for batch
                codes_batch = self.solver.solve(self._dictionary, X_batch, self.penalty)
                
                # Online dictionary update
                self._dictionary = self.updater.step(self._dictionary, X_batch, codes_batch)
                
                self._n_samples_seen += X_batch.shape[1]
                
                if verbose and batch_idx % 50 == 0:
                    objective = self._compute_objective(X_batch, codes_batch)
                    print(f"  Batch {batch_idx}/{n_batches}, Objective: {objective:.6f}")
        
        self._is_fitted = True
        return self
    
    def partial_fit(self, X: ArrayLike, **kwargs) -> 'OnlineLearner':
        """Incremental learning step for streaming data."""
        X = ensure_array(X) 
        
        if self._dictionary is None:
            self._init_dictionary(X, self.n_atoms)
        
        # Single online update step
        codes = self.solver.solve(self._dictionary, X, self.penalty)
        self._dictionary = self.updater.step(self._dictionary, X, codes)
        
        self._n_samples_seen += X.shape[1]
        self._is_fitted = True
        
        return self
    
    def encode(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """Encode data using learned dictionary."""
        if not self._is_fitted:
            raise ValueError("Learner not fitted. Call fit() first.")
        
        X = ensure_array(X)
        return self.solver.solve(self._dictionary, X, self.penalty, **kwargs)
    
    def decode(self, A: ArrayLike) -> ArrayLike:
        """Decode sparse codes."""
        if not self._is_fitted:
            raise ValueError("Learner not fitted. Call fit() first.")
        
        A = ensure_array(A)
        return self._dictionary @ A
    
    def reset(self) -> None:
        """Reset online learner state."""
        self._dictionary = None
        self._n_samples_seen = 0
        self._is_fitted = False
        if hasattr(self.updater, 'reset'):
            self.updater.reset()
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        """Learned dictionary matrix."""
        return self._dictionary
    
    @property  
    def n_samples_seen(self) -> int:
        """Number of samples processed."""
        return self._n_samples_seen
    
    def get_config(self) -> Dict[str, Any]:
        """Get learner configuration."""
        return {
            'class_name': 'OnlineLearner', 
            'n_atoms': self.n_atoms,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'n_samples_seen': self._n_samples_seen,
            'is_fitted': self._is_fitted
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set learner configuration."""
        self.n_atoms = config.get('n_atoms', self.n_atoms)
        self.batch_size = config.get('batch_size', self.batch_size) 
        self.learning_rate = config.get('learning_rate', self.learning_rate)
        self._n_samples_seen = config.get('n_samples_seen', 0)
        self._is_fitted = config.get('is_fitted', False)
    
    def _init_dictionary(self, X: ArrayLike, n_atoms: int):
        """Initialize dictionary from random data samples."""
        n_features, n_samples = X.shape
        indices = np.random.choice(n_samples, min(n_atoms, n_samples), replace=False)
        self._dictionary = X[:, indices].copy()
        
        norms = np.linalg.norm(self._dictionary, axis=0, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        self._dictionary = self._dictionary / norms
    
    def _compute_objective(self, X: ArrayLike, A: ArrayLike) -> float:
        """Compute objective function value."""
        residual = X - self._dictionary @ A
        data_fidelity = 0.5 * np.sum(residual**2) 
        penalty_term = self.penalty.value(A)
        return data_fidelity + penalty_term


# SOLUTION 3: Composite Learner Pattern 
@dataclass
class CompositeLearner:
    """
    Generic dictionary learner using composition pattern.
    
    Combines penalty, solver, and updater components for flexible
    dictionary learning. Allows mix-and-match of different algorithms.
    """
    penalty: Penalty
    solver: InferenceSolver  
    updater: DictUpdater
    n_atoms: int = 64
    n_steps: int = 100
    verbose: bool = False
    
    _dictionary: Optional[ArrayLike] = field(default=None, init=False)
    _is_fitted: bool = field(default=False, init=False)
    
    def fit(self, X: ArrayLike, **kwargs) -> 'CompositeLearner':
        """
        Generic alternating minimization using composed components.
        
        Alternates between sparse coding (solver) and dictionary update (updater)
        while optimizing the specified penalty function.
        """
        X = ensure_array(X)
        n_features, n_samples = X.shape
        
        n_steps = kwargs.get('n_steps', self.n_steps)
        verbose = kwargs.get('verbose', self.verbose)
        
        if self._dictionary is None:
            self._init_dictionary(X, self.n_atoms)
        
        # Generic alternating minimization
        for step in range(n_steps):
            if verbose and step % 10 == 0:
                algorithm_name = f"{self.solver.name}-{self.updater.name}"
                print(f"Composite step {step}/{n_steps} ({algorithm_name})")
            
            # Sparse coding using composed solver
            codes = self.solver.solve(self._dictionary, X, self.penalty)
            
            # Dictionary update using composed updater  
            self._dictionary = self.updater.step(self._dictionary, X, codes)
            
            # Post-normalization if required by updater
            if self.updater.requires_normalization:
                self._dictionary = self._normalize_dictionary(self._dictionary)
            
            if verbose and step % 10 == 0:
                objective = self._compute_objective(X, codes)
                print(f"  Objective: {objective:.6f}")
        
        self._is_fitted = True
        return self
    
    def encode(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """Encode data using composed solver."""
        if not self._is_fitted:
            raise ValueError("Learner not fitted. Call fit() first.")
        
        X = ensure_array(X)
        return self.solver.solve(self._dictionary, X, self.penalty, **kwargs)
    
    def decode(self, A: ArrayLike) -> ArrayLike:
        """Decode sparse codes."""
        if not self._is_fitted:
            raise ValueError("Learner not fitted. Call fit() first.")
        
        A = ensure_array(A)
        return self._dictionary @ A
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        """Learned dictionary matrix."""
        return self._dictionary
    
    def get_config(self) -> Dict[str, Any]:
        """Get learner configuration."""
        return {
            'class_name': 'CompositeLearner',
            'penalty_type': type(self.penalty).__name__,
            'solver_type': type(self.solver).__name__,
            'updater_type': type(self.updater).__name__,
            'n_atoms': self.n_atoms,
            'n_steps': self.n_steps,
            'is_fitted': self._is_fitted
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set learner configuration."""
        self.n_atoms = config.get('n_atoms', self.n_atoms)
        self.n_steps = config.get('n_steps', self.n_steps)
        self._is_fitted = config.get('is_fitted', False)
    
    def _init_dictionary(self, X: ArrayLike, n_atoms: int):
        """Initialize dictionary from random data samples."""
        n_features, n_samples = X.shape
        indices = np.random.choice(n_samples, min(n_atoms, n_samples), replace=False)
        self._dictionary = X[:, indices].copy()
        self._dictionary = self._normalize_dictionary(self._dictionary)
    
    def _normalize_dictionary(self, D: ArrayLike) -> ArrayLike:
        """Normalize dictionary columns."""
        norms = np.linalg.norm(D, axis=0, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return D / norms
    
    def _compute_objective(self, X: ArrayLike, A: ArrayLike) -> float:
        """Compute objective function value."""
        residual = X - self._dictionary @ A
        data_fidelity = 0.5 * np.sum(residual**2)
        penalty_term = self.penalty.value(A)
        return data_fidelity + penalty_term


# Configuration system for learner creation
@dataclass  
class LearnerConfig:
    """Configuration for learner creation with solution selection."""
    
    # Core learner parameters
    learner_type: str = 'composite'  # 'ksvd', 'mod', 'online', 'composite'
    n_atoms: int = 64
    n_steps: int = 100
    verbose: bool = False
    
    # Online specific
    batch_size: int = 100
    learning_rate: float = 0.01
    n_passes: int = 5
    
    # Component configurations (for composite learner)
    penalty_config: Optional[Dict[str, Any]] = None
    solver_config: Optional[SolverConfig] = None
    updater_config: Optional[DictUpdaterConfig] = None


def create_learner(config: LearnerConfig) -> Learner:
    """Create learner with configurable components and solution patterns."""
    
    if config.learner_type == 'ksvd':
        # K-SVD learner with defaults
        learner = KsvdLearner(
            n_atoms=config.n_atoms,
            n_steps=config.n_steps,
            verbose=config.verbose
        )
    
    elif config.learner_type == 'mod':
        # MOD learner with defaults
        learner = ModLearner(
            n_atoms=config.n_atoms,
            n_steps=config.n_steps,
            verbose=config.verbose
        )
    
    elif config.learner_type == 'online':
        # Online learner
        learner = OnlineLearner(
            n_atoms=config.n_atoms,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            n_passes=config.n_passes,
            verbose=config.verbose
        )
    
    elif config.learner_type == 'composite':
        # Composite learner with configurable components
        penalty = create_penalty(config.penalty_config or PenaltyConfig())
        solver = create_solver(config.solver_config or SolverConfig())  
        updater = create_dict_updater(config.updater_config or DictUpdaterConfig())
        
        learner = CompositeLearner(
            penalty=penalty,
            solver=solver,
            updater=updater,
            n_atoms=config.n_atoms,
            n_steps=config.n_steps,
            verbose=config.verbose
        )
    
    else:
        raise ValueError(f"Unknown learner type: {config.learner_type}")
    
    return learner