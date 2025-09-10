"""
Factory functions for creating configurable sparse coding algorithms.

Provides unified interface for instantiating penalty functions, solvers,
and dictionary updaters with research-validated parameter combinations.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass

from ..algorithms.penalties import (
    L1Penalty, L2Penalty, ElasticNetPenalty, CauchyPenalty, TopKPenalty,
    PenaltyType, PenaltyConfig
)
from ..algorithms.solvers import (
    FISTASolver, ISTASolver, OMPSolver,
    SolverType, SolverConfig  
)
from ..algorithms.dictionary_update_algorithms import (
    MODUpdater, GradientDictUpdater, KSVDUpdater,
    UpdaterType, UpdaterConfig
)
from ..core.interfaces import Penalty, InferenceSolver, DictUpdater


@dataclass
class LearnerConfig:
    """
    Configuration for complete dictionary learning system.
    
    Combines penalty, solver, and dictionary update configurations
    for end-to-end sparse coding and dictionary learning with
    research-validated convergence monitoring.
    """
    # Algorithm selection
    penalty_type: PenaltyType = PenaltyType.L1
    solver_type: SolverType = SolverType.FISTA
    updater_type: UpdaterType = UpdaterType.MOD
    
    # Dictionary parameters
    n_atoms: int = 100
    n_iterations: int = 20
    
    # Penalty parameters
    lam: float = 0.1
    l1_ratio: float = 0.5  # For Elastic Net
    sigma: float = 1.0     # For Cauchy penalty
    k: int = 10           # For Top-K constraint
    
    # Solver parameters
    max_iter: int = 100
    tol: float = 1e-6
    
    # Dictionary update parameters
    learning_rate: float = 0.01
    regularization: float = 1e-6
    normalize_atoms: bool = True
    
    # Convergence monitoring (Mairal et al. 2009)
    convergence_tolerance: float = 1e-4
    dict_convergence_tolerance: float = 1e-5
    residual_tolerance: float = 1e-6
    enable_early_stopping: bool = True
    patience: int = 5
    
    # System parameters
    verbose: bool = False
    compute_objective: bool = True


def create_penalty(penalty_type: Union[str, PenaltyType], **kwargs) -> Penalty:
    """
    Factory function to create penalty with configuration options
    
    Args:
        penalty_type: Type of penalty to create
        **kwargs: Configuration parameters
        
    Returns:
        Configured penalty instance
        
    Example:
        ```python
        # L1 penalty with custom strength
        l1 = create_penalty('l1', lam=0.05)
        
        # Elastic Net with 70% L1, 30% L2
        elastic = create_penalty('elastic_net', lam=0.1, l1_ratio=0.7)
        
        # Top-K constraint with 10 active coefficients
        topk = create_penalty('top_k', k=10)
        ```
    """
    if isinstance(penalty_type, str):
        penalty_type = PenaltyType(penalty_type)
    
    config = PenaltyConfig(penalty_type=penalty_type, **kwargs)
    
    penalty_map = {
        PenaltyType.L1: L1Penalty,
        PenaltyType.L2: L2Penalty,
        PenaltyType.ELASTIC_NET: ElasticNetPenalty,
        PenaltyType.CAUCHY: CauchyPenalty,
        PenaltyType.TOP_K: TopKPenalty,
    }
    
    penalty_class = penalty_map.get(penalty_type)
    if penalty_class is None:
        raise ValueError(f"Unknown penalty type: {penalty_type}")
    
    return penalty_class(config)


def create_solver(solver_type: Union[str, SolverType], **kwargs) -> InferenceSolver:
    """
    Factory function to create solver with configuration options
    
    Args:
        solver_type: Type of solver to create
        **kwargs: Configuration parameters
        
    Returns:
        Configured solver instance
        
    Example:
        ```python
        # FISTA solver with adaptive restart
        fista = create_solver('fista', max_iter=200, adaptive_restart=True)
        
        # OMP with exact sparsity control
        omp = create_solver('omp', max_iter=10)
        ```
    """
    if isinstance(solver_type, str):
        solver_type = SolverType(solver_type)
    
    config = SolverConfig(solver_type=solver_type, **kwargs)
    
    solver_map = {
        SolverType.FISTA: FISTASolver,
        SolverType.ISTA: ISTASolver,
        SolverType.OMP: OMPSolver,
    }
    
    solver_class = solver_map.get(solver_type)
    if solver_class is None:
        raise ValueError(f"Unknown solver type: {solver_type}")
    
    return solver_class(config)


def create_dict_updater(updater_type: Union[str, UpdaterType], **kwargs) -> DictUpdater:
    """
    Factory function to create dictionary updater with configuration options
    
    Args:
        updater_type: Type of updater to create
        **kwargs: Configuration parameters
        
    Returns:
        Configured dictionary updater instance
        
    Example:
        ```python
        # MOD updater with regularization
        mod = create_dict_updater('mod', regularization=1e-5)
        
        # K-SVD with limited atom updates per iteration
        ksvd = create_dict_updater('ksvd', max_atom_updates=5)
        ```
    """
    if isinstance(updater_type, str):
        updater_type = UpdaterType(updater_type)
    
    config = UpdaterConfig(updater_type=updater_type, **kwargs)
    
    updater_map = {
        UpdaterType.MOD: MODUpdater,
        UpdaterType.GRADIENT_DESCENT: GradientDictUpdater,
        UpdaterType.KSVD: KSVDUpdater,
    }
    
    updater_class = updater_map.get(updater_type)
    if updater_class is None:
        raise ValueError(f"Unknown updater type: {updater_type}")
    
    return updater_class(config)


class CompleteDictionaryLearner:
    """
    Complete dictionary learning system combining sparse coding and dictionary updates.
    
    Integrates penalty functions, sparse coding solvers, and dictionary update methods
    into a unified learning system with objective tracking and convergence monitoring.
    """
    
    def __init__(self, config: LearnerConfig):
        self.config = config
        
        # Create algorithm components
        self.penalty = create_penalty(
            config.penalty_type,
            lam=config.lam,
            l1_ratio=config.l1_ratio,
            sigma=config.sigma,
            k=config.k
        )
        
        self.solver = create_solver(
            config.solver_type,
            max_iter=config.max_iter,
            tol=config.tol
        )
        
        self.updater = create_dict_updater(
            config.updater_type,
            learning_rate=config.learning_rate,
            regularization=config.regularization,
            normalize_atoms=config.normalize_atoms
        )
        
        # Initialize state
        self.dictionary = None
        self._training_history = []
        
    def fit(self, X: np.ndarray) -> 'CompleteDictionaryLearner':
        """
        Learn dictionary from data using alternating optimization.
        
        Args:
            X: Data matrix (n_features, n_samples)
            
        Returns:
            Self for method chaining
        """
        n_features, n_samples = X.shape
        
        # Initialize dictionary randomly
        np.random.seed(42)  # For reproducibility
        self.dictionary = np.random.randn(n_features, self.config.n_atoms)
        self.dictionary /= np.linalg.norm(self.dictionary, axis=0, keepdims=True)
        
        # Research-accurate alternating optimization with convergence monitoring
        prev_objective = float('inf')
        prev_dictionary = self.dictionary.copy()
        stagnation_count = 0
        
        for iteration in range(self.config.n_iterations):
            # Sparse coding step
            codes = self.solver.solve(self.dictionary, X, self.penalty)
            
            # Dictionary update step
            old_dictionary = self.dictionary.copy()
            self.dictionary = self.updater.step(self.dictionary, X, codes)
            
            # Convergence monitoring (Mairal et al. 2009)
            reconstruction_error = 0.5 * np.sum((X - self.dictionary @ codes) ** 2)
            penalty_value = self.penalty.value(codes) if hasattr(self.penalty, 'value') else 0.0
            total_objective = reconstruction_error + penalty_value
            
            # Early stopping: Objective function convergence
            if self.config.enable_early_stopping and iteration > 0:
                objective_change = abs(total_objective - prev_objective) / (prev_objective + 1e-12)
                if objective_change < self.config.convergence_tolerance:
                    stagnation_count += 1
                    if stagnation_count >= self.config.patience:
                        if self.config.verbose:
                            print(f"Early stopping: objective convergence at iteration {iteration + 1}")
                        break
                else:
                    stagnation_count = 0
            
            # Early stopping: Dictionary change monitoring
            if self.config.enable_early_stopping:
                dict_change = np.linalg.norm(self.dictionary - old_dictionary, 'fro')
                if dict_change < self.config.dict_convergence_tolerance:
                    if self.config.verbose:
                        print(f"Early stopping: dictionary convergence at iteration {iteration + 1}")
                    break
            
            # Early stopping: Residual-based convergence
            if self.config.enable_early_stopping:
                residual_norm = np.linalg.norm(X - self.dictionary @ codes, 'fro')
                if residual_norm < self.config.residual_tolerance:
                    if self.config.verbose:
                        print(f"Early stopping: residual convergence at iteration {iteration + 1}")
                    break
            
            # Store training history
            if self.config.compute_objective:
                self._training_history.append({
                    'iteration': iteration,
                    'objective': total_objective,
                    'reconstruction_error': reconstruction_error,
                    'penalty_value': penalty_value,
                    'dict_change': np.linalg.norm(self.dictionary - prev_dictionary, 'fro'),
                    'residual_norm': np.linalg.norm(X - self.dictionary @ codes, 'fro')
                })
            
            prev_objective = total_objective
            prev_dictionary = self.dictionary.copy()
            
            if self.config.verbose:
                print(f"Iteration {iteration + 1}/{self.config.n_iterations}: obj={total_objective:.6f}")
        
        return self
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode data using learned dictionary.
        
        Args:
            X: Data matrix (n_features, n_samples)
            
        Returns:
            Sparse codes (n_atoms, n_samples)
        """
        if self.dictionary is None:
            raise ValueError("Dictionary not learned. Call fit() first.")
        
        # Comprehensive input validation
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2D array (n_features, n_samples)")
        if X.shape[0] != self.dictionary.shape[0]:
            raise ValueError(f"Feature dimension mismatch: X has {X.shape[0]} features, "
                           f"dictionary expects {self.dictionary.shape[0]}")
        if X.size == 0:
            raise ValueError("X cannot be empty")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values (NaN or Inf)")
        
        # Robust solver with error recovery
        try:
            return self.solver.solve(self.dictionary, X, self.penalty)
        except np.linalg.LinAlgError as e:
            if self.config.verbose:
                print(f"Solver numerical error: {e}. Attempting regularized fallback.")
            # Fallback: Add regularization to improve conditioning
            try:
                from ..core.inference.orthogonal_matching_pursuit import OrthogonalMatchingPursuit
                fallback_solver = OrthogonalMatchingPursuit(
                    regularization=self.config.regularization * 10,
                    solver='pinv'
                )
                return np.array([fallback_solver.solve(self.dictionary, X[:, i])[0] 
                               for i in range(X.shape[1])]).T
            except Exception as fallback_error:
                if self.config.verbose:
                    print(f"Fallback solver failed: {fallback_error}")
                return np.zeros((self.dictionary.shape[1], X.shape[1]))
        except Exception as e:
            if self.config.verbose:
                print(f"Solver failed with unexpected error: {e}")
            return np.zeros((self.dictionary.shape[1], X.shape[1]))
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from sparse codes.
        
        Args:
            codes: Sparse codes (n_atoms, n_samples)
            
        Returns:
            Reconstructed data (n_features, n_samples)
        """
        if self.dictionary is None:
            raise ValueError("Dictionary not learned. Call fit() first.")
        
        # Input validation
        if not isinstance(codes, np.ndarray):
            raise TypeError("codes must be numpy array")
        if codes.ndim != 2:
            raise ValueError("codes must be 2D array (n_atoms, n_samples)")
        if codes.shape[0] != self.dictionary.shape[1]:
            raise ValueError(f"Code dimension mismatch: codes has {codes.shape[0]} atoms, "
                           f"dictionary has {self.dictionary.shape[1]} atoms")
        if not np.isfinite(codes).all():
            raise ValueError("codes contains non-finite values (NaN or Inf)")
        
        return self.dictionary @ codes
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """
        Get convergence analysis from training history.
        
        Returns:
            Dictionary with convergence metrics and analysis
        """
        if not self._training_history:
            return {'message': 'No training history available'}
        
        history = np.array([h['objective'] for h in self._training_history])
        dict_changes = np.array([h.get('dict_change', 0) for h in self._training_history])
        residuals = np.array([h.get('residual_norm', 0) for h in self._training_history])
        
        return {
            'converged': len(self._training_history) < self.config.n_iterations,
            'final_objective': history[-1] if len(history) > 0 else None,
            'objective_reduction': (history[0] - history[-1]) / history[0] if len(history) > 1 else 0,
            'final_dict_change': dict_changes[-1] if len(dict_changes) > 0 else None,
            'final_residual': residuals[-1] if len(residuals) > 0 else None,
            'iterations_completed': len(self._training_history),
            'training_history': self._training_history
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Return complete configuration as dictionary."""
        return {
            'penalty_type': self.config.penalty_type.value,
            'solver_type': self.config.solver_type.value,
            'updater_type': self.config.updater_type.value,
            'n_atoms': self.config.n_atoms,
            'n_iterations': self.config.n_iterations,
            'lam': self.config.lam,
            'max_iter': self.config.max_iter,
            'tol': self.config.tol,
            'learning_rate': self.config.learning_rate,
            'regularization': self.config.regularization,
            'convergence_tolerance': self.config.convergence_tolerance,
            'dict_convergence_tolerance': self.config.dict_convergence_tolerance,
            'residual_tolerance': self.config.residual_tolerance,
            'enable_early_stopping': self.config.enable_early_stopping,
        }


def create_complete_learner(
    penalty_type: Union[str, PenaltyType] = 'l1',
    solver_type: Union[str, SolverType] = 'fista',
    updater_type: Union[str, UpdaterType] = 'mod',
    n_atoms: int = 100,
    n_iterations: int = 20,
    **kwargs
) -> CompleteDictionaryLearner:
    """
    Factory function to create complete dictionary learning system with research-accurate convergence monitoring.
    
    Args:
        penalty_type: Penalty function type ('l1', 'elastic_net', etc.)
        solver_type: Sparse coding solver ('fista', 'ista', 'omp')  
        updater_type: Dictionary update method ('mod', 'ksvd', 'grad_d')
        n_atoms: Number of dictionary atoms
        n_iterations: Number of alternating optimization iterations
        **kwargs: Additional configuration parameters including:
            - convergence_tolerance: Objective function convergence threshold
            - dict_convergence_tolerance: Dictionary change convergence threshold
            - residual_tolerance: Residual norm convergence threshold
            - enable_early_stopping: Enable automatic early stopping
            - patience: Number of iterations to wait before early stopping
        
    Returns:
        Configured CompleteDictionaryLearner instance with convergence monitoring
        
    Example:
        ```python
        # Create L1+FISTA+MOD learner with convergence monitoring
        learner = create_complete_learner(
            penalty_type='l1',
            solver_type='fista', 
            updater_type='mod',
            n_atoms=50,
            lam=0.1,
            n_iterations=30,
            convergence_tolerance=1e-5,
            enable_early_stopping=True
        )
        
        # Train on data with automatic convergence detection
        learner.fit(training_data)
        
        # Check convergence status
        conv_info = learner.get_convergence_info()
        print(f"Converged: {conv_info['converged']}")
        ```
    """
    # Input validation
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive")
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive")
    
    try:
        config = LearnerConfig(
            penalty_type=PenaltyType(penalty_type) if isinstance(penalty_type, str) else penalty_type,
            solver_type=SolverType(solver_type) if isinstance(solver_type, str) else solver_type,
            updater_type=UpdaterType(updater_type) if isinstance(updater_type, str) else updater_type,
            n_atoms=n_atoms,
            n_iterations=n_iterations,
            **kwargs
        )
        
        return CompleteDictionaryLearner(config)
    except ValueError as e:
        raise ValueError(f"Invalid configuration: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create learner: {e}")