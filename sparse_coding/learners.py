"""
Comprehensive Dictionary Learning Implementations

High-level learners that orchestrate inference solvers and dictionary updaters.
Implements research-accurate dictionary learning algorithms with configuration options.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Any, Dict, Union, Tuple
from .core.interfaces import Learner, StreamingLearner, Penalty, InferenceSolver, DictUpdater
from .core.array import ArrayLike
from .penalties import L1, L2, ElasticNet, CauchyPenalty, TopKConstraint, HuberPenalty, GroupLasso
from .solvers import get_solver
from .dict_updaters import get_dict_updater


class ComprehensiveDictionaryLearner:
    """Comprehensive dictionary learning with configurable components.
    
    Provides a unified interface to all sparse coding algorithms with research-accurate
    implementations. Users can configure penalty functions, solvers, and updaters.
    
    References:
    - Olshausen & Field (1996): Emergence of simple-cell receptive field properties
    - Aharon et al. (2006): K-SVD algorithm for overcomplete dictionaries  
    - Mairal et al. (2010): Online dictionary learning for sparse coding
    """
    
    def __init__(
        self,
        n_atoms: int = 100,
        penalty: Union[str, Penalty] = 'l1',
        penalty_params: Optional[Dict[str, Any]] = None,
        solver: str = 'fista',
        solver_params: Optional[Dict[str, Any]] = None,
        dict_updater: str = 'mod',
        updater_params: Optional[Dict[str, Any]] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize comprehensive dictionary learner.
        
        Args:
            n_atoms: Number of dictionary atoms
            penalty: Penalty function ('l1', 'l2', 'elastic_net', 'cauchy', 'topk', 'huber', 'group_lasso')
            penalty_params: Parameters for penalty function
            solver: Inference solver ('fista', 'ista', 'coordinate_descent', 'omp', 'ncg')
            solver_params: Parameters for solver
            dict_updater: Dictionary update method ('mod', 'grad_d', 'ksvd', 'online_sgd', 'adam')
            updater_params: Parameters for dictionary updater
            max_iter: Maximum dictionary learning iterations
            tol: Convergence tolerance
            random_state: Random seed for reproducibility
            verbose: Print training progress
        """
        self.n_atoms = n_atoms
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Configure penalty function
        self.penalty = self._create_penalty(penalty, penalty_params or {})
        
        # Configure solver
        self.solver = get_solver(solver, **(solver_params or {}))
        
        # Configure dictionary updater
        self.dict_updater = get_dict_updater(dict_updater, **(updater_params or {}))
        
        # Learner state
        self._dictionary = None
        self._fitted = False
        self.training_history_ = {
            'reconstruction_errors': [],
            'sparsity_levels': [],
            'objective_values': []
        }
    
    def _create_penalty(self, penalty: Union[str, Penalty], params: Dict[str, Any]) -> Penalty:
        """Create penalty function from string specification."""
        if isinstance(penalty, str):
            if penalty == 'l1':
                return L1(lam=params.get('lam', 0.1))
            elif penalty == 'l2':
                return L2(lam=params.get('lam', 0.1))
            elif penalty == 'elastic_net':
                return ElasticNet(
                    l1=params.get('l1', 0.1),
                    l2=params.get('l2', 0.1)
                )
            elif penalty == 'cauchy':
                return CauchyPenalty(
                    lam=params.get('lam', 0.1),
                    sigma=params.get('sigma', 1.0)
                )
            elif penalty == 'topk':
                return TopKConstraint(k=params.get('k', 10))
            elif penalty == 'huber':
                return HuberPenalty(
                    lam=params.get('lam', 0.1),
                    delta=params.get('delta', 1.0)
                )
            elif penalty == 'group_lasso':
                if 'groups' not in params:
                    raise ValueError("Group Lasso requires 'groups' parameter")
                return GroupLasso(
                    lam=params.get('lam', 0.1),
                    groups=params['groups']
                )
            else:
                raise ValueError(f"Unknown penalty: {penalty}")
        else:
            return penalty
    
    def fit(self, X: ArrayLike, **kwargs) -> 'ComprehensiveDictionaryLearner':
        \"\"\"
        Learn dictionary from data.
        
        Args:
            X: Training data (n_features, n_samples)
            
        Returns:
            Self for chaining
        \"\"\"
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_features, n_samples = X.shape
        
        # Initialize dictionary
        if self._dictionary is None:
            self._dictionary = np.random.randn(n_features, self.n_atoms)
            self._normalize_dictionary()
        
        if self.verbose:
            print(f\"Dictionary learning with {self.n_atoms} atoms on {n_samples} samples\")
            print(f\"Penalty: {type(self.penalty).__name__}\")
            print(f\"Solver: {self.solver.name}\")
            print(f\"Updater: {self.dict_updater.name}\")
        
        # Main learning loop
        obj_prev = float('inf')
        
        for iteration in range(self.max_iter):
            # Step 1: Sparse coding (E-step)
            A = self.solver.solve(self._dictionary, X, self.penalty)
            
            # Step 2: Dictionary update (M-step)  
            self._dictionary = self.dict_updater.step(self._dictionary, X, A)
            
            # Normalize dictionary if updater requires it
            if self.dict_updater.requires_normalization:
                self._normalize_dictionary()
            
            # Compute metrics
            reconstruction = self._dictionary @ A
            reconstruction_error = 0.5 * np.mean((X - reconstruction)**2)
            sparsity_level = np.mean(np.abs(A) > 1e-6)
            
            # Compute objective value
            penalty_val = np.mean([self.penalty.value(A[:, i]) for i in range(n_samples)])
            objective = reconstruction_error + penalty_val
            
            # Store history
            self.training_history_['reconstruction_errors'].append(reconstruction_error)
            self.training_history_['sparsity_levels'].append(sparsity_level)
            self.training_history_['objective_values'].append(objective)
            
            # Print progress
            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(f\"Iter {iteration:3d}: obj={objective:.6f}, \"\n                      f\"recon_err={reconstruction_error:.6f}, sparsity={sparsity_level:.3f}\")
            
            # Convergence check
            if abs(objective - obj_prev) / max(abs(objective), 1e-8) < self.tol:
                if self.verbose:
                    print(f\"Converged after {iteration} iterations\")
                break
            
            obj_prev = objective
        
        self._fitted = True
        return self
    
    def encode(self, X: ArrayLike, **kwargs) -> ArrayLike:
        \"\"\"
        Encode data using learned dictionary.
        
        Args:
            X: Data to encode (n_features, n_samples)
            
        Returns:
            Sparse codes (n_atoms, n_samples)
        \"\"\"
        if not self._fitted:
            raise ValueError(\"Must call fit() before encode()\")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.solver.solve(self._dictionary, X, self.penalty)
    
    def decode(self, A: ArrayLike) -> ArrayLike:
        \"\"\"
        Decode sparse codes back to data space.
        
        Args:
            A: Sparse codes (n_atoms, n_samples)
            
        Returns:
            Reconstructed data (n_features, n_samples)
        \"\"\"
        if not self._fitted:
            raise ValueError(\"Must call fit() before decode()\")
        
        A = np.asarray(A)
        return self._dictionary @ A
    
    def fit_transform(self, X: ArrayLike, **kwargs) -> ArrayLike:
        \"\"\"Fit dictionary and encode data.\"\"\"\n        return self.fit(X, **kwargs).encode(X)
    
    def _normalize_dictionary(self):
        \"\"\"Normalize dictionary atoms to unit norm.\"\"\"
        if self._dictionary is not None:
            norms = np.linalg.norm(self._dictionary, axis=0)
            norms[norms == 0] = 1
            self._dictionary = self._dictionary / norms[np.newaxis, :]
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        \"\"\"Learned dictionary matrix.\"\"\"
        return self._dictionary
    
    def get_config(self) -> Dict[str, Any]:
        \"\"\"Get learner configuration for serialization.\"\"\"
        return {
            'n_atoms': self.n_atoms,
            'penalty_type': type(self.penalty).__name__,
            'penalty_params': getattr(self.penalty, '__dict__', {}),
            'solver_name': self.solver.name,
            'solver_params': getattr(self.solver, '__dict__', {}),
            'updater_name': self.dict_updater.name,
            'updater_params': getattr(self.dict_updater, '__dict__', {}),
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        \"\"\"Set learner configuration from serialization.\"\"\"
        self.n_atoms = config['n_atoms']
        self.max_iter = config['max_iter']
        self.tol = config['tol']
        self.random_state = config['random_state']
        
        # Recreate penalty
        penalty_class = globals()[config['penalty_type']]
        self.penalty = penalty_class(**config['penalty_params'])
        
        # Recreate solver and updater  
        self.solver = get_solver(config['solver_name'], **config['solver_params'])
        self.dict_updater = get_dict_updater(config['updater_name'], **config['updater_params'])


class OnlineDictionaryLearner:
    \"\"\"Online/streaming dictionary learning.
    
    Reference: Mairal et al. (2010). Online dictionary learning for sparse coding.
    
    Learns dictionary incrementally from streaming data using stochastic approximation.
    \"\"\"
    
    def __init__(
        self,
        n_atoms: int = 100,
        penalty: Union[str, Penalty] = 'l1',
        penalty_params: Optional[Dict[str, Any]] = None,
        solver: str = 'fista',
        solver_params: Optional[Dict[str, Any]] = None,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.n_atoms = n_atoms
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Configure components like ComprehensiveDictionaryLearner
        self.penalty = self._create_penalty(penalty, penalty_params or {})
        self.solver = get_solver(solver, **(solver_params or {}))
        
        # Online learning state
        self._dictionary = None
        self._fitted = False
        self._n_samples_seen = 0
        self._A_avg = None  # Running average of coefficient outer products
        self._B_avg = None  # Running average of data-coefficient cross products
    
    def _create_penalty(self, penalty: Union[str, Penalty], params: Dict[str, Any]) -> Penalty:
        \"\"\"Same as ComprehensiveDictionaryLearner.\"\"\"
        if isinstance(penalty, str):
            if penalty == 'l1':
                return L1(lam=params.get('lam', 0.1))
            elif penalty == 'l2':
                return L2(lam=params.get('lam', 0.1))
            # Add other penalties as needed
        return penalty
    
    def partial_fit(self, X: ArrayLike, **kwargs) -> 'OnlineDictionaryLearner':
        \"\"\"
        Incremental learning step.
        
        Args:
            X: Mini-batch of data (n_features, batch_size)
            
        Returns:
            Self for chaining
        \"\"\"
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_features, batch_size = X.shape
        
        # Initialize dictionary on first batch
        if self._dictionary is None:
            self._dictionary = np.random.randn(n_features, self.n_atoms)
            self._normalize_dictionary()
            self._A_avg = np.zeros((self.n_atoms, self.n_atoms))
            self._B_avg = np.zeros((n_features, self.n_atoms))
        
        # Sparse coding step
        A = self.solver.solve(self._dictionary, X, self.penalty)
        
        # Update running averages
        momentum = min(0.99, self._n_samples_seen / (self._n_samples_seen + batch_size))
        
        self._A_avg = momentum * self._A_avg + (1 - momentum) * (A @ A.T) / batch_size
        self._B_avg = momentum * self._B_avg + (1 - momentum) * (X @ A.T) / batch_size
        
        # Dictionary update using running averages
        try:
            self._dictionary = self._B_avg @ np.linalg.inv(self._A_avg + 1e-7 * np.eye(self.n_atoms))
        except np.linalg.LinAlgError:
            self._dictionary = self._B_avg @ np.linalg.pinv(self._A_avg)
        
        self._normalize_dictionary()
        
        self._n_samples_seen += batch_size
        self._fitted = True
        
        if self.verbose and self._n_samples_seen % 1000 == 0:
            print(f\"Processed {self._n_samples_seen} samples\")
        
        return self
    
    def fit(self, X: ArrayLike, **kwargs) -> 'OnlineDictionaryLearner':
        \"\"\"
        Fit using mini-batch processing.
        
        Args:
            X: Training data (n_features, n_samples)
            
        Returns:
            Self for chaining
        \"\"\"
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[1]
        
        # Process in mini-batches
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            X_batch = X[:, start:end]
            self.partial_fit(X_batch)
        
        return self
    
    def encode(self, X: ArrayLike, **kwargs) -> ArrayLike:
        \"\"\"Encode data using learned dictionary.\"\"\"
        if not self._fitted:
            raise ValueError(\"Must call fit() or partial_fit() before encode()\")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.solver.solve(self._dictionary, X, self.penalty)
    
    def decode(self, A: ArrayLike) -> ArrayLike:
        \"\"\"Decode sparse codes back to data space.\"\"\"
        return self._dictionary @ A if self._fitted else None
    
    def reset(self) -> None:
        \"\"\"Reset learner state for fresh training.\"\"\"
        self._dictionary = None
        self._fitted = False
        self._n_samples_seen = 0
        self._A_avg = None
        self._B_avg = None
    
    @property
    def n_samples_seen(self) -> int:
        \"\"\"Number of samples processed.\"\"\"
        return self._n_samples_seen
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        \"\"\"Learned dictionary matrix.\"\"\"
        return self._dictionary
    
    def _normalize_dictionary(self):
        \"\"\"Normalize dictionary atoms to unit norm.\"\"\"
        if self._dictionary is not None:
            norms = np.linalg.norm(self._dictionary, axis=0)
            norms[norms == 0] = 1
            self._dictionary = self._dictionary / norms[np.newaxis, :]


# Registry of available learners
LEARNERS = {
    'comprehensive': ComprehensiveDictionaryLearner,
    'online': OnlineDictionaryLearner,
}


def get_learner(name: str, **kwargs):
    \"\"\"Factory function for learner instantiation.\"\"\"
    if name not in LEARNERS:
        raise ValueError(f\"Unknown learner '{name}'. Available: {list(LEARNERS.keys())}\")
    return LEARNERS[name](**kwargs)