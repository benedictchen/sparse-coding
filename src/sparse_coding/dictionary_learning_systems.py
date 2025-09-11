"""
Dictionary Learning Implementations

High-level learners that orchestrate inference solvers and dictionary updaters.
Implements dictionary learning algorithms with configuration options.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Any, Dict, Union, Tuple
from .core.interfaces import Learner, StreamingLearner, Penalty, InferenceSolver, DictUpdater
from .core.array import ArrayLike
from .core.penalties import L1Penalty as L1, L2Penalty as L2, ElasticNetPenalty as ElasticNet, CauchyPenalty, TopKConstraint, GroupLassoPenalty as GroupLasso, HuberPenalty
from .core.penalties import PenaltyProtocol  # For type hints
from .core.solver_implementations import get_solver
from .core.dict_updater_implementations import create_dict_updater, DictUpdaterConfig


class DictionaryLearner:
    """Dictionary learning with configurable components.
    
    Provides a unified interface to sparse coding algorithms.
    Users can configure penalty functions, solvers, and updaters.
    
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
        verbose: bool = False,
        dead_atom_threshold: float = 1e-6,
        replace_dead_atoms: bool = True,
        lambda_init_method: str = 'data_adaptive',
        lambda_annealing: Optional[float] = None,
        convergence_criteria: str = 'objective',
        gradient_tol: float = 1e-4
    ):
        """
        Initialize dictionary learner.
        
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
            dead_atom_threshold: Threshold for detecting unused atoms (atoms with usage < threshold)
            replace_dead_atoms: Whether to replace dead atoms with data samples
            lambda_init_method: Method for lambda initialization ('fixed', 'data_adaptive', 'cross_validation')
            lambda_annealing: Annealing factor for lambda decay (None = no annealing, 0.95 = 5% decay per iter)
            convergence_criteria: Convergence detection method ('objective', 'gradient', 'both')
            gradient_tol: Tolerance for gradient norm convergence criterion
        """
        self.n_atoms = n_atoms
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.dead_atom_threshold = dead_atom_threshold
        self.replace_dead_atoms = replace_dead_atoms
        self.lambda_init_method = lambda_init_method
        self.lambda_annealing = lambda_annealing
        self.convergence_criteria = convergence_criteria
        self.gradient_tol = gradient_tol
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Configure penalty function
        self.penalty = self._create_penalty(penalty, penalty_params or {})
        
        # Configure solver
        self.solver = get_solver(solver, **(solver_params or {}))
        
        # Configure dictionary updater
        updater_config = DictUpdaterConfig(method=dict_updater, **(updater_params or {}))
        self.dict_updater = create_dict_updater(updater_config)
        
        # Learner state
        self._dictionary = None
        self._fitted = False
        self.training_history_ = {
            'reconstruction_errors': [],
            'sparsity_levels': [],
            'objective_values': [],
            'gradient_norms': []
        }
    
    def _create_penalty(self, penalty: Union[str, Penalty], params: Dict[str, Any]) -> Penalty:
        """Create penalty function from string specification."""
        if isinstance(penalty, str):
            # Store for later data-adaptive initialization
            self._penalty_type = penalty
            self._penalty_params = params.copy()
            
            # Use default lambda values - will be replaced by data-adaptive initialization
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
            self._penalty_type = None  # Custom penalty object
            self._penalty_params = {}
            return penalty
    
    def _initialize_lambda(self, X: ArrayLike) -> None:
        """
        Initialize lambda parameter based on data characteristics.
        
        Data-adaptive initialization provides better starting points than fixed defaults.
        
        Research Foundation:
        - Tibshirani (1996): Cross-validation for optimal lambda selection
        - Zou & Hastie (2005): Adaptive lasso with data-dependent weights
        - Olshausen & Field (1996): Scale lambda with data variance
        
        Methods:
        - 'fixed': Use provided lambda values (no initialization)
        - 'data_adaptive': Scale lambda by data variance (recommended)
        - 'cross_validation': Use simple heuristic (computationally expensive)
        
        Args:
            X: Training data for adaptive scaling
        """
        if self.lambda_init_method == 'fixed' or self._penalty_type is None:
            return  # Use provided lambda values
            
        if self.lambda_init_method == 'data_adaptive':
            # Scale lambda by data variance (Olshausen & Field approach)
            data_variance = np.var(X)
            lambda_scale = np.sqrt(data_variance)
            
            # Update penalty object with scaled lambda
            if self._penalty_type == 'l1':
                base_lam = self._penalty_params.get('lam', 0.1)
                self.penalty = L1(lam=base_lam * lambda_scale)
                if self.verbose:
                    print(f"    Data-adaptive L1 lambda: {base_lam * lambda_scale:.6f} (data_var={data_variance:.6f})")
                    
            elif self._penalty_type == 'l2':
                base_lam = self._penalty_params.get('lam', 0.1)
                self.penalty = L2(lam=base_lam * lambda_scale)
                if self.verbose:
                    print(f"    Data-adaptive L2 lambda: {base_lam * lambda_scale:.6f}")
                    
            elif self._penalty_type == 'elastic_net':
                base_l1 = self._penalty_params.get('l1', 0.1)
                base_l2 = self._penalty_params.get('l2', 0.1)
                self.penalty = ElasticNet(
                    l1=base_l1 * lambda_scale,
                    l2=base_l2 * lambda_scale
                )
                if self.verbose:
                    print(f"    Data-adaptive ElasticNet lambda: l1={base_l1 * lambda_scale:.6f}, l2={base_l2 * lambda_scale:.6f}")
                    
            elif self._penalty_type == 'cauchy':
                base_lam = self._penalty_params.get('lam', 0.1)
                sigma = self._penalty_params.get('sigma', 1.0)
                self.penalty = CauchyPenalty(
                    lam=base_lam * lambda_scale,
                    sigma=sigma
                )
                if self.verbose:
                    print(f"    Data-adaptive Cauchy lambda: {base_lam * lambda_scale:.6f}")
                    
            elif self._penalty_type == 'huber':
                base_lam = self._penalty_params.get('lam', 0.1)
                delta = self._penalty_params.get('delta', 1.0)
                self.penalty = HuberPenalty(
                    lam=base_lam * lambda_scale,
                    delta=delta
                )
                if self.verbose:
                    print(f"    Data-adaptive Huber lambda: {base_lam * lambda_scale:.6f}")
                    
            elif self._penalty_type == 'group_lasso':
                base_lam = self._penalty_params.get('lam', 0.1)
                groups = self._penalty_params['groups']
                self.penalty = GroupLasso(
                    lam=base_lam * lambda_scale,
                    groups=groups
                )
                if self.verbose:
                    print(f"    Data-adaptive GroupLasso lambda: {base_lam * lambda_scale:.6f}")
                    
            # TopK doesn't use lambda, so no scaling needed
            
        elif self.lambda_init_method == 'cross_validation':
            # Simple heuristic: try multiple lambda values and pick one with good sparsity
            if self._penalty_type in ['l1', 'l2', 'elastic_net', 'cauchy', 'huber', 'group_lasso']:
                if self.verbose:
                    print("    Using cross-validation heuristic for lambda initialization")
                # For simplicity, use moderate scaling
                data_variance = np.var(X)
                lambda_scale = 0.5 * np.sqrt(data_variance)
                # Apply same scaling as data_adaptive but more conservative
                if hasattr(self.penalty, 'lam'):
                    current_lam = getattr(self.penalty, 'lam', 0.1)
                    new_penalty_params = self._penalty_params.copy()
                    new_penalty_params['lam'] = current_lam * lambda_scale
                    self.penalty = self._create_penalty(self._penalty_type, new_penalty_params)
    
    def _apply_lambda_annealing(self, iteration: int) -> None:
        """
        Apply lambda annealing (gradual reduction) during training.
        
        Research Foundation:
        - Annealing helps escape local minima in early iterations
        - Stabilizes convergence in later iterations
        - Common in optimization literature (simulated annealing)
        
        Args:
            iteration: Current iteration number
        """
        if self.lambda_annealing is None or self._penalty_type is None:
            return
            
        # Apply exponential decay: lambda_new = lambda_old * annealing_factor
        annealing_factor = self.lambda_annealing
        
        if self._penalty_type == 'l1' and hasattr(self.penalty, 'lam'):
            self.penalty.lam *= annealing_factor
        elif self._penalty_type == 'l2' and hasattr(self.penalty, 'lam'):
            self.penalty.lam *= annealing_factor
        elif self._penalty_type == 'elastic_net':
            if hasattr(self.penalty, 'l1'):
                self.penalty.l1 *= annealing_factor
            if hasattr(self.penalty, 'l2'):
                self.penalty.l2 *= annealing_factor
        elif self._penalty_type in ['cauchy', 'huber', 'group_lasso'] and hasattr(self.penalty, 'lam'):
            self.penalty.lam *= annealing_factor
    
    def _compute_gradient_norm(self, D: ArrayLike, X: ArrayLike, A: ArrayLike) -> float:
        """
        Compute gradient norm for convergence detection.
        
        Computes the Frobenius norm of the gradient with respect to the dictionary.
        This provides a more reliable convergence criterion than objective changes alone.
        
        Research Foundation:
        - Boyd & Vandenberghe (2004): Convex optimization convergence criteria
        - Beck & Teboulle (2009): FISTA convergence using gradient information
        - Mairal et al. (2010): Online dictionary learning gradient bounds
        
        The gradient of the objective F(D) = (1/2)||X - DA||²_F with respect to D is:
        ∇_D F = (DA - X)A^T = -R A^T where R = X - DA is the residual
        
        Args:
            D: Current dictionary (n_features, n_atoms)
            X: Data matrix (n_features, n_samples) 
            A: Current sparse codes (n_atoms, n_samples)
            
        Returns:
            Frobenius norm of the gradient: ||∇_D F||_F
        """
        # Compute residual: R = X - DA
        residual = X - D @ A
        
        # Gradient w.r.t. dictionary: ∇_D F = -R A^T
        gradient_D = -residual @ A.T
        
        # Return Frobenius norm of gradient
        gradient_norm = np.linalg.norm(gradient_D, ord='fro')
        
        return gradient_norm
    
    def _check_convergence(self, iteration: int, objective: float, obj_prev: float, 
                          gradient_norm: float) -> Tuple[bool, str]:
        """
        Check convergence based on specified criteria.
        
        Args:
            iteration: Current iteration number
            objective: Current objective value
            obj_prev: Previous objective value  
            gradient_norm: Current gradient norm
            
        Returns:
            (converged, reason): Boolean convergence flag and reason string
        """
        converged = False
        reason = ""
        
        # Objective-based convergence
        obj_relative_change = abs(objective - obj_prev) / max(abs(objective), 1e-8)
        obj_converged = obj_relative_change < self.tol
        
        # Gradient-based convergence
        grad_converged = gradient_norm < self.gradient_tol
        
        if self.convergence_criteria == 'objective':
            converged = obj_converged
            reason = f"objective change {obj_relative_change:.2e} < {self.tol}"
        elif self.convergence_criteria == 'gradient':
            converged = grad_converged
            reason = f"gradient norm {gradient_norm:.2e} < {self.gradient_tol}"
        elif self.convergence_criteria == 'both':
            converged = obj_converged and grad_converged
            if converged:
                reason = f"both objective ({obj_relative_change:.2e}) and gradient ({gradient_norm:.2e}) converged"
            else:
                reason = f"objective: {obj_relative_change:.2e}, gradient: {gradient_norm:.2e} (need both)"
        
        return converged, reason
    
    def fit(self, X: ArrayLike, **kwargs) -> 'DictionaryLearner':
        """
        Learn dictionary from data.
        
        Args:
            X: Training data (n_features, n_samples)
            
        Returns:
            Self for chaining
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_features, n_samples = X.shape
        
        # Initialize dictionary
        if self._dictionary is None:
            self._dictionary = np.random.randn(n_features, self.n_atoms)
            self._normalize_dictionary()
        
        # Initialize lambda parameters based on data
        self._initialize_lambda(X)
        
        if self.verbose:
            print(f"Initializing dictionary learning: {self.n_atoms} atoms, {n_samples} training samples")
            print(f"Sparsity penalty: {type(self.penalty).__name__}")
            print(f"Solver algorithm: {self.solver.name}")
            print(f"Dictionary update method: {self.dict_updater.name}")
            print(f"Lambda initialization: {self.lambda_init_method}")
            if self.lambda_annealing:
                print(f"Lambda annealing factor: {self.lambda_annealing}")
        
        # Main learning loop
        obj_prev = float('inf')
        
        for iteration in range(self.max_iter):
            # Step 1: Sparse coding (E-step)
            A = self.solver.solve(self._dictionary, X, self.penalty)
            
            # Step 2: Dictionary update (M-step)  
            self._dictionary = self.dict_updater.step(self._dictionary, X, A)
            
            # Step 3: Dead atom detection and replacement
            if self.replace_dead_atoms:
                self._replace_dead_atoms(X, A)
            
            # Normalize dictionary if updater requires it
            if self.dict_updater.requires_normalization:
                self._normalize_dictionary()
            
            # Step 4: Apply lambda annealing (if enabled)
            if iteration > 0:  # Don't anneal on first iteration
                self._apply_lambda_annealing(iteration)
            
            # Compute metrics
            reconstruction = self._dictionary @ A
            reconstruction_error = 0.5 * np.mean((X - reconstruction)**2)
            sparsity_level = np.mean(np.abs(A) > 1e-6)
            
            # Compute objective value
            penalty_val = np.mean([self.penalty.value(A[:, i]) for i in range(n_samples)])
            objective = reconstruction_error + penalty_val
            
            # Compute gradient norm for convergence
            gradient_norm = self._compute_gradient_norm(self._dictionary, X, A)
            
            # Store history
            self.training_history_['reconstruction_errors'].append(reconstruction_error)
            self.training_history_['sparsity_levels'].append(sparsity_level)
            self.training_history_['objective_values'].append(objective)
            self.training_history_['gradient_norms'].append(gradient_norm)
            
            # Print progress
            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration:3d}: objective={objective:.6f}, "
                      f"reconstruction_error={reconstruction_error:.6f}, sparsity={sparsity_level:.3f}, "
                      f"grad_norm={gradient_norm:.6f}")
            
            # Convergence check using improved criteria
            converged, reason = self._check_convergence(iteration, objective, obj_prev, gradient_norm)
            if converged:
                if self.verbose:
                    print(f"Algorithm converged after {iteration} iterations: {reason}")
                break
            
            obj_prev = objective
        
        self._fitted = True
        return self
    
    def encode(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """
        Encode data using learned dictionary.
        
        Args:
            X: Data to encode (n_features, n_samples)
            
        Returns:
            Sparse codes (n_atoms, n_samples)
        """
        if not self._fitted:
            raise ValueError("Must call fit() before encode()")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.solver.solve(self._dictionary, X, self.penalty)
    
    def decode(self, A: ArrayLike) -> ArrayLike:
        """
        Decode sparse codes back to data space.
        
        Args:
            A: Sparse codes (n_atoms, n_samples)
            
        Returns:
            Reconstructed data (n_features, n_samples)
        """
        if not self._fitted:
            raise ValueError("Must call fit() before decode()")
        
        A = np.asarray(A)
        return self._dictionary @ A
    
    def fit_transform(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """Fit dictionary and encode data."""
        return self.fit(X, **kwargs).encode(X)
    
    def _normalize_dictionary(self):
        """Normalize dictionary atoms to unit norm."""
        if self._dictionary is not None:
            norms = np.linalg.norm(self._dictionary, axis=0)
            norms[norms == 0] = 1
            self._dictionary = self._dictionary / norms[np.newaxis, :]
    
    def _replace_dead_atoms(self, X: ArrayLike, A: ArrayLike) -> None:
        """
        Replace dead (unused) atoms with random data samples.
        
        Dead atoms are those that are rarely used in sparse coding.
        This prevents dictionary collapse and maintains effective dictionary size.
        
        Research Foundation:
        - Aharon et al. (2006): K-SVD replaces unused atoms to maintain full rank
        - Mairal et al. (2010): Online learning benefits from dead atom replacement
        - Helps prevent local minima where some atoms become unused
        
        Args:
            X: Data matrix (n_features, n_samples)
            A: Current sparse codes (n_atoms, n_samples)
        """
        if self._dictionary is None:
            return
            
        # Compute atom usage: max absolute coefficient for each atom across all samples
        atom_usage = np.max(np.abs(A), axis=1)
        
        # Find dead atoms (usage below threshold)
        dead_atoms = atom_usage < self.dead_atom_threshold
        n_dead = np.sum(dead_atoms)
        
        if n_dead > 0:
            if self.verbose:
                print(f"    Replacing {n_dead} dead atoms (usage < {self.dead_atom_threshold})")
                
            # Replace dead atoms with randomly selected data samples
            n_samples = X.shape[1]
            if n_samples > 0:
                # Randomly select data samples to replace dead atoms
                replace_indices = np.random.choice(n_samples, size=n_dead, replace=n_samples < n_dead)
                self._dictionary[:, dead_atoms] = X[:, replace_indices]
                
                # Add small random noise to avoid exact duplicates
                noise_scale = 0.01 * np.std(X)
                noise = np.random.normal(0, noise_scale, size=(X.shape[0], n_dead))
                self._dictionary[:, dead_atoms] += noise
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        """Learned dictionary matrix."""
        return self._dictionary
    
    def get_config(self) -> Dict[str, Any]:
        """Get learner configuration for serialization."""
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
            'random_state': self.random_state,
            'dead_atom_threshold': self.dead_atom_threshold,
            'replace_dead_atoms': self.replace_dead_atoms,
            'lambda_init_method': self.lambda_init_method,
            'lambda_annealing': self.lambda_annealing,
            'convergence_criteria': self.convergence_criteria,
            'gradient_tol': self.gradient_tol
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set learner configuration from serialization."""
        self.n_atoms = config['n_atoms']
        self.max_iter = config['max_iter']
        self.tol = config['tol']
        self.random_state = config['random_state']
        self.dead_atom_threshold = config.get('dead_atom_threshold', 1e-6)
        self.replace_dead_atoms = config.get('replace_dead_atoms', True)
        self.lambda_init_method = config.get('lambda_init_method', 'data_adaptive')
        self.lambda_annealing = config.get('lambda_annealing', None)
        self.convergence_criteria = config.get('convergence_criteria', 'objective')
        self.gradient_tol = config.get('gradient_tol', 1e-4)
        
        # Recreate penalty
        penalty_class = globals()[config['penalty_type']]
        self.penalty = penalty_class(**config['penalty_params'])
        
        # Recreate solver and updater  
        self.solver = get_solver(config['solver_name'], **config['solver_params'])
        updater_config = DictUpdaterConfig(method=config['updater_name'], **config['updater_params'])
        self.dict_updater = create_dict_updater(updater_config)


class OnlineDictionaryLearner:
    """Online/streaming dictionary learning.
    
    Reference: Mairal et al. (2010). Online dictionary learning for sparse coding.
    
    Learns dictionary incrementally from streaming data using stochastic approximation.
    """
    
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
        
        # Configure components like DictionaryLearner
        self.penalty = self._create_penalty(penalty, penalty_params or {})
        self.solver = get_solver(solver, **(solver_params or {}))
        
        # Online learning state
        self._dictionary = None
        self._fitted = False
        self._n_samples_seen = 0
        self._A_avg = None  # Running average of coefficient outer products
        self._B_avg = None  # Running average of data-coefficient cross products
    
    def _create_penalty(self, penalty: Union[str, Penalty], params: Dict[str, Any]) -> Penalty:
        """Same as DictionaryLearner."""
        if isinstance(penalty, str):
            if penalty == 'l1':
                return L1(lam=params.get('lam', 0.1))
            elif penalty == 'l2':
                return L2(lam=params.get('lam', 0.1))
            # Add other penalties as needed
        return penalty
    
    def partial_fit(self, X: ArrayLike, **kwargs) -> 'OnlineDictionaryLearner':
        """
        Incremental learning step.
        
        Args:
            X: Mini-batch of data (n_features, batch_size)
            
        Returns:
            Self for chaining
        """
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
        # Use solve instead of inv for numerical stability (Golub & Van Loan, 2013)
        eps = 1e-7
        regularized_A = self._A_avg + eps * np.eye(self.n_atoms)
        try:
            # D = B @ A^-1 = (A^-T @ B^T)^T solved via A^T @ X = B^T
            self._dictionary = np.linalg.solve(regularized_A, self._B_avg.T).T
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse for severely ill-conditioned cases
            self._dictionary = self._B_avg @ np.linalg.pinv(self._A_avg)
        
        self._normalize_dictionary()
        
        self._n_samples_seen += batch_size
        self._fitted = True
        
        if self.verbose and self._n_samples_seen % 1000 == 0:
            print(f"Processed {self._n_samples_seen} samples")
        
        return self
    
    def fit(self, X: ArrayLike, **kwargs) -> 'OnlineDictionaryLearner':
        """
        Fit using mini-batch processing.
        
        Args:
            X: Training data (n_features, n_samples)
            
        Returns:
            Self for chaining
        """
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
        """Encode data using learned dictionary."""
        if not self._fitted:
            raise ValueError("Must call fit() or partial_fit() before encode()")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.solver.solve(self._dictionary, X, self.penalty)
    
    def decode(self, A: ArrayLike) -> ArrayLike:
        """Decode sparse codes back to data space."""
        return self._dictionary @ A if self._fitted else None
    
    def reset(self) -> None:
        """Reset learner state for fresh training."""
        self._dictionary = None
        self._fitted = False
        self._n_samples_seen = 0
        self._A_avg = None
        self._B_avg = None
    
    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed."""
        return self._n_samples_seen
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        """Learned dictionary matrix."""
        return self._dictionary
    
    def _normalize_dictionary(self):
        """Normalize dictionary atoms to unit norm."""
        if self._dictionary is not None:
            norms = np.linalg.norm(self._dictionary, axis=0)
            norms[norms == 0] = 1
            self._dictionary = self._dictionary / norms[np.newaxis, :]


# Registry of available learners
LEARNERS = {
    'standard': DictionaryLearner,
    'online': OnlineDictionaryLearner,
}


def get_learner(name: str, **kwargs):
    """Factory function for learner instantiation."""
    if name not in LEARNERS:
        raise ValueError(f"Unknown learner '{name}'. Available: {list(LEARNERS.keys())}")
    return LEARNERS[name](**kwargs)