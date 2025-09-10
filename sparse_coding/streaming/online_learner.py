"""
Online/streaming dictionary learning implementation.

Provides incremental dictionary updates with partial_fit interface,
adaptive learning rates, and streaming statistics tracking.
"""

import numpy as np
import warnings
from typing import Optional, Dict, Any, Union, List, Tuple
from dataclasses import dataclass
from threading import Lock
import time

from ..core.array import ArrayLike, xp, as_same, ensure_array
from ..core.interfaces import StreamingLearner, Penalty, InferenceSolver, DictUpdater
from ..api.registry import create_from_config


@dataclass
class StreamingConfig:
    """
    Configuration for streaming dictionary learning.
    
    Attributes
    ----------
    buffer_size : int, default=1000
        Size of internal sample buffer for batch processing
    learning_rate : float, default=0.01
        Initial dictionary learning rate
    lr_decay : float, default=0.99
        Learning rate decay factor per batch
    min_lr : float, default=1e-5
        Minimum learning rate
    momentum : float, default=0.9
        Momentum factor for dictionary updates
    adaptive_lr : bool, default=True
        Whether to use adaptive learning rates per atom
    warm_start_batches : int, default=5
        Number of batches for warm start (higher learning rate)
    forgetting_factor : float, default=0.99
        Exponential forgetting for running statistics
    dead_atom_threshold : float, default=1e-6
        Threshold for detecting dead atoms
    reinit_dead_atoms : bool, default=True
        Whether to reinitialize dead atoms
    thread_safe : bool, default=True
        Enable thread-safe operations
    """
    buffer_size: int = 1000
    learning_rate: float = 0.01
    lr_decay: float = 0.99
    min_lr: float = 1e-5
    momentum: float = 0.9
    adaptive_lr: bool = True
    warm_start_batches: int = 5
    forgetting_factor: float = 0.99
    dead_atom_threshold: float = 1e-6
    reinit_dead_atoms: bool = True
    thread_safe: bool = True


class OnlineSparseCoderLearner(StreamingLearner):
    """
    Online dictionary learning with partial_fit interface.
    
    Implements streaming dictionary learning with adaptive learning rates,
    momentum, and automatic dead atom handling. Thread-safe for concurrent access.
    
    Parameters
    ----------
    n_atoms : int
        Number of dictionary atoms
    penalty_config : dict
        Penalty configuration
    solver_config : dict  
        Solver configuration
    streaming_config : StreamingConfig, optional
        Streaming-specific configuration
    random_state : int or None, default=None
        Random seed for reproducibility
    
    Examples
    --------
    >>> from sparse_coding.streaming import OnlineSparseCoderLearner
    >>> import numpy as np
    
    >>> # Setup streaming learner
    >>> learner = OnlineSparseCoderLearner(n_atoms=64)
    >>> 
    >>> # Streaming training loop
    >>> for batch in data_stream:
    ...     learner.partial_fit(batch)
    ...     
    ...     if learner.n_samples_seen % 1000 == 0:
    ...         print(f"Processed {learner.n_samples_seen} samples")
    >>> 
    >>> # Use learned dictionary
    >>> codes = learner.encode(test_data)
    """
    
    def __init__(
        self,
        n_atoms: int,
        penalty_config: Optional[Dict[str, Any]] = None,
        solver_config: Optional[Dict[str, Any]] = None,
        streaming_config: Optional[StreamingConfig] = None,
        random_state: Optional[int] = None
    ):
        # Default configurations
        if penalty_config is None:
            penalty_config = {"name": "l1", "params": {"lam": 0.1}}
        if solver_config is None:
            solver_config = {"name": "fista", "params": {"max_iter": 50}}
        if streaming_config is None:
            streaming_config = StreamingConfig()
        
        self.n_atoms = n_atoms
        self.penalty_config = penalty_config
        self.solver_config = solver_config
        self.streaming_config = streaming_config
        self.random_state = random_state
        
        # Initialize random number generator
        self.rng = np.random.default_rng(random_state)
        
        # Create components
        self._penalty = create_from_config({"kind": "penalty", **penalty_config})
        self._solver = create_from_config({"kind": "solver", **solver_config})
        
        # State variables
        self._dictionary = None
        self._n_features = None
        self._n_samples_seen = 0
        self._n_batches_seen = 0
        
        # Adaptive learning rate state
        self._current_lr = streaming_config.learning_rate
        self._atom_lrs = None
        self._momentum_buffer = None
        
        # Running statistics
        self._atom_usage_stats = None
        self._reconstruction_error_ema = None
        self._sparsity_level_ema = None
        
        # Sample buffer for batch processing
        self._sample_buffer = []
        
        # Thread safety
        self._lock = Lock() if streaming_config.thread_safe else None
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        """Get current dictionary matrix."""
        return self._dictionary
    
    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed."""
        return self._n_samples_seen
    
    def _safe_operation(self, func, *args, **kwargs):
        """Execute operation with thread safety if enabled."""
        if self._lock is not None:
            with self._lock:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _initialize_dictionary(self, X: ArrayLike) -> None:
        """Initialize dictionary from first batch."""
        backend = xp(X)
        n_features, n_samples = X.shape
        
        self._n_features = n_features
        
        # Initialize dictionary - sample from data + random
        n_data_atoms = min(self.n_atoms, n_samples)
        if n_data_atoms > 0:
            # Sample columns from data
            indices = self.rng.choice(n_samples, n_data_atoms, replace=False)
            D = X[:, indices].copy()
        else:
            D = backend.zeros((n_features, 0))
        
        # Add random atoms if needed
        if self.n_atoms > n_data_atoms:
            n_random = self.n_atoms - n_data_atoms
            random_atoms = self.rng.normal(
                scale=backend.std(X), 
                size=(n_features, n_random)
            )
            random_atoms = as_same(random_atoms, X)
            D = backend.concatenate([D, random_atoms], axis=1)
        
        # Normalize dictionary
        norms = backend.linalg.norm(D, axis=0, keepdims=True)
        norms = backend.where(norms < 1e-12, 1.0, norms)
        self._dictionary = D / norms
        
        # Initialize adaptive learning rate state
        if self.streaming_config.adaptive_lr:
            self._atom_lrs = backend.full(self.n_atoms, self._current_lr)
        
        # Initialize momentum buffer
        if self.streaming_config.momentum > 0:
            self._momentum_buffer = backend.zeros_like(self._dictionary)
        
        # Initialize statistics tracking
        self._atom_usage_stats = backend.zeros(self.n_atoms)
        self._reconstruction_error_ema = 0.0
        self._sparsity_level_ema = 0.0
    
    def _update_learning_rate(self) -> None:
        """Update learning rate with decay and adaptation."""
        config = self.streaming_config
        
        # Global learning rate decay
        if self._n_batches_seen >= config.warm_start_batches:
            self._current_lr = max(
                config.min_lr,
                self._current_lr * config.lr_decay
            )
        
        # Adaptive per-atom learning rates
        if config.adaptive_lr and self._atom_lrs is not None:
            backend = xp(self._dictionary)
            
            # Increase LR for underused atoms, decrease for overused
            target_usage = 1.0 / self.n_atoms
            usage_ratios = self._atom_usage_stats / (target_usage + 1e-12)
            
            # Adjust learning rates
            lr_multipliers = backend.where(
                usage_ratios < 0.5, 1.2,  # Increase LR for underused atoms
                backend.where(usage_ratios > 2.0, 0.8, 1.0)  # Decrease for overused
            )
            
            self._atom_lrs = backend.clip(
                self._atom_lrs * lr_multipliers,
                config.min_lr, 
                config.learning_rate * 2.0
            )
    
    def _detect_and_reinit_dead_atoms(self, X: ArrayLike) -> None:
        """Detect and reinitialize dead atoms."""
        if not self.streaming_config.reinit_dead_atoms:
            return
        
        backend = xp(self._dictionary)
        
        # Find dead atoms (very low usage)
        dead_mask = self._atom_usage_stats < self.streaming_config.dead_atom_threshold
        n_dead = backend.sum(dead_mask)
        
        if n_dead > 0:
            # Reinitialize dead atoms with random samples from current batch
            n_samples = X.shape[1]
            if n_samples > 0:
                # Sample random columns from X
                n_reinit = min(int(n_dead), n_samples)
                indices = self.rng.choice(n_samples, n_reinit, replace=False)
                new_atoms = X[:, indices]
                
                # Normalize new atoms
                norms = backend.linalg.norm(new_atoms, axis=0, keepdims=True)
                norms = backend.where(norms < 1e-12, 1.0, norms)
                new_atoms = new_atoms / norms
                
                # Replace dead atoms
                dead_indices = backend.where(dead_mask)[0][:n_reinit]
                self._dictionary = self._dictionary.at[:, dead_indices].set(new_atoms)
                
                # Reset usage stats for reinitialized atoms
                self._atom_usage_stats = self._atom_usage_stats.at[dead_indices].set(0.0)
                
                if self.streaming_config.adaptive_lr and self._atom_lrs is not None:
                    # Reset learning rates for reinitialized atoms
                    self._atom_lrs = self._atom_lrs.at[dead_indices].set(self._current_lr)
    
    def _update_statistics(self, X: ArrayLike, A: ArrayLike) -> None:
        """Update running statistics."""
        backend = xp(X)
        config = self.streaming_config
        
        # Update atom usage statistics
        atom_activity = backend.mean(backend.abs(A), axis=1)
        self._atom_usage_stats = (
            config.forgetting_factor * self._atom_usage_stats + 
            (1 - config.forgetting_factor) * atom_activity
        )
        
        # Update reconstruction error EMA
        reconstruction = self._dictionary @ A
        error = backend.mean((X - reconstruction) ** 2)
        self._reconstruction_error_ema = (
            config.forgetting_factor * self._reconstruction_error_ema +
            (1 - config.forgetting_factor) * float(error)
        )
        
        # Update sparsity level EMA
        sparsity = backend.mean(backend.abs(A) < 1e-6)
        self._sparsity_level_ema = (
            config.forgetting_factor * self._sparsity_level_ema +
            (1 - config.forgetting_factor) * float(sparsity)
        )
    
    def _dictionary_update_step(self, X: ArrayLike, A: ArrayLike) -> None:
        """Single dictionary update step with momentum."""
        backend = xp(self._dictionary)
        config = self.streaming_config
        
        # Compute gradient: dD = (X - D*A) * A^T
        residual = X - self._dictionary @ A
        gradient = residual @ A.T
        
        # Normalize gradient by batch size
        gradient = gradient / A.shape[1]
        
        # Apply momentum if enabled
        if config.momentum > 0 and self._momentum_buffer is not None:
            self._momentum_buffer = (
                config.momentum * self._momentum_buffer +
                (1 - config.momentum) * gradient
            )
            effective_gradient = self._momentum_buffer
        else:
            effective_gradient = gradient
        
        # Apply learning rate (per-atom if adaptive)
        if config.adaptive_lr and self._atom_lrs is not None:
            lr_matrix = self._atom_lrs[backend.newaxis, :]  # (1, n_atoms)
            update = lr_matrix * effective_gradient
        else:
            update = self._current_lr * effective_gradient
        
        # Update dictionary
        self._dictionary = self._dictionary + update
        
        # Normalize dictionary columns
        norms = backend.linalg.norm(self._dictionary, axis=0, keepdims=True)
        norms = backend.where(norms < 1e-12, 1.0, norms)
        self._dictionary = self._dictionary / norms
    
    def partial_fit(self, X: ArrayLike, **kwargs) -> 'OnlineSparseCoderLearner':
        """
        Incremental learning step.
        
        Parameters
        ----------
        X : ArrayLike of shape (n_features, n_samples) or (n_features,)
            Mini-batch of training data
        **kwargs : dict
            Additional parameters (n_steps, etc.)
            
        Returns
        -------
        self : OnlineSparseCoderLearner
        """
        def _partial_fit_impl():
            X_processed = ensure_array(X)
            if X_processed.ndim == 1:
                X_processed = X_processed[:, np.newaxis]
            
            n_features, n_samples = X_processed.shape
            
            # Initialize dictionary on first batch
            if self._dictionary is None:
                self._initialize_dictionary(X_processed)
            elif n_features != self._n_features:
                raise ValueError(
                    f"Expected {self._n_features} features, got {n_features}"
                )
            
            # Add to buffer if using buffering
            if self.streaming_config.buffer_size > 1:
                self._sample_buffer.append(X_processed)
                total_buffered = sum(batch.shape[1] for batch in self._sample_buffer)
                
                if total_buffered < self.streaming_config.buffer_size:
                    return self  # Wait for more samples
                
                # Process buffered samples
                X_batch = xp(X_processed).concatenate(self._sample_buffer, axis=1)
                self._sample_buffer.clear()
            else:
                X_batch = X_processed
            
            # Sparse coding step
            A = self._solver.solve(self._dictionary, X_batch, self._penalty)
            
            # Dictionary update step
            n_steps = kwargs.get('n_steps', 1)
            for _ in range(n_steps):
                self._dictionary_update_step(X_batch, A)
                
                # Re-encode after dictionary update (for better convergence)
                if n_steps > 1:
                    A = self._solver.solve(self._dictionary, X_batch, self._penalty)
            
            # Update statistics and learning rate
            self._update_statistics(X_batch, A)
            self._update_learning_rate()
            
            # Handle dead atoms
            self._detect_and_reinit_dead_atoms(X_batch)
            
            # Update counters
            self._n_samples_seen += X_batch.shape[1]
            self._n_batches_seen += 1
            
            return self
        
        return self._safe_operation(_partial_fit_impl)
    
    def fit(self, X: ArrayLike, **kwargs) -> 'OnlineSparseCoderLearner':
        """
        Fit dictionary using batch learning (convenience method).
        
        Parameters
        ----------
        X : ArrayLike of shape (n_features, n_samples)
            Training data
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        self : OnlineSparseCoderLearner
        """
        # Reset state
        self.reset()
        
        # Process as single large batch
        return self.partial_fit(X, **kwargs)
    
    def encode(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """
        Encode data using current dictionary.
        
        Parameters
        ----------
        X : ArrayLike of shape (n_features, n_samples)
            Data to encode
        **kwargs : dict
            Additional solver parameters
            
        Returns
        -------
        codes : ArrayLike of shape (n_atoms, n_samples)
            Sparse codes
        """
        if self._dictionary is None:
            raise RuntimeError("Learner not fitted. Call partial_fit() first.")
        
        def _encode_impl():
            X_processed = ensure_array(X)
            if X_processed.ndim == 1:
                X_processed = X_processed[:, np.newaxis]
            
            return self._solver.solve(self._dictionary, X_processed, self._penalty)
        
        return self._safe_operation(_encode_impl)
    
    def decode(self, A: ArrayLike) -> ArrayLike:
        """
        Decode sparse codes.
        
        Parameters
        ----------
        A : ArrayLike of shape (n_atoms, n_samples)
            Sparse codes
            
        Returns
        -------
        reconstructed : ArrayLike of shape (n_features, n_samples)
            Reconstructed data
        """
        if self._dictionary is None:
            raise RuntimeError("Learner not fitted. Call partial_fit() first.")
        
        def _decode_impl():
            A_processed = ensure_array(A)
            return self._dictionary @ A_processed
        
        return self._safe_operation(_decode_impl)
    
    def reset(self) -> None:
        """Reset learner state for fresh training."""
        def _reset_impl():
            self._dictionary = None
            self._n_features = None
            self._n_samples_seen = 0
            self._n_batches_seen = 0
            self._current_lr = self.streaming_config.learning_rate
            self._atom_lrs = None
            self._momentum_buffer = None
            self._atom_usage_stats = None
            self._reconstruction_error_ema = None
            self._sparsity_level_ema = None
            self._sample_buffer.clear()
        
        self._safe_operation(_reset_impl)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current learning statistics.
        
        Returns
        -------
        stats : dict
            Dictionary with learning statistics
        """
        def _get_stats_impl():
            return {
                'n_samples_seen': self._n_samples_seen,
                'n_batches_seen': self._n_batches_seen,
                'current_lr': self._current_lr,
                'reconstruction_error': self._reconstruction_error_ema,
                'sparsity_level': self._sparsity_level_ema,
                'dictionary_shape': self._dictionary.shape if self._dictionary is not None else None,
                'atom_usage_mean': float(np.mean(self._atom_usage_stats)) if self._atom_usage_stats is not None else None,
                'atom_usage_std': float(np.std(self._atom_usage_stats)) if self._atom_usage_stats is not None else None,
                'n_dead_atoms': int(np.sum(self._atom_usage_stats < self.streaming_config.dead_atom_threshold)) if self._atom_usage_stats is not None else 0
            }
        
        return self._safe_operation(_get_stats_impl)
    
    def get_config(self) -> Dict[str, Any]:
        """Get learner configuration."""
        return {
            'n_atoms': self.n_atoms,
            'penalty_config': self.penalty_config,
            'solver_config': self.solver_config,
            'streaming_config': self.streaming_config.__dict__,
            'random_state': self.random_state
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set learner configuration."""
        self.n_atoms = config['n_atoms']
        self.penalty_config = config['penalty_config']
        self.solver_config = config['solver_config']
        self.streaming_config = StreamingConfig(**config['streaming_config'])
        self.random_state = config['random_state']
        
        # Recreate components
        self._penalty = create_from_config({"kind": "penalty", **self.penalty_config})
        self._solver = create_from_config({"kind": "solver", **self.solver_config})
        self.rng = np.random.default_rng(self.random_state)