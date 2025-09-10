"""
Configurable learner that can be composed from registry components.

This module provides a learner class that dynamically creates sparse coding
systems from configuration dictionaries, using the component registry.
"""

from typing import Dict, Any, Optional
import numpy as np

from ..api.registry import create_from_config
from ..core.interfaces import Learner, ArrayLike


class ConfigurableLearner:
    """
    Configurable sparse coding learner.
    
    Creates a complete sparse coding system from configuration,
    dynamically composing penalties, solvers, and dictionary updaters
    from the component registry.
    
    This enables flexible configuration without hard dependencies on
    specific algorithm implementations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize learner from configuration.
        
        Args:
            config: Complete system configuration including penalty,
                   solver, dict_updater, and other parameters
        """
        self.config = config
        self._dictionary = None
        self._fitted = False
        
        # Create components from config
        penalty_config = config.get('penalty', {'name': 'l1', 'params': {}})
        if isinstance(penalty_config, str):
            penalty_config = {'name': penalty_config, 'params': {}}
        self.penalty = create_from_config({'kind': 'penalty', **penalty_config})
        
        solver_config = config.get('solver', {'name': 'fista', 'params': {}})
        if isinstance(solver_config, str):
            solver_config = {'name': solver_config, 'params': {}}
        self.solver = create_from_config({'kind': 'solver', **solver_config})
        
        dict_updater_config = config.get('dict_updater', {'name': 'mod', 'params': {}})
        if isinstance(dict_updater_config, str):
            dict_updater_config = {'name': dict_updater_config, 'params': {}}
        self.dict_updater = create_from_config({'kind': 'dict_updater', **dict_updater_config})
        
        # Extract learning parameters
        self.n_atoms = config.get('n_atoms', 144)
        self.max_iter = config.get('max_iter', 30)
        self.tol = config.get('tol', 1e-6)
        self.random_state = config.get('random_state', None)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ConfigurableLearner':
        """Create learner from configuration dictionary."""
        return cls(config)
    
    def fit(self, X: ArrayLike, **kwargs) -> 'ConfigurableLearner':
        """
        Learn dictionary from data.
        
        Args:
            X: Training data (n_features, n_samples)
            **kwargs: Additional training parameters
            
        Returns:
            Self (for chaining)
        """
        X = np.asarray(X)
        n_features, n_samples = X.shape
        
        # Initialize dictionary randomly  
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self._dictionary = np.random.randn(n_features, self.n_atoms)
        self._dictionary /= np.linalg.norm(self._dictionary, axis=0, keepdims=True)
        
        # Alternating optimization
        for iteration in range(self.max_iter):
            # Sparse coding step
            codes = self.solver.solve(self._dictionary, X, self.penalty)
            
            # Dictionary update step  
            self._dictionary = self.dict_updater.step(self._dictionary, X, codes)
            
            # Check convergence (optional)
            if iteration > 0 and kwargs.get('check_convergence', False):
                # Simple convergence check based on reconstruction error
                reconstruction = self._dictionary @ codes
                error = np.mean((X - reconstruction) ** 2)
                if error < self.tol:
                    break
        
        self._fitted = True
        return self
    
    def encode(self, X: ArrayLike) -> ArrayLike:
        """
        Encode data using learned dictionary.
        
        Args:
            X: Data to encode (n_features, n_samples)
            
        Returns:
            Sparse codes (n_atoms, n_samples)
        """
        if not self._fitted:
            raise ValueError("Learner must be fitted before encoding")
        
        X = np.asarray(X)
        return self.solver.solve(self._dictionary, X, self.penalty)
    
    def decode(self, codes: ArrayLike) -> ArrayLike:
        """
        Decode sparse codes back to data space.
        
        Args:
            codes: Sparse codes (n_atoms, n_samples)
            
        Returns:
            Reconstructed data (n_features, n_samples)
        """
        if not self._fitted:
            raise ValueError("Learner must be fitted before decoding")
        
        codes = np.asarray(codes)
        return self._dictionary @ codes
    
    @property
    def dictionary(self) -> Optional[ArrayLike]:
        """Learned dictionary matrix."""
        return self._dictionary
    
    def get_config(self) -> Dict[str, Any]:
        """Get learner configuration for serialization."""
        config = self.config.copy()
        
        # Add runtime state if fitted
        if self._fitted and self._dictionary is not None:
            config['_fitted'] = True
            config['_dictionary_shape'] = self._dictionary.shape
        
        return config
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set learner configuration from serialization."""
        self.config = config
        self._fitted = config.get('_fitted', False)
        
        # Recreate components
        penalty_config = config.get('penalty', {'name': 'l1', 'params': {}})
        if isinstance(penalty_config, str):
            penalty_config = {'name': penalty_config, 'params': {}}
        self.penalty = create_from_config({'kind': 'penalty', **penalty_config})
        
        solver_config = config.get('solver', {'name': 'fista', 'params': {}})
        if isinstance(solver_config, str):
            solver_config = {'name': solver_config, 'params': {}}
        self.solver = create_from_config({'kind': 'solver', **solver_config})
        
        dict_updater_config = config.get('dict_updater', {'name': 'mod', 'params': {}})
        if isinstance(dict_updater_config, str):
            dict_updater_config = {'name': dict_updater_config, 'params': {}}
        self.dict_updater = create_from_config({'kind': 'dict_updater', **dict_updater_config})
        
        # Update parameters
        self.n_atoms = config.get('n_atoms', self.n_atoms)
        self.max_iter = config.get('max_iter', self.max_iter)
        self.tol = config.get('tol', self.tol)
        self.random_state = config.get('random_state', self.random_state)