"""
Olshausen & Field sparse coding learner orchestration.

Implements the complete sparse coding framework from Olshausen & Field (1996)
with configurable inference and dictionary update algorithms.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Union, Optional
from dataclasses import dataclass, asdict

from ..penalties.penalty_protocol import PenaltyProtocol
from ..inference.fista_accelerated_solver import FISTASolver
from ..inference.ista_basic_solver import ISTASolver
from ..inference.nonlinear_conjugate_gradient import NonlinearConjugateGradient
from ..inference.orthogonal_matching_pursuit import OrthogonalMatchingPursuit
from ..dictionary.method_optimal_directions import MethodOptimalDirections
from ..dictionary.ksvd_dictionary_learning import KSVDDictionaryLearning
from ..dictionary.gradient_descent_update import GradientDescentUpdate
from ..dictionary.online_dictionary_learning import OnlineDictionaryLearning


@dataclass
class SparseCodingConfig:
    """Configuration for sparse coding learning."""
    
    # Inference algorithm
    inference_method: str = "fista"
    inference_max_iter: int = 1000
    inference_tol: float = 1e-6
    
    # Dictionary update algorithm  
    dict_update_method: str = "mod"
    dict_learning_rate: float = 0.01
    dict_regularization: float = 1e-6
    
    # Training parameters
    n_iterations: int = 100
    regularization_param: float = 0.1
    
    # Initialization
    n_atoms: int = 144
    random_seed: Optional[int] = None


class OlshausenFieldLearner:
    """Sparse coding learner following Olshausen & Field (1996).
    
    Reference: Olshausen & Field (1996). Natural image statistics and efficient 
    coding.
    """
    
    def __init__(self, config: Optional[SparseCodingConfig] = None):
        self.config = config or SparseCodingConfig()
        self.dictionary = None
        self._setup_algorithms()
    
    def _setup_algorithms(self):
        """Initialize inference and dictionary update algorithms."""
        
        # Setup inference solver
        if self.config.inference_method == "fista":
            self.inference_solver = FISTASolver(
                max_iter=self.config.inference_max_iter,
                tol=self.config.inference_tol
            )
        elif self.config.inference_method == "ista":
            self.inference_solver = ISTASolver(
                max_iter=self.config.inference_max_iter,
                tol=self.config.inference_tol
            )
        elif self.config.inference_method == "ncg":
            self.inference_solver = NonlinearConjugateGradient(
                max_iter=self.config.inference_max_iter,
                tol=self.config.inference_tol
            )
        elif self.config.inference_method == "omp":
            self.inference_solver = OrthogonalMatchingPursuit(
                tol=self.config.inference_tol
            )
        else:
            raise ValueError(f"Unknown inference method: {self.config.inference_method}")
        
        # Setup dictionary updater
        if self.config.dict_update_method == "mod":
            self.dict_updater = MethodOptimalDirections(
                regularization=self.config.dict_regularization
            )
        elif self.config.dict_update_method == "ksvd":
            self.dict_updater = KSVDDictionaryLearning()
        elif self.config.dict_update_method == "gradient":
            self.dict_updater = GradientDescentUpdate(
                learning_rate=self.config.dict_learning_rate
            )
        elif self.config.dict_update_method == "online":
            self.dict_updater = OnlineDictionaryLearning(
                regularization=self.config.dict_regularization
            )
        else:
            raise ValueError(f"Unknown dict update method: {self.config.dict_update_method}")
    
    def fit(self, 
            X: np.ndarray, 
            penalty: PenaltyProtocol) -> 'OlshausenFieldLearner':
        """Learn dictionary from training data.
        
        Args:
            X: Training data (n_features, n_samples)
            penalty: Penalty function for sparsity
            
        Returns:
            Self for method chaining
        """
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        n_features, n_samples = X.shape
        
        # Initialize dictionary
        if self.dictionary is None:
            # Random initialization normalized to unit columns
            self.dictionary = np.random.randn(n_features, self.config.n_atoms)
            norms = np.linalg.norm(self.dictionary, axis=0, keepdims=True)
            self.dictionary = self.dictionary / (norms + 1e-12)
        
        # Alternating optimization loop
        for iteration in range(self.config.n_iterations):
            # Sparse inference step
            if self.config.inference_method == "omp":
                # OMP doesn't use penalty, only sparsity constraint
                sparse_codes = np.zeros((self.config.n_atoms, n_samples))
                for i in range(n_samples):
                    sparse_codes[:, i], _ = self.inference_solver.solve(
                        self.dictionary, X[:, i]
                    )
            else:
                sparse_codes = np.zeros((self.config.n_atoms, n_samples))
                for i in range(n_samples):
                    sparse_codes[:, i], _ = self.inference_solver.solve(
                        self.dictionary, X[:, i], penalty, self.config.regularization_param
                    )
            
            # Dictionary update step
            if self.config.dict_update_method == "ksvd":
                self.dictionary, sparse_codes = self.dict_updater.update(
                    self.dictionary, X, sparse_codes
                )
            else:
                self.dictionary = self.dict_updater.update(
                    self.dictionary, X, sparse_codes
                )
        
        return self
    
    def encode(self, 
               X: np.ndarray, 
               penalty: PenaltyProtocol) -> np.ndarray:
        """Encode signals using learned dictionary.
        
        Args:
            X: Input signals (n_features, n_samples)
            penalty: Penalty function for sparsity
            
        Returns:
            Sparse codes (n_atoms, n_samples)
        """
        if self.dictionary is None:
            raise ValueError("Must fit model before encoding")
        
        n_samples = X.shape[1]
        sparse_codes = np.zeros((self.config.n_atoms, n_samples))
        
        for i in range(n_samples):
            if self.config.inference_method == "omp":
                sparse_codes[:, i], _ = self.inference_solver.solve(
                    self.dictionary, X[:, i]
                )
            else:
                sparse_codes[:, i], _ = self.inference_solver.solve(
                    self.dictionary, X[:, i], penalty, self.config.regularization_param
                )
        
        return sparse_codes
    
    def decode(self, sparse_codes: np.ndarray) -> np.ndarray:
        """Reconstruct signals from sparse codes.
        
        Args:
            sparse_codes: Sparse codes (n_atoms, n_samples)
            
        Returns:
            Reconstructed signals (n_features, n_samples)
        """
        if self.dictionary is None:
            raise ValueError("Must fit model before decoding")
        
        return self.dictionary @ sparse_codes
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return asdict(self.config)
    
    def set_config(self, config: Union[Dict[str, Any], SparseCodingConfig]):
        """Set new configuration and reinitialize algorithms."""
        if isinstance(config, dict):
            # Update existing config with provided values
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    raise ValueError(f"Unknown config parameter: {key}")
        else:
            self.config = config
        
        self._setup_algorithms()