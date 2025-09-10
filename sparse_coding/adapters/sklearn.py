"""
Scikit-learn compatible estimators for sparse coding.

Provides BaseEstimator and TransformerMixin interfaces with proper
parameter validation, fit/transform pattern, and pipeline compatibility.
"""

import numpy as np
from typing import Optional, Dict, Any, Union
import warnings

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_array, check_is_fitted
    from sklearn.utils import check_random_state
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    # Provide stubs for development without sklearn
    class BaseEstimator:
        def get_params(self, deep=True):
            return {}
        def set_params(self, **params):
            return self
    
    class TransformerMixin:
        def fit_transform(self, X, y=None, **fit_params):
            return self.fit(X, y, **fit_params).transform(X)
    
    def check_array(X, **kwargs):
        return np.asarray(X)
    
    def check_is_fitted(estimator, attributes=None):
        pass
    
    def check_random_state(seed):
        return np.random.RandomState(seed)

from ..api.registry import create_from_config
from ..api.config import create_default_config, validate_config


class SparseCoderEstimator(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible sparse coding estimator.
    
    Implements fit/transform pattern for sparse coding with configurable
    penalties, solvers, and dictionary updaters via plugin system.
    
    Parameters
    ----------
    n_atoms : int, default=144
        Number of dictionary atoms
    penalty : str or dict, default="l1"  
        Penalty configuration (name or full config dict)
    solver : str or dict, default="fista"
        Solver configuration (name or full config dict)
    dict_updater : str or dict, default="mod"
        Dictionary updater configuration (name or full config dict)
    max_iter : int, default=30
        Maximum number of dictionary learning iterations
    tol : float, default=1e-6
        Convergence tolerance
    random_state : int, RandomState or None, default=None
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print training progress
    
    Attributes
    ----------
    dictionary_ : ndarray of shape (n_features, n_atoms)
        Learned dictionary matrix
    components_ : ndarray of shape (n_atoms, n_features)
        Dictionary components (sklearn convention: transposed dictionary)
    n_features_in_ : int
        Number of features seen during fit
    feature_names_in_ : ndarray
        Names of features seen during fit (if input is DataFrame)
    
    Examples
    --------
    >>> from sparse_coding.adapters import SparseCoderEstimator
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> import numpy as np
    
    >>> X = np.random.randn(100, 50)  # (samples, features)
    >>> 
    >>> # Basic usage
    >>> coder = SparseCoderEstimator(n_atoms=20)
    >>> codes = coder.fit_transform(X)
    >>> reconstructed = coder.inverse_transform(codes)
    >>>
    >>> # Pipeline integration
    >>> pipe = Pipeline([
    ...     ('scale', StandardScaler()),
    ...     ('sparse', SparseCoderEstimator(n_atoms=20, penalty='l1'))
    ... ])
    >>> sparse_features = pipe.fit_transform(X)
    >>>
    >>> # Custom configuration
    >>> config_coder = SparseCoderEstimator(
    ...     penalty={'name': 'elastic_net', 'params': {'l1_ratio': 0.5}},
    ...     solver={'name': 'fista', 'params': {'max_iter': 500}}
    ... )
    """
    
    def __init__(
        self,
        n_atoms: int = 144,
        penalty: Union[str, Dict[str, Any]] = "l1",
        solver: Union[str, Dict[str, Any]] = "fista", 
        dict_updater: Union[str, Dict[str, Any]] = "mod",
        max_iter: int = 30,
        tol: float = 1e-6,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: bool = False
    ):
        self.n_atoms = n_atoms
        self.penalty = penalty
        self.solver = solver
        self.dict_updater = dict_updater
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
    
    def _build_config(self) -> Dict[str, Any]:
        """Build configuration from parameters."""
        def _normalize_component_config(component, component_type):
            if isinstance(component, str):
                return {"name": component, "params": {}}
            elif isinstance(component, dict):
                if "name" not in component:
                    raise ValueError(f"{component_type} config must have 'name' field")
                return component
            else:
                raise ValueError(f"{component_type} must be string or dict, got {type(component)}")
        
        config = {
            "penalty": _normalize_component_config(self.penalty, "penalty"),
            "solver": _normalize_component_config(self.solver, "solver"),
            "dict_updater": _normalize_component_config(self.dict_updater, "dict_updater"),
            "learner": {
                "n_atoms": self.n_atoms,
                "max_iterations": self.max_iter,
                "tolerance": self.tol,
                "random_seed": self.random_state,
                "verbose": self.verbose
            }
        }
        
        return config
    
    def fit(self, X, y=None):
        """
        Learn dictionary from data.
        
        Parameters
        ---------- 
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for sklearn compatibility
            
        Returns
        -------
        self : object
            Returns self for method chaining
        """
        # Input validation
        X = check_array(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        if n_features < self.n_atoms:
            warnings.warn(
                f"n_atoms ({self.n_atoms}) > n_features ({n_features}). "
                f"Consider reducing n_atoms or using more features."
            )
        
        # Store input info for sklearn compatibility
        self.n_features_in_ = n_features
        if hasattr(X, 'columns'):  # pandas DataFrame
            self.feature_names_in_ = np.array(X.columns)
        
        # Build and validate configuration
        config = self._build_config()
        validate_config(config, strict=True)
        
        # Create learner from config
        from ..learners.configurable import ConfigurableLearner
        self._learner = ConfigurableLearner.from_config(config)
        
        # Fit (transpose for internal convention: features x samples)
        self._learner.fit(X.T)
        
        # Store results in sklearn format
        self.dictionary_ = self._learner.dictionary.copy()
        self.components_ = self.dictionary_.T  # sklearn convention
        
        return self
    
    def transform(self, X):
        """
        Encode data using learned dictionary.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to encode
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_atoms)
            Sparse codes
        """
        check_is_fitted(self, 'dictionary_')
        X = check_array(X, dtype=np.float64)
        
        # Encode (transpose for internal convention)
        codes = self._learner.encode(X.T)
        
        # Return in sklearn format (samples x atoms)
        return codes.T
    
    def inverse_transform(self, X):
        """
        Reconstruct data from sparse codes.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_atoms)
            Sparse codes
            
        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)  
            Reconstructed data
        """
        check_is_fitted(self, 'dictionary_')
        X = check_array(X, dtype=np.float64)
        
        # Decode (transpose for internal convention)
        reconstructed = self._learner.decode(X.T)
        
        # Return in sklearn format (samples x features)
        return reconstructed.T
    
    def score(self, X, y=None):
        """
        Return the reconstruction score (negative MSE).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        y : Ignored
            
        Returns
        -------
        score : float
            Negative mean squared reconstruction error
        """
        X_transformed = self.transform(X) 
        X_reconstructed = self.inverse_transform(X_transformed)
        mse = np.mean((X - X_reconstructed) ** 2)
        return -mse
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names
            
        Returns
        -------
        feature_names_out : ndarray of str
            Output feature names
        """
        check_is_fitted(self, 'dictionary_')
        return np.array([f'atom_{i}' for i in range(self.n_atoms)])
    
    def _more_tags(self):
        """Additional sklearn tags for compatibility."""
        return {
            'requires_fit': True,
            'requires_positive_X': False,
            'X_types': ['2darray'],
            'allow_nan': False,
            'stateless': False,
            'poor_score': True  # Reconstruction score not great for model selection
        }


class DictionaryLearnerEstimator(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible dictionary learning for image patches.
    
    Specialized estimator for learning dictionaries from image patches
    with spatial pooling for feature extraction.
    
    Parameters
    ----------
    n_components : int, default=100
        Number of dictionary atoms  
    patch_size : tuple of int, default=(8, 8)
        Size of image patches
    sparsity_penalty : float, default=0.1
        L1 regularization strength
    learning_rate : float, default=0.01
        Dictionary update learning rate
    max_iterations : int, default=100
        Maximum training iterations
    pooling : str, default='max'
        Pooling method ('max', 'mean', 'sum') for feature extraction
    overlap_factor : float, default=0.5
        Patch overlap factor (0=no overlap, 0.5=50% overlap)
    random_state : int, RandomState or None, default=None
        Random seed
    verbose : bool, default=False
        Whether to print progress
    
    Attributes
    ----------
    dictionary_ : ndarray of shape (patch_h*patch_w, n_components)
        Learned dictionary matrix
    components_ : ndarray of shape (n_components, patch_h*patch_w)  
        Dictionary atoms reshaped as patches
    training_history_ : dict
        Training metrics history
    
    Examples
    --------
    >>> import numpy as np
    >>> from sparse_coding.adapters import DictionaryLearnerEstimator
    
    >>> # Generate random "images"
    >>> images = np.random.randn(10, 32, 32)  # 10 images of 32x32
    >>> 
    >>> # Learn dictionary and extract features
    >>> learner = DictionaryLearnerEstimator(
    ...     n_components=50, 
    ...     patch_size=(8, 8),
    ...     max_iterations=10
    ... )
    >>> features = learner.fit_transform(images)  # (10, 50)
    >>>
    >>> # Get learned atoms as images
    >>> atoms = learner.get_dictionary_atoms()  # (50, 8, 8)
    """
    
    def __init__(
        self,
        n_components: int = 100,
        patch_size: tuple = (8, 8),
        sparsity_penalty: float = 0.1,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        pooling: str = 'max',
        overlap_factor: float = 0.5,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: bool = False
    ):
        self.n_components = n_components
        self.patch_size = patch_size  
        self.sparsity_penalty = sparsity_penalty
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.pooling = pooling
        self.overlap_factor = overlap_factor
        self.random_state = random_state
        self.verbose = verbose
    
    def fit(self, X, y=None):
        """
        Learn dictionary from images.
        
        Parameters
        ----------
        X : array-like of shape (n_images, height, width)
            Input images
        y : Ignored
            
        Returns
        -------
        self : object
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim not in [2, 3]:
            raise ValueError(f"X must be 2D (single image) or 3D (multiple images), got {X.ndim}D")
        
        if X.ndim == 2:
            X = X[np.newaxis, :, :]  # Add batch dimension
        
        # Create dictionary learner
        from ..dictionary_learner import DictionaryLearner
        
        self._learner = DictionaryLearner(
            n_components=self.n_components,
            patch_size=self.patch_size,
            sparsity_penalty=self.sparsity_penalty,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
            random_seed=self.random_state
        )
        
        # Train
        self.training_history_ = self._learner.fit(X, 
            overlap_factor=self.overlap_factor, 
            verbose=self.verbose
        )
        
        # Store results in sklearn format
        self.dictionary_ = self._learner.dictionary.copy()
        self.components_ = self.dictionary_.T
        
        return self
    
    def transform(self, X):
        """
        Extract sparse features from images.
        
        Parameters
        ----------
        X : array-like of shape (n_images, height, width)
            Input images
            
        Returns
        -------
        X_transformed : ndarray of shape (n_images, n_components)
            Sparse features
        """
        check_is_fitted(self, 'dictionary_')
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 2:
            X = X[np.newaxis, :, :]
        
        # Extract features
        features = self._learner.transform(X, pooling=self.pooling)
        
        return features
    
    def get_dictionary_atoms(self):
        """
        Get dictionary atoms reshaped as image patches.
        
        Returns
        -------
        atoms : ndarray of shape (n_components, patch_h, patch_w)
            Dictionary atoms as image patches
        """
        check_is_fitted(self, 'dictionary_')
        return self._learner.get_dictionary_atoms()
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self, 'dictionary_')
        return np.array([f'dict_atom_{i}' for i in range(self.n_components)])