"""
Model persistence with robust serialization.

Handles saving/loading of dictionary learner state with metadata,
version control, and data integrity checks.
"""

import json
import hashlib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
import numpy as np

import joblib

from ..core.array import ArrayLike, get_array_info
try:
    from .. import __version__
except ImportError:
    __version__ = "2.5.0"


@dataclass 
class ModelState:
    """
    Complete model state for serialization.
    
    Attributes
    ----------
    dictionary : np.ndarray
        Learned dictionary matrix
    config : dict
        Model configuration
    metadata : dict
        Additional metadata (training history, etc.)
    version : str
        Library version used to create model
    created : str
        ISO timestamp of creation
    checksum : str
        SHA256 checksum of dictionary for integrity
    """
    dictionary: np.ndarray
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    version: str
    created: str
    checksum: str
    
    @classmethod
    def from_learner(cls, learner, metadata: Optional[Dict[str, Any]] = None) -> 'ModelState':
        """
        Create ModelState from learner instance.
        
        Parameters
        ----------
        learner : object
            Sparse coding learner with dictionary and get_config() method
        metadata : dict, optional
            Additional metadata to store
            
        Returns
        -------
        state : ModelState
            Serializable model state
        """
        # Get dictionary (handle different learner types)
        if hasattr(learner, 'dictionary') and learner.dictionary is not None:
            dictionary = np.asarray(learner.dictionary)
        elif hasattr(learner, 'D') and learner.D is not None:
            dictionary = np.asarray(learner.D)
        else:
            raise ValueError("Learner has no fitted dictionary")
        
        # Get configuration
        if hasattr(learner, 'get_config'):
            config = learner.get_config()
        else:
            # Fallback: extract common attributes
            config = {}
            for attr in ['n_atoms', 'n_components', 'penalty_config', 'solver_config']:
                if hasattr(learner, attr):
                    config[attr] = getattr(learner, attr)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add training statistics if available
        if hasattr(learner, 'training_history') and learner.training_history:
            metadata['training_history'] = learner.training_history
        
        if hasattr(learner, 'get_stats'):
            metadata['stats'] = learner.get_stats()
        
        # Add array info
        metadata['dictionary_info'] = get_array_info(dictionary)
        
        # Compute checksum
        checksum = hashlib.sha256(dictionary.tobytes()).hexdigest()
        
        return cls(
            dictionary=dictionary,
            config=config,
            metadata=metadata,
            version=__version__,
            created=datetime.now().isoformat(),
            checksum=checksum
        )
    
    def verify_integrity(self) -> bool:
        """
        Verify data integrity using checksum.
        
        Returns
        -------
        valid : bool
            True if checksum matches
        """
        current_checksum = hashlib.sha256(self.dictionary.tobytes()).hexdigest()
        return current_checksum == self.checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large arrays)."""
        result = asdict(self)
        # Replace dictionary with shape info for JSON serialization
        result['dictionary'] = {
            'shape': self.dictionary.shape,
            'dtype': str(self.dictionary.dtype),
            'checksum': self.checksum
        }
        return result


def save_model(learner, path: Union[str, Path], 
               metadata: Optional[Dict[str, Any]] = None,
               compress: bool = True,
               include_sklearn: bool = True) -> None:
    """
    Save sparse coding model to disk.
    
    Creates a directory with:
    - model.npz: Dictionary and numerical data
    - config.json: Configuration and metadata
    - sklearn_model.joblib: sklearn-compatible wrapper (if available)
    
    Parameters
    ----------
    learner : object
        Sparse coding learner to save
    path : str or Path
        Directory path to save model
    metadata : dict, optional
        Additional metadata to include
    compress : bool, default=True
        Whether to compress npz file
    include_sklearn : bool, default=True
        Whether to save sklearn-compatible wrapper
        
    Examples
    --------
    >>> from sparse_coding import SparseCoder
    >>> from sparse_coding.serialization import save_model, load_model
    
    >>> # Train model
    >>> learner = SparseCoder(n_atoms=64)
    >>> learner.fit(X)
    >>> 
    >>> # Save model
    >>> save_model(learner, 'my_model', metadata={'dataset': 'CIFAR-10'})
    >>> 
    >>> # Load model
    >>> loaded_learner = load_model('my_model')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Create model state
    state = ModelState.from_learner(learner, metadata)
    
    # Save dictionary and arrays to npz
    arrays = {'dictionary': state.dictionary}
    
    # Add any other arrays from metadata
    for key, value in state.metadata.items():
        if isinstance(value, np.ndarray):
            arrays[f'metadata_{key}'] = value
    
    npz_path = path / 'model.npz'
    if compress:
        np.savez_compressed(npz_path, **arrays)
    else:
        np.savez(npz_path, **arrays)
    
    # Save configuration and metadata to JSON
    config_data = state.to_dict()
    config_path = path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2, default=_json_serialize_helper)
    
    # Save sklearn-compatible wrapper if requested
    if include_sklearn:
        try:
            from ..adapters.sklearn import SparseCoderEstimator
            
            # Create sklearn wrapper
            sklearn_model = SparseCoderEstimator(
                n_atoms=state.config.get('n_atoms', 144),
                penalty=state.config.get('penalty_config', 'l1'),
                solver=state.config.get('solver_config', 'fista')
            )
            
            # Set fitted state
            sklearn_model._learner = learner
            sklearn_model.dictionary_ = state.dictionary.copy()
            sklearn_model.components_ = state.dictionary.T
            sklearn_model.n_features_in_ = state.dictionary.shape[0]
            
            # Save using joblib
            joblib_path = path / 'sklearn_model.joblib'
            joblib.dump(sklearn_model, joblib_path)
            
        except Exception as e:
            warnings.warn(f"Failed to save sklearn wrapper: {e}")
    
    print(f"Model saved to {path}")
    print(f"Dictionary shape: {state.dictionary.shape}")
    print(f"Model version: {state.version}")


def load_model(path: Union[str, Path], 
               verify_integrity: bool = True,
               backend: str = 'auto') -> Any:
    """
    Load sparse coding model from disk.
    
    Parameters
    ----------
    path : str or Path
        Directory containing saved model
    verify_integrity : bool, default=True
        Whether to verify data integrity using checksums
    backend : str, default='auto'
        Backend to use ('auto', 'numpy', 'sklearn')
        
    Returns
    -------
    learner : object
        Loaded sparse coding learner
        
    Examples
    --------
    >>> learner = load_model('my_model')
    >>> codes = learner.encode(test_data)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model directory not found: {path}")
    
    # Load configuration
    config_path = path / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Load arrays
    npz_path = path / 'model.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f"Model data not found: {npz_path}")
    
    data = np.load(npz_path)
    
    # Reconstruct model state
    dictionary = data['dictionary']
    
    # Load metadata arrays
    metadata = config_data.get('metadata', {})
    for key in data.files:
        if key.startswith('metadata_'):
            original_key = key[9:]  # Remove 'metadata_' prefix
            metadata[original_key] = data[key]
    
    state = ModelState(
        dictionary=dictionary,
        config=config_data['config'],
        metadata=metadata,
        version=config_data['version'],
        created=config_data['created'],
        checksum=config_data['checksum']
    )
    
    # Verify integrity
    if verify_integrity and not state.verify_integrity():
        raise ValueError("Model integrity check failed. Data may be corrupted.")
    
    # Determine which loader to use
    if backend == 'sklearn' or (backend == 'auto' and (path / 'sklearn_model.joblib').exists()):
        return _load_sklearn_model(path)
    else:
        return _load_native_model(state)


def _load_sklearn_model(path: Path):
    """Load sklearn-compatible model."""
    joblib_path = path / 'sklearn_model.joblib'
    return joblib.load(joblib_path)


def _load_native_model(state: ModelState):
    """Load native sparse coding model."""
    # Try to determine model type from config
    config = state.config
    
    # Handle streaming learner
    if 'streaming_config' in config:
        from ..streaming.online_learner import OnlineSparseCoderLearner
        
        learner = OnlineSparseCoderLearner(
            n_atoms=config['n_atoms'],
            penalty_config=config.get('penalty_config'),
            solver_config=config.get('solver_config'),
            streaming_config=config.get('streaming_config')
        )
        learner._dictionary = state.dictionary
        learner._n_features = state.dictionary.shape[0]
        return learner
    
    # Handle standard dictionary learner
    elif 'patch_size' in config:
        from ..dictionary_learner import DictionaryLearner
        
        learner = DictionaryLearner(
            n_components=config.get('n_components', config.get('n_atoms', 144)),
            patch_size=config.get('patch_size', (8, 8)),
            sparsity_penalty=config.get('sparsity_penalty', 0.1),
            learning_rate=config.get('learning_rate', 0.01),
            max_iterations=config.get('max_iterations', 1000),
            mode=config.get('mode', 'l1')
        )
        learner.dictionary = state.dictionary
        learner.sparse_coder.D = state.dictionary
        return learner
    
    # Handle basic sparse coder
    else:
        from ..sparse_coder import SparseCoder
        
        learner = SparseCoder(
            n_atoms=config.get('n_atoms', 144),
            lam=config.get('lam'),
            mode=config.get('mode', 'l1'),
            max_iter=config.get('max_iter', 200)
        )
        learner.D = state.dictionary
        return learner


def list_saved_models(directory: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    List saved models in directory with metadata.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search for models
        
    Returns
    -------
    models : list
        List of model info dictionaries
    """
    directory = Path(directory)
    models = []
    
    for path in directory.iterdir():
        if path.is_dir():
            config_path = path / 'config.json'
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    info = {
                        'path': str(path),
                        'name': path.name,
                        'version': config.get('version', 'unknown'),
                        'created': config.get('created'),
                        'dictionary_shape': config.get('metadata', {}).get('dictionary_info', {}).get('shape'),
                        'config_summary': {
                            k: v for k, v in config.get('config', {}).items() 
                            if k in ['n_atoms', 'n_components', 'mode']
                        }
                    }
                    models.append(info)
                except Exception as e:
                    warnings.warn(f"Failed to read model {path}: {e}")
    
    return models


def _json_serialize_helper(obj):
    """Helper for JSON serialization of numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")