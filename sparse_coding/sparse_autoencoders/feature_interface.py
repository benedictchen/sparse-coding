"""
Unified feature interface bridging classical sparse coding and modern SAEs.

Provides consistent API for feature extraction using dictionary learning,
sparse autoencoders, or hybrid approaches. Enables seamless switching
between methods for research and interpretability workflows.
"""

import numpy as np
from typing import Union, Dict, Any, Optional, Literal, Tuple
from dataclasses import dataclass
import warnings

from ..core.array import ArrayLike, xp, as_same, ensure_array
from ..sparse_coder import SparseCoder
from ..dictionary_learner import DictionaryLearner

import torch
from .torch_sae import SAE, L1SAE, TopKSAE


@dataclass
class Features:
    """
    Unified feature representation for different sparse coding methods.
    
    Attributes
    ----------
    dictionary : ArrayLike, shape (n_features, n_atoms)
        Learned dictionary/decoder matrix
    method : str
        Method used ('dict', 'sae', 'hybrid')
    metadata : dict
        Additional method-specific information
    """
    dictionary: ArrayLike
    method: str
    metadata: Dict[str, Any]
    
    @property
    def n_features(self) -> int:
        """Number of input features."""
        return self.dictionary.shape[0]
    
    @property 
    def n_atoms(self) -> int:
        """Number of dictionary atoms/latents."""
        return self.dictionary.shape[1]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Dictionary shape (n_features, n_atoms)."""
        return self.dictionary.shape


class FeatureExtractor:
    """
    Unified feature extraction supporting multiple sparse coding methods.
    
    Provides consistent interface for dictionary learning, SAE training,
    and hybrid approaches. Handles backend compatibility automatically.
    
    Parameters
    ----------
    method : str, default='dict'
        Extraction method ('dict', 'sae', 'hybrid')
    n_atoms : int, default=256
        Number of dictionary atoms/SAE latents
    sparsity : float, default=0.1
        Sparsity parameter (lambda for dict, L1 penalty for SAE)
    **kwargs
        Method-specific parameters
    """
    
    def __init__(
        self,
        method: Literal['dict', 'sae', 'hybrid'] = 'dict',
        n_atoms: int = 256,
        sparsity: float = 0.1,
        **kwargs
    ):
        self.method = method
        self.n_atoms = n_atoms
        self.sparsity = sparsity
        self.kwargs = kwargs
        self._fitted_features: Optional[Features] = None
        
    
    def fit(self, X: ArrayLike, **fit_kwargs) -> 'FeatureExtractor':
        """
        Fit feature extraction method on data.
        
        Parameters
        ----------
        X : ArrayLike, shape (n_samples, n_features)
            Training data
        **fit_kwargs
            Additional fitting parameters
            
        Returns
        -------
        self : FeatureExtractor
            Fitted extractor
        """
        X = ensure_array(X)
        
        if self.method == 'dict':
            features = self._fit_dictionary(X, **fit_kwargs)
        elif self.method == 'sae':
            features = self._fit_sae(X, **fit_kwargs)
        elif self.method == 'hybrid':
            features = self._fit_hybrid(X, **fit_kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self._fitted_features = features
        return self
    
    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Transform data to sparse codes using fitted features.
        
        Parameters
        ----------
        X : ArrayLike, shape (n_samples, n_features)
            Data to encode
            
        Returns
        -------
        A : ArrayLike, shape (n_samples, n_atoms)
            Sparse codes/activations
        """
        if self._fitted_features is None:
            raise ValueError("Must call fit() before transform()")
        
        return encode_features(X, self._fitted_features)
    
    def fit_transform(self, X: ArrayLike, **fit_kwargs) -> ArrayLike:
        """Fit features and transform data in one step."""
        return self.fit(X, **fit_kwargs).transform(X)
    
    def inverse_transform(self, A: ArrayLike) -> ArrayLike:
        """
        Reconstruct data from sparse codes.
        
        Parameters
        ----------  
        A : ArrayLike, shape (n_samples, n_atoms)
            Sparse codes/activations
            
        Returns
        -------
        X_hat : ArrayLike, shape (n_samples, n_features)
            Reconstructed data
        """
        if self._fitted_features is None:
            raise ValueError("Must call fit() before inverse_transform()")
            
        return decode_features(A, self._fitted_features)
    
    @property
    def features_(self) -> Features:
        """Access fitted features."""
        if self._fitted_features is None:
            raise ValueError("Must call fit() first")
        return self._fitted_features
    
    def _fit_dictionary(self, X: ArrayLike, **kwargs) -> Features:
        """Fit classical dictionary learning."""
        # Map common parameter names to SparseCoder interface
        fit_params = {}
        if 'max_iter' in kwargs:
            fit_params['n_steps'] = kwargs['max_iter']
        if 'n_iterations' in kwargs:
            fit_params['n_steps'] = kwargs['n_iterations']
        if 'n_steps' in kwargs:
            fit_params['n_steps'] = kwargs['n_steps']
        if 'lr' in kwargs:
            fit_params['lr'] = kwargs['lr']
        if 'learning_rate' in kwargs:
            fit_params['lr'] = kwargs['learning_rate']
        
        # Filter out non-SparseCoder constructor parameters from self.kwargs
        valid_sparse_coder_params = {
            'mode', 'max_iter', 'tol', 'seed', 'anneal'
        }
        sparse_coder_kwargs = {k: v for k, v in self.kwargs.items() 
                             if k in valid_sparse_coder_params}
        
        # Use SparseCoder for batch dictionary learning
        learner = SparseCoder(
            n_atoms=self.n_atoms,
            lam=self.sparsity,
            **sparse_coder_kwargs
        )
        
        # Only pass valid fit parameters (n_steps, lr)
        valid_fit_params = {k: v for k, v in fit_params.items() 
                           if k in {'n_steps', 'lr'}}
        learner.fit(X.T, **valid_fit_params)  # SparseCoder expects (n_features, n_samples)
        
        return Features(
            dictionary=learner.D,
            method='dict',
            metadata={
                'learner': learner,
                'sparsity_param': self.sparsity,
                'solver': getattr(learner, 'solver', 'fista'),
                'n_iterations': getattr(learner, 'n_iterations_', None)
            }
        )
    
    def _fit_sae(self, X: ArrayLike, **kwargs) -> Features:
        """Fit sparse autoencoder."""
        
        # Convert to torch tensor
        X_torch = torch.from_numpy(as_same(X, np.array([]))).float()
        n_features = X_torch.shape[1]
        
        # Create SAE
        sae_type = self.kwargs.get('sae_type', 'L1SAE')
        
        # Separate constructor args from training args
        constructor_args = {
            'tie_weights', 'normalize_decoder', 'bias', 'device', 'dtype'
        }
        sae_constructor_kwargs = {k: v for k, v in self.kwargs.items() 
                                if k in constructor_args}
        
        if sae_type == 'L1SAE':
            sae = L1SAE(
                n_features=n_features,
                n_latents=self.n_atoms,
                l1_penalty=self.sparsity,
                **sae_constructor_kwargs
            )
        elif sae_type == 'TopKSAE':
            k = self.kwargs.get('k', max(1, int(0.1 * self.n_atoms)))
            sae = TopKSAE(
                n_features=n_features,
                n_latents=self.n_atoms,
                k=k,
                **sae_constructor_kwargs
            )
        else:
            raise ValueError(f"Unknown sae_type: {sae_type}")
        
        # Training setup
        device = self.kwargs.get('device', None)
        if device is not None:
            sae = sae.to(device)
            X_torch = X_torch.to(device)
        
        # Simple training (can be extended with DataLoader for large data)
        optimizer = torch.optim.Adam(sae.parameters(), lr=kwargs.get('lr', 1e-3))
        n_epochs = kwargs.get('n_epochs', 100)
        
        sae.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            x_hat, a = sae(X_torch)
            losses = sae.compute_loss(X_torch, x_hat, a, 
                                    kwargs.get('sparsity_weight', 1.0))
            losses['total'].backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                sae.normalize_decoder_weights()
        
        # Extract dictionary
        with torch.no_grad():
            dictionary = sae.decoder_weights.cpu().numpy()
        
        return Features(
            dictionary=dictionary,
            method='sae', 
            metadata={
                'sae': sae,
                'sae_type': sae_type,
                'sparsity_param': self.sparsity,
                'n_epochs': n_epochs,
                'device': str(device) if device else 'cpu'
            }
        )
    
    def _fit_hybrid(self, X: ArrayLike, **kwargs) -> Features:
        """Fit hybrid dictionary + SAE approach."""
        # Filter out SAE-specific parameters for dictionary learning
        dict_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in {'n_epochs', 'lr', 'sparsity_weight', 'sae_type', 'device'}}
        
        # First fit dictionary learning for initialization
        dict_features = self._fit_dictionary(X, **dict_kwargs)
        
        # Then fine-tune with SAE
        sae_kwargs = {**self.kwargs, **kwargs}
        sae_kwargs['init_dict'] = dict_features.dictionary
        
        temp_extractor = FeatureExtractor(method='sae', n_atoms=self.n_atoms, 
                                        sparsity=self.sparsity, **sae_kwargs)
        sae_features = temp_extractor._fit_sae(X, **kwargs)
        
        # Combine metadata
        metadata = {
            'dict_features': dict_features,
            'sae_features': sae_features,
            'method_sequence': ['dict', 'sae']
        }
        
        return Features(
            dictionary=sae_features.dictionary,
            method='hybrid',
            metadata=metadata
        )


def fit_features(
    X: ArrayLike,
    method: Literal['dict', 'sae', 'hybrid'] = 'dict',
    n_atoms: int = 256,
    sparsity: float = 0.1,
    **kwargs
) -> Features:
    """
    Fit sparse features using specified method.
    
    Unified interface for dictionary learning, SAE training, or hybrid approaches.
    Automatically handles backend compatibility and method-specific optimizations.
    
    Parameters
    ----------
    X : ArrayLike, shape (n_samples, n_features)
        Training data
    method : str, default='dict'
        Feature extraction method ('dict', 'sae', 'hybrid')
    n_atoms : int, default=256
        Number of dictionary atoms/SAE latents  
    sparsity : float, default=0.1
        Sparsity parameter (lambda for dict, L1 penalty for SAE)
    **kwargs
        Method-specific parameters
        
    Returns
    -------
    features : Features
        Fitted sparse features
        
    Examples
    --------
    >>> # Classical dictionary learning
    >>> features_dict = fit_features(X, method='dict', n_atoms=128, sparsity=0.1)
    >>> 
    >>> # Sparse autoencoder (requires PyTorch)
    >>> features_sae = fit_features(X, method='sae', n_atoms=128, sparsity=1e-3,
    ...                           sae_type='L1SAE', n_epochs=200)
    >>> 
    >>> # Hybrid approach (dict init + SAE fine-tuning)
    >>> features_hybrid = fit_features(X, method='hybrid', n_atoms=128, sparsity=0.1)
    """
    extractor = FeatureExtractor(method=method, n_atoms=n_atoms, 
                               sparsity=sparsity, **kwargs)
    extractor.fit(X, **kwargs)
    return extractor.features_


def encode_features(X: ArrayLike, features: Features) -> ArrayLike:
    """
    Encode data using fitted sparse features.
    
    Parameters
    ----------
    X : ArrayLike, shape (n_samples, n_features)
        Data to encode
    features : Features
        Fitted sparse features
        
    Returns
    -------
    A : ArrayLike, shape (n_samples, n_atoms)
        Sparse codes/activations
    """
    X = ensure_array(X)
    
    if features.method == 'dict':
        # Use classical sparse coding
        learner = features.metadata['learner']
        return learner.encode(X.T).T  # Handle transpose convention
    
    elif features.method == 'sae':
        # Use SAE encoding
        
        sae = features.metadata['sae']
        X_torch = torch.from_numpy(as_same(X, np.array([]))).float()
        
        if hasattr(sae, 'device') and sae.device != X_torch.device:
            X_torch = X_torch.to(sae.device)
        
        with torch.no_grad():
            sae.eval()
            A_torch = sae.encode(X_torch)
            return A_torch.cpu().numpy()
    
    elif features.method == 'hybrid':
        # Use SAE from hybrid training
        sae_features = features.metadata['sae_features']
        return encode_features(X, sae_features)
    
    else:
        raise ValueError(f"Unknown method: {features.method}")


def decode_features(A: ArrayLike, features: Features) -> ArrayLike:
    """
    Decode sparse codes to reconstruct data.
    
    Parameters
    ----------
    A : ArrayLike, shape (n_samples, n_atoms)
        Sparse codes/activations
    features : Features
        Fitted sparse features
        
    Returns
    -------
    X_hat : ArrayLike, shape (n_samples, n_features)  
        Reconstructed data
    """
    A = ensure_array(A)
    backend = xp(A)
    
    if features.method in ['dict', 'hybrid']:
        # Simple matrix multiplication: X_hat = A @ D.T
        D = as_same(features.dictionary, A)
        return backend.matmul(A, D.T) if hasattr(backend, 'matmul') else A @ D.T
    
    elif features.method == 'sae':
        # Use SAE decoding
        
        sae = features.metadata['sae']
        A_torch = torch.from_numpy(as_same(A, np.array([]))).float()
        
        if hasattr(sae, 'device') and sae.device != A_torch.device:
            A_torch = A_torch.to(sae.device)
        
        with torch.no_grad():
            sae.eval()
            X_hat_torch = sae.decode(A_torch)
            return X_hat_torch.cpu().numpy()
    
    else:
        raise ValueError(f"Unknown method: {features.method}")


def compare_features(
    X: ArrayLike,
    methods: list = ['dict', 'sae'], 
    n_atoms: int = 256,
    sparsity: float = 0.1,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different feature extraction methods on same data.
    
    Parameters
    ----------
    X : ArrayLike, shape (n_samples, n_features)
        Test data
    methods : list, default=['dict', 'sae']
        Methods to compare
    n_atoms : int, default=256
        Number of atoms for all methods
    sparsity : float, default=0.1
        Sparsity parameter for all methods
    **kwargs
        Additional parameters passed to all methods
        
    Returns
    -------
    results : dict
        Comparison results with reconstruction errors, sparsity levels, etc.
    """
    results = {}
    
    for method in methods:
        try:
            # Fit features
            features = fit_features(X, method=method, n_atoms=n_atoms, 
                                  sparsity=sparsity, **kwargs)
            
            # Encode and decode
            A = encode_features(X, features)
            X_hat = decode_features(A, features)
            
            # Compute metrics
            backend = xp(X)
            mse = float(backend.mean((X - X_hat)**2))
            sparsity_level = float(backend.mean(backend.abs(A) < 1e-6))
            
            results[method] = {
                'features': features,
                'codes': A,
                'reconstruction': X_hat,
                'mse': mse,
                'sparsity_level': sparsity_level,
                'n_nonzero': float(backend.mean(backend.sum(backend.abs(A) > 1e-6, axis=1)))
            }
            
        except Exception as e:
            results[method] = {'error': str(e)}
    
    return results


def visualize_features(
    features: Features,
    n_show: int = 64,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize learned dictionary atoms/SAE features.
    
    Parameters
    ----------
    features : Features
        Fitted sparse features to visualize
    n_show : int, default=64
        Number of features to show
    figsize : tuple, default=(12, 8)
        Figure size for matplotlib
    """
    import matplotlib.pyplot as plt
    
    D = features.dictionary
    n_features, n_atoms = D.shape
    n_show = min(n_show, n_atoms)
    
    # Determine visualization layout
    if n_features == 28*28 or n_features == 32*32:
        # Image patches
        patch_size = int(np.sqrt(n_features))
        grid_size = int(np.ceil(np.sqrt(n_show)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = axes.flatten()
        
        for i in range(n_show):
            atom = D[:, i].reshape(patch_size, patch_size)
            axes[i].imshow(atom, cmap='gray')
            axes[i].set_title(f'Atom {i}')
            axes[i].axis('off')
            
        for i in range(n_show, len(axes)):
            axes[i].axis('off')
            
    else:
        # 1D features - show as line plots
        fig, ax = plt.subplots(figsize=figsize)
        
        for i in range(n_show):
            ax.plot(D[:, i], alpha=0.7, label=f'Atom {i}' if i < 10 else '')
        
        ax.set_xlabel('Feature dimension')
        ax.set_ylabel('Activation')
        ax.set_title(f'{features.method.upper()} Dictionary Atoms')
        if n_show <= 10:
            ax.legend()
    
    plt.suptitle(f'{features.method.upper()} Features: {n_atoms} atoms, '
                f'{n_features} dimensions')
    plt.tight_layout()
    plt.show()