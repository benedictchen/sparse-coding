"""
PyTorch integration for sparse coding.

Provides nn.Module wrappers that integrate with PyTorch training loops,
automatic differentiation, and GPU acceleration via DLPack.
"""

import warnings
from typing import Optional, Dict, Any, Union, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available. Torch adapters disabled.")
    
    # Provide stubs
    class nn:
        class Module:
            def __init__(self): pass
            def forward(self, x): return x
            def parameters(self): return []
            def state_dict(self): return {}
            def load_state_dict(self, d): pass
    
    torch = None

import numpy as np
from ..core.array import as_same, get_array_info


class SparseCodingModule(nn.Module):
    """
    PyTorch module for sparse coding inference.
    
    Wraps sparse coding solvers as PyTorch modules for integration
    with training loops, automatic differentiation, and GPU acceleration.
    
    Parameters
    ----------
    dictionary : torch.Tensor or None
        Dictionary matrix of shape (n_features, n_atoms)
    penalty_config : dict
        Penalty configuration for solver
    solver_config : dict  
        Solver configuration
    learnable_dict : bool, default=False
        Whether dictionary should be a learnable parameter
    device : str or torch.device, default='cpu'
        Device for computations
    
    Examples
    --------
    >>> import torch
    >>> from sparse_coding.adapters import SparseCodingModule
    
    >>> # Create module with fixed dictionary
    >>> D = torch.randn(64, 32)  # (features, atoms)
    >>> module = SparseCodingModule(dictionary=D)
    >>> 
    >>> # Encode batch of signals
    >>> X = torch.randn(10, 64)  # (batch, features)
    >>> codes = module(X)  # (batch, atoms)
    >>> 
    >>> # Learnable dictionary for end-to-end training
    >>> learnable_module = SparseCodingModule(
    ...     dictionary=D, 
    ...     learnable_dict=True
    ... )
    >>> optimizer = torch.optim.Adam(learnable_module.parameters())
    """
    
    def __init__(
        self,
        dictionary: Optional[torch.Tensor] = None,
        penalty_config: Dict[str, Any] = None,
        solver_config: Dict[str, Any] = None,
        learnable_dict: bool = False,
        device: Union[str, torch.device] = 'cpu'
    ):
        super().__init__()
        
        if not HAS_TORCH:
            raise ImportError("PyTorch required for SparseCodingModule")
        
        # Default configurations
        if penalty_config is None:
            penalty_config = {"name": "l1", "params": {"lam": 0.1}}
        if solver_config is None:
            solver_config = {"name": "fista", "params": {"max_iter": 100}}
        
        self.penalty_config = penalty_config
        self.solver_config = solver_config
        self.learnable_dict = learnable_dict
        self.device = torch.device(device)
        
        # Initialize dictionary
        if dictionary is not None:
            dictionary = dictionary.to(self.device)
            if learnable_dict:
                self.dictionary = nn.Parameter(dictionary)
            else:
                self.register_buffer('dictionary', dictionary)
        else:
            self.dictionary = None
        
        # Create penalty and solver from config
        self._setup_components()
    
    def _setup_components(self):
        """Initialize penalty and solver components."""
        from ..api.registry import create_from_config
        
        # Create components (these work with numpy/backend-agnostic arrays)
        self._penalty = create_from_config(
            {"kind": "penalty", **self.penalty_config}
        )
        self._solver = create_from_config(
            {"kind": "solver", **self.solver_config}
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Encode input using sparse coding.
        
        Parameters
        ---------- 
        X : torch.Tensor of shape (batch_size, n_features) or (n_features,)
            Input signals to encode
            
        Returns
        -------
        codes : torch.Tensor of shape (batch_size, n_atoms) or (n_atoms,)
            Sparse codes
        """
        if self.dictionary is None:
            raise RuntimeError("Dictionary not initialized. Set dictionary or call initialize_dictionary().")
        
        # Handle single sample
        single_sample = X.ndim == 1
        if single_sample:
            X = X.unsqueeze(0)
        
        batch_size = X.shape[0]
        n_features, n_atoms = self.dictionary.shape
        
        # Convert to backend-agnostic arrays for processing
        X_np = X.detach().cpu().numpy().T  # (features, batch)
        D_np = self.dictionary.detach().cpu().numpy()
        
        # Solve sparse coding (batch processing)
        codes_np = self._solver.solve(D_np, X_np, self._penalty)
        
        # Convert back to torch tensor on same device as input
        codes = torch.from_numpy(codes_np.T).to(X.device, X.dtype)  # (batch, atoms)
        
        if single_sample:
            codes = codes.squeeze(0)
        
        return codes
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse codes back to signal space.
        
        Parameters
        ----------
        codes : torch.Tensor of shape (batch_size, n_atoms) or (n_atoms,)
            Sparse codes to decode
            
        Returns
        -------
        reconstructed : torch.Tensor of shape (batch_size, n_features) or (n_features,)
            Reconstructed signals
        """
        if self.dictionary is None:
            raise RuntimeError("Dictionary not initialized.")
        
        single_sample = codes.ndim == 1
        if single_sample:
            codes = codes.unsqueeze(0)
        
        # Simple matrix multiplication: X = D @ A
        reconstructed = torch.mm(codes, self.dictionary.T)  # (batch, features)
        
        if single_sample:
            reconstructed = reconstructed.squeeze(0)
        
        return reconstructed
    
    def initialize_dictionary(self, n_features: int, n_atoms: int, method: str = 'random'):
        """
        Initialize dictionary.
        
        Parameters
        ----------
        n_features : int
            Number of features (signal dimension)
        n_atoms : int
            Number of dictionary atoms
        method : str, default='random'
            Initialization method ('random', 'orthogonal')
        """
        if method == 'random':
            dictionary = torch.randn(n_features, n_atoms, device=self.device)
            # Normalize columns
            dictionary = dictionary / torch.norm(dictionary, dim=0, keepdim=True)
        elif method == 'orthogonal':
            dictionary = torch.empty(n_features, n_atoms, device=self.device)
            nn.init.orthogonal_(dictionary[:, :min(n_features, n_atoms)])
            if n_atoms > n_features:
                # Fill remaining with random normalized vectors
                extra = dictionary[:, n_features:]
                nn.init.normal_(extra)
                extra = extra / torch.norm(extra, dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        
        if self.learnable_dict:
            self.dictionary = nn.Parameter(dictionary)
        else:
            self.register_buffer('dictionary', dictionary)
    
    def reconstruction_loss(self, X: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Parameters
        ----------
        X : torch.Tensor
            Input signals
        reduction : str, default='mean'
            Loss reduction ('mean', 'sum', 'none')
            
        Returns
        -------
        loss : torch.Tensor
            Reconstruction loss
        """
        codes = self.forward(X)
        reconstructed = self.decode(codes)
        
        mse = (X - reconstructed) ** 2
        
        if reduction == 'mean':
            return mse.mean()
        elif reduction == 'sum':
            return mse.sum()
        else:
            return mse
    
    def sparsity_loss(self, X: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Compute sparsity loss.
        
        Parameters
        ----------
        X : torch.Tensor
            Input signals
        reduction : str
            Loss reduction
            
        Returns
        -------
        loss : torch.Tensor
            Sparsity loss
        """
        codes = self.forward(X)
        
        # Simple L1 penalty (could be extended based on penalty_config)
        l1_loss = torch.abs(codes)
        
        if reduction == 'mean':
            return l1_loss.mean()
        elif reduction == 'sum':
            return l1_loss.sum()
        else:
            return l1_loss
    
    def total_loss(self, X: torch.Tensor, sparsity_weight: float = 1.0) -> torch.Tensor:
        """
        Compute total loss (reconstruction + sparsity).
        
        Parameters
        ----------
        X : torch.Tensor
            Input signals
        sparsity_weight : float
            Weight for sparsity term
            
        Returns
        -------
        loss : torch.Tensor
            Total loss
        """
        recon_loss = self.reconstruction_loss(X)
        sparse_loss = self.sparsity_loss(X)
        return recon_loss + sparsity_weight * sparse_loss
    
    def extra_repr(self) -> str:
        """String representation for print()."""
        if self.dictionary is not None:
            n_features, n_atoms = self.dictionary.shape
            return f"n_features={n_features}, n_atoms={n_atoms}, learnable_dict={self.learnable_dict}"
        return "dictionary=None"


class DictionaryLearningModule(nn.Module):
    """
    PyTorch module for end-to-end dictionary learning.
    
    Combines sparse coding with learnable dictionary updates for
    end-to-end training in PyTorch pipelines.
    
    Parameters
    ----------
    n_features : int
        Input signal dimension
    n_atoms : int
        Number of dictionary atoms
    penalty_config : dict
        Penalty configuration
    solver_config : dict
        Solver configuration  
    dict_lr : float, default=0.01
        Learning rate for dictionary updates
    normalize_dict : bool, default=True
        Whether to normalize dictionary columns
    
    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from sparse_coding.adapters import DictionaryLearningModule
    
    >>> # Create learnable dictionary learning module
    >>> module = DictionaryLearningModule(n_features=64, n_atoms=32)
    >>> 
    >>> # Use in larger model
    >>> class SparseAutoencoder(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.encoder = DictionaryLearningModule(64, 32)
    ...         self.decoder = nn.Linear(32, 10)  # classifier
    ...     
    ...     def forward(self, x):
    ...         sparse_codes = self.encoder(x)
    ...         return self.decoder(sparse_codes)
    >>> 
    >>> model = SparseAutoencoder()
    >>> optimizer = torch.optim.Adam(model.parameters())
    """
    
    def __init__(
        self,
        n_features: int,
        n_atoms: int,
        penalty_config: Dict[str, Any] = None,
        solver_config: Dict[str, Any] = None,
        dict_lr: float = 0.01,
        normalize_dict: bool = True,
        device: Union[str, torch.device] = 'cpu'
    ):
        super().__init__()
        
        if not HAS_TORCH:
            raise ImportError("PyTorch required for DictionaryLearningModule")
        
        self.n_features = n_features
        self.n_atoms = n_atoms
        self.dict_lr = dict_lr
        self.normalize_dict = normalize_dict
        self.device = torch.device(device)
        
        # Initialize learnable dictionary
        dictionary = torch.randn(n_features, n_atoms, device=self.device)
        dictionary = dictionary / torch.norm(dictionary, dim=0, keepdim=True)
        self.dictionary = nn.Parameter(dictionary)
        
        # Create sparse coding module
        self.sparse_coder = SparseCodingModule(
            dictionary=None,  # Will use our parameter
            penalty_config=penalty_config,
            solver_config=solver_config,
            learnable_dict=False,
            device=device
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode using current dictionary.
        
        Parameters
        ----------
        X : torch.Tensor
            Input signals
            
        Returns
        -------
        codes : torch.Tensor
            Sparse codes
        """
        # Update sparse coder with current dictionary
        self.sparse_coder.dictionary = self.dictionary
        
        # Normalize dictionary if requested
        if self.normalize_dict:
            with torch.no_grad():
                dict_norms = torch.norm(self.dictionary, dim=0, keepdim=True)
                self.dictionary.data = self.dictionary / (dict_norms + 1e-12)
        
        return self.sparse_coder(X)
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode sparse codes."""
        return torch.mm(codes, self.dictionary.T)
    
    def dictionary_update_step(self, X: torch.Tensor, codes: torch.Tensor):
        """
        Manual dictionary update step (for custom training loops).
        
        Parameters
        ----------
        X : torch.Tensor
            Input signals (batch, features)
        codes : torch.Tensor  
            Sparse codes (batch, atoms)
        """
        with torch.no_grad():
            # Gradient-based dictionary update
            # D <- D + lr * (X - D*A) * A^T
            residual = X - self.decode(codes)
            gradient = torch.mm(residual.T, codes)  # (features, atoms)
            self.dictionary.data += self.dict_lr * gradient
            
            # Normalize if requested
            if self.normalize_dict:
                dict_norms = torch.norm(self.dictionary, dim=0, keepdim=True)
                self.dictionary.data = self.dictionary / (dict_norms + 1e-12)


def sparse_encode_batch(X: torch.Tensor, dictionary: torch.Tensor, 
                       penalty_config: Dict[str, Any] = None,
                       solver_config: Dict[str, Any] = None) -> torch.Tensor:
    """
    Standalone function for batch sparse encoding.
    
    Parameters
    ----------
    X : torch.Tensor of shape (batch_size, n_features)
        Input signals
    dictionary : torch.Tensor of shape (n_features, n_atoms)
        Dictionary matrix
    penalty_config : dict, optional
        Penalty configuration
    solver_config : dict, optional
        Solver configuration
        
    Returns
    -------
    codes : torch.Tensor of shape (batch_size, n_atoms)
        Sparse codes
        
    Examples
    --------
    >>> import torch
    >>> from sparse_coding.adapters.torch import sparse_encode_batch
    
    >>> X = torch.randn(32, 64)      # batch of 32 signals, 64 features
    >>> D = torch.randn(64, 32)      # dictionary: 64 features, 32 atoms
    >>> codes = sparse_encode_batch(X, D)  # (32, 32)
    """
    module = SparseCodingModule(
        dictionary=dictionary,
        penalty_config=penalty_config,
        solver_config=solver_config,
        learnable_dict=False
    )
    return module(X)


def create_sparse_autoencoder(input_dim: int, sparse_dim: int, output_dim: int,
                             sparsity_weight: float = 0.1) -> nn.Module:
    """
    Create sparse autoencoder with dictionary learning.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    sparse_dim : int
        Sparse representation dimension
    output_dim : int
        Output dimension
    sparsity_weight : float
        Sparsity regularization weight
        
    Returns
    -------
    model : nn.Module
        Sparse autoencoder model
        
    Examples
    --------
    >>> model = create_sparse_autoencoder(784, 128, 10)  # MNIST classifier
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> 
    >>> # Training loop
    >>> for batch in dataloader:
    ...     x, y = batch
    ...     codes = model.encoder(x)
    ...     logits = model.decoder(codes)
    ...     
    ...     ce_loss = F.cross_entropy(logits, y)
    ...     sparse_loss = model.encoder.sparsity_loss(x)
    ...     loss = ce_loss + sparsity_weight * sparse_loss
    ...     
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step()
    """
    class SparseAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = DictionaryLearningModule(input_dim, sparse_dim)
            self.decoder = nn.Linear(sparse_dim, output_dim)
            self.sparsity_weight = sparsity_weight
        
        def forward(self, x):
            codes = self.encoder(x)
            return self.decoder(codes)
        
        def loss(self, x, y):
            codes = self.encoder(x)
            logits = self.decoder(codes)
            
            ce_loss = torch.nn.functional.cross_entropy(logits, y)
            sparse_loss = torch.abs(codes).mean()
            
            return ce_loss + self.sparsity_weight * sparse_loss
    
    return SparseAutoencoder()