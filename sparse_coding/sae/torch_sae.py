"""
PyTorch-based Sparse Autoencoders (SAEs) for interpretability workflows.

Implements L1 and TopK sparse autoencoders with unified interface compatible
with classical sparse coding. Designed for LLM feature extraction and analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import warnings

from ..core.array import ArrayLike, xp, as_same


class SAE(nn.Module):
    """
    Base Sparse Autoencoder with configurable sparsity constraints.
    
    Implements the standard SAE architecture:
    - Encoder: x -> ReLU(Wx + b_enc) 
    - Decoder: a -> W^T a + b_dec
    - Sparsity: Applied via penalty or constraint
    
    Parameters
    ----------
    n_features : int
        Input feature dimension
    n_latents : int  
        Latent/dictionary dimension
    tie_weights : bool, default=True
        Whether to tie decoder weights as encoder transpose
    normalize_decoder : bool, default=True
        Whether to normalize decoder columns to unit norm
    bias : bool, default=True
        Whether to include bias terms
    """
    
    def __init__(
        self,
        n_features: int,
        n_latents: int,
        tie_weights: bool = True,
        normalize_decoder: bool = True,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.n_features = n_features
        self.n_latents = n_latents
        self.tie_weights = tie_weights
        self.normalize_decoder = normalize_decoder
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Encoder weights and bias
        self.W_enc = nn.Parameter(
            torch.randn(n_latents, n_features, **factory_kwargs) * 0.1
        )
        
        if not tie_weights:
            self.W_dec = nn.Parameter(
                torch.randn(n_features, n_latents, **factory_kwargs) * 0.1
            )
        else:
            self.register_parameter('W_dec', None)
        
        if bias:
            self.b_enc = nn.Parameter(torch.zeros(n_latents, **factory_kwargs))
            self.b_dec = nn.Parameter(torch.zeros(n_features, **factory_kwargs))
        else:
            self.register_parameter('b_enc', None)
            self.register_parameter('b_dec', None)
    
    @property
    def decoder_weights(self) -> torch.Tensor:
        """Get decoder weight matrix."""
        if self.tie_weights:
            return self.W_enc.T
        else:
            return self.W_dec
    
    def normalize_decoder_weights(self):
        """Normalize decoder columns to unit norm."""
        if not self.normalize_decoder:
            return
        
        with torch.no_grad():
            if self.tie_weights:
                # Normalize rows of encoder (columns of decoder)
                norms = torch.norm(self.W_enc, dim=1, keepdim=True)
                norms = torch.clamp(norms, min=1e-8)
                self.W_enc.div_(norms)
            else:
                # Normalize columns of decoder directly
                norms = torch.norm(self.W_dec, dim=0, keepdim=True)
                norms = torch.clamp(norms, min=1e-8)
                self.W_dec.div_(norms)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent activations.
        
        Parameters
        ----------
        x : torch.Tensor, shape (..., n_features)
            Input data
            
        Returns
        -------
        a : torch.Tensor, shape (..., n_latents)
            Latent activations
        """
        # Linear projection
        a = F.linear(x, self.W_enc, self.b_enc)
        
        # Apply sparsity (implemented in subclasses)
        return self.apply_sparsity(a)
    
    def decode(self, a: torch.Tensor) -> torch.Tensor:
        """
        Decode latent activations to reconstruction.
        
        Parameters
        ----------
        a : torch.Tensor, shape (..., n_latents)
            Latent activations
            
        Returns
        -------
        x_hat : torch.Tensor, shape (..., n_features)
            Reconstructed input
        """
        # decoder_weights is (n_features, n_latents)
        # For F.linear, we need weight matrix of shape (out_features, in_features)
        # decoder_weights is already in the correct shape: (n_features, n_latents)
        return F.linear(a, self.decoder_weights, self.b_dec)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.
        
        Parameters
        ----------
        x : torch.Tensor, shape (..., n_features)
            Input data
            
        Returns
        -------
        x_hat : torch.Tensor, shape (..., n_features)
            Reconstructed input
        a : torch.Tensor, shape (..., n_latents)  
            Latent activations
        """
        a = self.encode(x)
        x_hat = self.decode(a)
        return x_hat, a
    
    def apply_sparsity(self, a: torch.Tensor) -> torch.Tensor:
        """Apply sparsity constraint. Implemented in subclasses."""
        raise NotImplementedError("Subclasses must implement apply_sparsity")
    
    def sparsity_loss(self, a: torch.Tensor) -> torch.Tensor:
        """Compute sparsity penalty. Implemented in subclasses."""
        raise NotImplementedError("Subclasses must implement sparsity_loss")
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        x_hat: torch.Tensor, 
        a: torch.Tensor,
        sparsity_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full SAE loss with reconstruction and sparsity terms.
        
        Parameters
        ----------
        x : torch.Tensor
            Original input
        x_hat : torch.Tensor  
            Reconstructed input
        a : torch.Tensor
            Latent activations
        sparsity_weight : float, default=1.0
            Weight for sparsity penalty
            
        Returns
        -------
        losses : dict
            Dictionary with 'reconstruction', 'sparsity', and 'total' losses
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        
        # Sparsity loss (penalty-specific)
        sparse_loss = self.sparsity_loss(a)
        
        # Total loss
        total_loss = recon_loss + sparsity_weight * sparse_loss
        
        return {
            'reconstruction': recon_loss,
            'sparsity': sparse_loss, 
            'total': total_loss
        }


class L1SAE(SAE):
    """
    L1-regularized Sparse Autoencoder.
    
    Applies L1 penalty to latent activations for sparsity.
    Compatible with classical L1 sparse coding.
    
    Parameters
    ----------
    l1_penalty : float, default=1e-3
        L1 regularization strength
    **kwargs
        Passed to base SAE class
    """
    
    def __init__(self, l1_penalty: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty
    
    def apply_sparsity(self, a: torch.Tensor) -> torch.Tensor:
        """Apply ReLU activation (soft L1 constraint)."""
        return F.relu(a)
    
    def sparsity_loss(self, a: torch.Tensor) -> torch.Tensor:
        """Compute L1 penalty on activations."""
        return self.l1_penalty * torch.mean(torch.abs(a))


class TopKSAE(SAE):
    """
    TopK Sparse Autoencoder.
    
    Keeps only the top-k largest activations, setting others to zero.
    Provides exact sparsity control (k-sparse constraint).
    
    Parameters
    ----------
    k : int or float
        Number of active units. If float, treated as fraction of n_latents
    **kwargs
        Passed to base SAE class
    """
    
    def __init__(self, k: int, **kwargs):
        super().__init__(**kwargs)
        
        if isinstance(k, float):
            if not 0 < k <= 1:
                raise ValueError("If k is float, must be in (0, 1]")
            self.k = max(1, int(k * self.n_latents))
        else:
            if not 1 <= k <= self.n_latents:
                raise ValueError(f"k must be between 1 and {self.n_latents}")
            self.k = k
    
    def apply_sparsity(self, a: torch.Tensor) -> torch.Tensor:
        """Keep top-k activations, zero out rest."""
        # Apply ReLU first
        a_pos = F.relu(a)
        
        # Find top-k values along last dimension
        if self.k >= a_pos.shape[-1]:
            return a_pos
        
        # Get top-k indices
        _, top_indices = torch.topk(a_pos, self.k, dim=-1)
        
        # Create mask
        mask = torch.zeros_like(a_pos)
        mask.scatter_(-1, top_indices, 1.0)
        
        return a_pos * mask
    
    def sparsity_loss(self, a: torch.Tensor) -> torch.Tensor:
        """No additional sparsity loss needed (hard constraint)."""
        return torch.tensor(0.0, device=a.device, dtype=a.dtype)


def train_sae(
    sae: SAE,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 100,
    sparsity_weight: float = 1.0,
    normalize_freq: int = 10,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train SAE with standard training loop.
    
    Parameters
    ----------
    sae : SAE
        Sparse autoencoder to train
    data_loader : DataLoader
        Training data loader
    optimizer : Optimizer
        PyTorch optimizer
    n_epochs : int, default=100
        Number of training epochs
    sparsity_weight : float, default=1.0
        Weight for sparsity penalty
    normalize_freq : int, default=10
        Frequency to normalize decoder weights (epochs)
    device : torch.device, optional
        Device to train on
    verbose : bool, default=True
        Whether to print training progress
        
    Returns
    -------
    history : dict
        Training history with losses
    """
    if device is not None:
        sae = sae.to(device)
    
    history = {
        'total_loss': [],
        'reconstruction_loss': [],
        'sparsity_loss': []
    }
    
    sae.train()
    
    for epoch in range(n_epochs):
        epoch_losses = {'total': 0.0, 'reconstruction': 0.0, 'sparsity': 0.0}
        n_batches = 0
        
        for batch_x in data_loader:
            if device is not None:
                batch_x = batch_x.to(device)
            
            # Forward pass
            x_hat, a = sae(batch_x)
            
            # Compute losses
            losses = sae.compute_loss(batch_x, x_hat, a, sparsity_weight)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += losses['total'].item()
            epoch_losses['reconstruction'] += losses['reconstruction'].item()
            epoch_losses['sparsity'] += losses['sparsity'].item()
            n_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        # Record history
        history['total_loss'].append(epoch_losses['total'])
        history['reconstruction_loss'].append(epoch_losses['reconstruction'])
        history['sparsity_loss'].append(epoch_losses['sparsity'])
        
        # Normalize decoder weights periodically
        if (epoch + 1) % normalize_freq == 0:
            sae.normalize_decoder_weights()
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"Total={epoch_losses['total']:.6f}, "
                  f"Recon={epoch_losses['reconstruction']:.6f}, "
                  f"Sparse={epoch_losses['sparsity']:.6f}")
    
    return history


def convert_sae_to_dict(sae: SAE) -> ArrayLike:
    """
    Convert trained SAE to dictionary format for classical sparse coding.
    
    Parameters
    ----------
    sae : SAE
        Trained sparse autoencoder
        
    Returns
    -------
    D : ArrayLike
        Dictionary matrix, shape (n_features, n_latents)
    """
    with torch.no_grad():
        return sae.decoder_weights.detach().cpu().numpy()


def convert_dict_to_sae(
    D: ArrayLike,
    sae_type: str = 'L1SAE',
    **sae_kwargs
) -> SAE:
    """
    Convert classical dictionary to SAE initialization.
    
    Parameters
    ----------
    D : ArrayLike, shape (n_features, n_latents)
        Dictionary matrix
    sae_type : str, default='L1SAE'
        Type of SAE to create ('L1SAE' or 'TopKSAE')
    **sae_kwargs
        Additional arguments for SAE constructor
        
    Returns
    -------
    sae : SAE
        Initialized sparse autoencoder
    """
    backend = xp(D)
    
    # Convert to torch if needed
    if not isinstance(D, torch.Tensor):
        D_torch = torch.from_numpy(as_same(D, backend.array([])))
    else:
        D_torch = D
    
    n_features, n_latents = D_torch.shape
    
    # Create SAE
    if sae_type == 'L1SAE':
        sae = L1SAE(n_features=n_features, n_latents=n_latents, **sae_kwargs)
    elif sae_type == 'TopKSAE':
        sae = TopKSAE(n_features=n_features, n_latents=n_latents, **sae_kwargs)
    else:
        raise ValueError(f"Unknown sae_type: {sae_type}")
    
    # Initialize with dictionary
    with torch.no_grad():
        if sae.tie_weights:
            sae.W_enc.data = D_torch.T.clone()
        else:
            sae.W_dec.data = D_torch.clone()
    
    return sae