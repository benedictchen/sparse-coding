"""
PyTorch-based Sparse Autoencoders (SAEs) for interpretability workflows.

Implements L1 and TopK sparse autoencoders with unified interface compatible
with classical sparse coding. Designed for LLM feature extraction and analysis.

Key References
--------------
- Bricken et al. (2023): "Towards Monosemanticity: Decomposing Language Models
  With Dictionary Learning" - modern SAE techniques for LLM interpretability
- Gao et al. (2024): "Scaling and evaluating sparse autoencoders" - TopK SAEs
  for exact sparsity control and improved training stability
- Templeton et al. (2024): "Scaling Monosemanticity: Extracting Interpretable
  Features from Claude 3 Sonnet" - large-scale SAE deployment
- Sharkey et al. (2022): "Taking features out of superposition with sparse
  autoencoders" - SAE fundamentals and training techniques
- Lee & Seung (1999): "Learning the parts of objects by non-negative matrix
  factorization" - theoretical foundations of sparse decomposition
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
    
    def normalize_decoder_weights(self, dead_feature_threshold: float = 1e-6):
        """
        Normalize decoder columns to unit norm and handle dead features.
        
        Parameters
        ----------
        dead_feature_threshold : float, default=1e-6
            Threshold below which features are considered "dead" and re-initialized
        """
        if not self.normalize_decoder:
            return
        
        with torch.no_grad():
            if self.tie_weights:
                # Normalize rows of encoder (columns of decoder)
                norms = torch.norm(self.W_enc, dim=1, keepdim=True)
                
                # Identify dead features
                dead_mask = (norms < dead_feature_threshold).squeeze()
                n_dead = dead_mask.sum().item()
                
                if n_dead > 0:
                    import warnings
                    warnings.warn(f"Reinitializing {n_dead} dead features")
                    
                    # Reinitialize dead features with small random weights
                    self.W_enc.data[dead_mask] = torch.randn_like(
                        self.W_enc.data[dead_mask]
                    ) * 0.01
                    
                    # Recompute norms after reinitialization
                    norms = torch.norm(self.W_enc, dim=1, keepdim=True)
                
                # Normalize all features
                norms = torch.clamp(norms, min=1e-8)
                self.W_enc.div_(norms)
                
            else:
                # Normalize columns of decoder directly
                norms = torch.norm(self.W_dec, dim=0, keepdim=True)
                
                # Identify dead features
                dead_mask = (norms < dead_feature_threshold).squeeze()
                n_dead = dead_mask.sum().item()
                
                if n_dead > 0:
                    import warnings
                    warnings.warn(f"Reinitializing {n_dead} dead features")
                    
                    # Reinitialize dead features
                    self.W_dec.data[:, dead_mask] = torch.randn_like(
                        self.W_dec.data[:, dead_mask]
                    ) * 0.01
                    
                    # Recompute norms after reinitialization
                    norms = torch.norm(self.W_dec, dim=0, keepdim=True)
                
                # Normalize all features
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
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        if x.size(-1) != self.n_features:
            raise ValueError(
                f"Input feature dimension {x.size(-1)} doesn't match "
                f"expected {self.n_features}"
            )
        
        if not torch.isfinite(x).all():
            raise ValueError("Input contains non-finite values (NaN or Inf)")
        
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
        # Input validation
        if not isinstance(a, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(a)}")
        
        if a.size(-1) != self.n_latents:
            raise ValueError(
                f"Latent dimension {a.size(-1)} doesn't match "
                f"expected {self.n_latents}"
            )
        
        if not torch.isfinite(a).all():
            raise ValueError("Activations contain non-finite values (NaN or Inf)")
        
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
        """
        Apply sparsity constraint to activations.
        
        Default implementation uses ReLU activation for non-negative sparsity.
        Subclasses can override for other sparsity types:
        - TopKSAE: keep only top-k largest activations
        - L0SAE: use straight-through estimator for discrete sparsity
        - GroupSAE: apply group-wise sparsity constraints
        """
        import torch.nn.functional as F
        # Default: ReLU activation for non-negative sparse activations
        return F.relu(a)
    
    def sparsity_loss(self, a: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity penalty for regularization.
        
        Default implementation uses L1 penalty (standard sparse autoencoder).
        Subclasses can override for other penalty types:
        - L1SAE: return self.l1_penalty * torch.mean(torch.abs(a))
        - L2SAE: return self.l2_penalty * torch.mean(a**2)
        - TopKSAE: return torch.tensor(0.0) # Hard constraint, no additional loss
        - L0SAE: return self.l0_penalty * torch.mean(torch.sigmoid(a * temperature))
        - BatchSparsenessSAE: return torch.abs(torch.mean(a, dim=0) - target_sparsity).mean()
        """
        import torch
        # Default: L1 penalty with standard coefficient
        l1_penalty = getattr(self, 'l1_penalty', 0.001)
        return l1_penalty * torch.mean(torch.abs(a))
    
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
        
        # Validate k for this tensor
        latent_dim = a_pos.shape[-1]
        if self.k >= latent_dim:
            if self.k > latent_dim:
                import warnings
                warnings.warn(
                    f"k={self.k} > latent_dim={latent_dim}, using all activations. "
                    f"Consider reducing k for meaningful sparsity."
                )
            return a_pos
        
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        
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
    verbose: bool = True,
    validation_data: Optional[torch.utils.data.DataLoader] = None,
    early_stopping_patience: int = 10,
    early_stopping_threshold: float = 1e-6
) -> Dict[str, Any]:
    """
    Train SAE with standard training loop and optional early stopping.
    
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
    validation_data : DataLoader, optional
        Validation data for early stopping
    early_stopping_patience : int, default=10
        Number of epochs to wait for improvement before stopping
    early_stopping_threshold : float, default=1e-6
        Minimum improvement threshold for early stopping
        
    Returns
    -------
    history : dict
        Training history with losses and early stopping info
    """
    if device is not None:
        sae = sae.to(device)
    
    history = {
        'total_loss': [],
        'reconstruction_loss': [],
        'sparsity_loss': [],
        'validation_loss': [],
        'early_stopped': False,
        'best_epoch': 0
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
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
        
        # Record training history
        history['total_loss'].append(epoch_losses['total'])
        history['reconstruction_loss'].append(epoch_losses['reconstruction'])
        history['sparsity_loss'].append(epoch_losses['sparsity'])
        
        # Validation and early stopping
        val_loss = None
        if validation_data is not None:
            sae.eval()
            val_losses = {'total': 0.0, 'reconstruction': 0.0, 'sparsity': 0.0}
            n_val_batches = 0
            
            with torch.no_grad():
                for val_batch_x in validation_data:
                    if device is not None:
                        val_batch_x = val_batch_x.to(device)
                    
                    val_x_hat, val_a = sae(val_batch_x)
                    val_losses_batch = sae.compute_loss(val_batch_x, val_x_hat, val_a, sparsity_weight)
                    
                    val_losses['total'] += val_losses_batch['total'].item()
                    val_losses['reconstruction'] += val_losses_batch['reconstruction'].item()
                    val_losses['sparsity'] += val_losses_batch['sparsity'].item()
                    n_val_batches += 1
            
            # Average validation losses
            for key in val_losses:
                val_losses[key] /= n_val_batches
            
            val_loss = val_losses['total']
            history['validation_loss'].append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - early_stopping_threshold:
                best_val_loss = val_loss
                patience_counter = 0
                history['best_epoch'] = epoch
                best_model_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}
            else:
                patience_counter += 1
            
            sae.train()
        else:
            # Use training loss for early stopping if no validation data
            val_loss = epoch_losses['total']
            history['validation_loss'].append(val_loss)
            
            if val_loss < best_val_loss - early_stopping_threshold:
                best_val_loss = val_loss
                patience_counter = 0
                history['best_epoch'] = epoch
                best_model_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}
            else:
                patience_counter += 1
        
        # Normalize decoder weights periodically
        if (epoch + 1) % normalize_freq == 0:
            sae.normalize_decoder_weights()
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            val_str = f", Val={val_loss:.6f}" if val_loss is not None else ""
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"Total={epoch_losses['total']:.6f}, "
                  f"Recon={epoch_losses['reconstruction']:.6f}, "
                  f"Sparse={epoch_losses['sparsity']:.6f}"
                  f"{val_str}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} (patience: {early_stopping_patience})")
            history['early_stopped'] = True
            break
    
    # Restore best model if early stopping was used
    if best_model_state is not None and validation_data is not None:
        sae.load_state_dict({k: v.to(device if device else 'cpu') for k, v in best_model_state.items()})
        if verbose:
            print(f"Restored best model from epoch {history['best_epoch'] + 1}")
    
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