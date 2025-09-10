"""
Interpretability tools for sparse coding and SAE analysis.

Provides feature analysis, activation patterns, and visualization tools
for understanding learned representations in both classical dictionary
learning and modern sparse autoencoders.

Key References
--------------
- Olshausen & Field (1996): "Emergence of simple-cell receptive field properties
  by learning a sparse code for natural images" - foundational sparse coding
- Elhage et al. (2022): "Toy Models of Superposition" - polysemanticity concepts
- Cunningham et al. (2023): "Sparse Autoencoders Find Highly Interpretable 
  Features in Language Models" - modern SAE analysis techniques
- Bricken et al. (2023): "Towards Monosemanticity: Decomposing Language Models
  With Dictionary Learning" - feature interpretability metrics
- Conmy et al. (2023): "Towards Automated Circuit Discovery for Mechanistic 
  Interpretability" - feature interaction analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings

from ..core.array import ArrayLike, xp, as_same, ensure_array
from .feature_interface import Features, encode_features
def detect_polysemantic_features(
    activations: ArrayLike,
    silhouette_threshold: float = 0.3,
    min_cluster_size: int = 5,
    n_clusters: Optional[int] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Enhanced polysemanticity detection with configurable parameters and robust fallbacks.
    
    Uses clustering analysis to identify features that respond to multiple
    distinct input patterns, indicating polysemantic behavior.
    
    Parameters
    ----------
    activations : ArrayLike, shape (n_samples, n_features)
    Feature activation patterns
    silhouette_threshold : float, default=0.3
    Minimum silhouette score for considering clusters valid
    min_cluster_size : int, default=5
    Minimum samples required per cluster
    n_clusters : int, optional
    Number of clusters to use. If None, estimated automatically
    random_state : int, default=42
    Random seed for reproducibility
        
    Returns
    -------
    results : dict
    Dictionary containing:
    - polysemantic_features: List of feature indices identified as polysemantic
    - feature_scores: Per-feature polysemanticity scores (0-1)
    - cluster_analysis: Detailed clustering results per feature
    - method_used: Which method was used ('sklearn' or 'fallback')
        
    References
    ----------
    Elhage et al. (2022): "Toy Models of Superposition" - polysemanticity concepts
    Cunningham et al. (2023): "Sparse Autoencoders Find Highly Interpretable Features"
    """
    activations = ensure_array(activations)
    n_samples, n_features = activations.shape
    
    if n_samples < min_cluster_size * 2:
        warnings.warn("Too few samples for reliable clustering analysis")
        
    # Use sklearn-based clustering
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    method_used = 'sklearn'
    
    polysemantic_features = []
    feature_scores = np.zeros(n_features)
    cluster_analysis = {}
    
    for feat_idx in range(n_features):
        feat_activations = activations[:, feat_idx].reshape(-1, 1)
        
        # Skip if feature is mostly inactive
        active_mask = np.abs(feat_activations.flatten()) > 1e-6
        if np.sum(active_mask) < min_cluster_size * 2:
            continue
                
        active_activations = feat_activations[active_mask]
            
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            max_k = min(8, len(active_activations) // min_cluster_size)
            best_k = 2
            best_silhouette = -1
                
            for k in range(2, max_k + 1):
                if len(active_activations) >= k * min_cluster_size:
                    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                    labels = kmeans.fit_predict(active_activations)
                        
                    if len(np.unique(labels)) == k:  # All clusters have points
                        silhouette = silhouette_score(active_activations, labels)
                        if silhouette > best_silhouette:
                            best_silhouette = best_silhouette
                            best_k = k
            use_k = best_k
        else:
            use_k = min(n_clusters, len(active_activations) // min_cluster_size)
                
        if use_k < 2:
            continue
                
        # Perform final clustering
        kmeans = KMeans(n_clusters=use_k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(active_activations)
            
        # Check cluster validity
        unique_labels = np.unique(labels)
        valid_clusters = []
            
        for label in unique_labels:
            cluster_size = np.sum(labels == label)
            if cluster_size >= min_cluster_size:
                valid_clusters.append(label)
                    
        if len(valid_clusters) >= 2:
            try:
                silhouette = silhouette_score(active_activations, labels)
                    
                # Calculate comprehensive polysemanticity score
                cluster_balance = len(valid_clusters) / use_k
                cluster_separation = max(0.0, silhouette)
                activation_frequency = np.sum(active_mask) / n_samples
                    
                # Multi-factor score combining separation quality and cluster properties
                poly_score = (
                    0.4 * cluster_separation +
                    0.3 * cluster_balance + 
                    0.2 * min(1.0, activation_frequency * 10) +
                    0.1 * min(1.0, len(valid_clusters) / 4)
                )
                    
                feature_scores[feat_idx] = poly_score
                    
                # Store detailed analysis
                cluster_analysis[feat_idx] = {
                    'silhouette_score': silhouette,
                    'n_clusters': len(valid_clusters),
                    'cluster_sizes': [np.sum(labels == label) for label in valid_clusters],
                    'activation_frequency': activation_frequency,
                    'cluster_balance': cluster_balance,
                    'polysemanticity_score': poly_score,
                    'interpretability_difficulty': 'High' if poly_score > 0.7 else 'Medium' if poly_score > 0.4 else 'Low'
                }
                    
                # Feature is polysemantic if it meets threshold
                if silhouette > silhouette_threshold and poly_score > 0.3:
                    polysemantic_features.append(feat_idx)
                        
            except ValueError:
                # Silhouette calculation failed
                continue
                    
    
    return {
    'polysemantic_features': polysemantic_features,
    'feature_scores': feature_scores,
    'cluster_analysis': cluster_analysis,
    'method_used': method_used,
    'parameters': {
        'silhouette_threshold': silhouette_threshold,
        'min_cluster_size': min_cluster_size,
        'n_clusters': n_clusters
    }
    }


def _compute_feature_correlations(activations: ArrayLike, threshold: float = 1e-6) -> ArrayLike:
    """
    Compute numerically stable feature correlation matrix.
    
    Handles dead features and numerical instabilities that can occur
    when computing correlations of sparse activation patterns.
    
    Parameters
    ---------- 
    activations : ArrayLike, shape (n_samples, n_features)
    Feature activation patterns
    threshold : float, default=1e-6
    Variance threshold for identifying dead features
        
    Returns
    -------
    correlation_matrix : ArrayLike, shape (n_features, n_features)
    Correlation matrix with proper handling of dead features
    """
    activations = ensure_array(activations)
    n_samples, n_features = activations.shape
    
    # Identify dead features (zero variance)
    feature_vars = xp.var(activations, axis=0)
    dead_mask = feature_vars < threshold
    
    # Initialize correlation matrix
    corr_matrix = xp.eye(n_features)
    
    # Compute correlations only for active features 
    active_mask = ~dead_mask
    active_indices = xp.where(active_mask)[0]
    
    if len(active_indices) > 1:
        active_features = activations[:, active_indices]
        
        # Standardize features (mean=0, std=1)
        means = xp.mean(active_features, axis=0, keepdims=True)
        stds = xp.std(active_features, axis=0, keepdims=True)
        stds = xp.maximum(stds, threshold)  # Avoid division by very small numbers
        
        standardized = (active_features - means) / stds
        
        # Compute correlation matrix for active features
        active_corr = xp.dot(standardized.T, standardized) / n_samples
        
        # Fill in correlations for active features
        for i, idx_i in enumerate(active_indices):
            for j, idx_j in enumerate(active_indices):
                corr_matrix[idx_i, idx_j] = active_corr[i, j]
    
    # Set correlations involving dead features to 0
    corr_matrix[dead_mask, :] = 0
    corr_matrix[:, dead_mask] = 0
    
    # Ensure diagonal is 1 for active features, 0 for dead
    xp.fill_diagonal(corr_matrix, ~dead_mask)
    
    # Clamp to valid correlation range [-1, 1]
    corr_matrix = xp.clip(corr_matrix, -1.0, 1.0)
    
    return corr_matrix