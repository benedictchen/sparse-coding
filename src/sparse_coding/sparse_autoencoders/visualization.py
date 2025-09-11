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
def create_feature_atlas(
    features: Features,
    X_samples: ArrayLike,
    n_examples: int = 5,
    sort_by: str = 'usage'
) -> Dict[str, Any]:
    """
    Create atlas of learned features with examples.
    
    Parameters
    ----------
    features : Features
        Fitted sparse features
    X_samples : ArrayLike, shape (n_samples, n_features)
        Sample data for finding feature examples
    n_examples : int, default=5
        Number of top examples per feature
    sort_by : str, default='usage'
        How to sort features ('usage', 'max_activation', 'selectivity')
        
    Returns
    -------
    atlas : dict
        Feature atlas with examples and statistics
    """
    analyzer = FeatureAnalyzer(features)
    
    # Encode samples
    A = encode_features(X_samples, features)
    
    # Analyze codes and individual features
    code_analysis = analyzer.analyze_codes(X_samples, A)
    feature_stats = analyzer.analyze_individual_features(X_samples, A, top_k=n_examples)
    
    # Sort features by specified criterion
    if sort_by == 'usage':
        sort_key = lambda fs: fs.activation_freq
    elif sort_by == 'max_activation':
        sort_key = lambda fs: fs.max_activation
    elif sort_by == 'selectivity':
        sort_key = lambda fs: fs.selectivity
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")
    
    sorted_features = sorted(feature_stats, key=sort_key, reverse=True)
    
    # Create atlas structure
    atlas = {
        'features': features,
        'overall_analysis': code_analysis,
        'feature_ranking': [fs.feature_idx for fs in sorted_features],
        'feature_details': {fs.feature_idx: fs for fs in feature_stats},
        'top_examples': {},
        'metadata': {
            'n_features': features.n_features,
            'n_atoms': features.n_atoms,
            'method': features.method,
            'n_samples_analyzed': X_samples.shape[0],
            'sort_criterion': sort_by
        }
    }
    
    # Add top examples for each feature
    backend = xp(A)
    for fs in feature_stats:
        idx = fs.feature_idx
        top_sample_indices = fs.top_activating_samples
        
        atlas['top_examples'][idx] = {
            'sample_indices': top_sample_indices,
            'activations': [float(A[int(si), idx]) for si in top_sample_indices],
            'inputs': X_samples[as_same(top_sample_indices, np.array([]))]
        }
    
    return atlas


def summarize_atlas(atlas: Dict[str, Any], n_top: int = 20) -> str:
    """
    Generate text summary of feature atlas.
    
    Parameters
    ----------
    atlas : dict
        Feature atlas from create_feature_atlas()
    n_top : int, default=20
        Number of top features to include in summary
        
    Returns
    -------
    summary : str
        Text summary of the atlas
    """
    meta = atlas['metadata']
    analysis = atlas['overall_analysis']
    
    summary_lines = [
        f"Feature Atlas Summary",
        f"=" * 50,
        f"Method: {meta['method'].upper()}",
        f"Dictionary size: {meta['n_features']} → {meta['n_atoms']}",
        f"Samples analyzed: {meta['n_samples_analyzed']:,}",
        f"",
        f"Overall Statistics:",
        f"- Sparsity level: {analysis['sparsity_level']:.1%}",
        f"- Mean active per sample: {analysis['mean_active_per_sample']:.1f}",
        f"- Dead features: {analysis['dead_features']} / {meta['n_atoms']}",
        f"- Reconstruction MSE: {analysis['reconstruction_error']:.6f}",
        f""
    ]
    
    if analysis['highly_correlated_pairs']:
        summary_lines.append(f"- Highly correlated pairs: {len(analysis['highly_correlated_pairs'])}")
    
    summary_lines.extend([
        f"",
        f"Top {n_top} Features (by {meta['sort_criterion']}):",
        f"-" * 40
    ])
    
    for i, feat_idx in enumerate(atlas['feature_ranking'][:n_top]):
        stats = atlas['feature_details'][feat_idx]
        summary_lines.append(
            f"{i+1:2d}. Feature {feat_idx:3d}: "
            f"usage={stats.activation_freq:.1%}, "
            f"max_act={stats.max_activation:.3f}, "
            f"selectivity={stats.selectivity:.3f}"
        )
    
    return "\n".join(summary_lines)


def _estimate_mutual_info(X: ArrayLike, Y: ArrayLike) -> float:
    """
    Mutual information estimation with adaptive binning and numerical stability.
    
    Implements Scott's rule for optimal bin selection and robust handling of
    edge cases including zero-variance features and log(0) errors.
    
    Parameters
    ----------
    X : ArrayLike, shape (n_samples, n_features)
        First variable
    Y : ArrayLike, shape (n_samples, n_features) 
        Second variable
        
    Returns
    -------
    mutual_info : float
        Estimated mutual information in nats
        
    References
    ----------
    Scott, D.W. (1979). "On optimal and data-based histograms."
    Biometrika, 66(3), 605-610.
    """
    X = ensure_array(X)
    Y = ensure_array(Y) 
    
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have same shape, got {X.shape} vs {Y.shape}")
    
    n_samples = X.shape[0]
    if n_samples < 10:
        warnings.warn("Too few samples for reliable MI estimation")
        return 0.0
    
    # Scott's rule for optimal bin number: n_bins = ceil(2 * n^(1/3))
    n_bins = max(5, min(50, int(np.ceil(2 * n_samples**(1/3)))))
    
    total_mi = 0.0
    n_features = X.shape[1]
    
    for i in range(n_features):
        x_i = X[:, i].flatten()
        y_i = Y[:, i].flatten()
        
        # Handle zero-variance features
        if np.var(x_i) < 1e-12 or np.var(y_i) < 1e-12:
            continue
            
        # Normalize to [0,1] for uniform bin distribution
        x_norm = (x_i - np.min(x_i)) / (np.max(x_i) - np.min(x_i) + 1e-12)
        y_norm = (y_i - np.min(y_i)) / (np.max(y_i) - np.min(y_i) + 1e-12)
        
        # Compute 2D histogram
        hist_xy, _, _ = np.histogram2d(x_norm, y_norm, bins=n_bins, 
                                     range=[[0, 1], [0, 1]])
        
        # Compute marginal histograms
        hist_x = np.sum(hist_xy, axis=1) 
        hist_y = np.sum(hist_xy, axis=0)
        
        # Normalize to probabilities
        hist_xy = hist_xy / n_samples
        hist_x = hist_x / n_samples  
        hist_y = hist_y / n_samples
        
        # Compute MI with numerical stability
        mi_feature = 0.0
        for j in range(n_bins):
            for k in range(n_bins):
                if hist_xy[j, k] > 1e-12:  # Avoid log(0)
                    p_joint = hist_xy[j, k]
                    p_x = hist_x[j] + 1e-12  # Epsilon for stability
                    p_y = hist_y[k] + 1e-12
                    
                    mi_feature += p_joint * np.log(p_joint / (p_x * p_y))
        
        total_mi += mi_feature
    
    # Average over features and ensure non-negative
    avg_mi = total_mi / n_features
    return max(0.0, avg_mi)


