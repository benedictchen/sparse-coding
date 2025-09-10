"""
Interpretability tools for sparse coding and SAE analysis.

Provides feature analysis, activation patterns, and visualization tools
for understanding learned representations in both classical dictionary
learning and modern sparse autoencoders.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings

from ..core.array import ArrayLike, xp, as_same, ensure_array
from .feature_interface import Features, encode_features


@dataclass
class FeatureStats:
    """
    Statistical analysis of individual dictionary atoms/SAE features.
    
    Attributes
    ----------
    feature_idx : int
        Index of the feature
    activation_freq : float  
        Fraction of samples where feature is active (>threshold)
    mean_activation : float
        Mean activation strength when active
    max_activation : float
        Maximum activation observed
    selectivity : float
        Measure of feature selectivity (entropy-based)
    top_activating_samples : np.ndarray
        Indices of samples with highest activations
    """
    feature_idx: int
    activation_freq: float
    mean_activation: float  
    max_activation: float
    selectivity: float
    top_activating_samples: np.ndarray


class FeatureAnalyzer:
    """
    Comprehensive analysis of sparse coding features and activations.
    
    Provides tools for understanding what features have learned,
    identifying redundancy, measuring sparsity patterns, and
    analyzing feature interactions.
    
    Parameters
    ----------
    features : Features
        Fitted sparse features to analyze
    activation_threshold : float, default=1e-6
        Threshold for considering a feature "active"
    """
    
    def __init__(self, features: Features, activation_threshold: float = 1e-6):
        self.features = features
        self.threshold = activation_threshold
        self._cached_codes: Optional[ArrayLike] = None
        self._cached_data: Optional[ArrayLike] = None
    
    def analyze_codes(
        self, 
        X: ArrayLike, 
        A: Optional[ArrayLike] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of sparse codes for given data.
        
        Parameters
        ----------
        X : ArrayLike, shape (n_samples, n_features)
            Input data
        A : ArrayLike, optional, shape (n_samples, n_atoms)
            Precomputed sparse codes. If None, computed from X.
            
        Returns
        -------
        analysis : dict
            Dictionary containing various analysis results
        """
        if A is None:
            A = encode_features(X, self.features)
        
        self._cached_codes = A
        self._cached_data = X
        
        backend = xp(A)
        
        # Basic sparsity statistics
        active_mask = backend.abs(A) > self.threshold
        sparsity_level = float(backend.mean(~active_mask))
        active_per_sample = backend.sum(active_mask, axis=1)
        
        # Feature usage statistics  
        feature_usage = backend.mean(active_mask, axis=0)
        dead_features = backend.sum(feature_usage < 1e-8)
        
        # Activation strength statistics
        A_active = backend.where(active_mask, A, 0)
        mean_active_strength = backend.sum(A_active, axis=0) / (backend.sum(active_mask, axis=0) + 1e-8)
        max_activations = backend.max(backend.abs(A), axis=0)
        
        # Feature correlation analysis
        feature_corr = self._compute_feature_correlations(A)
        high_corr_pairs = self._find_correlated_features(feature_corr, threshold=0.8)
        
        return {
            'sparsity_level': sparsity_level,
            'mean_active_per_sample': float(backend.mean(active_per_sample)),
            'std_active_per_sample': float(backend.std(active_per_sample)),
            'feature_usage_freq': feature_usage,
            'dead_features': int(dead_features),
            'mean_activation_strength': mean_active_strength,
            'max_activations': max_activations,
            'feature_correlations': feature_corr,
            'highly_correlated_pairs': high_corr_pairs,
            'reconstruction_error': self._compute_reconstruction_error(X, A)
        }
    
    def analyze_individual_features(
        self, 
        X: ArrayLike, 
        A: Optional[ArrayLike] = None,
        top_k: int = 10
    ) -> List[FeatureStats]:
        """
        Analyze each feature individually.
        
        Parameters
        ----------
        X : ArrayLike, shape (n_samples, n_features)
            Input data
        A : ArrayLike, optional, shape (n_samples, n_atoms)
            Precomputed sparse codes
        top_k : int, default=10
            Number of top activating samples to track per feature
            
        Returns
        -------
        feature_stats : list
            List of FeatureStats for each feature
        """
        if A is None:
            A = encode_features(X, self.features)
        
        backend = xp(A)
        n_samples, n_atoms = A.shape
        feature_stats = []
        
        for i in range(n_atoms):
            activations = A[:, i]
            active_mask = backend.abs(activations) > self.threshold
            
            # Basic statistics
            activation_freq = float(backend.mean(active_mask))
            mean_activation = float(backend.mean(backend.abs(activations[active_mask]))) if backend.any(active_mask) else 0.0
            max_activation = float(backend.max(backend.abs(activations)))
            
            # Selectivity (based on activation distribution entropy)
            selectivity = self._compute_selectivity(activations)
            
            # Top activating samples
            top_indices = backend.argsort(backend.abs(activations))[-top_k:][::-1]
            
            stats = FeatureStats(
                feature_idx=i,
                activation_freq=activation_freq,
                mean_activation=mean_activation,
                max_activation=max_activation,
                selectivity=selectivity,
                top_activating_samples=as_same(top_indices, np.array([]))
            )
            feature_stats.append(stats)
        
        return feature_stats
    
    def find_feature_interactions(
        self,
        A: ArrayLike,
        method: str = 'coactivation',
        threshold: float = 0.1
    ) -> Dict[Tuple[int, int], float]:
        """
        Find interactions between features.
        
        Parameters
        ----------
        A : ArrayLike, shape (n_samples, n_atoms)
            Sparse codes
        method : str, default='coactivation'
            Method for measuring interactions ('coactivation', 'mutual_info')
        threshold : float, default=0.1
            Minimum interaction strength to report
            
        Returns
        -------
        interactions : dict
            Dictionary mapping feature pairs to interaction strengths
        """
        backend = xp(A)
        n_samples, n_atoms = A.shape
        interactions = {}
        
        if method == 'coactivation':
            active_mask = backend.abs(A) > self.threshold
            
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    # Compute coactivation frequency
                    both_active = backend.logical_and(active_mask[:, i], active_mask[:, j])
                    coactivation = float(backend.mean(both_active))
                    
                    # Normalize by individual activation frequencies
                    freq_i = float(backend.mean(active_mask[:, i]))
                    freq_j = float(backend.mean(active_mask[:, j]))
                    
                    if freq_i > 0 and freq_j > 0:
                        normalized_coactivation = coactivation / (freq_i * freq_j)
                        
                        if normalized_coactivation > threshold:
                            interactions[(i, j)] = normalized_coactivation
        
        elif method == 'mutual_info':
            # Simplified mutual information estimate
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    mi = self._estimate_mutual_info(A[:, i], A[:, j])
                    if mi > threshold:
                        interactions[(i, j)] = mi
        
        return interactions
    
    def detect_polysemantic_features(
        self,
        X: ArrayLike,
        A: Optional[ArrayLike] = None,
        n_clusters: int = 5
    ) -> Dict[int, Dict[str, Any]]:
        """
        Detect potentially polysemantic features (responding to multiple concepts).
        
        Parameters
        ----------
        X : ArrayLike, shape (n_samples, n_features)
            Input data
        A : ArrayLike, optional, shape (n_samples, n_atoms)
            Sparse codes
        n_clusters : int, default=5
            Number of clusters for analyzing feature responses
            
        Returns
        -------
        polysemantic : dict
            Dictionary mapping feature indices to polysemanticity analysis
        """
        if A is None:
            A = encode_features(X, self.features)
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            warnings.warn("scikit-learn required for polysemantic detection")
            return {}
        
        backend = xp(A)
        n_samples, n_atoms = A.shape
        polysemantic_features = {}
        
        for i in range(n_atoms):
            activations = A[:, i]
            active_mask = backend.abs(activations) > self.threshold
            
            if backend.sum(active_mask) < n_clusters * 2:  # Not enough active samples
                continue
            
            # Get inputs that activate this feature
            active_inputs = X[as_same(active_mask, np.array([]))]
            active_strengths = activations[active_mask]
            
            if len(active_inputs) < n_clusters * 2:
                continue
            
            # Cluster the activating inputs
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(active_inputs)
            
            # Compute silhouette score (higher = more distinct clusters)
            silhouette = silhouette_score(active_inputs, cluster_labels)
            
            # Analyze cluster properties
            cluster_analysis = {}
            for c in range(n_clusters):
                cluster_mask = cluster_labels == c
                cluster_analysis[c] = {
                    'size': int(np.sum(cluster_mask)),
                    'mean_activation': float(np.mean(as_same(active_strengths, np.array([]))[cluster_mask])),
                    'centroid': kmeans.cluster_centers_[c]
                }
            
            # Consider polysemantic if clusters are well-separated
            if silhouette > 0.3:  # Threshold for "good" clustering
                polysemantic_features[i] = {
                    'silhouette_score': silhouette,
                    'n_clusters': n_clusters,
                    'clusters': cluster_analysis,
                    'activation_freq': float(backend.mean(active_mask))
                }
        
        return polysemantic_features
    
    def _compute_feature_correlations(self, A: ArrayLike) -> ArrayLike:
        """Compute pairwise feature correlations."""
        backend = xp(A)
        
        # Compute correlation matrix
        A_centered = A - backend.mean(A, axis=0, keepdims=True)
        cov_matrix = backend.matmul(A_centered.T, A_centered) / (A.shape[0] - 1)
        
        # Normalize to get correlations
        std_devs = backend.sqrt(backend.diag(cov_matrix))
        std_matrix = backend.outer(std_devs, std_devs)
        
        corr_matrix = cov_matrix / (std_matrix + 1e-8)
        
        return corr_matrix
    
    def _find_correlated_features(
        self, 
        corr_matrix: ArrayLike, 
        threshold: float = 0.8
    ) -> List[Tuple[int, int, float]]:
        """Find highly correlated feature pairs."""
        backend = xp(corr_matrix)
        n_atoms = corr_matrix.shape[0]
        correlated_pairs = []
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                corr_val = float(backend.abs(corr_matrix[i, j]))
                if corr_val > threshold:
                    correlated_pairs.append((i, j, corr_val))
        
        return sorted(correlated_pairs, key=lambda x: x[2], reverse=True)
    
    def _compute_selectivity(self, activations: ArrayLike) -> float:
        """Compute feature selectivity based on activation entropy."""
        backend = xp(activations)
        
        # Discretize activations into bins
        active_vals = activations[backend.abs(activations) > self.threshold]
        if len(active_vals) < 2:
            return 0.0
        
        hist, _ = backend.histogram(active_vals, bins=10)
        hist = hist + 1e-8  # Avoid log(0)
        probs = hist / backend.sum(hist)
        
        # Compute entropy (lower entropy = higher selectivity)
        entropy = -backend.sum(probs * backend.log(probs))
        max_entropy = backend.log(len(hist))
        
        return float(1.0 - entropy / max_entropy)
    
    def _compute_reconstruction_error(self, X: ArrayLike, A: ArrayLike) -> float:
        """Compute mean squared reconstruction error."""
        from .feature_interface import decode_features
        
        X_hat = decode_features(A, self.features)
        backend = xp(X)
        
        return float(backend.mean((X - X_hat)**2))
    
    def _estimate_mutual_info(self, x: ArrayLike, y: ArrayLike) -> float:
        """Simplified mutual information estimate using binning."""
        backend = xp(x)
        
        # Discretize to categorical
        x_discrete = backend.digitize(x, backend.linspace(backend.min(x), backend.max(x), 10))
        y_discrete = backend.digitize(y, backend.linspace(backend.min(y), backend.max(y), 10))
        
        # Compute joint and marginal histograms
        joint_hist = backend.histogram2d(x_discrete, y_discrete, bins=10)[0]
        x_hist = backend.sum(joint_hist, axis=1)
        y_hist = backend.sum(joint_hist, axis=0)
        
        # Normalize to probabilities
        joint_prob = joint_hist / backend.sum(joint_hist)
        x_prob = x_hist / backend.sum(x_hist)
        y_prob = y_hist / backend.sum(y_hist)
        
        # Compute mutual information
        mi = 0.0
        for i in range(len(x_prob)):
            for j in range(len(y_prob)):
                if joint_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                    mi += joint_prob[i, j] * backend.log(
                        joint_prob[i, j] / (x_prob[i] * y_prob[j])
                    )
        
        return float(mi)


def create_feature_atlas(
    features: Features,
    X_samples: ArrayLike,
    n_examples: int = 5,
    sort_by: str = 'usage'
) -> Dict[str, Any]:
    """
    Create comprehensive atlas of learned features with examples.
    
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
        Comprehensive feature atlas with examples and statistics
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