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
        n_clusters: int = 5,
        silhouette_threshold: float = 0.3,
        min_cluster_size: int = 10
    ) -> Dict[int, Dict[str, Any]]:
        """
        Detect potentially polysemantic features (responding to multiple concepts).
        
        Uses clustering analysis of activating inputs to identify features that
        respond to distinct input patterns, following Elhage et al. (2022) and
        Cunningham et al. (2023) approaches to polysemanticity detection.
        
        Parameters
        ----------
        X : ArrayLike, shape (n_samples, n_features)
            Input data
        A : ArrayLike, optional, shape (n_samples, n_atoms)
            Sparse codes (computed if not provided)
        n_clusters : int, default=5
            Number of clusters for analyzing feature responses
        silhouette_threshold : float, default=0.3
            Minimum silhouette score to consider clusters well-separated
            Based on interpretation: >0.5 good, 0.25-0.5 weak, <0.25 poor
        min_cluster_size : int, default=10
            Minimum samples per cluster for robust analysis
            
        Returns
        -------
        polysemantic : dict
            Dictionary mapping feature indices to polysemanticity analysis:
            - 'silhouette_score': cluster separation quality
            - 'n_clusters': number of detected concept clusters
            - 'clusters': detailed per-cluster statistics
            - 'activation_freq': feature activation frequency
            - 'polysemanticity_score': normalized measure of polysemanticity
        """
        if A is None:
            A = encode_features(X, self.features)
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            warnings.warn(
                "scikit-learn required for polysemantic detection. "
                "Install with: pip install scikit-learn"
            )
            # Fallback to simple variance-based heuristic
            return self._simple_polysemantic_fallback(X, A, n_clusters)
        
        backend = xp(A)
        n_samples, n_atoms = A.shape
        polysemantic_features = {}
        
        # Validate inputs
        if n_samples < min_cluster_size * n_clusters:
            warnings.warn(
                f"Insufficient samples ({n_samples}) for robust clustering "
                f"with {n_clusters} clusters requiring {min_cluster_size} samples each"
            )
            return {}
        
        for i in range(n_atoms):
            activations = A[:, i]
            active_mask = backend.abs(activations) > self.threshold
            n_active = int(backend.sum(active_mask))
            
            # Require minimum active samples for robust analysis
            min_required = max(n_clusters * min_cluster_size, n_clusters * 2)
            if n_active < min_required:
                continue
            
            # Get inputs that activate this feature
            active_inputs = X[as_same(active_mask, np.array([]))]
            active_strengths = activations[active_mask]
            
            # Additional validation
            if len(active_inputs) != n_active:
                continue
            
            try:
                # Cluster the activating inputs with error handling
                kmeans = KMeans(
                    n_clusters=n_clusters, 
                    random_state=42, 
                    n_init=10,
                    max_iter=300
                )
                cluster_labels = kmeans.fit_predict(active_inputs)
                
                # Validate clustering result
                unique_labels = np.unique(cluster_labels)
                if len(unique_labels) < 2:  # Degenerate clustering
                    continue
                
                # Compute silhouette score (higher = more distinct clusters)
                silhouette = silhouette_score(active_inputs, cluster_labels)
                
                # Analyze cluster properties with validation
                cluster_analysis = {}
                valid_clusters = 0
                active_strengths_np = as_same(active_strengths, np.array([]))
                
                for c in range(n_clusters):
                    cluster_mask = cluster_labels == c
                    cluster_size = int(np.sum(cluster_mask))
                    
                    # Skip tiny clusters
                    if cluster_size < min_cluster_size // 2:
                        continue
                        
                    valid_clusters += 1
                    cluster_activations = active_strengths_np[cluster_mask]
                    
                    cluster_analysis[c] = {
                        'size': cluster_size,
                        'mean_activation': float(np.mean(cluster_activations)),
                        'std_activation': float(np.std(cluster_activations)),
                        'centroid': kmeans.cluster_centers_[c].tolist(),
                        'cluster_coherence': float(np.mean([
                            np.linalg.norm(active_inputs[j] - kmeans.cluster_centers_[c])
                            for j in np.where(cluster_mask)[0][:min(10, cluster_size)]
                        ]))
                    }
                
                # Enhanced polysemanticity scoring
                if silhouette > silhouette_threshold and valid_clusters >= 2:
                    # Compute comprehensive polysemanticity score
                    activation_freq = float(backend.mean(active_mask))
                    cluster_size_variance = np.var([
                        cluster_analysis[c]['size'] for c in cluster_analysis
                    ])
                    
                    # Normalized polysemanticity score (0-1)
                    poly_score = min(1.0, (
                        0.4 * silhouette +  # Cluster separation (0-1)
                        0.3 * min(1.0, valid_clusters / n_clusters) +  # Cluster validity
                        0.2 * activation_freq +  # Feature activity
                        0.1 * (1 - min(1.0, cluster_size_variance / n_active))  # Balanced clusters
                    ))
                    
                    polysemantic_features[i] = {
                        'silhouette_score': float(silhouette),
                        'n_clusters': valid_clusters,
                        'clusters': cluster_analysis,
                        'activation_freq': activation_freq,
                        'polysemanticity_score': poly_score,
                        'cluster_size_variance': float(cluster_size_variance),
                        'interpretability_difficulty': 'high' if poly_score > 0.7 else 'medium' if poly_score > 0.4 else 'low'
                    }
                    
            except Exception as e:
                warnings.warn(f"Clustering failed for feature {i}: {e}")
                continue
        
        return polysemantic_features
    
    def _simple_polysemantic_fallback(
        self, 
        X: ArrayLike, 
        A: ArrayLike, 
        n_clusters: int
    ) -> Dict[int, Dict[str, Any]]:
        """
        Simple polysemanticity detection without sklearn.
        
        Uses activation pattern variance as a heuristic for polysemanticity.
        Features with high variance in their activation patterns across different
        input regions may be polysemantic.
        """
        backend = xp(A)
        n_samples, n_atoms = A.shape
        polysemantic_features = {}
        
        # Divide data into spatial regions for variance analysis
        regions = min(n_clusters, n_samples // 10)
        region_size = n_samples // regions
        
        for i in range(n_atoms):
            activations = A[:, i]
            active_mask = backend.abs(activations) > self.threshold
            
            if backend.sum(active_mask) < regions * 2:
                continue
            
            # Compute activation statistics across regions
            region_activations = []
            for r in range(regions):
                start_idx = r * region_size
                end_idx = min((r + 1) * region_size, n_samples)
                
                region_mask = active_mask[start_idx:end_idx]
                if backend.sum(region_mask) > 0:
                    region_mean = float(backend.mean(
                        backend.abs(activations[start_idx:end_idx][region_mask])
                    ))
                    region_activations.append(region_mean)
            
            if len(region_activations) < 2:
                continue
                
            # High variance suggests polysemanticity
            activation_variance = float(np.var(region_activations))
            activation_mean = float(np.mean(region_activations))
            
            # Coefficient of variation as polysemanticity heuristic
            if activation_mean > 0:
                cv = activation_variance / (activation_mean ** 2)
                if cv > 0.5:  # High variability threshold
                    polysemantic_features[i] = {
                        'activation_variance': activation_variance,
                        'activation_cv': cv,
                        'activation_freq': float(backend.mean(active_mask)),
                        'polysemanticity_score': min(1.0, cv),
                        'method': 'variance_fallback',
                        'interpretability_difficulty': 'medium' if cv > 0.8 else 'low'
                    }
        
        return polysemantic_features
    
    def _compute_feature_correlations(self, A: ArrayLike) -> ArrayLike:
        """
        Compute pairwise feature correlations with robust handling of dead features.
        
        Returns correlation matrix with proper masking for zero-variance features.
        """
        backend = xp(A)
        
        # Compute correlation matrix with centered data
        A_centered = A - backend.mean(A, axis=0, keepdims=True)
        cov_matrix = backend.matmul(A_centered.T, A_centered) / (A.shape[0] - 1)
        
        # Compute standard deviations with numerical stability
        variances = backend.diag(cov_matrix)
        std_devs = backend.sqrt(backend.maximum(variances, 1e-12))
        
        # Identify dead features (zero variance)
        dead_mask = variances < 1e-10
        
        # Create correlation matrix with proper normalization
        std_matrix = backend.outer(std_devs, std_devs)
        corr_matrix = backend.where(
            std_matrix > 1e-10,
            cov_matrix / std_matrix,
            0.0  # Set correlations involving dead features to 0
        )
        
        # Ensure diagonal is 1.0 for active features, 0.0 for dead features
        n_atoms = A.shape[1]
        diag_mask = backend.eye(n_atoms)
        corr_matrix = backend.where(
            diag_mask > 0,
            backend.where(dead_mask[:, None], 0.0, 1.0),
            corr_matrix
        )
        
        # Clamp correlations to valid range [-1, 1]
        corr_matrix = backend.clip(corr_matrix, -1.0, 1.0)
        
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
        """
        Improved mutual information estimate using adaptive binning.
        
        Uses Scott's rule for optimal bin selection and normalizes inputs
        for better numerical stability.
        """
        backend = xp(x)
        
        # Normalize inputs to [0, 1] for better bin uniformity
        x_min, x_max = backend.min(x), backend.max(x)
        y_min, y_max = backend.min(y), backend.max(y)
        
        # Handle constant variables (zero variance)
        if x_max == x_min or y_max == y_min:
            return 0.0
            
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)
        
        # Adaptive binning using Scott's rule: n_bins = ceil(2 * n^(1/3))
        n_samples = len(x)
        n_bins = max(5, min(50, int(backend.ceil(2 * n_samples**(1/3)))))
        
        # Create bins with small epsilon to handle edge cases
        eps = 1e-10
        x_bins = backend.linspace(-eps, 1 + eps, n_bins + 1)
        y_bins = backend.linspace(-eps, 1 + eps, n_bins + 1)
        
        # Discretize using normalized values
        x_discrete = backend.digitize(x_norm, x_bins) - 1  # 0-indexed
        y_discrete = backend.digitize(y_norm, y_bins) - 1  # 0-indexed
        
        # Clamp to valid range (handle edge cases)
        x_discrete = backend.clip(x_discrete, 0, n_bins - 1)
        y_discrete = backend.clip(y_discrete, 0, n_bins - 1)
        
        # Compute joint histogram using more stable approach
        joint_hist = backend.zeros((n_bins, n_bins))
        for i in range(n_samples):
            joint_hist[x_discrete[i], y_discrete[i]] += 1
        
        # Add small epsilon to avoid log(0)
        joint_hist = joint_hist + 1e-12
        
        # Marginal histograms
        x_hist = backend.sum(joint_hist, axis=1)
        y_hist = backend.sum(joint_hist, axis=0)
        
        # Normalize to probabilities
        total = backend.sum(joint_hist)
        joint_prob = joint_hist / total
        x_prob = x_hist / total
        y_prob = y_hist / total
        
        # Vectorized mutual information computation
        # MI = sum(P(x,y) * log(P(x,y) / (P(x) * P(y))))
        outer_prod = backend.outer(x_prob, y_prob)
        # Avoid log(0) with maximum
        ratio = backend.maximum(joint_prob / backend.maximum(outer_prod, 1e-12), 1e-12)
        mi = backend.sum(joint_prob * backend.log(ratio))
        
        return float(backend.maximum(mi, 0.0))  # Ensure non-negative


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


def _estimate_mutual_info(X: ArrayLike, Y: ArrayLike) -> float:
    """
    Enhanced mutual information estimation with adaptive binning and numerical stability.
    
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
        
    # Try sklearn-based clustering first
    try:
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
                    
    except ImportError:
        warnings.warn(
            "scikit-learn not available. Using variance-based fallback for polysemanticity detection. "
            "Install sklearn for more accurate clustering analysis: pip install scikit-learn"
        )
        method_used = 'fallback'
        
        # Variance-based heuristic fallback
        polysemantic_features = []
        feature_scores = np.zeros(n_features)
        cluster_analysis = {}
        
        for feat_idx in range(n_features):
            feat_activations = activations[:, feat_idx]
            
            # Skip mostly inactive features
            active_mask = np.abs(feat_activations) > 1e-6
            if np.sum(active_mask) < min_cluster_size:
                continue
                
            active_vals = feat_activations[active_mask]
            
            # Simple heuristic: high variance suggests multiple modes
            if len(active_vals) >= min_cluster_size:
                # Normalize by mean to get relative variance
                rel_variance = np.var(active_vals) / (np.mean(np.abs(active_vals)) + 1e-12)
                activation_freq = np.sum(active_mask) / len(feat_activations)
                
                # Simple polysemanticity score based on relative variance
                poly_score = min(1.0, rel_variance / 2.0) * min(1.0, activation_freq * 5)
                
                feature_scores[feat_idx] = poly_score
                cluster_analysis[feat_idx] = {
                    'relative_variance': rel_variance,
                    'activation_frequency': activation_freq,
                    'polysemanticity_score': poly_score,
                    'method': 'variance_heuristic'
                }
                
                # Conservative threshold for fallback method
                if poly_score > 0.5:
                    polysemantic_features.append(feat_idx)
    
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