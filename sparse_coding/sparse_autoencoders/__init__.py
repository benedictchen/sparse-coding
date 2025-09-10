"""
Sparse Autoencoders (SAEs) for modern interpretability workflows.

Integrates SAEs with the sparse coding framework, providing unified
feature interfaces and interpretability tools.
"""

from .torch_sae import SAE, TopKSAE, L1SAE, train_sae, convert_sae_to_dict, convert_dict_to_sae
from .feature_interface import (
    Features, FeatureExtractor, fit_features, encode_features, decode_features,
    compare_features, visualize_features
)
from .analysis import FeatureStats, FeatureAnalyzer
from .visualization import create_feature_atlas, summarize_atlas
from .detection import detect_polysemantic_features

__all__ = [
    # Core SAE classes
    'SAE', 'TopKSAE', 'L1SAE',
    
    # Training and conversion utilities
    'train_sae', 'convert_sae_to_dict', 'convert_dict_to_sae',
    
    # Unified feature interface
    'Features', 'FeatureExtractor', 'fit_features', 'encode_features', 'decode_features',
    'compare_features', 'visualize_features',
    
    # Interpretability tools
    'FeatureStats', 'FeatureAnalyzer', 'create_feature_atlas', 'summarize_atlas',
    'detect_polysemantic_features'
]