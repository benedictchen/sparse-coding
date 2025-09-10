"""
Sparse Autoencoders (SAEs) for modern interpretability workflows.

Integrates SAEs with the sparse coding framework, providing unified
feature interfaces and interpretability tools.
"""

from .torch_sae import SAE, TopKSAE, L1SAE
from .feature_interface import FeatureExtractor, fit_features, encode_features, decode_features
from .interpretability import FeatureAnalyzer, create_feature_atlas

__all__ = [
    'SAE', 'TopKSAE', 'L1SAE',
    'FeatureExtractor', 'fit_features', 'encode_features', 'decode_features',
    'FeatureAnalyzer', 'create_feature_atlas'
]