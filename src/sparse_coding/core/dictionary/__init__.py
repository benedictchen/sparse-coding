"""
Dictionary learning algorithms for sparse coding.

This module implements dictionary update methods following Olshausen & Field (1996)
and subsequent advances in dictionary learning.
"""

from .method_optimal_directions import MethodOptimalDirections
from .ksvd_dictionary_learning import KSVDDictionaryLearning  
from .gradient_descent_update import GradientDescentUpdate
from .online_dictionary_learning import OnlineDictionaryLearning

__all__ = [
    'MethodOptimalDirections',
    'KSVDDictionaryLearning',
    'GradientDescentUpdate', 
    'OnlineDictionaryLearning'
]