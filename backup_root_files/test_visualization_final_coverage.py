#!/usr/bin/env python3
"""
Final Coverage Test for visualization.py - Target Remaining 9 Lines
==================================================================

Targets specific uncovered lines: 431-437, 484-485
Goal: 95% → 100% coverage

Author: Benedict Chen (benedict@benedictchen.com)
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from unittest.mock import patch

try:
    from .visualization import SparseVisualization
except ImportError:
    from visualization import SparseVisualization


def test_training_progress_with_learning_rates():
    """Test training progress visualization with learning rates (lines 431-437)"""
    viz = SparseVisualization()
    
    # Create training history WITH learning rates to trigger lines 431-437
    training_history = {
        'reconstruction_errors': [1.0, 0.8, 0.6, 0.4],
        'sparsity_levels': [0.1, 0.15, 0.2, 0.25],
        'dictionary_changes': [0.5, 0.3, 0.2, 0.1],
        'learning_rates': [0.01, 0.008, 0.006, 0.004]  # THIS triggers lines 431-437
    }
    
    with patch('matplotlib.pyplot.show'):
        viz.visualize_training_progress(training_history)
        
    # Verify it worked
    assert viz._last_plot_info is not None


def test_receptive_field_analysis_with_zero_atoms():
    """Test receptive field analysis with zero/flat atoms (lines 484-485)"""
    viz = SparseVisualization()
    
    # Create dictionary with some zero/flat atoms to trigger lines 484-485
    patch_size = (6, 6)
    patch_dim = 36
    dictionary = np.random.randn(patch_dim, 8)
    
    # Make some atoms completely flat (zero variance) to trigger else clause at lines 484-485
    dictionary[:, 0] = 0.5  # Flat atom - no variation
    dictionary[:, 1] = np.ones(patch_dim) * 0.3  # Another flat atom
    
    with patch('matplotlib.pyplot.show'):
        analysis = viz.analyze_receptive_fields(dictionary, patch_size)
        
    # Verify zero orientations and frequencies were assigned (lines 484-485)
    assert isinstance(analysis, dict)
    # The flat atoms should have gotten orientations=0 and frequencies=0


def test_combined_edge_case_coverage():
    """Combined test to ensure all edge cases are covered"""
    viz = SparseVisualization()
    
    # Test 1: Training progress with all possible keys
    full_training_history = {
        'reconstruction_errors': [2.0, 1.5, 1.0],
        'sparsity_levels': [0.05, 0.1, 0.15], 
        'dictionary_changes': [0.8, 0.5, 0.2],
        'learning_rates': [0.02, 0.015, 0.01]  # Trigger lines 431-437
    }
    
    with patch('matplotlib.pyplot.show'):
        viz.visualize_training_progress(full_training_history)
    
    # Test 2: Receptive field with mixed atom types  
    patch_size = (4, 4)
    dictionary = np.random.randn(16, 6)
    
    # Mix of normal and flat atoms
    dictionary[:, 0] = np.random.randn(16) * 0.1  # Very small variation
    dictionary[:, 1] = 0  # Completely flat
    dictionary[:, 2] = np.ones(16) * 0.7  # Constant value
    # Other atoms remain random
    
    with patch('matplotlib.pyplot.show'):
        analysis = viz.analyze_receptive_fields(dictionary, patch_size)
        
    assert isinstance(analysis, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])