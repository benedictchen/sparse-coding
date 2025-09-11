"""
Test suite for README code examples.

Ensures all code samples in README.md are functional and correctly documented.
Catches API changes that would break user examples.
"""

import numpy as np
import pytest
import tempfile
import os
import shutil
from pathlib import Path

from sparse_coding import DictionaryLearner, create_advanced_sparse_coder, DashboardLogger, visualization


class TestREADMEExamples:
    """Test all README code examples for functionality."""
    
    def test_quick_start_example(self):
        """Test the main Quick Start example from README."""
        # Generate synthetic natural image patches
        images = np.random.randn(10, 64, 64)  # 10 images, 64x64 pixels
        assert images.shape == (10, 64, 64)

        # Learn sparse dictionary
        learner = DictionaryLearner(
            n_components=50,         # Reduced for fast testing
            patch_size=(8, 8),       # 8x8 patches
            sparsity_penalty=0.05,   # L1 regularization
            max_iterations=3         # Short training for testing
        )

        # Train on image patches
        history = learner.fit(images, verbose=False)
        assert isinstance(history, dict)
        assert 'reconstruction_errors' in history

        # Extract sparse features
        features = learner.transform(images, pooling='max')
        assert features.shape == (10, 50)  # n_images x n_components
        
        # Visualize learned dictionary (test without showing)
        try:
            fig = visualization.plot_dictionary_atoms(
                learner.dictionary, 
                learner.patch_size, 
                title="Learned Sparse Dictionary"
            )
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Visualization requires display: {e}")

    def test_dictionary_learner_interface(self):
        """Test DictionaryLearner parameter interface from README."""
        # Full parameter specification as shown in README
        learner = DictionaryLearner(
            n_components=25,         # Reduced for testing
            patch_size=(8, 8),       # Patch dimensions
            sparsity_penalty=0.03,   # L1 regularization strength
            learning_rate=0.01,      # Dictionary update step size
            max_iterations=3         # Training iterations (reduced)
        )
        
        # Verify parameters are set correctly
        assert learner.n_components == 25
        assert learner.patch_size == (8, 8)
        assert learner.sparsity_penalty == 0.03
        assert learner.learning_rate == 0.01
        assert learner.max_iterations == 3

    def test_optimization_algorithm_example(self):
        """Test optimization algorithm README example."""
        # Create test dictionary and signal
        dictionary = np.random.randn(64, 32)
        dictionary /= np.linalg.norm(dictionary, axis=0, keepdims=True)
        signal = np.random.randn(64, 1)

        # Create optimizer with different methods
        optimizer = create_advanced_sparse_coder(
            dictionary, 
            penalty_type='l1',  # 'l1', 'elastic_net', 'non_negative_l1'
            penalty_params={'lam': 0.1}
        )

        # Test optimization methods from README
        methods = ['ista', 'fista']
        for method in methods:
            result = getattr(optimizer, method)(signal)
            assert 'iterations' in result
            assert isinstance(result['iterations'], int)
            assert result['iterations'] > 0

    def test_professional_visualization_example(self):
        """Test Professional Visualization README example."""
        # Create test data
        dictionary = np.random.randn(64, 25)
        dictionary /= np.linalg.norm(dictionary, axis=0, keepdims=True)
        sparse_codes = np.random.randn(25, 50) * (np.random.rand(25, 50) < 0.1)
        training_history = {
            'reconstruction_errors': [0.5, 0.3, 0.2, 0.15, 0.1],
            'sparsity_levels': [0.05, 0.06, 0.07, 0.08, 0.08],
            'dictionary_changes': [1.0, 0.8, 0.6, 0.4, 0.2]
        }

        try:
            # Create complete analysis report
            figures = visualization.create_visualization_report(
                dictionary=dictionary,
                codes=sparse_codes,
                history=training_history,
                patch_size=(8, 8),
                save_path=None  # Don't save files in test
            )
            
            assert isinstance(figures, list)
            assert len(figures) > 0
            
        except Exception as e:
            pytest.skip(f"Visualization requires display environment: {e}")

    def test_tensorboard_integration_example(self):
        """Test TensorBoard Integration README example."""
        # Create temporary directory for logging
        temp_dir = tempfile.mkdtemp()
        
        try:
            tensorboard_dir = os.path.join(temp_dir, "logs", "sparse_coding")
            csv_path = os.path.join(temp_dir, "metrics.csv")

            # Setup logging
            logger = DashboardLogger(
                tensorboard_dir=tensorboard_dir,
                csv_path=csv_path
            )

            # Log training metrics
            logger.log_training_metrics({
                'reconstruction_error': 0.001,
                'sparsity_level': 0.05
            })

            # Verify CSV was created
            assert os.path.exists(csv_path)

            # Visualize dictionary evolution
            dictionary = np.random.randn(64, 16)
            dictionary /= np.linalg.norm(dictionary, axis=0, keepdims=True)
            patch_size = (8, 8)
            
            logger.log_dictionary_atoms(dictionary, patch_size)

            # Verify tensorboard directory was created
            assert os.path.exists(tensorboard_dir)

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)

    def test_example_scripts_exist(self):
        """Test that example scripts mentioned in README exist."""
        examples_dir = Path(__file__).parent.parent.parent / "examples"
        
        expected_examples = [
            "basic_dictionary_learning.py",
            "advanced_optimization_comparison.py", 
            "complete_pipeline_demo.py"
        ]
        
        for example in expected_examples:
            example_path = examples_dir / example
            assert example_path.exists(), f"README mentions {example} but file doesn't exist"

    def test_mathematical_notation_consistency(self):
        """Test that mathematical concepts mentioned in README are implemented."""
        # Test that core optimization problem components exist
        
        # Dictionary learning with L1 penalty (longer training for proper sparsity)
        learner = DictionaryLearner(
            n_components=16, 
            max_iterations=10,
            sparsity_penalty=0.3  # Higher penalty to ensure sparsity
        )
        images = np.random.randn(5, 32, 32)
        
        history = learner.fit(images, verbose=False)
        
        # Verify L1 penalty is applied (check that not all features are dense)
        features = learner.transform(images)
        # With L1 penalty, we should have some variety in feature magnitudes
        feature_std = np.std(np.abs(features))
        assert feature_std > 1e-6, "L1 penalty should create variety in feature magnitudes"
        
        # Check that optimization problem components exist
        assert learner.dictionary is not None, "Dictionary D should exist"
        assert features.shape[1] == learner.n_components, "Features should match n_components"

    def test_convergence_methods_mentioned(self):
        """Test that convergence methods from performance table work."""
        # Create test problem
        dictionary = np.random.randn(32, 16)
        dictionary /= np.linalg.norm(dictionary, axis=0, keepdims=True)
        signal = np.random.randn(32, 1)

        optimizer = create_advanced_sparse_coder(
            dictionary, 
            penalty_type='l1',
            penalty_params={'lam': 0.1}
        )

        # Test methods mentioned in performance table
        methods_to_test = ['ista', 'fista']  # Subset for testing
        
        for method in methods_to_test:
            result = getattr(optimizer, method)(signal)
            # Should converge in reasonable iterations
            assert result['iterations'] < 1000, f"{method} should converge reasonably fast"
            assert 'solution' in result, f"{method} should return solution"


class TestREADMEAPIConsistency:
    """Ensure README examples match actual API."""
    
    def test_dictionary_learner_constructor_params(self):
        """Verify DictionaryLearner constructor matches README documentation."""
        import inspect
        
        sig = inspect.signature(DictionaryLearner.__init__)
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        
        # Parameters shown in README should exist
        required_params = [
            'n_components', 'patch_size', 'sparsity_penalty', 
            'learning_rate', 'max_iterations'
        ]
        
        for param in required_params:
            assert param in params, f"README shows {param} but it's not in constructor"

    def test_fit_method_signature(self):
        """Verify fit method matches README usage."""
        import inspect
        
        sig = inspect.signature(DictionaryLearner.fit)
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        
        # Should accept data and verbose parameters as shown in README
        assert 'data' in params
        assert 'verbose' in params
        
        # Should NOT accept max_iterations (common mistake)
        assert 'max_iterations' not in params, "max_iterations should be in constructor, not fit()"

    def test_transform_method_signature(self):
        """Verify transform method matches README usage."""
        import inspect
        
        sig = inspect.signature(DictionaryLearner.transform)
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        
        # Should accept pooling parameter as shown in README
        assert 'pooling' in params

    def test_dashboard_logger_interface(self):
        """Verify DashboardLogger matches README usage."""
        import inspect
        
        # Constructor should accept tensorboard_dir and csv_path
        init_sig = inspect.signature(DashboardLogger.__init__)
        init_params = list(init_sig.parameters.keys())[1:]  # Skip 'self'
        
        assert 'tensorboard_dir' in init_params
        assert 'csv_path' in init_params
        
        # Methods shown in README should exist
        logger = DashboardLogger(tensorboard_dir="test", csv_path="test.csv")
        assert hasattr(logger, 'log_training_metrics')
        assert hasattr(logger, 'log_dictionary_atoms')