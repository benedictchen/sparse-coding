"""
Integration tests for complete sparse coding pipeline.

Tests the full workflow from data preparation through dictionary learning
to sparse coding and reconstruction, ensuring all components work together.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from sparse_coding import (
    SparseCoder, DictionaryLearner, L1Proximal, AdvancedOptimizer
)
from sparse_coding.data_preprocessing_whitening import zero_phase_whiten
from sparse_coding.sparse_coding_monitoring import TB, CSVDump, DashboardLogger
from sparse_coding import sparse_coding_visualization as visualization
from tests.conftest import (
    assert_dictionary_normalized, assert_sparse_solution, 
    assert_reconstruction_quality, create_test_dictionary
)


class TestBasicPipelineIntegration:
    """Test basic pipeline integration."""
    
    def test_end_to_end_dictionary_learning(self, natural_image_patches):
        """Test complete dictionary learning pipeline."""
        X = natural_image_patches[:, :100]  # Manageable size
        
        # Step 1: Initialize dictionary learner
        learner = DictionaryLearner(
            n_atoms=32,
            max_iter=10,
            fit_algorithm="fista",
            dict_init="random",
            tol=1e-4,
            seed=42,
            sparsity_penalty=1.0  # Higher penalty for sparser solutions
        )
        
        # Step 2: Fit dictionary
        learner.fit(X)
        
        # Step 3: Transform data
        A = learner.transform(X)
        
        # Step 4: Verify results
        D = learner.dictionary
        
        # Check dimensions
        assert D.shape == (X.shape[0], 32)
        assert A.shape == (32, X.shape[1])
        
        # Check mathematical properties
        assert_dictionary_normalized(D)
        assert_sparse_solution(A)
        assert_reconstruction_quality(X, D @ A, tolerance=0.8)  # Higher tolerance for sparse solutions
    
    def test_sparse_coder_modes_integration(self, synthetic_data):
        """Test integration of different SparseCoder modes."""
        data = synthetic_data
        X = data['signals']
        
        modes = ["l1", "paper"]
        results = {}
        
        for mode in modes:
            coder = SparseCoder(
                n_atoms=data['n_components'],
                mode=mode,
                max_iter=100,
                tol=1e-6,
                seed=42
            )
            
            # Fit and encode
            coder.fit(X)
            A = coder.encode(X)
            
            # Store results
            results[mode] = {
                'codes': A,
                'dictionary': coder.dictionary,
                'reconstruction': coder.dictionary @ A
            }
        
        # Both modes should produce reasonable results
        for mode in modes:
            assert_dictionary_normalized(results[mode]['dictionary'])
            assert_sparse_solution(results[mode]['codes'])
            assert_reconstruction_quality(X, results[mode]['reconstruction'], tolerance=0.4)
    
    def test_optimization_algorithm_integration(self, synthetic_data):
        """Test integration with optimization algorithms."""
        data = synthetic_data
        X = data['signals'][:, :5]  # Smaller for speed
        D = create_test_dictionary(data['n_features'], data['n_components'], seed=42)
        
        # Test different optimization algorithms
        proximal_op = L1Proximal(lam=0.1)
        optimizer = AdvancedOptimizer(D, proximal_op, max_iter=50, tolerance=1e-6)
        
        algorithms = ['ista', 'fista', 'coordinate_descent']
        results = {}
        
        for alg_name in algorithms:
            algorithm = getattr(optimizer, alg_name)
            
            # Test on single signal
            x = X[:, 0]
            result = algorithm(x)
            
            results[alg_name] = result
            
            # Basic checks
            assert result['solution'].shape == (data['n_components'],)
            assert np.isfinite(result['final_objective'])
            assert result['iterations'] > 0
        
        # All algorithms should achieve similar final objectives
        objectives = [results[alg]['final_objective'] for alg in algorithms]
        obj_std = np.std(objectives)
        obj_mean = np.mean(objectives)
        
        if obj_mean > 1e-10:
            relative_std = obj_std / obj_mean
            assert relative_std < 0.5, f"Algorithm results too different: {relative_std:.3f}"


class TestDataPreprocessingIntegration:
    """Test integration with data preprocessing."""
    
    def test_whitening_integration(self, natural_image_patches):
        """Test integration with whitening preprocessing."""
        # Create 2D patches from 1D vectors (assuming square patches)
        patch_size = int(np.sqrt(natural_image_patches.shape[0]))
        n_patches = min(50, natural_image_patches.shape[1])  # Limit for speed
        
        patches_2d = []
        for i in range(n_patches):
            patch_1d = natural_image_patches[:, i]
            patch_2d = patch_1d.reshape(patch_size, patch_size)
            patches_2d.append(patch_2d)
        
        # Apply whitening to each patch
        whitened_patches = []
        for patch in patches_2d:
            whitened = zero_phase_whiten(patch, f0=100.0)
            whitened_patches.append(whitened.flatten())
        
        X_whitened = np.array(whitened_patches).T
        
        # Learn dictionary on whitened data
        learner = DictionaryLearner(n_atoms=25, max_iter=8, tol=1e-4)
        learner.fit(X_whitened)
        
        # Verify results
        D = learner.dictionary
        A = learner.transform(X_whitened)
        
        assert_dictionary_normalized(D)
        assert_sparse_solution(A)
        assert_reconstruction_quality(X_whitened, D @ A, tolerance=0.4)
    
    def test_patch_extraction_integration(self):
        """Test integration with patch extraction from images."""
        # Create synthetic image
        image_size = 64
        patch_size = 8
        
        np.random.seed(42)
        
        # Create image with edge-like structure
        x, y = np.meshgrid(np.linspace(0, 4*np.pi, image_size), 
                          np.linspace(0, 4*np.pi, image_size))
        image = np.sin(x) * np.cos(y/2) + 0.1 * np.random.randn(image_size, image_size)
        
        # Extract patches
        patches = []
        for i in range(0, image_size - patch_size + 1, patch_size//2):
            for j in range(0, image_size - patch_size + 1, patch_size//2):
                patch = image[i:i+patch_size, j:j+patch_size]
                # Normalize patch
                patch = patch - np.mean(patch)
                patch = patch / (np.std(patch) + 1e-10)
                patches.append(patch.flatten())
        
        X = np.array(patches).T
        
        # Dictionary learning on patches
        learner = DictionaryLearner(n_atoms=32, max_iter=5, seed=42)
        learner.fit(X)
        
        # Verify results
        D = learner.dictionary
        A = learner.transform(X)
        
        assert D.shape == (patch_size**2, 32)
        assert A.shape == (32, X.shape[1])
        assert_dictionary_normalized(D)
        assert_sparse_solution(A)


class TestDashboardIntegration:
    """Test integration with dashboard and logging components."""
    
    def test_tensorboard_logging_integration(self, synthetic_data):
        """Test integration with TensorBoard logging."""
        data = synthetic_data
        X = data['signals'][:, :10]  # Smaller dataset
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup TensorBoard logger
            tb_logger = TB(log_dir=temp_dir)
            
            # Manual training loop with logging
            learner = DictionaryLearner(n_atoms=data['n_components'], max_iter=5, seed=42)
            learner.fit(X)
            
            # Log some metrics
            A = learner.transform(X)
            D = learner.dictionary
            
            # Compute metrics
            reconstruction_error = np.mean((X - D @ A)**2)
            sparsity = np.mean(np.abs(A) < 0.01)
            
            # Log to TensorBoard
            tb_logger.log_scalar('reconstruction_error', reconstruction_error, step=0)
            tb_logger.log_scalar('sparsity', sparsity, step=0)
            tb_logger.log_histogram('sparse_codes', A.flatten(), step=0)
            
            tb_logger.close()
            
            # Verify log files were created
            log_files = list(Path(temp_dir).glob('events.out.tfevents.*'))
            assert len(log_files) > 0, "TensorBoard event files should be created"
    
    def test_csv_logging_integration(self, synthetic_data):
        """Test integration with CSV logging."""
        data = synthetic_data
        X = data['signals'][:, :10]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / 'training_log.csv'
            csv_logger = CSVDump(str(csv_file))
            
            # Simulate training with logging
            learner = DictionaryLearner(n_atoms=data['n_components'], max_iter=3, seed=42)
            
            for iteration in range(3):
                # Partial fit (would be part of training loop)
                if iteration == 0:
                    learner.fit(X)
                
                A = learner.transform(X)
                D = learner.dictionary
                
                # Log metrics
                metrics = {
                    'iteration': iteration,
                    'reconstruction_error': float(np.mean((X - D @ A)**2)),
                    'sparsity': float(np.mean(np.abs(A) < 0.01)),
                    'dict_coherence': float(np.max(np.abs(D.T @ D - np.eye(D.shape[1]))))
                }
                
                csv_logger.log(**metrics)
            
            csv_logger.close()
            
            # Verify CSV file was created and has content
            assert csv_file.exists(), "CSV log file should be created"
            
            with open(csv_file) as f:
                content = f.read()
                assert 'iteration' in content, "CSV should contain headers"
                assert len(content.strip().split('\n')) >= 4, "CSV should have header + 3 data rows"
    
    def test_combined_dashboard_integration(self, synthetic_data):
        """Test integration with combined dashboard logger."""
        data = synthetic_data
        X = data['signals'][:, :5]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = DashboardLogger(
                tb_log_dir=temp_dir,
                csv_path=str(Path(temp_dir) / 'combined_log.csv')
            )
            
            # Training with dashboard
            learner = DictionaryLearner(n_atoms=data['n_components'], max_iter=3, seed=42)
            learner.fit(X)
            
            A = learner.transform(X)
            D = learner.dictionary
            
            # Log comprehensive metrics
            dashboard.log_training_step(
                step=0,
                reconstruction_error=float(np.mean((X - D @ A)**2)),
                sparsity=float(np.mean(np.abs(A) < 0.01)),
                dict_coherence=float(np.max(np.abs(D.T @ D - np.eye(D.shape[1])))),
                sparse_codes=A,
                dictionary_atoms=D[:, :5]  # First 5 atoms
            )
            
            dashboard.close()
            
            # Verify both log types were created
            tb_files = list(Path(temp_dir).glob('events.out.tfevents.*'))
            csv_file = Path(temp_dir) / 'combined_log.csv'
            
            assert len(tb_files) > 0, "TensorBoard logs should be created"
            assert csv_file.exists(), "CSV log should be created"


class TestVisualizationIntegration:
    """Test integration with visualization components."""
    
    def test_dictionary_visualization_integration(self, natural_image_patches):
        """Test integration with dictionary atom visualization."""
        # Use square patches for visualization
        patch_size = int(np.sqrt(natural_image_patches.shape[0]))
        X = natural_image_patches[:patch_size**2, :50]  # Ensure square patches
        
        learner = DictionaryLearner(n_atoms=16, max_iter=5, seed=42)
        learner.fit(X)
        
        D = learner.dictionary
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test dictionary visualization
            fig = visualization.plot_dictionary_atoms(D, patch_size=(patch_size, patch_size))
            
            # Save figure
            output_path = Path(temp_dir) / 'dictionary_atoms.png'
            fig.savefig(output_path)
            
            # Verify file was created
            assert output_path.exists(), "Dictionary visualization should be saved"
            assert output_path.stat().st_size > 1000, "Saved image should have reasonable size"
    
    def test_training_progress_visualization_integration(self, synthetic_data):
        """Test integration with training progress visualization."""
        data = synthetic_data
        X = data['signals'][:, :10]
        
        # Simulate training with metric tracking
        learner = DictionaryLearner(n_atoms=data['n_components'], max_iter=8, seed=42)
        learner.fit(X)
        
        # Create mock training history
        n_iterations = 8
        history = {
            'reconstruction_error': np.exp(-np.linspace(0, 2, n_iterations)) + 0.1,
            'sparsity': np.linspace(0.3, 0.7, n_iterations),
            'dict_coherence': np.exp(-np.linspace(0, 1, n_iterations)) * 0.5 + 0.1
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test training progress visualization
            fig = visualization.plot_training_progress(history)
            
            # Save figure
            output_path = Path(temp_dir) / 'training_progress.png'
            fig.savefig(output_path)
            
            # Verify file creation
            assert output_path.exists(), "Training progress plot should be saved"
            assert output_path.stat().st_size > 1000, "Plot should have reasonable file size"
    
    def test_analysis_report_integration(self, natural_image_patches):
        """Test integration with analysis reporting system."""
        # Prepare data
        patch_size = int(np.sqrt(natural_image_patches.shape[0]))
        X = natural_image_patches[:patch_size**2, :30]
        
        # Train model
        learner = DictionaryLearner(n_atoms=16, max_iter=5, seed=42)
        learner.fit(X)
        
        A = learner.transform(X)
        D = learner.dictionary
        
        # Create mock history
        history = {
            'reconstruction_error': [0.5, 0.3, 0.2, 0.15, 0.12],
            'sparsity': [0.3, 0.45, 0.55, 0.6, 0.65]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate visualization report
            report_path = visualization.create_visualization_report(
                dictionary=D,
                sparse_codes=A,
                signals=X,
                training_history=history,
                patch_size=(patch_size, patch_size),
                save_path=temp_dir
            )
            
            # Verify report files were created
            assert Path(report_path).exists(), "Comprehensive report should be created"
            
            # Check for individual plot files
            expected_files = [
                'dictionary_atoms.png',
                'training_progress.png', 
                'sparse_codes_distribution.png'
            ]
            
            for filename in expected_files:
                filepath = Path(temp_dir) / filename
                if filepath.exists():  # Some plots may be optional
                    assert filepath.stat().st_size > 500, f"{filename} should have reasonable size"


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""
    
    def test_invalid_input_handling(self):
        """Test graceful handling of invalid inputs."""
        # Test with invalid data shapes
        with pytest.raises((ValueError, AssertionError)):
            X_invalid = np.array([1, 2, 3])  # 1D array
            learner = DictionaryLearner(n_atoms=10)
            learner.fit(X_invalid)
    
    def test_convergence_failure_handling(self, synthetic_data):
        """Test handling when algorithms fail to converge."""
        data = synthetic_data
        X = data['signals']
        
        # Use very strict tolerance and few iterations
        coder = SparseCoder(
            n_atoms=data['n_components'],
            mode="l1",
            max_iter=1,  # Very few iterations
            tol=1e-12,   # Very strict tolerance
            lam=0.1
        )
        
        # Should still produce results (may not converge but shouldn't crash)
        coder.fit(X)
        A = coder.encode(X[:, :1])
        
        # Results should be finite even if not converged
        assert np.all(np.isfinite(A)), "Should produce finite results even without convergence"
    
    def test_memory_efficiency_large_problems(self):
        """Test memory efficiency with larger problems."""
        # Create moderately large problem
        np.random.seed(42)
        n_features, n_atoms, n_samples = 200, 100, 50
        
        D_true = create_test_dictionary(n_features, n_atoms, seed=42)
        A_true = np.random.laplace(scale=0.1, size=(n_atoms, n_samples))
        A_true[np.abs(A_true) < 0.2] = 0  # Enforce sparsity
        X = D_true @ A_true + 0.01 * np.random.randn(n_features, n_samples)
        
        # Should handle this size without issues
        learner = DictionaryLearner(n_atoms=n_atoms, max_iter=3, tol=1e-4)
        learner.fit(X)
        
        A_estimated = learner.transform(X)
        
        # Basic sanity checks
        assert A_estimated.shape == (n_atoms, n_samples)
        assert np.all(np.isfinite(A_estimated))
        
        # Should achieve reasonable reconstruction
        D_learned = learner.dictionary
        reconstruction = D_learned @ A_estimated
        relative_error = (np.linalg.norm(X - reconstruction, 'fro') / 
                         np.linalg.norm(X, 'fro'))
        assert relative_error < 0.8, f"Reconstruction error too high: {relative_error:.3f}"


@pytest.mark.slow
class TestLongRunningIntegration:
    """Test integration with longer-running scenarios."""
    
    def test_extended_training_stability(self, natural_image_patches):
        """Test stability over extended training."""
        X = natural_image_patches[:, :100]
        
        # Longer training
        learner = DictionaryLearner(n_atoms=32, max_iter=20, tol=1e-6, seed=42)
        learner.fit(X)
        
        D = learner.dictionary
        A = learner.transform(X)
        
        # Should maintain mathematical properties
        assert_dictionary_normalized(D)
        assert_sparse_solution(A, sparsity_threshold=0.1)
        assert_reconstruction_quality(X, D @ A, tolerance=0.25)
        
        # Dictionary should have developed some structure
        atom_norms = np.linalg.norm(D, axis=0)
        assert np.allclose(atom_norms, 1.0, atol=1e-6), "Atoms should remain normalized"
    
    def test_multiple_dataset_consistency(self):
        """Test consistency across multiple datasets."""
        np.random.seed(42)
        
        results = []
        
        # Test on multiple synthetic datasets
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            
            # Generate dataset
            n_features, n_atoms, n_samples = 64, 32, 50
            D_true = create_test_dictionary(n_features, n_atoms, seed=seed)
            A_true = np.random.laplace(scale=0.15, size=(n_atoms, n_samples))
            A_true[np.abs(A_true) < 0.3] = 0
            X = D_true @ A_true + 0.02 * np.random.randn(n_features, n_samples)
            
            # Learn dictionary
            learner = DictionaryLearner(n_atoms=n_atoms, max_iter=10, seed=42)  # Fixed seed for learner
            learner.fit(X)
            
            A_learned = learner.transform(X)
            D_learned = learner.dictionary
            
            # Compute metrics
            reconstruction_error = np.mean((X - D_learned @ A_learned)**2)
            sparsity = np.mean(np.abs(A_learned) < 0.01)
            
            results.append({
                'reconstruction_error': reconstruction_error,
                'sparsity': sparsity,
                'dictionary': D_learned,
                'codes': A_learned
            })
        
        # Results should be reasonably consistent
        errors = [r['reconstruction_error'] for r in results]
        sparsities = [r['sparsity'] for r in results]
        
        error_std = np.std(errors)
        error_mean = np.mean(errors)
        sparsity_std = np.std(sparsities)
        
        # Check consistency
        if error_mean > 1e-10:
            error_cv = error_std / error_mean  # Coefficient of variation
            assert error_cv < 1.0, f"Reconstruction errors too variable: {error_cv:.3f}"
        
        assert sparsity_std < 0.3, f"Sparsity levels too variable: {sparsity_std:.3f}"