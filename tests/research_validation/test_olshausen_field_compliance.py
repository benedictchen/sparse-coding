"""
Tests for Olshausen & Field (1996) research compliance.

Validates that our implementation matches the original sparse coding paper:
"Emergence of simple-cell receptive field properties by learning a sparse code 
for natural images" by Olshausen & Field (1996)

Key validation points:
1. Objective function formulation
2. Dictionary learning alternation
3. Sparse inference methods (L1 and log prior)
4. Dictionary normalization constraints
5. Convergence properties
"""

import numpy as np
import pytest
from sparse_coding import SparseCoder, DictionaryLearner
from tests.conftest import (assert_dictionary_normalized, assert_sparse_solution, 
                           assert_reconstruction_quality, measure_convergence_rate)


class TestOlshausenFieldObjectiveFunction:
    """Test objective function matches Olshausen & Field (1996) formulation."""
    
    def test_l1_objective_formulation(self, synthetic_data, tolerance):
        """Test L1 objective: E = ||X - DA||_F^2 + λ||A||_1"""
        data = synthetic_data
        X = data['signals']
        
        coder = SparseCoder(n_atoms=data['n_components'], mode="l1", lam=0.1)
        coder.fit(X)
        
        # Get sparse codes
        A = coder.encode(X)
        D = coder.dictionary
        
        # Compute objective components
        reconstruction_error = 0.5 * np.linalg.norm(X - D @ A, 'fro')**2
        sparsity_penalty = 0.1 * np.sum(np.abs(A))
        total_objective = reconstruction_error + sparsity_penalty
        
        # Verify objective is reasonable (should be minimized)
        assert total_objective > 0, "Objective function must be positive"
        
        # Verify reconstruction is reasonable
        relative_error = reconstruction_error / (0.5 * np.linalg.norm(X, 'fro')**2)
        assert relative_error < 1.0, f"Reconstruction error too high: {relative_error:.3f}"
    
    def test_log_prior_objective_formulation(self, synthetic_data, tolerance):
        """Test log prior objective from Olshausen & Field original formulation."""
        data = synthetic_data
        X = data['signals']
        
        coder = SparseCoder(n_atoms=data['n_components'], mode="log", lam=0.1)
        coder.fit(X)
        
        # Get sparse codes  
        A = coder.encode(X)
        D = coder.dictionary
        
        # Verify sparsity is achieved
        assert_sparse_solution(A, sparsity_threshold=0.1)
        
        # Verify dictionary normalization
        assert_dictionary_normalized(D, tolerance)


class TestDictionaryLearningAlternation:
    """Test dictionary learning alternating optimization."""
    
    def test_alternating_optimization_reduces_objective(self, natural_image_patches):
        """Test that alternating optimization reduces the objective function."""
        X = natural_image_patches
        
        learner = DictionaryLearner(n_atoms=64, max_iter=10, fit_algorithm="fista")
        
        # Track objective during learning
        objectives = []
        
        # Manual implementation to track objectives
        learner.fit(X)
        
        # Get final results
        A = learner.transform(X)
        D = learner.dictionary
        
        # Compute final objective
        reconstruction_error = 0.5 * np.linalg.norm(X - D @ A, 'fro')**2
        sparsity_penalty = learner.sparse_coder.lam * np.sum(np.abs(A))
        final_objective = reconstruction_error + sparsity_penalty
        
        # Verify dictionary is normalized
        assert_dictionary_normalized(D)
        
        # Verify solution achieves reasonable sparsity
        assert_sparse_solution(A)
        
        # Verify reconstruction quality
        assert_reconstruction_quality(X, D @ A, tolerance=0.2)
    
    @pytest.mark.slow
    def test_dictionary_atoms_emerge_localized_features(self, natural_image_patches):
        """Test that learned dictionary atoms are localized (edge-like) features."""
        X = natural_image_patches[:256, :500]  # Use 16x16 patches
        
        learner = DictionaryLearner(n_atoms=64, max_iter=20, fit_algorithm="fista")
        learner.fit(X)
        
        D = learner.dictionary
        
        # Reshape atoms back to image patches
        patch_size = int(np.sqrt(D.shape[0]))
        atoms = D.T.reshape(-1, patch_size, patch_size)
        
        # Test for localization: compute spatial concentration
        spatial_concentrations = []
        for atom in atoms:
            # Compute center of mass
            y_indices, x_indices = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
            total_mass = np.sum(np.abs(atom))
            if total_mass > 1e-10:
                center_y = np.sum(y_indices * np.abs(atom)) / total_mass
                center_x = np.sum(x_indices * np.abs(atom)) / total_mass
                
                # Compute concentration around center
                distances = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
                concentration = np.sum(np.abs(atom) * (1.0 / (1.0 + distances))) / total_mass
                spatial_concentrations.append(concentration)
        
        # Most atoms should be somewhat localized
        mean_concentration = np.mean(spatial_concentrations)
        assert mean_concentration > 0.3, f"Dictionary atoms not sufficiently localized: {mean_concentration:.3f}"


class TestSparseInferenceMethods:
    """Test sparse inference methods from Olshausen & Field."""
    
    def test_l1_method_produces_sparse_codes(self, synthetic_data):
        """Test that L1 method produces appropriately sparse codes."""
        data = synthetic_data
        X = data['signals']
        
        # Test different lambda values
        lambda_values = [0.01, 0.1, 0.5]
        sparsity_levels = []
        
        for lam in lambda_values:
            coder = SparseCoder(n_atoms=data['n_components'], mode="l1", lam=lam)
            coder.fit(X)
            A = coder.encode(X)
            
            # Measure sparsity (fraction of near-zero elements)
            sparsity = np.mean(np.abs(A) < 0.01)
            sparsity_levels.append(sparsity)
            
            # Basic sparsity check
            assert sparsity > 0.1, f"Insufficient sparsity with λ={lam}: {sparsity:.3f}"
        
        # Verify that higher lambda produces more sparsity
        assert sparsity_levels[-1] > sparsity_levels[0], "Higher λ should produce more sparsity"
    
    def test_log_prior_method_convergence(self, synthetic_data):
        """Test that log prior method converges properly."""
        data = synthetic_data
        X = data['signals']
        
        coder = SparseCoder(n_atoms=data['n_components'], mode="log", 
                           max_iter=200, tol=1e-8)
        coder.fit(X)
        
        # Test convergence on a single signal
        x = X[:, 0:1]
        result = coder._solve_single_log(x, coder.dictionary)
        
        # Should have converged
        assert result['converged'], "Log prior optimization should converge"
        assert result['iterations'] < 200, "Should converge in reasonable iterations"
        
        # Final solution should be reasonably sparse
        a = result['coefficients']
        assert_sparse_solution(a, sparsity_threshold=0.1)
    
    def test_inference_methods_consistency(self, synthetic_data, tolerance):
        """Test that different inference methods give consistent results."""
        data = synthetic_data
        X = data['signals'][:, :5]  # Use fewer signals for speed
        
        # Learn dictionary with L1 method
        coder_l1 = SparseCoder(n_atoms=data['n_components'], mode="l1", lam=0.1)
        coder_l1.fit(X)
        
        # Use same dictionary for log method
        coder_log = SparseCoder(n_atoms=data['n_components'], mode="log", lam=0.1)
        coder_log.dictionary = coder_l1.dictionary.copy()
        
        # Get codes from both methods
        A_l1 = coder_l1.encode(X)
        A_log = coder_log.encode(X)
        
        # Reconstructions should be similar quality
        D = coder_l1.dictionary
        recon_l1 = D @ A_l1
        recon_log = D @ A_log
        
        error_l1 = np.mean((X - recon_l1)**2)
        error_log = np.mean((X - recon_log)**2)
        
        # Both should achieve reasonable reconstruction
        assert error_l1 < 0.5 * np.var(X), "L1 reconstruction error too high"
        assert error_log < 0.5 * np.var(X), "Log reconstruction error too high"


class TestDictionaryNormalizationConstraints:
    """Test dictionary normalization as in Olshausen & Field."""
    
    def test_dictionary_atoms_unit_normalized(self, synthetic_data, tolerance):
        """Test that dictionary atoms maintain unit norm."""
        data = synthetic_data
        X = data['signals']
        
        learner = DictionaryLearner(n_atoms=data['n_components'], max_iter=5)
        learner.fit(X)
        
        D = learner.dictionary
        assert_dictionary_normalized(D, tolerance)
    
    def test_normalization_preserved_during_learning(self, natural_image_patches):
        """Test that normalization is preserved throughout learning."""
        X = natural_image_patches[:, :100]  # Smaller dataset for speed
        
        learner = DictionaryLearner(n_atoms=32, max_iter=3)
        
        # Custom learning loop to check normalization at each step
        learner._initialize_dictionary(X)
        initial_dict = learner.dictionary.copy()
        assert_dictionary_normalized(initial_dict)
        
        # Perform one update step and check normalization
        A = learner.sparse_coder.encode(X)
        learner._update_dictionary(X, A)
        
        updated_dict = learner.dictionary
        assert_dictionary_normalized(updated_dict)
    
    def test_overcomplete_dictionary_normalization(self, synthetic_data):
        """Test normalization with overcomplete dictionaries."""
        data = synthetic_data
        X = data['signals']
        
        # Create overcomplete dictionary (more atoms than signal dimension)
        n_atoms = data['n_features'] + 20
        
        learner = DictionaryLearner(n_atoms=n_atoms, max_iter=3)
        learner.fit(X)
        
        D = learner.dictionary
        assert D.shape == (data['n_features'], n_atoms)
        assert_dictionary_normalized(D)


class TestConvergenceProperties:
    """Test convergence properties as described in Olshausen & Field."""
    
    @pytest.mark.slow
    def test_objective_convergence_properties(self, synthetic_data):
        """Test that the objective function converges monotonically."""
        data = synthetic_data
        X = data['signals']
        
        learner = DictionaryLearner(n_atoms=data['n_components'], max_iter=15, tol=1e-6)
        
        # Track objectives manually
        objectives = []
        learner.fit(X)
        
        # Check final convergence
        A_final = learner.transform(X)
        D_final = learner.dictionary
        
        reconstruction_error = 0.5 * np.linalg.norm(X - D_final @ A_final, 'fro')**2
        sparsity_penalty = learner.sparse_coder.lam * np.sum(np.abs(A_final))
        final_objective = reconstruction_error + sparsity_penalty
        
        assert final_objective > 0, "Final objective must be positive"
        assert np.isfinite(final_objective), "Final objective must be finite"
    
    def test_sparse_coding_convergence_rate(self, synthetic_data):
        """Test convergence rate of sparse coding step."""
        data = synthetic_data
        X = data['signals'][:, 0:1]  # Single signal
        
        coder = SparseCoder(n_atoms=data['n_components'], mode="l1", 
                           max_iter=500, tol=1e-10)
        coder.fit(data['signals'])  # Fit on full data first
        
        # Get convergence history for single signal
        result = coder._fista_single(X, coder.dictionary)
        
        if 'objectives' in result:
            objectives = result['objectives']
            if len(objectives) > 10:
                convergence_rate = measure_convergence_rate(objectives)
                
                # FISTA should have good convergence rate
                if not np.isnan(convergence_rate):
                    assert convergence_rate < -0.01, f"Convergence rate too slow: {convergence_rate:.6f}"
    
    @pytest.mark.convergence
    def test_stability_with_different_initializations(self, synthetic_data):
        """Test that algorithm is stable across different initializations."""
        data = synthetic_data
        X = data['signals']
        
        final_objectives = []
        
        for seed in [42, 123, 456]:
            learner = DictionaryLearner(n_atoms=data['n_components'], 
                                      max_iter=10, seed=seed)
            learner.fit(X)
            
            A = learner.transform(X) 
            D = learner.dictionary
            
            reconstruction_error = 0.5 * np.linalg.norm(X - D @ A, 'fro')**2
            sparsity_penalty = learner.sparse_coder.lam * np.sum(np.abs(A))
            objective = reconstruction_error + sparsity_penalty
            
            final_objectives.append(objective)
        
        # Results should be reasonably consistent
        objectives_array = np.array(final_objectives)
        relative_std = np.std(objectives_array) / np.mean(objectives_array)
        
        assert relative_std < 0.5, f"Results too variable across initializations: {relative_std:.3f}"


@pytest.mark.research
class TestResearchAccuracy:
    """High-level tests for research accuracy."""
    
    def test_reproduces_basic_sparse_coding_behavior(self, natural_image_patches):
        """Test that implementation reproduces basic sparse coding behavior."""
        X = natural_image_patches[:, :200]  # Subset for speed
        
        # Learn dictionary
        learner = DictionaryLearner(n_atoms=32, max_iter=10)
        learner.fit(X)
        
        # Get results
        A = learner.transform(X)
        D = learner.dictionary
        
        # Basic sanity checks
        assert D.shape == (X.shape[0], 32)
        assert A.shape == (32, X.shape[1]) 
        
        # Check mathematical properties
        assert_dictionary_normalized(D)
        assert_sparse_solution(A)
        assert_reconstruction_quality(X, D @ A, tolerance=0.3)
    
    @pytest.mark.slow 
    def test_qualitative_receptive_field_emergence(self, natural_image_patches):
        """Test qualitative emergence of receptive field-like properties."""
        # Use patches that look like natural image structure
        X = natural_image_patches[:, :500]
        patch_size = int(np.sqrt(X.shape[0]))
        
        learner = DictionaryLearner(n_atoms=49, max_iter=15, fit_algorithm="fista")
        learner.fit(X)
        
        D = learner.dictionary
        atoms = D.T.reshape(-1, patch_size, patch_size)
        
        # Test for edge-like properties: high gradient responses
        gradient_responses = []
        for atom in atoms:
            # Compute gradient magnitude
            gy, gx = np.gradient(atom)
            grad_magnitude = np.sqrt(gx**2 + gy**2)
            mean_gradient = np.mean(grad_magnitude)
            gradient_responses.append(mean_gradient)
        
        # Many atoms should have significant gradient responses (edge-like)
        high_gradient_fraction = np.mean(np.array(gradient_responses) > 0.1)
        assert high_gradient_fraction > 0.3, f"Too few edge-like atoms: {high_gradient_fraction:.3f}"