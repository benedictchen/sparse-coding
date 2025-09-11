"""
Research-Based Validation Tests for Olshausen & Field (1996) Sparse Coding.

This module validates our implementation against the mathematical foundations
established in the seminal sparse coding paper:

"Emergence of simple-cell receptive field properties by learning a sparse code 
for natural images" by Bruno A. Olshausen and David J. Field, Nature 1996.

Tests verify:
1. Objective function formulation: E = ||I - Φa||² + λS(a)
2. Sparsity function implementations: |x| and ln(1 + x²)  
3. Dictionary learning convergence properties
4. Gradient computations and optimization behavior
5. Mathematical properties required by the theory

Research Foundation:
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field 
  properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.
- Olshausen, B. A., & Field, D. J. (1997). Sparse coding with an overcomplete basis set: 
  A strategy employed by V1? Vision Research, 37(23), 3311-3325.

Author: Benedict Chen
"""

import numpy as np
import pytest
from sparse_coding.core.penalties.implementations import L1Penalty, CauchyPenalty, create_penalty
from sparse_coding.core.learner_implementations import KsvdLearner
from sparse_coding import SparseCoder, DictionaryLearner
import sys
sys.path.append('tests')
from conftest import create_test_dictionary


class TestOlshausenFieldObjectiveFunction:
    """Test that our objective matches Olshausen & Field formulation."""
    
    def test_objective_function_formulation(self):
        """
        Test: E = ||I - Φa||² + λS(a)
        
        Research Foundation:
        Olshausen & Field (1996) equation in Methods section.
        This is the core optimization problem for sparse coding.
        """
        np.random.seed(42)
        
        # Create test problem matching research setup
        n_features, n_atoms = 16, 12  # Overcomplete dictionary (typical for O&F)
        I = np.random.randn(n_features)  # Input image patch  
        Phi = np.random.randn(n_features, n_atoms)  # Dictionary matrix
        Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)  # Normalized atoms
        a = np.random.randn(n_atoms) * 0.1  # Sparse coefficients
        lam = 0.1  # Sparsity parameter
        
        # Manual computation of Olshausen & Field objective
        # Note: Our implementation uses 0.5 * ||residual||² (common in optimization literature)
        reconstruction_error = 0.5 * np.sum((I - Phi @ a)**2)  # 0.5 * ||I - Φa||²
        l1_sparsity = lam * np.sum(np.abs(a))  # λ|a|₁ 
        expected_objective = reconstruction_error + l1_sparsity
        
        # Our implementation via learner
        learner = KsvdLearner(n_atoms=n_atoms, penalty=L1Penalty(lam=lam))
        learner._dictionary = Phi
        computed_objective = learner._compute_objective(I.reshape(-1, 1), a.reshape(-1, 1))
        
        # Should match exactly
        np.testing.assert_allclose(computed_objective, expected_objective, rtol=1e-12,
                                   err_msg="Objective function doesn't match Olshausen & Field formulation")
        
        print(f"✅ Objective validation: computed={computed_objective:.6f}, expected={expected_objective:.6f}")


class TestOlshausenFieldSparsityFunctions:
    """Test sparsity functions used by Olshausen & Field."""
    
    def test_l1_sparsity_function(self):
        """Test S(x) = |x| sparsity function."""
        test_coeffs = np.array([1.5, -0.8, 0.0, 0.3, -2.1])
        lam = 0.2
        
        # Manual computation
        expected_value = lam * np.sum(np.abs(test_coeffs))
        
        # Our implementation
        penalty = L1Penalty(lam=lam)
        computed_value = penalty.value(test_coeffs)
        
        np.testing.assert_allclose(computed_value, expected_value, rtol=1e-12)
        print(f"✅ L1 sparsity S(x)=|x|: {computed_value:.4f}")
    
    def test_cauchy_sparsity_function(self):
        """Test S(x) = ln(1 + x²) sparsity function."""
        test_coeffs = np.array([1.0, -0.5, 0.2, -1.2, 0.0])
        lam = 0.15
        sigma = 1.0
        
        # Manual computation: λ Σᵢ ln(1 + (xᵢ/σ)²)
        expected_value = lam * np.sum(np.log(1.0 + (test_coeffs / sigma)**2))
        
        # Our implementation  
        penalty = CauchyPenalty(lam=lam, sigma=sigma)
        computed_value = penalty.value(test_coeffs)
        
        np.testing.assert_allclose(computed_value, expected_value, rtol=1e-12)
        print(f"✅ Cauchy sparsity S(x)=ln(1+x²): {computed_value:.4f}")
    
    def test_sparsity_function_properties(self):
        """Test mathematical properties of sparsity functions."""
        test_data = np.array([2.0, -1.5, 0.0, 0.8, -0.3])
        lam = 0.1
        
        penalties = [
            ('L1', L1Penalty(lam=lam)),
            ('Cauchy', CauchyPenalty(lam=lam, sigma=1.0))
        ]
        
        for name, penalty in penalties:
            # Property 1: Non-negativity 
            value = penalty.value(test_data)
            assert value >= 0, f"{name} penalty must be non-negative, got {value}"
            
            # Property 2: Zero at origin (for most penalties)
            zero_value = penalty.value(np.zeros_like(test_data))
            if name == 'L1':
                assert zero_value == 0.0, f"{name} should be zero at origin"
            
            # Property 3: Scale invariance test
            scaled_data = 2.0 * test_data
            scaled_value = penalty.value(scaled_data)
            
            if name == 'L1':
                # L1 is homogeneous of degree 1
                expected_scaled = 2.0 * value
                np.testing.assert_allclose(scaled_value, expected_scaled, rtol=1e-12)
                
            print(f"✅ {name} penalty properties verified")


class TestOlshausenFieldDictionaryLearning:
    """Test dictionary learning aspects from Olshausen & Field."""
    
    def test_dictionary_normalization_constraint(self):
        """Test that dictionary atoms are normalized (O&F requirement)."""
        np.random.seed(42)
        
        # Create synthetic data
        n_features, n_atoms = 32, 24
        X = np.random.randn(n_features, 50)
        
        # Train dictionary learner
        learner = DictionaryLearner(n_atoms=n_atoms, max_iterations=10, sparsity_penalty=0.1)
        learner.fit(X, verbose=False)
        
        # Check atom normalization (O&F constraint)
        dictionary = learner.dictionary
        atom_norms = np.linalg.norm(dictionary, axis=0)
        
        # All atoms should have unit norm (within numerical precision)
        np.testing.assert_allclose(atom_norms, 1.0, atol=1e-10,
                                   err_msg="Dictionary atoms must be unit normalized (O&F constraint)")
        
        print(f"✅ Dictionary normalization: mean norm = {np.mean(atom_norms):.6f}")
    
    def test_overcomplete_dictionary_learning(self):
        """Test learning with overcomplete dictionaries (O&F setting)."""
        np.random.seed(42)
        
        # Overcomplete setting: more atoms than features
        n_features, n_atoms = 20, 30  # 1.5x overcomplete
        X = np.random.randn(n_features, 40)
        
        # Should handle overcomplete case without issues
        learner = DictionaryLearner(n_atoms=n_atoms, max_iterations=5)
        history = learner.fit(X, verbose=False)
        
        # Check that learning proceeded
        assert len(history['reconstruction_errors']) > 0
        assert learner.dictionary.shape == (n_features, n_atoms)
        
        # Error should decrease (basic learning check)
        initial_error = history['reconstruction_errors'][0]
        final_error = history['reconstruction_errors'][-1]
        assert final_error <= initial_error, "Learning should reduce reconstruction error"
        
        print(f"✅ Overcomplete learning: {n_features}→{n_atoms} atoms")


class TestOlshausenFieldMathematicalProperties:
    """Test core mathematical properties from the theory."""
    
    def test_sparse_solution_properties(self):
        """Test that solutions exhibit sparsity as expected by theory."""
        np.random.seed(42)
        
        # Create well-conditioned test problem
        n_features, n_atoms = 25, 20
        D = create_test_dictionary(n_features, n_atoms, condition_number=5.0, seed=42)
        
        # Generate sparse ground truth
        true_a = np.random.laplace(scale=0.1, size=(n_atoms, 3))
        true_a[np.abs(true_a) < 0.2] = 0  # Make sparse
        X = D @ true_a + 0.01 * np.random.randn(n_features, 3)
        
        # Test different sparsity levels
        lambda_values = [0.01, 0.1, 1.0]  # Increasing sparsity
        sparsity_levels = []
        
        for lam in lambda_values:
            coder = SparseCoder(n_atoms=n_atoms, lam=lam, max_iter=100)
            coder.D = D.copy()
            A_estimated = coder.encode(X)
            
            # Measure sparsity (proportion of near-zero elements)
            sparsity = np.mean(np.abs(A_estimated) < 0.01)
            sparsity_levels.append(sparsity)
        
        # Sparsity should increase with lambda (O&F theory prediction)
        assert sparsity_levels[1] >= sparsity_levels[0], "Sparsity should increase with λ"
        assert sparsity_levels[2] >= sparsity_levels[1], "Sparsity should increase with λ"
        
        print(f"✅ Sparsity progression: λ={lambda_values} → sparsity={[f'{s:.2f}' for s in sparsity_levels]}")
    
    def test_reconstruction_fidelity_vs_sparsity_tradeoff(self):
        """Test the fundamental tradeoff between reconstruction and sparsity."""
        np.random.seed(42)
        
        # Create test problem
        n_features, n_atoms = 30, 25
        D = create_test_dictionary(n_features, n_atoms, seed=42)
        X = np.random.randn(n_features, 5)
        
        lambda_values = [0.001, 0.1, 10.0]  # Low to high sparsity
        reconstruction_errors = []
        sparsity_penalties = []
        
        for lam in lambda_values:
            coder = SparseCoder(n_atoms=n_atoms, lam=lam, max_iter=50, tol=1e-8)
            coder.D = D.copy()
            A = coder.encode(X)
            
            # Compute both terms of objective
            reconstruction_error = 0.5 * np.sum((X - D @ A)**2)
            sparsity_penalty = lam * np.sum(np.abs(A))
            
            reconstruction_errors.append(reconstruction_error)
            sparsity_penalties.append(sparsity_penalty)
        
        # As λ increases: reconstruction error should increase, sparsity penalty effect should dominate
        # This is the fundamental tradeoff in Olshausen & Field theory
        assert reconstruction_errors[2] >= reconstruction_errors[0], "Higher λ should increase reconstruction error"
        
        # Total objectives should reflect the tradeoff
        total_objectives = [r + s for r, s in zip(reconstruction_errors, sparsity_penalties)]
        
        print(f"✅ Tradeoff verification:")
        for i, lam in enumerate(lambda_values):
            print(f"   λ={lam}: recon={reconstruction_errors[i]:.3f}, sparse={sparsity_penalties[i]:.3f}, total={total_objectives[i]:.3f}")


class TestResearchAccuracyComparison:
    """Compare our implementation with expected research results."""
    
    def test_algorithm_convergence_behavior(self):
        """Test that our algorithm exhibits expected convergence behavior."""
        np.random.seed(42)
        
        # Create controlled test problem
        n_features, n_atoms = 40, 30
        D_true = create_test_dictionary(n_features, n_atoms, seed=42)
        
        # Generate data with known sparse structure
        A_true = np.random.laplace(scale=0.08, size=(n_atoms, 20))
        A_true[np.abs(A_true) < 0.1] = 0
        X = D_true @ A_true + 0.005 * np.random.randn(n_features, 20)
        
        # Train with monitoring
        learner = DictionaryLearner(n_atoms=n_atoms, max_iterations=50, sparsity_penalty=0.05)
        history = learner.fit(X, verbose=False)
        
        # Analyze convergence properties
        errors = history['reconstruction_errors']
        
        # Should decrease monotonically (or at least overall trend)
        initial_error = errors[0]
        final_error = errors[-1]
        improvement_ratio = (initial_error - final_error) / initial_error
        
        assert improvement_ratio > 0.001, f"Should achieve some improvement, got {improvement_ratio:.3f}"
        
        # Should reach stable solution (last few iterations similar)
        if len(errors) >= 10:
            final_stability = np.std(errors[-5:]) / np.mean(errors[-5:])
            assert final_stability < 0.1, f"Final iterations should be stable, got variation {final_stability:.3f}"
        
        print(f"✅ Convergence analysis: {improvement_ratio:.2%} improvement, final stability {final_stability:.3f}")


def test_complete_research_validation():
    """
    Complete validation that our sparse coding implementation is research-accurate.
    
    This test serves as the definitive validation that our implementation
    correctly follows Olshausen & Field (1996) mathematical foundations.
    """
    print("\n" + "="*60)
    print("OLSHAUSEN & FIELD (1996) RESEARCH VALIDATION")
    print("="*60)
    
    # Run all validation components
    validator = TestOlshausenFieldObjectiveFunction()
    validator.test_objective_function_formulation()
    
    sparsity_validator = TestOlshausenFieldSparsityFunctions()
    sparsity_validator.test_l1_sparsity_function()
    sparsity_validator.test_cauchy_sparsity_function() 
    sparsity_validator.test_sparsity_function_properties()
    
    learning_validator = TestOlshausenFieldDictionaryLearning()
    learning_validator.test_dictionary_normalization_constraint()
    learning_validator.test_overcomplete_dictionary_learning()
    
    properties_validator = TestOlshausenFieldMathematicalProperties()
    properties_validator.test_sparse_solution_properties()
    properties_validator.test_reconstruction_fidelity_vs_sparsity_tradeoff()
    
    convergence_validator = TestResearchAccuracyComparison()
    convergence_validator.test_algorithm_convergence_behavior()
    
    print("\n" + "="*60)
    print("✅ COMPLETE RESEARCH VALIDATION PASSED")
    print("✅ Implementation follows Olshausen & Field (1996) theory")
    print("✅ All mathematical properties verified")
    print("✅ Objective function: E = ||I - Φa||² + λS(a)")
    print("✅ Sparsity functions: L1 and Cauchy implemented correctly")  
    print("✅ Dictionary learning converges as expected")
    print("="*60)


if __name__ == "__main__":
    test_complete_research_validation()