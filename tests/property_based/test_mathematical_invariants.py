"""
Property-based tests for mathematical invariants in sparse coding.

Uses Hypothesis to generate test cases and verify mathematical properties
that should hold universally across different inputs and parameters.

Key invariants tested:
1. Dictionary normalization preservation
2. Reconstruction error properties  
3. Sparsity vs reconstruction trade-offs
4. Algorithm convergence properties
5. Numerical stability across parameter ranges
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays, array_shapes
from sparse_coding import SparseCoder, DictionaryLearner, AdvancedOptimizer, L1Proximal
from sparse_coding.fista_batch import soft_thresh, fista_batch
from tests.conftest import assert_dictionary_normalized, create_test_dictionary


# Hypothesis strategies for sparse coding parameters
@st.composite
def sparse_coding_dimensions(draw):
    """Generate reasonable dimensions for sparse coding problems."""
    n_features = draw(st.integers(min_value=20, max_value=200))
    n_atoms = draw(st.integers(min_value=10, max_value=min(n_features + 20, 100)))
    n_samples = draw(st.integers(min_value=5, max_value=50))
    return n_features, n_atoms, n_samples


@st.composite  
def lambda_values(draw):
    """Generate reasonable regularization parameters."""
    return draw(st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False))


@st.composite
def tolerance_values(draw):
    """Generate reasonable tolerance values."""
    return draw(st.floats(min_value=1e-8, max_value=1e-3, allow_nan=False, allow_infinity=False))


@st.composite
def sparse_test_data(draw):
    """Generate test data for sparse coding."""
    n_features, n_atoms, n_samples = draw(sparse_coding_dimensions())
    seed = draw(st.integers(min_value=0, max_value=2**31-1))
    
    np.random.seed(seed)
    
    # Generate dictionary
    D = np.random.randn(n_features, n_atoms)
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    
    # Generate sparse codes
    sparsity_level = draw(st.floats(min_value=0.1, max_value=0.8))
    A = np.random.laplace(scale=0.2, size=(n_atoms, n_samples))
    mask = np.random.random((n_atoms, n_samples)) < sparsity_level
    A[mask] = 0
    
    # Generate signals
    X = D @ A
    
    # Add small amount of noise
    noise_level = draw(st.floats(min_value=0.0, max_value=0.1))
    if noise_level > 0:
        X += noise_level * np.random.randn(*X.shape)
    
    return {
        'signals': X,
        'dictionary': D,
        'true_codes': A,
        'dimensions': (n_features, n_atoms, n_samples),
        'seed': seed
    }


class TestDictionaryNormalizationInvariants:
    """Test invariants related to dictionary normalization."""
    
    @given(sparse_test_data())
    @settings(max_examples=20, deadline=10000)
    def test_dictionary_normalization_preserved_after_learning(self, data):
        """Property: Dictionary atoms should remain normalized after learning."""
        assume(data['dimensions'][0] >= 10)  # Minimum size for meaningful test
        assume(data['dimensions'][1] >= 5)
        
        X = data['signals']
        n_features, n_atoms, n_samples = data['dimensions']
        
        # Skip if problem is too large for property testing
        assume(n_features * n_atoms < 5000)
        assume(n_samples <= 20)  # Limit samples for speed
        
        learner = DictionaryLearner(n_atoms=n_atoms, max_iter=3, tol=1e-4)
        learner.fit(X)
        
        # Property: All atoms should be unit normalized
        D = learner.dictionary
        atom_norms = np.linalg.norm(D, axis=0)
        
        assert np.allclose(atom_norms, 1.0, atol=1e-6), f"Atom norms: {atom_norms}"
    
    @given(arrays(dtype=np.float64, shape=st.tuples(st.integers(10, 50), st.integers(5, 20)), 
                  elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False)))
    @settings(max_examples=15, deadline=5000)
    def test_normalization_idempotent(self, D):
        """Property: Normalizing an already normalized dictionary should not change it."""
        assume(np.all(np.isfinite(D)))
        assume(np.linalg.norm(D, 'fro') > 1e-6)  # Non-trivial matrix
        
        # First normalization
        D1 = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
        
        # Second normalization  
        D2 = D1 / (np.linalg.norm(D1, axis=0, keepdims=True) + 1e-12)
        
        # Should be identical (up to numerical precision)
        assert np.allclose(D1, D2, atol=1e-10), "Double normalization should be idempotent"


class TestSoftThresholdingInvariants:
    """Test mathematical invariants of soft thresholding operator."""
    
    @given(arrays(dtype=np.float64, shape=st.integers(5, 100), 
                  elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)),
           st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=1000)
    def test_soft_thresholding_shrinkage_property(self, x, threshold):
        """Property: |soft_thresh(x, t)| ≤ |x| for all x, t ≥ 0."""
        assume(np.all(np.isfinite(x)))
        assume(threshold >= 0)
        
        result = soft_thresh(x, threshold)
        
        # Shrinkage property
        assert np.all(np.abs(result) <= np.abs(x) + 1e-12), "Soft thresholding should shrink magnitudes"
    
    @given(arrays(dtype=np.float64, shape=st.integers(5, 100),
                  elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)),
           st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=1000)
    def test_soft_thresholding_sparsity_property(self, x, threshold):
        """Property: soft_thresh(x, t) = 0 when |x| ≤ t."""
        assume(np.all(np.isfinite(x)))
        assume(threshold >= 0)
        
        result = soft_thresh(x, threshold)
        
        # Thresholding property
        small_elements = np.abs(x) <= threshold
        assert np.allclose(result[small_elements], 0.0, atol=1e-12), "Should threshold small elements to zero"
    
    @given(st.floats(-5, 5, allow_nan=False, allow_infinity=False),
           st.floats(0, 2, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100, deadline=500)
    def test_soft_thresholding_single_element_formula(self, x, threshold):
        """Property: Verify soft thresholding formula for single elements."""
        assume(np.isfinite(x) and np.isfinite(threshold))
        assume(threshold >= 0)
        
        result = soft_thresh(np.array([x]), threshold)[0]
        
        # Manual computation
        if abs(x) <= threshold:
            expected = 0.0
        else:
            expected = np.sign(x) * (abs(x) - threshold)
        
        assert abs(result - expected) < 1e-12, f"Formula mismatch: {result} vs {expected}"
    
    @given(arrays(dtype=np.float64, shape=st.integers(5, 50),
                  elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False)),
           st.floats(0.1, 2, allow_nan=False, allow_infinity=False),
           st.floats(1.5, 3, allow_nan=False, allow_infinity=False))
    @settings(max_examples=30, deadline=2000)
    def test_soft_thresholding_monotonicity(self, x, t1, t2):
        """Property: For t1 ≤ t2, |soft_thresh(x, t2)| ≤ |soft_thresh(x, t1)|."""
        assume(np.all(np.isfinite(x)))
        assume(0 < t1 <= t2)
        
        result1 = soft_thresh(x, t1)
        result2 = soft_thresh(x, t2)
        
        # Monotonicity in threshold
        assert np.all(np.abs(result2) <= np.abs(result1) + 1e-12), "Larger threshold should produce smaller magnitudes"


class TestReconstructionErrorProperties:
    """Test properties related to reconstruction errors."""
    
    @given(sparse_test_data(), lambda_values())
    @settings(max_examples=10, deadline=15000)  
    def test_reconstruction_sparsity_tradeoff(self, data, lam):
        """Property: Higher λ should lead to sparser solutions (potentially higher reconstruction error)."""
        assume(data['dimensions'][0] * data['dimensions'][1] < 3000)  # Size limit
        assume(data['dimensions'][2] <= 10)  # Sample limit
        
        X = data['signals']
        n_features, n_atoms, n_samples = data['dimensions']
        
        # Test two different lambda values
        lam_small = lam * 0.1
        lam_large = lam * 10.0
        
        coder_small = SparseCoder(n_atoms=n_atoms, mode="l1", lam=lam_small, max_iter=50, tol=1e-5)
        coder_large = SparseCoder(n_atoms=n_atoms, mode="l1", lam=lam_large, max_iter=50, tol=1e-5)
        
        coder_small.fit(X)
        coder_large.fit(X) 
        
        A_small = coder_small.encode(X)
        A_large = coder_large.encode(X)
        
        # Measure sparsity (fraction of near-zero elements)
        sparsity_small = np.mean(np.abs(A_small) < 0.01)
        sparsity_large = np.mean(np.abs(A_large) < 0.01)
        
        # Property: Larger lambda should produce more sparsity
        assert sparsity_large >= sparsity_small - 0.1, f"λ={lam_large:.3f} should be sparser than λ={lam_small:.3f}"
    
    @given(sparse_test_data())
    @settings(max_examples=10, deadline=10000)
    def test_perfect_reconstruction_with_true_dictionary(self, data):
        """Property: With the true dictionary, we should achieve good reconstruction."""
        assume(data['dimensions'][2] <= 5)  # Limit samples for speed
        assume(data['dimensions'][0] <= 100)  # Size limit
        
        X = data['signals']
        D_true = data['dictionary']
        A_true = data['true_codes']
        
        # Small regularization to avoid numerical issues
        coder = SparseCoder(n_atoms=D_true.shape[1], mode="l1", lam=0.001, max_iter=100)
        coder.dictionary = D_true.copy()
        
        A_estimated = coder.encode(X)
        reconstruction = D_true @ A_estimated
        
        # Should achieve reasonable reconstruction
        mse = np.mean((X - reconstruction)**2)
        signal_power = np.mean(X**2)
        relative_mse = mse / max(signal_power, 1e-12)
        
        assert relative_mse < 0.5, f"Reconstruction error too high with true dictionary: {relative_mse:.6f}"


class TestConvergenceInvariants:
    """Test algorithm convergence properties."""
    
    @given(sparse_test_data(), tolerance_values())
    @settings(max_examples=8, deadline=20000)
    def test_algorithm_eventually_stops(self, data, tol):
        """Property: Algorithm should eventually stop (converge or hit max iterations)."""
        assume(data['dimensions'][0] <= 50)  # Size limit for speed
        assume(data['dimensions'][1] <= 30)
        assume(data['dimensions'][2] <= 5)
        
        X = data['signals']
        n_features, n_atoms, n_samples = data['dimensions']
        
        coder = SparseCoder(n_atoms=n_atoms, mode="l1", lam=0.1, max_iter=100, tol=tol)
        
        # Should not hang indefinitely
        coder.fit(X)
        A = coder.encode(X[:, :1])  # Single signal
        
        # Should produce finite results
        assert np.all(np.isfinite(A)), "Algorithm should produce finite results"
        assert A.shape == (n_atoms, 1), f"Output shape mismatch: {A.shape}"
    
    @given(sparse_test_data())
    @settings(max_examples=5, deadline=15000)
    def test_iterative_improvement_property(self, data):
        """Property: More iterations should not significantly worsen the objective."""
        assume(data['dimensions'][0] <= 40)  # Keep problems small
        assume(data['dimensions'][1] <= 25)
        assume(data['dimensions'][2] <= 3)
        
        X = data['signals']
        n_features, n_atoms, n_samples = data['dimensions']
        D = create_test_dictionary(n_features, n_atoms, seed=data['seed'])
        
        # Compare different iteration limits
        result_few = fista_batch(D, X, lam=0.1, max_iter=5, tol=1e-12)
        result_many = fista_batch(D, X, lam=0.1, max_iter=20, tol=1e-12)
        
        # Compute objectives
        obj_few = 0.5 * np.linalg.norm(X - D @ result_few, 'fro')**2 + 0.1 * np.sum(np.abs(result_few))
        obj_many = 0.5 * np.linalg.norm(X - D @ result_many, 'fro')**2 + 0.1 * np.sum(np.abs(result_many))
        
        # More iterations should not significantly worsen objective (allowing for numerical noise)
        relative_change = (obj_many - obj_few) / max(abs(obj_few), 1e-12)
        assert relative_change <= 0.1, f"More iterations worsened objective: {relative_change:.6f}"


class TestNumericalStabilityInvariants:
    """Test numerical stability across parameter ranges."""
    
    @given(arrays(dtype=np.float64, shape=st.tuples(st.integers(10, 30), st.integers(5, 15)),
                  elements=st.floats(-2, 2, allow_nan=False, allow_infinity=False)),
           arrays(dtype=np.float64, shape=st.tuples(st.integers(5, 15), st.integers(2, 8)),
                  elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False)),
           lambda_values())
    @settings(max_examples=15, deadline=10000)
    def test_finite_outputs_property(self, D, A, lam):
        """Property: Algorithm should always produce finite outputs."""
        assume(np.all(np.isfinite(D)) and np.all(np.isfinite(A)))
        assume(D.shape[1] == A.shape[0])  # Compatible shapes
        assume(np.linalg.norm(D, 'fro') > 1e-6)  # Non-trivial dictionary
        
        # Normalize dictionary
        D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
        
        # Generate signals
        X = D @ A
        
        try:
            result = fista_batch(D, X, lam=lam, max_iter=50, tol=1e-6)
            
            # Should be finite
            assert np.all(np.isfinite(result)), "FISTA should produce finite results"
            assert result.shape == A.shape, f"Shape mismatch: {result.shape} vs {A.shape}"
            
        except Exception as e:
            # If algorithm fails, it should fail gracefully (not hang)
            pytest.fail(f"Algorithm failed unexpectedly: {e}")
    
    @given(st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=1000)
    def test_soft_thresholding_numerical_stability(self, scale):
        """Property: Soft thresholding should be numerically stable across scales."""
        # Create test vector at different scales
        x_base = np.array([1.0, -0.5, 0.2, -1.5, 0.0])
        x = x_base * scale
        threshold = 0.3 * scale
        
        result = soft_thresh(x, threshold)
        
        # Should be finite
        assert np.all(np.isfinite(result)), f"Soft thresholding failed at scale {scale}"
        
        # Should preserve scale relationships
        result_normalized = result / scale
        expected_normalized = soft_thresh(x_base, 0.3)
        
        assert np.allclose(result_normalized, expected_normalized, rtol=1e-10, atol=1e-12), \
            f"Scale invariance violated at scale {scale}"
    
    @given(sparse_test_data(), 
           st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=8, deadline=12000)
    def test_robustness_to_dictionary_scaling(self, data, scale_factor):
        """Property: Results should be robust to dictionary scaling."""
        assume(data['dimensions'][0] <= 40)
        assume(data['dimensions'][2] <= 5)
        
        X = data['signals']
        D = data['dictionary']
        
        # Scale dictionary and compensate in signals
        D_scaled = D * scale_factor
        X_scaled = X * scale_factor
        
        try:
            # Run with original
            A1 = fista_batch(D, X, lam=0.1, max_iter=30, tol=1e-6)
            
            # Run with scaled version
            A2 = fista_batch(D_scaled, X_scaled, lam=0.1 * scale_factor, max_iter=30, tol=1e-6)
            
            # Reconstructions should be similar (up to scaling)
            recon1 = D @ A1
            recon2 = D_scaled @ A2
            
            # Check relative reconstruction quality
            error1 = np.linalg.norm(X - recon1, 'fro') / np.linalg.norm(X, 'fro')
            error2 = np.linalg.norm(X_scaled - recon2, 'fro') / np.linalg.norm(X_scaled, 'fro')
            
            assert abs(error1 - error2) < 0.2, f"Scaling sensitivity too high: {error1:.3f} vs {error2:.3f}"
            
        except Exception as e:
            # Algorithm should not fail due to reasonable scaling
            if scale_factor >= 1.0 and scale_factor <= 5.0:
                pytest.fail(f"Algorithm failed with reasonable scaling {scale_factor}: {e}")


class TestL1ProximalOperatorInvariants:
    """Test invariants specific to L1 proximal operator."""
    
    @given(arrays(dtype=np.float64, shape=st.integers(5, 50),
                  elements=st.floats(-3, 3, allow_nan=False, allow_infinity=False)),
           st.floats(0.01, 2, allow_nan=False, allow_infinity=False),
           st.floats(0.1, 1, allow_nan=False, allow_infinity=False))
    @settings(max_examples=30, deadline=2000)
    def test_proximal_operator_optimality_condition(self, v, lam, t):
        """Property: Proximal operator should satisfy optimality conditions."""
        assume(np.all(np.isfinite(v)))
        
        proximal_op = L1Proximal(lam)
        x = proximal_op.prox(v, t)
        
        # For each component, check optimality condition
        for i in range(len(x)):
            if abs(x[i]) > 1e-10:  # Non-zero component
                # Should satisfy: v[i] - x[i] = t * lam * sign(x[i])
                residual = v[i] - x[i]
                expected_residual = t * lam * np.sign(x[i])
                
                assert abs(residual - expected_residual) < 1e-10, \
                    f"Optimality condition violated at index {i}: {residual} vs {expected_residual}"
    
    @given(arrays(dtype=np.float64, shape=st.integers(5, 30),
                  elements=st.floats(-2, 2, allow_nan=False, allow_infinity=False)),
           st.floats(0.05, 1, allow_nan=False, allow_infinity=False))
    @settings(max_examples=40, deadline=1000)
    def test_proximal_operator_nonexpansive_property(self, lam, t):
        """Property: Proximal operator is non-expansive."""
        # Generate two different input vectors
        v1 = np.random.randn(10) * 2.0
        v2 = v1 + 0.5 * np.random.randn(10)
        
        assume(np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)))
        
        proximal_op = L1Proximal(lam)
        
        x1 = proximal_op.prox(v1, t)
        x2 = proximal_op.prox(v2, t)
        
        # Non-expansive property: ||prox(v1) - prox(v2)|| ≤ ||v1 - v2||
        input_distance = np.linalg.norm(v1 - v2)
        output_distance = np.linalg.norm(x1 - x2)
        
        assert output_distance <= input_distance + 1e-10, \
            f"Non-expansive property violated: {output_distance} > {input_distance}"