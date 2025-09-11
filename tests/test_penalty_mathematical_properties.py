"""
Mathematical Property Tests for Penalty Functions

Tests validate mathematical properties required for optimization convergence
and research accuracy. These tests verify penalty functions behave according
to convex optimization theory and published research.

Research Foundation:
- Parikh & Boyd (2014) "Proximal Algorithms"
- Beck & Teboulle (2009) "A Fast Iterative Shrinkage-Thresholding Algorithm"
- Rockafellar (1970) "Convex Analysis"

Author: Benedict Chen
"""

import numpy as np
import pytest
from sparse_coding.core.penalties.implementations import (
    L1Penalty, L2Penalty, ElasticNetPenalty, TopKConstraint, 
    CauchyPenalty, SCADPenalty, HuberPenalty, create_penalty
)


class TestL1PenaltyMathematicalProperties:
    """Test L1 penalty satisfies required mathematical properties."""
    
    @pytest.fixture
    def penalty(self):
        return L1Penalty(lam=0.5)
    
    def test_non_negativity(self, penalty):
        """L1 penalty value must be non-negative."""
        test_vectors = [
            np.array([1.0, -2.0, 3.0]),
            np.array([0.0, 0.0, 0.0]),
            np.random.randn(100),
            np.array([-10.5, 15.2, -3.7])
        ]
        
        for a in test_vectors:
            value = penalty.value(a)
            assert value >= 0, f"L1 penalty must be non-negative, got {value} for input {a}"
    
    def test_positive_homogeneity_degree_1(self, penalty):
        """L1 penalty is positively homogeneous of degree 1: f(ca) = c*f(a) for c > 0."""
        np.random.seed(42)
        a = np.random.randn(10)
        scale_factors = [0.1, 0.5, 2.0, 5.0, 10.0]
        
        base_value = penalty.value(a)
        
        for c in scale_factors:
            scaled_value = penalty.value(c * a)
            expected = c * base_value
            
            assert abs(scaled_value - expected) < 1e-12, (
                f"L1 penalty not homogeneous: f({c}*a) = {scaled_value:.12f}, "
                f"expected {c} * f(a) = {expected:.12f}"
            )
    
    def test_triangle_inequality(self, penalty):
        """L1 penalty satisfies triangle inequality: ||a + b||₁ ≤ ||a||₁ + ||b||₁."""
        np.random.seed(42)
        
        for _ in range(10):
            a = np.random.randn(20)
            b = np.random.randn(20)
            
            combined_value = penalty.value(a + b)
            individual_sum = penalty.value(a) + penalty.value(b)
            
            assert combined_value <= individual_sum + 1e-12, (
                f"Triangle inequality violated: ||a+b||₁ = {combined_value:.12f}, "
                f"||a||₁ + ||b||₁ = {individual_sum:.12f}"
            )
    
    def test_proximal_operator_properties(self, penalty):
        """Test proximal operator satisfies mathematical properties."""
        np.random.seed(42)
        z = np.random.randn(15)
        t = 0.1
        
        # Property 1: Proximal optimality condition
        prox_result = penalty.prox(z, t)
        residual = z - prox_result
        
        # For L1: residual should be subgradient at prox_result
        for i, (res, prox_val) in enumerate(zip(residual, prox_result)):
            if abs(prox_val) > 1e-12:  # Non-zero case
                expected_subgrad = t * penalty.lam * np.sign(prox_val)
                assert abs(res - expected_subgrad) < 1e-12, (
                    f"Proximal optimality violated at index {i}: residual={res}, "
                    f"expected subgradient={expected_subgrad}"
                )
            else:  # Zero case
                assert abs(res) <= t * penalty.lam + 1e-12, (
                    f"Subgradient bound violated for zero element: residual={res}, "
                    f"bound={t * penalty.lam}"
                )
    
    def test_soft_thresholding_correctness(self, penalty):
        """Verify soft thresholding formula matches theoretical expectation."""
        test_cases = [
            (2.0, 0.5, 0.1),  # Above threshold
            (-1.5, 0.5, 0.1),  # Above threshold (negative)
            (0.05, 0.5, 0.1),  # Below threshold
            (0.0, 0.5, 0.1),   # Exactly zero
        ]
        
        for z_val, lam, t in test_cases:
            penalty_test = L1Penalty(lam=lam)
            z = np.array([z_val])
            
            result = penalty_test.prox(z, t)
            threshold = t * lam
            expected = np.sign(z_val) * max(abs(z_val) - threshold, 0.0)
            
            assert abs(result[0] - expected) < 1e-15, (
                f"Soft thresholding incorrect: prox({z_val}) = {result[0]}, "
                f"expected {expected}"
            )


class TestL2PenaltyMathematicalProperties:
    """Test L2 penalty mathematical properties."""
    
    @pytest.fixture
    def penalty(self):
        return L2Penalty(lam=0.3)
    
    def test_positive_homogeneity_degree_2(self, penalty):
        """L2 penalty is positively homogeneous of degree 2: f(ca) = c²*f(a)."""
        np.random.seed(42)
        a = np.random.randn(8)
        scale_factors = [0.1, 0.5, 2.0, 3.0]
        
        base_value = penalty.value(a)
        
        for c in scale_factors:
            scaled_value = penalty.value(c * a)
            expected = (c ** 2) * base_value
            
            assert abs(scaled_value - expected) < 1e-12, (
                f"L2 penalty not degree-2 homogeneous: f({c}*a) = {scaled_value:.12f}, "
                f"expected {c}² * f(a) = {expected:.12f}"
            )
    
    def test_strongly_convex(self, penalty):
        """Test L2 penalty strong convexity."""
        # Strong convexity: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y) - (μ/2)λ(1-λ)||x-y||²
        # For L2: μ = penalty.lam (strong convexity constant)
        
        np.random.seed(42)
        x = np.random.randn(6)
        y = np.random.randn(6)
        
        lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for lam in lambdas:
            point = lam * x + (1 - lam) * y
            
            actual = penalty.value(point)
            upper_bound = lam * penalty.value(x) + (1 - lam) * penalty.value(y)
            strong_convex_term = (penalty.lam / 2) * lam * (1 - lam) * np.sum((x - y)**2)
            expected_bound = upper_bound - strong_convex_term
            
            assert actual <= expected_bound + 1e-12, (
                f"Strong convexity violated: f({lam}x + {1-lam}y) = {actual:.12f}, "
                f"bound = {expected_bound:.12f}"
            )
    
    def test_proximal_shrinkage_formula(self, penalty):
        """Verify L2 proximal operator is simple shrinkage."""
        np.random.seed(42)
        z = np.random.randn(10)
        t = 0.2
        
        result = penalty.prox(z, t)
        shrinkage_factor = 1.0 / (1.0 + t * penalty.lam)
        expected = shrinkage_factor * z
        
        np.testing.assert_allclose(result, expected, rtol=1e-12,
                                 err_msg="L2 proximal operator should be simple shrinkage")


class TestElasticNetMathematicalProperties:
    """Test Elastic Net penalty mathematical properties."""
    
    @pytest.fixture
    def penalty(self):
        return ElasticNetPenalty(lam=0.1, l1_ratio=0.7)
    
    def test_convex_combination_property(self, penalty):
        """Elastic Net combines L1 and L2 penalties correctly."""
        np.random.seed(42)
        a = np.random.randn(5)
        
        # Create separate L1 and L2 penalties
        l1_penalty = L1Penalty(lam=penalty.lam_l1)
        l2_penalty = L2Penalty(lam=penalty.lam_l2)
        
        # Test penalty value
        elastic_value = penalty.value(a)
        expected_value = l1_penalty.value(a) + l2_penalty.value(a)
        
        assert abs(elastic_value - expected_value) < 1e-12, (
            f"Elastic Net value incorrect: got {elastic_value:.12f}, "
            f"expected {expected_value:.12f}"
        )
    
    def test_proximal_operator_two_step_process(self, penalty):
        """Elastic Net proximal operator should be shrinkage then soft thresholding."""
        np.random.seed(42)
        z = np.random.randn(8)
        t = 0.15
        
        # Manual two-step computation
        shrinkage_factor = 1.0 / (1.0 + t * penalty.lam_l2)
        z_shrunk = shrinkage_factor * z
        threshold = t * penalty.lam_l1 * shrinkage_factor
        manual_result = np.sign(z_shrunk) * np.maximum(np.abs(z_shrunk) - threshold, 0.0)
        
        # Method result
        method_result = penalty.prox(z, t)
        
        np.testing.assert_allclose(method_result, manual_result, rtol=1e-12,
                                 err_msg="Elastic Net proximal operator implementation incorrect")


class TestTopKConstraintMathematicalProperties:
    """Test Top-K constraint mathematical properties."""
    
    @pytest.fixture
    def penalty(self):
        return TopKConstraint(k=3)
    
    def test_indicator_function_behavior(self, penalty):
        """Top-K constraint should be 0 for sparse vectors, ∞ for dense."""
        # Sparse vector (≤ K nonzeros)
        sparse_vector = np.array([0.0, 2.0, 0.0, -1.5, 0.0])  # 2 nonzeros, K=3
        assert penalty.value(sparse_vector) == 0.0, "Sparse vector should have zero penalty"
        
        # Dense vector (> K nonzeros)  
        dense_vector = np.array([1.0, 2.0, -1.0, 3.0, 0.5])  # 5 nonzeros, K=3
        assert penalty.value(dense_vector) == np.inf, "Dense vector should have infinite penalty"
    
    def test_hard_thresholding_property(self, penalty):
        """Proximal operator should perform hard thresholding."""
        z = np.array([0.1, 2.5, -1.8, 0.3, -3.2, 0.7])  # 6 elements, K=3
        
        result = penalty.prox(z, 0.1)  # t value doesn't matter for indicator functions
        
        # Count nonzeros
        nonzero_count = np.sum(np.abs(result) > 1e-12)
        assert nonzero_count == penalty.k, (
            f"Hard thresholding should preserve exactly {penalty.k} elements, "
            f"got {nonzero_count}"
        )
        
        # Check that largest magnitude elements were preserved
        abs_z = np.abs(z)
        k_largest_indices = np.argpartition(abs_z, -penalty.k)[-penalty.k:]
        
        for i in k_largest_indices:
            assert abs(result[i] - z[i]) < 1e-12, (
                f"Element {i} should be preserved: z[{i}]={z[i]}, result[{i}]={result[i]}"
            )


class TestPenaltyFactoryIntegration:
    """Test penalty factory creates valid penalties with correct properties."""
    
    def test_all_penalty_types_created(self):
        """Factory should create all implemented penalty types."""
        penalty_configs = {
            'l1': {'lam': 0.1},
            'l2': {'lam': 0.1}, 
            'elastic_net': {'lam': 0.1},
            'top_k': {'k': 5},  # TopKConstraint needs k parameter, not lam
            'cauchy': {'lam': 0.1},
            'scad': {'lam': 0.1},
            'huber': {'lam': 0.1}
        }
        
        for ptype, config in penalty_configs.items():
            penalty = create_penalty(ptype, **config)
            
            # Test basic properties
            test_vector = np.array([1.0, -0.5, 0.0, 2.0])
            
            # Should compute value without error
            value = penalty.value(test_vector)
            assert isinstance(value, (float, int)), f"Penalty {ptype} should return numeric value"
            
            # Should have boolean properties
            assert hasattr(penalty, 'is_prox_friendly'), f"Penalty {ptype} missing is_prox_friendly"
            assert hasattr(penalty, 'is_differentiable'), f"Penalty {ptype} missing is_differentiable"
    
    def test_proximal_operators_work(self):
        """All prox-friendly penalties should have working proximal operators."""
        prox_friendly_penalties = ['l1', 'l2', 'elastic_net', 'scad', 'huber']
        
        np.random.seed(42)
        z = np.random.randn(5)
        t = 0.1
        
        for ptype in prox_friendly_penalties:
            penalty = create_penalty(ptype, lam=0.1)
            
            if penalty.is_prox_friendly:
                result = penalty.prox(z, t)
                
                assert result.shape == z.shape, f"Penalty {ptype} prox should preserve shape"
                assert np.all(np.isfinite(result)), f"Penalty {ptype} prox should return finite values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])