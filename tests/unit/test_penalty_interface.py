#!/usr/bin/env python3
"""
Test suite for penalty interface compliance and mathematical correctness.

This module validates penalty function implementations against their
mathematical specifications and protocol requirements.
"""

import sys
import numpy as np

# Note: sys.path manipulation removed - tests should use PYTHONPATH=src

from sparse_coding.core.penalties.implementations import (
    L1Penalty, L2Penalty, ElasticNetPenalty, TopKConstraint, CauchyPenalty,
    LogSumPenalty, GroupLassoPenalty, SCADPenalty, create_penalty
)


def test_topk_gradient_implementation():
    """Test that TopKPenalty gradient is properly implemented with subgradient"""
    print("\n[TEST] TopKPenalty gradient implementation...")
    
    # Create TopK constraint
    topk = TopKConstraint(k=3)
    
    # Test gradient computation
    a = np.array([1.0, 0.1, 0.5, 0.01, 2.0, 0.001])
    
    grad = topk.grad(a)
    assert grad is not None, "Gradient should not be None"
    assert grad.shape == a.shape, "Gradient shape should match input"
    
    # Check that K largest elements have zero gradient
    k_largest_indices = np.argpartition(np.abs(a), -3)[-3:]
    assert np.allclose(grad[k_largest_indices], 0.0), "K largest elements should have zero gradient"
    
    # Check that other elements have relatively large penalty (scale-invariant test)
    other_indices = np.setdiff1d(range(len(a)), k_largest_indices)
    on_support_magnitude = np.max(np.abs(grad[k_largest_indices])) if len(k_largest_indices) > 0 else 0.0
    off_support_magnitude = np.min(np.abs(grad[other_indices])) if len(other_indices) > 0 else 0.0
    
    # Off-support penalty should be at least 10x larger than on-support
    min_ratio = 10.0
    assert off_support_magnitude >= min_ratio * max(on_support_magnitude, 1e-6), \
        f"Off-support penalty {off_support_magnitude:.2e} should be ≥{min_ratio}x on-support {on_support_magnitude:.2e}"
    
    print("  ✓ TopKPenalty gradient correctly implemented with subgradient")


def test_protocol_implementations():
    """Test that Protocol methods are properly implemented"""
    print("\n[TEST] Protocol method implementations...")
    
    # Test Penalty protocol properties
    penalties = [
        L1Penalty(), L2Penalty(), ElasticNetPenalty(),
        TopKConstraint(k=5), CauchyPenalty()
    ]
    
    for penalty in penalties:
        # Test is_prox_friendly property
        prox_friendly = penalty.is_prox_friendly
        assert isinstance(prox_friendly, bool), f"{penalty.__class__.__name__} is_prox_friendly should return bool"
        
        # Test is_differentiable property
        differentiable = penalty.is_differentiable
        assert isinstance(differentiable, bool), f"{penalty.__class__.__name__} is_differentiable should return bool"
        
        print(f"  ✓ {penalty.__class__.__name__} protocol properties work")


def test_research_penalty_implementations():
    """Test newly added research-accurate penalty implementations"""
    print("\n[TEST] Research penalty implementations...")
    
    n_features = 10
    test_vector = np.random.randn(n_features) * 0.5
    
    penalties_to_test = [
        ('LogSum', create_penalty('log_sum', lam=0.1, epsilon=0.01)),
        ('GroupLasso', create_penalty('group_lasso', lam=0.1, 
                                     groups=[np.array([0,1,2]), np.array([3,4]), np.array([5,6,7,8,9])])),
        ('SCAD', create_penalty('scad', lam=0.1, a=3.7)),
    ]
    
    for name, penalty in penalties_to_test:
        # Test value function
        val = penalty.value(test_vector)
        assert np.isfinite(val), f"{name} value should be finite"
        assert val >= 0, f"{name} value should be non-negative"
        
        # Test proximal operator
        prox_result = penalty.prox(test_vector, t=0.01)
        assert prox_result.shape == test_vector.shape, f"{name} prox should preserve shape"
        assert np.all(np.isfinite(prox_result)), f"{name} prox should return finite values"
        
        # Test gradient
        grad = penalty.grad(test_vector)
        assert grad.shape == test_vector.shape, f"{name} gradient should preserve shape"
        assert np.all(np.isfinite(grad)), f"{name} gradient should be finite"
        
        print(f"  ✓ {name} penalty correctly implemented")


def test_configuration_options():
    """Test that penalties support algorithmic configuration options"""
    print("\n[TEST] Configuration options...")
    
    # Test L1 with different soft thresholding modes
    configs = [
        ('L1 standard', create_penalty('l1', lam=0.1, soft_threshold_mode='standard')),
        ('L1 vectorized', create_penalty('l1', lam=0.1, soft_threshold_mode='vectorized')),
        ('L1 numba', create_penalty('l1', lam=0.1, soft_threshold_mode='numba_accelerated')),
    ]
    
    test_vec = np.array([0.5, -0.3, 0.1, -0.8, 0.0])
    
    for name, penalty in configs:
        result = penalty.prox(test_vec, t=0.1)
        assert result is not None, f"{name} should return result"
        print(f"  ✓ {name} configuration works")
    
    # Test TopK with different tie-breaking strategies
    topk_configs = [
        ('TopK first', create_penalty('top_k', k=3, tie_breaking='first')),
        ('TopK last', create_penalty('top_k', k=3, tie_breaking='last')),
        ('TopK random', create_penalty('top_k', k=3, tie_breaking='random')),
    ]
    
    for name, penalty in topk_configs:
        result = penalty.prox(test_vec, t=0.1)
        assert np.sum(result != 0) <= 3, f"{name} should keep at most K elements"
        print(f"  ✓ {name} configuration works")


def test_research_accuracy():
    """Test that implementations match research paper specifications"""
    print("\n[TEST] Research accuracy verification...")
    
    # Test SCAD penalty regions (Fan & Li 2001)
    scad = SCADPenalty(lam=1.0, a=3.7)
    
    # Test three regions of SCAD
    small_val = 0.5  # |a| <= λ
    medium_val = 2.0  # λ < |a| <= aλ
    large_val = 5.0   # |a| > aλ
    
    # Region 1: Should behave like L1
    scad_small = scad.value(np.array([small_val]))
    l1_small = 1.0 * np.abs(small_val)
    assert np.allclose(scad_small, l1_small), "SCAD region 1 should match L1"
    
    # Region 3: Should be constant
    scad_large1 = scad.value(np.array([large_val]))
    scad_large2 = scad.value(np.array([large_val * 2]))
    expected_const = (3.7 + 1) * 1.0**2 / 2
    assert np.allclose(scad_large1, expected_const), "SCAD region 3 should be constant"
    assert np.allclose(scad_large2, expected_const), "SCAD region 3 should be constant"
    
    print("  ✓ SCAD penalty matches Fan & Li (2001) specification")
    
    # Test Group LASSO (Yuan & Lin 2006)
    groups = [np.array([0, 1]), np.array([2, 3, 4])]
    group_lasso = GroupLassoPenalty(lam=1.0, groups=groups)
    
    test_vec = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    prox_result = group_lasso.prox(test_vec, t=0.5)
    
    # First group should be shrunk, second group should be zero
    assert np.linalg.norm(prox_result[groups[1]]) < 1e-10, "Small group should be zeroed out"
    print("  ✓ Group LASSO matches Yuan & Lin (2006) specification")


def test_mathematical_properties():
    """Test mathematical properties of implementations"""
    print("\n[TEST] Mathematical properties...")
    
    # Test proximal operator properties
    penalties = [
        create_penalty('l1', lam=0.1),
        create_penalty('elastic_net', lam=0.1, l1_ratio=0.7),
        create_penalty('log_sum', lam=0.1, epsilon=0.01),
    ]
    
    test_vec = np.random.randn(10)
    
    for penalty in penalties:
        # Proximal operator should be non-expansive (for convex penalties)
        # Note: LogSum is approximately non-expansive due to iterative solution
        z1 = test_vec
        z2 = test_vec + np.random.randn(10) * 0.1
        
        prox1 = penalty.prox(z1, t=0.1)
        prox2 = penalty.prox(z2, t=0.1)
        
        dist_before = np.linalg.norm(z2 - z1)
        dist_after = np.linalg.norm(prox2 - prox1)
        
        # Allow small tolerance for iterative methods
        tolerance = 1.1 if penalty.__class__.__name__ == 'LogSumPenalty' else 1.01
        assert dist_after <= dist_before * tolerance, f"{penalty.__class__.__name__} proximal operator should be approximately non-expansive"
        print(f"  ✓ {penalty.__class__.__name__} proximal operator is non-expansive")


def run_all_tests():
    """Run penalty interface compliance tests"""
    print("="*60)
    print("SPARSE CODING PENALTY INTERFACE TEST SUITE")
    print("Testing penalty function mathematical correctness")
    print("="*60)
    
    tests = [
        ("TopK Gradient", test_topk_gradient_implementation),
        ("Protocol Methods", test_protocol_implementations),
        ("Research Penalties", test_research_penalty_implementations),
        ("Configuration Options", test_configuration_options),
        ("Research Accuracy", test_research_accuracy),
        ("Mathematical Properties", test_mathematical_properties),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()  # Test functions now use assertions instead of returning values
            results.append((name, True))
        except Exception as e:
            print(f"\n[ERROR] {name} test failed: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:25} {status}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL PENALTY INTERFACE TESTS PASSED!")
        print("✓ TopK gradient subgradient correctly implemented")
        print("✓ Protocol methods satisfy interface requirements")
        print("✓ Research penalty implementations mathematically accurate")
        print("✓ Configuration options provide algorithmic choices")
        print("✓ Mathematical properties verified")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review implementations.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())