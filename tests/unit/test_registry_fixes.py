"""
Test that the registry fixes solve the shared state and invalid routing issues.

Verifies that:
1. Registry uses factories, not shared instances
2. Invalid solver selection rules are fixed
3. Proper error handling for parameter mismatches
"""

import numpy as np
import pytest
from sparse_coding.core.solver_implementations import SOLVER_REGISTRY
from sparse_coding.core.penalties.implementations import L1Penalty, CauchyPenalty, TopKConstraint


class TestRegistryFixes:
    """Test registry shared state and routing fixes."""
    
    def test_registry_uses_factories_not_instances(self):
        """Test that registry creates fresh instances, avoiding shared state."""
        # Get two solver instances of the same type
        solver1 = SOLVER_REGISTRY.get_solver('fista', max_iter=100, tol=1e-6)
        solver2 = SOLVER_REGISTRY.get_solver('fista', max_iter=200, tol=1e-8)
        
        # They should be different instances
        assert solver1 is not solver2, "Registry should create fresh instances, not reuse shared ones"
        
        # They should have different configurations
        assert solver1.max_iter == 100
        assert solver2.max_iter == 200
        assert solver1.tol == 1e-6
        assert solver2.tol == 1e-8
        
        print("✅ Registry creates fresh instances - no shared state bug!")
        
    def test_invalid_solver_routing_fixed(self):
        """Test that non-prox/non-diff penalties route to NCG, not FISTA."""
        # Create a TopK penalty (non-prox, non-diff)
        topk_penalty = TopKConstraint(k=5)
        assert not topk_penalty.is_prox_friendly
        assert not topk_penalty.is_differentiable
        
        # Auto-select solver
        solver = SOLVER_REGISTRY.auto_select_solver(topk_penalty)
        
        # Should get NCG, not FISTA
        assert solver.name == 'ncg', f"TopK should route to NCG, got {solver.name}"
        
        print("✅ Fixed: non-prox/non-diff penalties route to NCG, not FISTA!")
        
    def test_proper_penalty_routing(self):
        """Test that all penalty types route to appropriate solvers."""
        test_cases = [
            (L1Penalty(lam=0.1), 'fista', 'L1 should route to FISTA'),
            (CauchyPenalty(lam=0.1), 'ncg', 'Cauchy should route to NCG'),
            (TopKConstraint(k=5), 'ncg', 'TopK should route to NCG'),
        ]
        
        for penalty, expected_solver, description in test_cases:
            solver = SOLVER_REGISTRY.auto_select_solver(penalty)
            assert solver.name == expected_solver, f"{description}, got {solver.name}"
            print(f"✅ {description}")
    
    def test_registry_parameter_validation(self):
        """Test that registry provides helpful error messages for parameter mismatches."""
        # Try to create FISTA with invalid parameter
        with pytest.raises(TypeError) as exc_info:
            SOLVER_REGISTRY.get_solver('fista', invalid_param=123)
        
        error_msg = str(exc_info.value)
        assert 'Invalid parameters for fista' in error_msg
        assert 'Expected parameters' in error_msg
        
        print("✅ Registry provides helpful parameter error messages!")
        
    def test_registry_separation_of_concerns(self):
        """Test that solver creation and solve parameters are properly separated."""
        # Create test data
        D = np.random.randn(32, 16)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        X = np.random.randn(32, 5)
        penalty = L1Penalty(lam=0.1)
        
        # Solve with mixed solver and solve parameters
        try:
            result = SOLVER_REGISTRY.solve(
                D, X, penalty,
                algorithm='fista',
                max_iter=50,  # Solver parameter
                tol=1e-5,     # Solver parameter
                verbose=False  # Solver parameter
            )
            
            assert result.shape == (16, 5), "Solve should return correct shape"
            assert np.all(np.isfinite(result)), "Solution should be finite"
            
            print("✅ Registry properly separates solver creation from solve parameters!")
            
        except Exception as e:
            pytest.fail(f"Registry solve failed: {e}")
    
    def test_no_shared_state_corruption(self):
        """Test that multiple simultaneous uses don't corrupt each other's state."""
        D = np.random.randn(20, 10)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        X1 = np.random.randn(20, 3)
        X2 = np.random.randn(20, 4)
        penalty = L1Penalty(lam=0.1)
        
        # Solve two problems "simultaneously" with different solvers
        solver1 = SOLVER_REGISTRY.get_solver('fista', max_iter=10)
        solver2 = SOLVER_REGISTRY.get_solver('fista', max_iter=20)
        
        result1 = solver1.solve(D, X1, penalty)
        result2 = solver2.solve(D, X2, penalty)
        
        # Results should have correct shapes
        assert result1.shape == (10, 3)
        assert result2.shape == (10, 4)
        
        # Solvers should still have their original configurations
        assert solver1.max_iter == 10
        assert solver2.max_iter == 20
        
        print("✅ No shared state corruption between solver instances!")


def test_registry_comprehensive_validation():
    """Integration test verifying all registry fixes work together."""
    print("\n🔧 COMPREHENSIVE REGISTRY VALIDATION")
    print("=" * 50)
    
    # Test 1: Factory pattern works
    solver_a = SOLVER_REGISTRY.get_solver('ista', max_iter=100)
    solver_b = SOLVER_REGISTRY.get_solver('ista', max_iter=200)
    assert solver_a is not solver_b
    assert solver_a.max_iter != solver_b.max_iter
    print("1. ✅ Factory pattern: No shared state between instances")
    
    # Test 2: Invalid routing fixed
    non_prox_penalty = TopKConstraint(k=3)
    auto_solver = SOLVER_REGISTRY.auto_select_solver(non_prox_penalty)
    assert auto_solver.name == 'ncg'
    print("2. ✅ Invalid routing fixed: TopK → NCG (not FISTA)")
    
    # Test 3: Proper error handling
    try:
        SOLVER_REGISTRY.get_solver('nonexistent')
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert 'not found' in str(e)
        print("3. ✅ Proper error handling for missing solvers")
    
    # Test 4: Parameter validation
    try:
        SOLVER_REGISTRY.get_solver('fista', bad_param=123)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Invalid parameters' in str(e)
        print("4. ✅ Parameter validation with helpful error messages")
    
    # Test 5: End-to-end functionality
    D = np.random.randn(16, 8)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    X = np.random.randn(16, 3)
    penalty = L1Penalty(lam=0.05)
    
    result = SOLVER_REGISTRY.solve(D, X, penalty, algorithm='auto', max_iter=50)
    assert result.shape == (8, 3)
    assert np.all(np.isfinite(result))
    print("5. ✅ End-to-end solve functionality preserved")
    
    print("\n🎯 ALL REGISTRY ARCHITECTURAL FIXES VERIFIED!")
    print("• Shared state bug eliminated")
    print("• Invalid penalty routing fixed") 
    print("• Proper factory pattern implemented")
    print("• Parameter validation improved")
    print("• Error handling enhanced")
    
    return True