#!/usr/bin/env python3
"""
Simple performance benchmark for sparse coding.

Tests basic functionality and timing without complex dependencies.
"""

import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from sparse_coding import SparseCoder

def test_basic_performance():
    """Test basic sparse coding performance."""
    print("🚀 Simple Sparse Coding Benchmark")
    print("=" * 50)
    
    # Test parameters
    np.random.seed(42)
    n_features = 64
    n_samples = 500
    n_atoms = 128
    
    # Generate test data
    X = np.random.randn(n_features, n_samples)
    print(f"Test data: {n_features} features × {n_samples} samples")
    print(f"Dictionary size: {n_atoms} atoms")
    
    # Test different modes
    modes = ['l1', 'paper', 'log']
    results = {}
    
    for mode in modes:
        print(f"\n--- Testing mode: {mode} ---")
        try:
            # Dictionary learning
            start_time = time.time()
            sc = SparseCoder(n_atoms=n_atoms, mode=mode, lam=0.1, max_iter=100)
            sc.fit(X, n_steps=10)
            dict_time = time.time() - start_time
            
            # Encoding
            start_time = time.time()
            codes = sc.encode(X[:, :100])  # Encode subset for speed
            encode_time = time.time() - start_time
            
            # Calculate metrics
            sparsity = (codes == 0).mean()
            
            # Store results
            results[mode] = {
                'dict_time': dict_time,
                'encode_time': encode_time,
                'sparsity': sparsity,
                'codes_shape': codes.shape
            }
            
            print(f"  Dictionary learning: {dict_time:.2f}s")
            print(f"  Encoding time: {encode_time:.2f}s")
            print(f"  Sparsity level: {sparsity:.2f}")
            print(f"  Codes shape: {codes.shape}")
            
        except Exception as e:
            print(f"  ❌ Mode {mode} failed: {e}")
            results[mode] = None
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    successful_modes = [mode for mode, result in results.items() if result is not None]
    if successful_modes:
        print(f"✅ Successful modes: {', '.join(successful_modes)}")
        
        # Find fastest mode
        fastest_mode = min(successful_modes, 
                          key=lambda m: results[m]['dict_time'] + results[m]['encode_time'])
        total_time = results[fastest_mode]['dict_time'] + results[fastest_mode]['encode_time']
        print(f"🏆 Fastest mode: {fastest_mode} ({total_time:.2f}s total)")
        
        # Average sparsity
        avg_sparsity = np.mean([results[m]['sparsity'] for m in successful_modes])
        print(f"📈 Average sparsity: {avg_sparsity:.2f}")
    else:
        print("❌ No modes completed successfully")
    
    return results

def test_scalability():
    """Test performance across different data sizes."""
    print(f"\n{'=' * 50}")
    print("📏 SCALABILITY TEST")
    print("=" * 50)
    
    data_sizes = [(32, 100), (64, 300), (128, 500)]
    
    for n_features, n_samples in data_sizes:
        print(f"\nTesting {n_features}×{n_samples} data...")
        
        try:
            np.random.seed(42)
            X = np.random.randn(n_features, n_samples)
            n_atoms = min(n_features * 2, 128)
            
            start_time = time.time()
            sc = SparseCoder(n_atoms=n_atoms, mode='l1', lam=0.1, max_iter=50)
            sc.fit(X, n_steps=5)
            codes = sc.encode(X[:, :min(50, n_samples)])
            total_time = time.time() - start_time
            
            sparsity = (codes == 0).mean()
            print(f"  Time: {total_time:.2f}s, Sparsity: {sparsity:.2f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")

def test_import_functionality():
    """Test that core imports and basic functionality work."""
    print(f"\n{'=' * 50}")
    print("🔧 FUNCTIONALITY TEST")
    print("=" * 50)
    
    try:
        # Test imports
        from sparse_coding import SparseCoder
        from sparse_coding.sparse_coding_monitoring import TB, CSVDump
        print("✅ Core imports successful")
        
        # Test basic functionality
        np.random.seed(42)
        X = np.random.randn(32, 100)
        
        sc = SparseCoder(n_atoms=64, mode='l1', lam=0.1)
        print("✅ SparseCoder creation successful")
        
        sc.fit(X, n_steps=3)
        print("✅ Dictionary learning successful")
        
        codes = sc.encode(X[:, :20])
        print(f"✅ Encoding successful: {codes.shape}")
        
        # Test monitoring (simplified)
        try:
            monitor = CSVDump("test_log.csv")
            print("✅ Monitoring imports work")
        except Exception as e:
            print(f"⚠️  Monitoring test skipped: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all benchmark tests."""
    print("🧪 Sparse Coding Performance Benchmark Suite")
    print("=" * 60)
    
    # Test basic functionality first
    if not test_import_functionality():
        print("\n❌ Basic functionality failed - aborting benchmark")
        return False
    
    # Run performance tests
    results = test_basic_performance()
    test_scalability()
    
    print(f"\n{'=' * 60}")
    print("🎉 Benchmark completed successfully!")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    main()