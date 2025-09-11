#!/usr/bin/env python3
"""
Performance benchmarks for sparse coding implementations.

Compares our implementation against scikit-learn and measures:
- Encoding speed
- Dictionary learning speed  
- Memory usage
- Convergence quality
- Sparsity levels

Author: Benedict Chen
"""

import time
import sys
import warnings
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Optional imports with fallbacks
try:
    from sklearn.decomposition import SparseCoder as SKSparseCoder, DictionaryLearning
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available for comparison")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    warnings.warn("psutil not available for memory monitoring")

from sparse_coding import SparseCoder
from sparse_coding.sparse_coding_monitoring import CSVDump

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = {}
        self.csv_logger = CSVDump(str(self.results_dir / "benchmark_results.csv"))
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if HAS_PSUTIL:
            process = psutil.Process()
            return process.memory_info().rss / 1024**2
        return 0.0
    
    def generate_test_data(self, n_features: int, n_samples: int, 
                          noise_level: float = 0.1) -> np.ndarray:
        """Generate synthetic sparse coding test data."""
        np.random.seed(42)  # Reproducible results
        
        # Create sparse ground truth
        n_atoms = min(n_features * 2, 256)  # Overcomplete dictionary
        true_dict = np.random.randn(n_features, n_atoms)
        true_dict /= np.linalg.norm(true_dict, axis=0)
        
        # Sparse codes (10% non-zero)
        sparsity = 0.1
        codes = np.random.randn(n_atoms, n_samples) * 0.5
        mask = np.random.rand(n_atoms, n_samples) > sparsity
        codes[mask] = 0
        
        # Generate data with noise
        X = true_dict @ codes + noise_level * np.random.randn(n_features, n_samples)
        
        return X, true_dict, codes
    
    def benchmark_encoding_speed(self, data_sizes: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Benchmark encoding speed across different data sizes."""
        print("=== Encoding Speed Benchmark ===")
        results = {'our_times': [], 'sklearn_times': [], 'data_sizes': []}
        
        for n_features, n_samples in data_sizes:
            print(f"Testing {n_features}×{n_samples} data...")
            
            X, _, _ = self.generate_test_data(n_features, n_samples)
            n_atoms = min(n_features * 2, 256)
            
            # Our implementation
            start_time = time.time()
            sc = SparseCoder(n_atoms=n_atoms, mode='l1', lam=0.1, max_iter=100)
            sc.fit(X, n_steps=5)  # Quick dictionary learning
            codes = sc.encode(X)
            our_time = time.time() - start_time
            
            our_sparsity = (codes == 0).mean()
            our_reconstruction_error = np.linalg.norm(X - sc.decode(codes)) / np.linalg.norm(X)
            
            print(f"  Our impl: {our_time:.2f}s, sparsity: {our_sparsity:.2f}, error: {our_reconstruction_error:.4f}")
            
            # Scikit-learn comparison
            sklearn_time = None
            if HAS_SKLEARN:
                try:
                    start_time = time.time()
                    sk_sc = SKSparseCoder(n_components=n_atoms, alpha=0.1, max_iter=100)
                    # Fit dictionary first (sklearn requires this)
                    sk_dict = np.random.randn(n_features, n_atoms)
                    sk_dict /= np.linalg.norm(sk_dict, axis=0)
                    sk_sc.set_params(dictionary=sk_dict.T)  # sklearn expects atoms as rows
                    sk_codes = sk_sc.transform(X.T)  # sklearn expects samples as rows
                    sklearn_time = time.time() - start_time
                    
                    sk_sparsity = (sk_codes == 0).mean()
                    print(f"  Sklearn:  {sklearn_time:.2f}s, sparsity: {sk_sparsity:.2f}")
                    
                except Exception as e:
                    print(f"  Sklearn failed: {e}")
                    sklearn_time = None
            
            results['our_times'].append(our_time)
            results['sklearn_times'].append(sklearn_time)
            results['data_sizes'].append(f"{n_features}×{n_samples}")
            
            # Log to CSV
            self.csv_logger.log_scalar('test_type', 'encoding_speed')
            self.csv_logger.log_scalar('n_features', n_features)
            self.csv_logger.log_scalar('n_samples', n_samples)
            self.csv_logger.log_scalar('our_time', our_time)
            self.csv_logger.log_scalar('sklearn_time', sklearn_time or 0)
            self.csv_logger.log_scalar('our_sparsity', our_sparsity)
            self.csv_logger.log_scalar('reconstruction_error', our_reconstruction_error)
        
        return results
    
    def benchmark_dictionary_learning(self, n_features: int = 64, n_samples: int = 1000) -> Dict[str, Any]:
        """Benchmark dictionary learning convergence and speed."""
        print(f"\n=== Dictionary Learning Benchmark ({n_features}×{n_samples}) ===")
        
        X, true_dict, _ = self.generate_test_data(n_features, n_samples)
        n_atoms = 128
        
        results = {}
        
        # Our implementation
        print("Testing our implementation...")
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        sc = SparseCoder(n_atoms=n_atoms, mode='l1', lam=0.1, max_iter=200)
        sc.fit(X, n_steps=20)
        
        our_time = time.time() - start_time
        end_memory = self.get_memory_usage()
        memory_usage = end_memory - start_memory
        
        # Final encoding
        final_codes = sc.encode(X)
        final_reconstruction = sc.decode(final_codes)
        reconstruction_error = np.linalg.norm(X - final_reconstruction) / np.linalg.norm(X)
        sparsity = (final_codes == 0).mean()
        
        print(f"  Time: {our_time:.2f}s")
        print(f"  Memory: {memory_usage:.1f} MB")
        print(f"  Final reconstruction error: {reconstruction_error:.4f}")
        print(f"  Final sparsity: {sparsity:.2f}")
        
        results['our'] = {
            'time': our_time,
            'memory': memory_usage,
            'reconstruction_error': reconstruction_error,
            'sparsity': sparsity
        }
        
        # Scikit-learn comparison
        if HAS_SKLEARN:
            print("Testing scikit-learn DictionaryLearning...")
            try:
                start_time = time.time()
                sk_dl = DictionaryLearning(
                    n_components=n_atoms, 
                    alpha=0.1, 
                    max_iter=20,
                    transform_max_iter=200
                )
                sk_codes = sk_dl.fit_transform(X.T)  # sklearn expects samples as rows
                sklearn_time = time.time() - start_time
                
                sk_reconstruction = (sk_dl.components_.T @ sk_codes.T)
                sk_error = np.linalg.norm(X - sk_reconstruction) / np.linalg.norm(X)
                sk_sparsity = (sk_codes == 0).mean()
                
                print(f"  Time: {sklearn_time:.2f}s")
                print(f"  Reconstruction error: {sk_error:.4f}")
                print(f"  Sparsity: {sk_sparsity:.2f}")
                
                results['sklearn'] = {
                    'time': sklearn_time,
                    'reconstruction_error': sk_error,
                    'sparsity': sk_sparsity
                }
                
                print(f"\nPerformance ratio: {our_time/sklearn_time:.2f}x")
                
            except Exception as e:
                print(f"  Scikit-learn failed: {e}")
                results['sklearn'] = None
        
        return results
    
    def benchmark_algorithm_modes(self, n_features: int = 64, n_samples: int = 500) -> Dict[str, Any]:
        """Benchmark different algorithm modes in our implementation."""
        print(f"\n=== Algorithm Modes Benchmark ({n_features}×{n_samples}) ===")
        
        X, _, _ = self.generate_test_data(n_features, n_samples)
        n_atoms = 128
        modes = ['l1', 'paper', 'olshausen_pure', 'log']
        
        results = {}
        
        for mode in modes:
            print(f"Testing mode: {mode}")
            try:
                start_time = time.time()
                sc = SparseCoder(n_atoms=n_atoms, mode=mode, lam=0.05, max_iter=100)
                sc.fit(X, n_steps=10)
                codes = sc.encode(X[:, :100])  # Test subset for speed
                time_taken = time.time() - start_time
                
                reconstruction = sc.decode(codes)
                error = np.linalg.norm(X[:, :100] - reconstruction) / np.linalg.norm(X[:, :100])
                sparsity = (codes == 0).mean()
                
                results[mode] = {
                    'time': time_taken,
                    'reconstruction_error': error,
                    'sparsity': sparsity
                }
                
                print(f"  Time: {time_taken:.2f}s, Error: {error:.4f}, Sparsity: {sparsity:.2f}")
                
            except Exception as e:
                print(f"  Mode {mode} failed: {e}")
                results[mode] = None
        
        return results
    
    def plot_results(self, encoding_results: Dict[str, Any]):
        """Create performance visualization plots."""
        print("\n=== Creating Performance Plots ===")
        
        if not encoding_results['our_times']:
            print("No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Encoding speed comparison
        data_sizes = encoding_results['data_sizes']
        our_times = encoding_results['our_times']
        sklearn_times = [t for t in encoding_results['sklearn_times'] if t is not None]
        
        x_pos = np.arange(len(data_sizes))
        
        ax1.bar(x_pos - 0.2, our_times, 0.4, label='Our Implementation', alpha=0.7)
        if sklearn_times:
            ax1.bar(x_pos + 0.2, sklearn_times[:len(x_pos)], 0.4, label='Scikit-learn', alpha=0.7)
        
        ax1.set_xlabel('Data Size (features×samples)')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Encoding Speed Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(data_sizes, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance ratio
        if sklearn_times:
            ratios = [our/sk for our, sk in zip(our_times, sklearn_times[:len(our_times)]) if sk > 0]
            ax2.plot(range(len(ratios)), ratios, 'o-', linewidth=2, markersize=8)
            ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Equal Performance')
            ax2.set_xlabel('Test Case')
            ax2.set_ylabel('Performance Ratio (Our/Sklearn)')
            ax2.set_title('Relative Performance')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Scikit-learn\nNot Available', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to {self.results_dir / 'performance_comparison.png'}")
    
    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("🚀 Starting Comprehensive Performance Benchmark")
        print("="*60)
        
        # Test different data sizes
        data_sizes = [
            (32, 100),   # Small
            (64, 500),   # Medium
            (128, 1000), # Large
        ]
        
        # Run benchmarks
        encoding_results = self.benchmark_encoding_speed(data_sizes)
        dict_results = self.benchmark_dictionary_learning()
        mode_results = self.benchmark_algorithm_modes()
        
        # Store results
        self.results = {
            'encoding': encoding_results,
            'dictionary_learning': dict_results,
            'algorithm_modes': mode_results
        }
        
        # Create visualizations
        self.plot_results(encoding_results)
        
        # Summary
        print("\n" + "="*60)
        print("📊 BENCHMARK SUMMARY")
        print("="*60)
        
        if encoding_results['our_times']:
            avg_our_time = np.mean(encoding_results['our_times'])
            sklearn_times = [t for t in encoding_results['sklearn_times'] if t is not None]
            if sklearn_times:
                avg_sklearn_time = np.mean(sklearn_times)
                ratio = avg_our_time / avg_sklearn_time
                print(f"Average encoding time ratio (Our/Sklearn): {ratio:.2f}x")
            else:
                print(f"Average encoding time (Our): {avg_our_time:.2f}s")
        
        if 'our' in dict_results:
            our_dict = dict_results['our']
            print(f"Dictionary learning time: {our_dict['time']:.2f}s")
            print(f"Final reconstruction error: {our_dict['reconstruction_error']:.4f}")
            print(f"Memory usage: {our_dict['memory']:.1f} MB")
        
        print(f"\nResults saved to: {self.results_dir}")
        print(f"CSV log: {self.results_dir / 'benchmark_results.csv'}")
        
        return self.results

def main():
    """Run benchmark script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sparse Coding Performance Benchmark')
    parser.add_argument('--output-dir', default='benchmark_results', 
                       help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark (smaller data sizes)')
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark(args.output_dir)
    
    if args.quick:
        print("🏃 Running quick benchmark...")
        # Quick test with smaller data
        encoding_results = benchmark.benchmark_encoding_speed([(32, 100), (64, 200)])
        benchmark.plot_results(encoding_results)
    else:
        benchmark.run_full_benchmark()

if __name__ == '__main__':
    main()