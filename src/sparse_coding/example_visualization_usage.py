#!/usr/bin/env python3
"""
Example: Using the Enhanced Visualization Module
===============================================

This example demonstrates how to use the new comprehensive visualization
capabilities extracted from SparseCoder into the VisualizationMixin.

The visualization module provides:
- Enhanced dictionary visualization with more options
- Comprehensive training curve analysis
- Sparse code pattern analysis
- Reconstruction quality assessment
- Dictionary property analysis
- Research-ready visualization exports

Author: Benedict Chen (benedict@benedictchen.com)
"""

import numpy as np
try:
    from .sparse_coder import SparseCoder
except ImportError:
    try:
        from sparse_coder import SparseCoder
    except ImportError:
        from ..sparse_coder import SparseCoder

def demo_enhanced_visualizations():
    """Demonstrate the enhanced visualization capabilities"""
    
    print("🎨 Enhanced Sparse Coding Visualization Demo")
    print("=" * 50)
    
    # Create a sparse coder
    sc = SparseCoder(
        n_components=64,  # Good number for visualization
        patch_size=(8, 8),
        sparsity_penalty=0.1
    )
    
    # The SparseCoder now automatically includes all visualization methods
    # from the VisualizationMixin through multiple inheritance
    
    print(f"✓ SparseCoder created with {sc.n_components} components")
    print(f"✓ Available visualization methods: {len([m for m in dir(sc) if 'visualize' in m or 'plot' in m])}")
    
    # 1. ENHANCED DICTIONARY VISUALIZATION
    print("\n🖼️  Enhanced Dictionary Visualization Features:")
    print("   • Custom titles and color schemes")
    print("   • Save to file capability")  
    print("   • Flexible element count display")
    print("   • Automatic normalization options")
    
    # Example: Visualize dictionary with custom options
    # sc.visualize_dictionary(
    #     figsize=(20, 20),
    #     max_elements=64,
    #     title="My Custom Dictionary",
    #     save_path="./my_dictionary.png",
    #     colormap="viridis"
    # )
    
    # 2. COMPREHENSIVE TRAINING ANALYSIS
    print("\n📊 Comprehensive Training Analysis:")
    print("   • Reconstruction error with trend lines")
    print("   • Sparsity evolution tracking") 
    print("   • Convergence rate analysis")
    print("   • Training statistics summary")
    
    # Add some dummy training history for demonstration
    sc.training_history = {
        'reconstruction_error': np.exp(-np.linspace(0, 3, 20)) + 0.01,
        'sparsity': 20 - np.linspace(0, 15, 20)
    }
    
    # Example: Plot enhanced training curves
    # sc.plot_training_curves(
    #     figsize=(15, 5),
    #     save_path="./training_analysis.png",
    #     show_statistics=True
    # )
    
    # 3. SPARSE CODE ANALYSIS
    print("\n🔍 Sparse Code Pattern Analysis:")
    print("   • Coefficient distribution heatmaps")
    print("   • Sparsity level histograms")
    print("   • Feature usage frequency analysis")
    print("   • Magnitude distribution studies")
    
    # Generate example sparse codes
    n_samples = 100
    sparse_coeffs = np.random.randn(n_samples, sc.n_components) * 0.3
    # Make them actually sparse
    sparse_coeffs[np.abs(sparse_coeffs) < 0.2] = 0
    
    print(f"✓ Generated {n_samples} sparse codes with {np.mean(np.sum(np.abs(sparse_coeffs) > 0, axis=1)):.1f} avg active elements")
    
    # Example: Analyze sparse codes
    # sc.visualize_sparse_codes(
    #     sparse_coeffs,
    #     figsize=(14, 10), 
    #     max_samples=50,
    #     save_path="./sparse_codes_analysis.png"
    # )
    
    # 4. RECONSTRUCTION QUALITY ASSESSMENT  
    print("\n🔧 Reconstruction Quality Assessment:")
    print("   • Side-by-side original vs reconstructed comparison")
    print("   • Error magnitude visualization")
    print("   • Best/worst/random example selection")
    print("   • Detailed reconstruction statistics")
    
    # Generate example patches and reconstructions
    dummy_patches = np.random.randn(n_samples, sc.patch_size[0] * sc.patch_size[1])
    
    # Example: Assess reconstruction quality
    # sc.visualize_reconstruction_quality(
    #     dummy_patches,
    #     sparse_coeffs,
    #     n_examples=8,
    #     figsize=(16, 6),
    #     save_path="./reconstruction_quality.png"
    # )
    
    # 5. DICTIONARY PROPERTY ANALYSIS
    print("\n⚙️  Dictionary Property Analysis:")
    print("   • Gram matrix visualization")
    print("   • Element norm distributions") 
    print("   • Coherence analysis")
    print("   • Condition number assessment")
    print("   • Singular value analysis")
    print("   • Feature correlation networks")
    
    # Example: Comprehensive dictionary analysis
    # sc.plot_dictionary_properties(
    #     figsize=(16, 12),
    #     save_path="./dictionary_properties.png"
    # )
    
    # 6. RESEARCH DOCUMENTATION
    print("\n📋 Research Documentation Features:")
    print("   • Comprehensive analysis reports")
    print("   • Dictionary export for external analysis")
    print("   • Batch visualization generation")
    
    # Example: Generate complete research report
    # sc.create_visualization_report(save_dir="./research_report/")
    
    # Example: Export dictionary for research
    # sc.export_dictionary_for_research(save_path="./dictionary_data.npz")
    
    # 7. BIOLOGICAL ANALYSIS
    print("\n🧠 Biological Plausibility Analysis:")
    print("   • Orientation selectivity measurement")
    print("   • Edge detector identification") 
    print("   • V1 similarity assessment")
    
    # This is automatically included in dictionary analysis
    properties = sc._analyze_dictionary_properties()
    print(f"✓ Dictionary analysis found {len(properties)} quantitative properties")
    
    # 8. COMPARATIVE ANALYSIS
    print("\n🔬 Comparative Analysis Features:")
    print("   • Multi-dictionary comparison")
    print("   • Algorithm performance comparison")
    print("   • Parameter sensitivity analysis")
    
    # Example: Compare dictionaries (would need multiple trained dictionaries)
    # other_dicts = [dict1, dict2, dict3]  
    # labels = ["Method A", "Method B", "Method C"]
    # sc.compare_dictionaries(other_dicts, labels, save_path="./comparison.png")
    
    print("\n🎉 Demo completed!")
    print("   All visualization methods handle matplotlib gracefully")
    print("   Methods print informative warnings if matplotlib unavailable")
    print("   Enhanced functionality maintains backward compatibility")
    print("   New features enable comprehensive research analysis")

def show_backward_compatibility():
    """Show that old code still works with enhanced features"""
    
    print("\n🔄 Backward Compatibility Demo")
    print("=" * 35)
    
    sc = SparseCoder(n_components=16, patch_size=(4, 4))
    
    # Old method calls still work exactly as before
    print("✓ Old method signatures still supported:")
    
    # Original calls (will show enhanced behavior)
    sc.visualize_dictionary()  # Now has more options but same basic call
    
    sc.training_history = {'reconstruction_error': [0.5, 0.3], 'sparsity': [10, 8]}
    sc.plot_training_curves()  # Enhanced with more analysis
    
    print("✅ All original code continues to work!")
    print("   Users get enhanced features automatically")
    print("   No breaking changes to existing workflows")

if __name__ == "__main__":
    demo_enhanced_visualizations()
    show_backward_compatibility()