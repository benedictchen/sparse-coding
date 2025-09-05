"""
📋 Optimization
================

🔬 Research Foundation:
======================
Based on foundational sparse coding research:
- Olshausen, B.A. & Field, D.J. (1996). "Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images"
- Field, D.J. (1994). "What Is the Goal of Sensory Coding?"
- Lewicki, M.S. & Sejnowski, T.J. (2000). "Learning Overcomplete Representations"
🎯 ELI5 Summary:
This file is an important component in our AI research system! Like different organs 
in your body that work together to keep you healthy, this file has a specific job that 
helps the overall algorithm work correctly and efficiently.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

"""
"""
🏗️ Sparse Coding Optimization Utilities - ISTA/FISTA Mathematical Tools
======================================================================

🧠 ELI5 Explanation:
Think of sparse coding like organizing your photo collection. You want to represent each photo 
using just a few "basic building blocks" (dictionary atoms) instead of storing every pixel. 
The optimization utilities here are like smart tools that help you decide:

1. **Thresholding**: "Is this building block important enough to keep?" - If a coefficient 
   is too small, set it to zero (sparsity). It's like deciding if a tiny dot of color in 
   your photo is worth storing or can be ignored.

2. **Lipschitz Constants**: "How fast can I safely adjust my guesses?" - Like knowing how 
   hard you can press the gas pedal without losing control of your car. Too fast and you 
   overshoot the best solution.

3. **Line Search**: "What's the best step size to take?" - Like deciding how big steps to 
   take when walking to a destination in the dark - small steps are safer but slower.

The math ensures you find the sparsest representation (fewest building blocks) that still 
captures the essence of your data, just like Olshausen & Field discovered in natural images.

📚 Research Foundation:  
- Olshausen, B. & Field, D. (1996) "Emergence of simple-cell receptive field properties"
- Beck, A. & Teboulle, M. (2009) "A Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)"
- Fan, J. & Li, R. (2001) "Variable selection via nonconcave penalized likelihood (SCAD)"
- Daubechies, I. et al. (2004) "An iterative thresholding algorithm (ISTA)"

Key mathematical insight: Proximal operators solve: prox_λf(x) = argmin_z {½||z-x||² + λf(z)}
For L1 penalty: prox_λ||·||₁(x) = sign(x) × max(|x| - λ, 0) (soft thresholding)

🏗️ Optimization Components Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    SPARSE CODING OPTIMIZATION                   │
├─────────────────────────────────────────────────────────────────┤
│  Input Data → Thresholding → Sparsity Control → Output          │
│       ↓            ↓              ↓               ↓             │
│   [x₁,x₂,...] → [threshold] → [λ penalty] → [sparse_x]         │
│                                                                 │
│  Soft: sign(x) × max(|x|-λ, 0)   Hard: x × (|x| ≥ λ)          │
│  SCAD: Smooth clipping            Garrote: Non-negative shrink  │
│                                                                 │
│  Gradient Method → Lipschitz → Step Size → Convergence         │
│       ↓              ↓           ↓            ↓                │
│   ∇f(x)       →  L = λₘₐₓ(AᵀA) → α ≤ 1/L → x_{k+1}           │
│                                                                 │
│  Line Search → Armijo Condition → Backtracking → Safe Steps    │
│       ↓              ↓               ↓             ↓           │
│   f(x+αd) ≤ f(x) + c₁α∇f(x)ᵀd  → α = βα → Guaranteed Descent │
└─────────────────────────────────────────────────────────────────┘

🔧 Usage Examples:
```python
# Sparse signal recovery with soft thresholding (ISTA step)
noisy_coefficients = np.array([0.1, 2.5, -0.05, 3.2, 0.08])
threshold = 0.1
sparse_coeffs = soft_threshold(noisy_coefficients, threshold)
# Result: [0.0, 2.4, 0.0, 3.1, 0.0] - small values zeroed out

# Compute Lipschitz constant for safe gradient steps
dictionary = np.random.randn(64, 256)  # 64 atoms, 256 dimensions
L = compute_lipschitz_constant(dictionary)  
safe_step_size = 0.9 / L  # Ensure convergence

# Advanced SCAD thresholding for better sparsity patterns
scad_result = shrinkage_threshold(noisy_coefficients, 0.1, 'scad')
# SCAD provides continuous shrinkage, reducing bias for large coefficients
```

⚙️ Mathematical Foundations:
- **Soft Thresholding**: prox_λ||·||₁(x) = sign(x) ⊙ max(|x| - λ, 0)
- **SCAD Penalty**: λ|x| if |x| ≤ λ; (2aλ|x| - x² - λ²)/(2(a-1)) if λ < |x| ≤ aλ; λ²(a+1)/2 if |x| > aλ
- **Lipschitz Constant**: L = λₘₐₓ(AᵀA) ensures ||∇f(x) - ∇f(y)|| ≤ L||x - y||
- **Armijo Condition**: f(x + αd) ≤ f(x) + c₁α∇f(x)ᵀd guarantees sufficient decrease

💰 FUNDING APPEAL - PLEASE DONATE! 💰
=====================================
🌟 This sparse coding optimization research is made possible by Benedict Chen
   📧 Contact: benedict@benedictchen.com
   
💳 PLEASE DONATE! Your support keeps this research alive! 💳
   🔗 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   🔗 GitHub Sponsors: https://github.com/sponsors/benedictchen
   
☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!
(Start small, dream big! Every donation helps advance AI research! 😄)

💡 Why donate? This optimization module took months of research, implementing 
   cutting-edge algorithms from 1996-2009 papers with mathematical precision!
   Your support enables more breakthrough AI implementations! 🚀
"""

"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Your support makes advanced AI research accessible to everyone! 🚀

Made possible by Benedict Chen (benedict@benedictchen.com)
"""

import numpy as np
from typing import Callable
import warnings


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Soft thresholding operator (proximal operator for L1 norm)
    
    Parameters
    ----------
    x : array
        Input array
    threshold : float
        Threshold parameter
        
    Returns
    -------
    thresholded : array, same shape as input
        Soft thresholded values
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def hard_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Hard thresholding operator
    
    Parameters
    ----------
    x : array
        Input array
    threshold : float
        Threshold parameter
        
    Returns
    -------
    thresholded : array, same shape as input
        Hard thresholded values (set to 0 if |x| < threshold)
    """
    return x * (np.abs(x) >= threshold)


def shrinkage_threshold(x: np.ndarray, threshold: float, shrinkage_type: str = 'soft') -> np.ndarray:
    """
    Generalized shrinkage/thresholding operator
    
    Parameters
    ----------
    x : array
        Input array
    threshold : float
        Threshold parameter
    shrinkage_type : str
        Type of shrinkage: 'soft', 'hard', 'garrote', 'scad'
        
    Returns
    -------
    thresholded : array, same shape as input
        Thresholded values
    """
    if shrinkage_type == 'soft':
        return soft_threshold(x, threshold)
    elif shrinkage_type == 'hard':
        return hard_threshold(x, threshold)
    elif shrinkage_type == 'garrote':
        # Non-negative garrote
        return np.maximum(1 - threshold / np.maximum(np.abs(x), 1e-8), 0) * x
    elif shrinkage_type == 'scad':
        # SCAD (Smoothly Clipped Absolute Deviation)
        a = 3.7  # SCAD parameter
        abs_x = np.abs(x)
        
        result = np.zeros_like(x)
        
        # Region 1: |x| <= threshold
        mask1 = abs_x <= threshold
        result[mask1] = 0
        
        # Region 2: threshold < |x| <= a*threshold  
        mask2 = (abs_x > threshold) & (abs_x <= a * threshold)
        result[mask2] = np.sign(x[mask2]) * (abs_x[mask2] - threshold)
        
        # Region 3: |x| > a*threshold
        mask3 = abs_x > a * threshold
        result[mask3] = np.sign(x[mask3]) * (abs_x[mask3] * (a - 1) - threshold * a) / (a - 2)
        
        return result
    else:
        raise ValueError(f"Unknown shrinkage type: {shrinkage_type}")


def compute_lipschitz_constant(A: np.ndarray) -> float:
    """
    🔢 Compute Lipschitz Constant for Sparse Coding Optimization
    
    Computes the Lipschitz constant L for the gradient of f(x) = ½||Ax - b||² 
    which equals the largest eigenvalue of A^T A. This is crucial for determining
    safe step sizes in ISTA/FISTA algorithms to guarantee convergence.
    
    🧠 Mathematical Context:
    The Lipschitz constant ensures ||∇f(x) - ∇f(y)|| ≤ L||x - y||, meaning
    the gradient doesn't change too quickly. For ISTA: step size α ≤ 1/L ensures
    the algorithm converges to the global optimum of the convex problem.
    
    Parameters
    ----------
    A : array, shape (m, n)
        Dictionary matrix for sparse coding (atoms as columns)
        
    Returns
    -------
    L : float
        Lipschitz constant (largest eigenvalue of A^T A)
        Always positive, with minimum value 1e-12 for numerical stability
        
    Raises
    ------
    ValueError
        If input matrix has invalid dimensions or properties
        
    Notes
    -----
    Uses efficient algorithms based on matrix size:
    - Small matrices: Direct eigendecomposition  
    - Large matrices: Power iteration or matrix norm approximation
    - Handles numerical issues with complex eigenvalues robustly
    """
    # Input validation (FIXME solutions implemented)
    if not isinstance(A, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if A.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional")
    if A.size == 0:
        return 1e-12  # Avoid zero Lipschitz constant
    if np.allclose(A, 0):
        return 1e-12  # Zero matrix case
        
    # Check for ill-conditioned matrices
    if np.linalg.cond(A) > 1e12:
        warnings.warn("Matrix appears ill-conditioned (condition number > 1e12). "
                     "Lipschitz constant may be inaccurate.", UserWarning)
    
    m, n = A.shape
    
    # Efficient computation based on matrix size (FIXME solutions implemented)
    if max(m, n) > 1000:  # Large matrix case
        # Use power iteration for efficiency O(mn) per iteration instead of O(n³)
        if m <= n:
            # More columns than rows: compute largest eigenvalue of AA^T
            return _power_iteration_largest_eigenvalue(A @ A.T, max_iter=20, tol=1e-6)
        else:
            # More rows than columns: compute largest eigenvalue of A^T A
            return _power_iteration_largest_eigenvalue(A.T @ A, max_iter=20, tol=1e-6)
    else:
        # Small matrix case: use direct eigendecomposition
        if m <= n:
            # More columns than rows: compute eigenvalues of AA^T
            eigenvals = np.linalg.eigvals(A @ A.T)
        else:
            # More rows than columns: compute eigenvalues of A^T A
            eigenvals = np.linalg.eigvals(A.T @ A)
        
        # Handle numerical precision issues (FIXME solutions implemented)
        max_eigenval = np.max(np.real(eigenvals))
        
        # Warn if significant imaginary components (numerical errors)
        max_imag = np.max(np.abs(np.imag(eigenvals)))
        if max_imag > 1e-10:
            warnings.warn(f"Complex eigenvalues detected (max imaginary part: {max_imag:.2e}). "
                         f"Taking real part. Consider checking matrix conditioning.", UserWarning)
        
        # Ensure positive minimum for numerical stability
        return max(max_eigenval, 1e-12)


def _power_iteration_largest_eigenvalue(M: np.ndarray, max_iter: int = 20, tol: float = 1e-6) -> float:
    """
    🔄 Power Iteration for Largest Eigenvalue (Efficient Implementation)
    
    Implements the power iteration method to find the largest eigenvalue of a 
    symmetric positive semidefinite matrix. This is much more efficient than
    full eigendecomposition for large matrices: O(mn) vs O(n³).
    
    Parameters
    ----------
    M : array, shape (k, k)
        Symmetric matrix (typically A^T A or AA^T)
    max_iter : int, default=20
        Maximum number of iterations
    tol : float, default=1e-6
        Convergence tolerance for eigenvalue estimate
        
    Returns
    -------
    eigenval : float
        Largest eigenvalue of M
    """
    n = M.shape[0]
    
    # Start with random vector
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    eigenval_old = 0
    
    for i in range(max_iter):
        # Power iteration step
        Mv = M @ v
        eigenval = np.dot(v, Mv)  # Rayleigh quotient
        
        # Check convergence
        if abs(eigenval - eigenval_old) < tol:
            break
            
        # Normalize for next iteration
        v = Mv / np.linalg.norm(Mv)
        eigenval_old = eigenval
    
    return max(eigenval, 1e-12)  # Ensure positive minimum


def line_search_backtrack(f: Callable, grad_f: Callable, x: np.ndarray, 
                         direction: np.ndarray, alpha: float = 1.0,
                         beta: float = 0.5, c1: float = 1e-4,
                         max_iter: int = 20) -> float:
    """
    Backtracking line search with Armijo condition
    
    Parameters
    ----------
    f : callable
        Objective function
    grad_f : callable  
        Gradient function
    x : array
        Current point
    direction : array
        Search direction
    alpha : float
        Initial step size
    beta : float
        Backtracking parameter (0 < beta < 1)
    c1 : float
        Armijo parameter (0 < c1 < 1)
    max_iter : int
        Maximum number of backtracking steps
        
    Returns
    -------
    step_size : float
        Selected step size
    """
    f_x = f(x)
    grad_f_x = grad_f(x)
    directional_derivative = np.dot(grad_f_x, direction)
    
    for _ in range(max_iter):
        if f(x + alpha * direction) <= f_x + c1 * alpha * directional_derivative:
            return alpha
        alpha *= beta
    
    return alpha


# Export functions
__all__ = [
    'soft_threshold',
    'hard_threshold', 
    'shrinkage_threshold',
    'compute_lipschitz_constant',
    'line_search_backtrack'
]


if __name__ == "__main__":
    # Removed print spam: "\n...
    print("   📧 Contact: benedict@benedictchen.com")
    print()
    print("💰 PLEASE DONATE! Your support keeps this research alive! 💰")
    print("   🔗 💳 CLICK HERE TO DONATE VIA PAYPAL")
    print("   🔗 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS")
    print()
    print("   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!")
    print("   (Start small, dream big! Every donation helps! 😄)")
    print()
    # print("🏗️ Sparse Coding - Optimization Utilities Module")
    print("=" * 50)
    # Removed print spam: "...
    print("  • Soft and hard thresholding operators")
    print("  • Advanced shrinkage operators (SCAD, non-negative garrote)")
    print("  • Lipschitz constant computation for gradient methods")
    print("  • Backtracking line search with Armijo condition")
    print("")
    print("Optimization utilities module loaded.")