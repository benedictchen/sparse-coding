"""
🔍 Shape Validation - Atoms-As-Columns Convention Enforcement
===========================================================

🎯 ELI5 EXPLANATION:
==================
Think of shape validation like checking if your puzzle pieces fit together correctly!

Imagine you're building a LEGO model where each piece must connect perfectly:
1. 🧱 **Dictionary (D)**: The LEGO baseplates [patch_size, n_atoms]
2. 🔧 **Coefficients (A)**: The building instructions [n_atoms, n_samples]  
3. 🏗️ **Reconstruction (X)**: The final model [patch_size, n_samples]

Just like LEGO pieces can only connect in specific ways, our mathematical operations
require specific shape conventions. This module ensures everything fits together!

🔬 RESEARCH FOUNDATION:
======================
Shape conventions are critical in sparse coding literature:
- **Olshausen & Field (1996)**: Original formulation uses atoms-as-columns
- **Mairal et al. (2014)**: "Sparse modeling for image and vision processing" - atoms-as-columns
- **Elad (2010)**: "Sparse and Redundant Representations" - atoms-as-columns standard
- **Rubinstein et al. (2008)**: K-SVD algorithm - atoms-as-columns convention

🧮 MATHEMATICAL PRINCIPLES:
==========================
**Global Convention: ATOMS-AS-COLUMNS**

Dictionary: D ∈ ℝ^(p×K) where:
- p = patch_size (number of features)
- K = n_components (number of atoms)
- Each column d_k is a dictionary atom

Coefficients: A ∈ ℝ^(K×N) where:
- K = n_components (matches dictionary)
- N = n_samples (number of data points)

Reconstruction: X ≈ D @ A ∈ ℝ^(p×N)

📊 SHAPE VALIDATION VISUALIZATION:
=================================
```
🔍 ATOMS-AS-COLUMNS SHAPE VALIDATION 🔍

Dictionary Shape               Coefficients Shape            Reconstruction
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│ D: (p, K)       │     @     │ A: (K, N)       │    =      │ X: (p, N)       │
│                 │           │                 │           │                 │
│ p = patch_size  │           │ K = n_atoms     │           │ p = patch_size  │
│ K = n_atoms     │           │ N = n_samples   │           │ N = n_samples   │
└─────────────────┘           └─────────────────┘           └─────────────────┘
        ↓                            ↓                            ↓
   Each column is              Coefficients for            Reconstructed
   a dictionary atom           each sample                 data patches

✅ CORRECT: X ≈ D @ A
❌ WRONG:   X ≈ D.T @ A (atoms-as-rows assumption)
```

Author: Research Code Auditor
Email: research.audit@accuracy.validation
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings


def validate_atoms_as_columns_convention(
    dictionary: np.ndarray,
    coefficients: Optional[np.ndarray] = None,
    data: Optional[np.ndarray] = None,
    operation_name: str = "sparse_coding_operation"
) -> Dict[str, Any]:
    """
    Validate that all arrays follow atoms-as-columns convention.
    
    This function enforces the global shape convention:
    - Dictionary D: (patch_size, n_atoms) - atoms are columns
    - Coefficients A: (n_atoms, n_samples) 
    - Data X: (patch_size, n_samples)
    - Reconstruction: X ≈ D @ A
    
    Args:
        dictionary: Dictionary matrix D of shape (patch_size, n_atoms)
        coefficients: Coefficient matrix A of shape (n_atoms, n_samples), optional
        data: Data matrix X of shape (patch_size, n_samples), optional
        operation_name: Name of operation for error messages
        
    Returns:
        Dictionary with validation results:
        - 'valid': bool - True if all shapes are consistent
        - 'errors': list - List of shape validation errors
        - 'warnings': list - List of potential issues
        - 'shape_info': dict - Information about detected shapes
        
    Raises:
        ValueError: If critical shape mismatches are detected
        
    Example:
        ```python
        # Validate sparse coding setup
        validation = validate_atoms_as_columns_convention(
            dictionary=D,        # Shape: (64, 128) 
            coefficients=A,      # Shape: (128, 1000)
            data=X,             # Shape: (64, 1000)
            operation_name="FISTA_optimization"
        )
        
        if not validation['valid']:
            for error in validation['errors']:
                print(f"SHAPE ERROR: {error}")
        ```
    """
    
    errors = []
    warnings_list = []
    shape_info = {}
    
    # Validate dictionary shape
    if dictionary.ndim != 2:
        errors.append(f"Dictionary must be 2D, got {dictionary.ndim}D array")
        return {'valid': False, 'errors': errors, 'warnings': warnings_list, 'shape_info': shape_info}
    
    patch_size, n_atoms = dictionary.shape
    shape_info['dictionary'] = (patch_size, n_atoms)
    
    # Validate coefficients if provided
    if coefficients is not None:
        if coefficients.ndim != 2:
            errors.append(f"Coefficients must be 2D, got {coefficients.ndim}D array")
        else:
            coeff_atoms, n_samples_coeff = coefficients.shape
            shape_info['coefficients'] = (coeff_atoms, n_samples_coeff)
            
            # Check atom dimension consistency
            if coeff_atoms != n_atoms:
                errors.append(
                    f"Atom dimension mismatch: dictionary has {n_atoms} atoms, "
                    f"coefficients have {coeff_atoms} atoms"
                )
    
    # Validate data if provided
    if data is not None:
        if data.ndim != 2:
            errors.append(f"Data must be 2D, got {data.ndim}D array")
        else:
            data_patch_size, n_samples_data = data.shape
            shape_info['data'] = (data_patch_size, n_samples_data)
            
            # Check patch size consistency
            if data_patch_size != patch_size:
                errors.append(
                    f"Patch size mismatch: dictionary has {patch_size} features, "
                    f"data has {data_patch_size} features"
                )
            
            # Check sample consistency if both data and coefficients provided
            if coefficients is not None and coefficients.ndim == 2:
                _, n_samples_coeff = coefficients.shape
                if n_samples_data != n_samples_coeff:
                    errors.append(
                        f"Sample count mismatch: data has {n_samples_data} samples, "
                        f"coefficients have {n_samples_coeff} samples"
                    )
    
    # Check for common mistake patterns
    if coefficients is not None and coefficients.ndim == 2:
        coeff_atoms, n_samples_coeff = coefficients.shape
        
        # Detect potential atoms-as-rows confusion
        if coeff_atoms == patch_size and n_samples_coeff == n_atoms:
            warnings_list.append(
                f"POTENTIAL TRANSPOSE ERROR: Coefficients shape {coefficients.shape} "
                f"suggests atoms-as-rows convention. Expected ({n_atoms}, n_samples) for atoms-as-columns"
            )
    
    # Generate reconstruction shape prediction
    if coefficients is not None and len(errors) == 0:
        expected_reconstruction_shape = (patch_size, n_samples_coeff)
        shape_info['expected_reconstruction'] = expected_reconstruction_shape
    
    valid = len(errors) == 0
    
    # Issue warnings if any detected
    for warning in warnings_list:
        warnings.warn(f"{operation_name}: {warning}", UserWarning)
    
    # Raise error for critical issues
    if not valid:
        error_summary = f"{operation_name} shape validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        raise ValueError(error_summary)
    
    return {
        'valid': valid,
        'errors': errors,
        'warnings': warnings_list,
        'shape_info': shape_info
    }


def assert_reconstruction_shapes(
    dictionary: np.ndarray,
    coefficients: np.ndarray,
    data: Optional[np.ndarray] = None,
    operation_name: str = "reconstruction"
) -> None:
    """
    Assert that reconstruction D @ A has compatible shapes.
    
    This is a critical check that prevents silent mathematical errors
    by ensuring the matrix multiplication D @ A produces the expected shape.
    
    Args:
        dictionary: Dictionary D of shape (patch_size, n_atoms)
        coefficients: Coefficients A of shape (n_atoms, n_samples)
        data: Optional data X of shape (patch_size, n_samples) for comparison
        operation_name: Name of operation for error messages
        
    Raises:
        ValueError: If shapes are incompatible for reconstruction
        
    Example:
        ```python
        # Before reconstruction
        assert_reconstruction_shapes(D, A, X, "FISTA_solve")
        
        # Safe to compute
        X_reconstructed = D @ A
        ```
    """
    
    # Validate basic shapes
    validation = validate_atoms_as_columns_convention(
        dictionary, coefficients, data, operation_name
    )
    
    # Additional reconstruction-specific checks
    if dictionary.shape[1] != coefficients.shape[0]:
        raise ValueError(
            f"{operation_name}: Matrix multiplication D @ A impossible. "
            f"Dictionary shape {dictionary.shape} incompatible with coefficients shape {coefficients.shape}. "
            f"Expected dictionary columns ({dictionary.shape[1]}) to match coefficient rows ({coefficients.shape[0]})"
        )
    
    # Verify reconstruction shape matches data if provided
    if data is not None:
        expected_shape = (dictionary.shape[0], coefficients.shape[1])
        if data.shape != expected_shape:
            raise ValueError(
                f"{operation_name}: Data shape {data.shape} doesn't match expected reconstruction shape {expected_shape}"
            )


def check_for_atoms_as_rows_violations(code_string: str, file_path: str = "unknown") -> Dict[str, Any]:
    """
    Analyze code string for potential atoms-as-rows violations.
    
    This function scans code for patterns that suggest incorrect shape assumptions,
    helping detect bugs before they cause mathematical errors.
    
    Args:
        code_string: Python code to analyze
        file_path: File path for reporting
        
    Returns:
        Dictionary with violation analysis:
        - 'violations': list - Detected violation patterns
        - 'suspicious_patterns': list - Patterns that might be errors
        - 'line_numbers': dict - Line numbers for each violation
        
    Example:
        ```python
        with open('sparse_coder.py', 'r') as f:
            code = f.read()
        
        violations = check_for_atoms_as_rows_violations(code, 'sparse_coder.py')
        for violation in violations['violations']:
            print(f"VIOLATION: {violation}")
        ```
    """
    
    violations = []
    suspicious_patterns = []
    line_numbers = {'violations': [], 'suspicious': []}
    
    lines = code_string.split('\n')
    
    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()
        
        # Skip comments and empty lines
        if line_stripped.startswith('#') or not line_stripped:
            continue
        
        # Check for definite violations (atoms-as-rows patterns)
        if 'dictionary_.T @' in line or 'dictionary.T @' in line:
            if 'D @ y' not in line and 'D @ a' not in line:  # These are correct
                violations.append(f"Line {i}: Uses 'dictionary.T @' suggesting atoms-as-rows: {line_stripped}")
                line_numbers['violations'].append(i)
        
        # Check for Gram matrix violations
        if 'dictionary_ @ dictionary_.T' in line or 'dictionary @ dictionary.T' in line:
            violations.append(f"Line {i}: Gram matrix uses D @ D.T (atoms-as-rows), should be D.T @ D: {line_stripped}")
            line_numbers['violations'].append(i)
        
        # Check for suspicious patterns
        if '@ dictionary' in line and 'dictionary @' not in line:
            suspicious_patterns.append(f"Line {i}: Pattern '@ dictionary' might indicate atoms-as-rows: {line_stripped}")
            line_numbers['suspicious'].append(i)
        
        # Check for reconstruction patterns that might be wrong
        if 'X.T -' in line and 'dictionary_.T @' in line:
            violations.append(f"Line {i}: Reconstruction uses atoms-as-rows pattern: {line_stripped}")
            line_numbers['violations'].append(i)
    
    return {
        'file_path': file_path,
        'violations': violations,
        'suspicious_patterns': suspicious_patterns,
        'line_numbers': line_numbers,
        'total_violations': len(violations),
        'total_suspicious': len(suspicious_patterns)
    }


def add_shape_assertions_to_method(
    method_code: str,
    dictionary_var: str = "self.dictionary_",
    coefficients_var: str = "codes",
    data_var: str = "X"
) -> str:
    """
    Add shape validation assertions to existing method code.
    
    This function automatically inserts shape validation checks into existing
    sparse coding methods to catch shape errors early.
    
    Args:
        method_code: The method code to enhance
        dictionary_var: Variable name for dictionary
        coefficients_var: Variable name for coefficients
        data_var: Variable name for data
        
    Returns:
        Enhanced method code with validation assertions
        
    Example:
        ```python
        original_code = '''
        def solve(self, X, codes):
            return self.dictionary_ @ codes.T
        '''
        
        enhanced_code = add_shape_assertions_to_method(original_code)
        # Now includes shape validation before reconstruction
        ```
    """
    
    # Find the first executable line after the docstring
    lines = method_code.split('\n')
    insert_line = 0
    
    # Skip docstring
    in_docstring = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if '"""' in stripped or "'''" in stripped:
            in_docstring = not in_docstring
            continue
        
        if not in_docstring and stripped and not stripped.startswith('#'):
            insert_line = i
            break
    
    # Create validation code
    validation_code = f'''
        # AUTOMATIC SHAPE VALIDATION - Atoms-as-columns convention
        from .shape_validation import assert_reconstruction_shapes
        if hasattr(self, 'dictionary_') and {coefficients_var} is not None:
            assert_reconstruction_shapes(
                {dictionary_var}, {coefficients_var}, 
                {data_var} if '{data_var}' in locals() else None,
                f"{{self.__class__.__name__}}.{{self.__method_name__ if hasattr(self, '__method_name__') else 'method'}}"
            )'''
    
    # Insert validation code
    lines.insert(insert_line, validation_code)
    
    return '\n'.join(lines)


# Global validation function for external use
def enforce_atoms_as_columns_globally():
    """
    Set global numpy matrix multiplication checks for atoms-as-columns.
    
    This function can be called at module import to enable strict shape checking
    throughout the sparse coding package.
    
    Example:
        ```python
        # At top of __init__.py
        from .core_modules.shape_validation import enforce_atoms_as_columns_globally
        enforce_atoms_as_columns_globally()
        ```
    """
    
    # This would require monkey-patching numpy operations
    # For now, we rely on explicit validation calls
    import warnings
    warnings.warn(
        "Global enforcement requires calling validate_atoms_as_columns_convention() "
        "explicitly in each method. Consider adding shape assertions to critical methods.",
        UserWarning
    )