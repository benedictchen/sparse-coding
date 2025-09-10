"""
Export utilities for sparse coding models.

Provides export to standard formats like ONNX, sklearn pipelines,
and other interoperability formats for deployment.
"""

import numpy as np
import warnings
from typing import Optional, Dict, Any, Union, Tuple
from pathlib import Path

import onnx
import onnxruntime as ort
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from ..core.array import ArrayLike, ensure_array


def export_to_onnx(
    learner,
    output_path: Union[str, Path],
    input_shape: Optional[Tuple[int, ...]] = None,
    opset_version: int = 11
) -> bool:
    """
    Export sparse coding model to ONNX format.
    
    Creates ONNX model for sparse encoding (inference only).
    The exported model performs matrix multiplication D.T @ x
    followed by soft thresholding for L1 penalty.
    
    Parameters
    ----------
    learner : object
        Fitted sparse coding learner with dictionary
    output_path : str or Path
        Path for output ONNX model
    input_shape : tuple, optional
        Input shape for ONNX model. If None, inferred from dictionary.
    opset_version : int, default=11
        ONNX opset version
        
    Returns
    -------
    success : bool
        True if export succeeded
        
    Examples
    --------
    >>> from sparse_coding import SparseCoder
    >>> from sparse_coding.serialization import export_to_onnx
    >>> 
    >>> # Train model
    >>> learner = SparseCoder(n_atoms=64)
    >>> learner.fit(X)
    >>> 
    >>> # Export to ONNX
    >>> export_to_onnx(learner, "sparse_coder.onnx", input_shape=(1, 256))
    """
    
    # Get dictionary
    if hasattr(learner, 'D') and learner.D is not None:
        dictionary = learner.D
    elif hasattr(learner, 'dictionary') and learner.dictionary is not None:
        dictionary = learner.dictionary
    else:
        raise ValueError("Learner has no fitted dictionary")
    
    dictionary = ensure_array(dictionary)
    n_features, n_atoms = dictionary.shape
    
    if input_shape is None:
        input_shape = (1, n_features)
    
    # Get sparsity parameter
    lam = getattr(learner, 'lam', 0.1)
    
    try:
        # Create ONNX graph manually (simplified approach)
        from onnx import helper, TensorProto, mapping
        
        # Define input
        input_tensor = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape
        )
        
        # Define dictionary as initializer
        dictionary_tensor = helper.make_tensor(
            'dictionary', TensorProto.FLOAT,
            dictionary.shape, dictionary.flatten().astype(np.float32)
        )
        
        # Define lambda (threshold) as initializer
        lambda_tensor = helper.make_tensor(
            'lambda', TensorProto.FLOAT, [], [float(lam)]
        )
        
        # Define output
        output_tensor = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [input_shape[0], n_atoms]
        )
        
        # Create nodes
        # Node 1: Matrix multiplication (input @ dictionary)
        matmul_node = helper.make_node(
            'MatMul',
            inputs=['input', 'dictionary'],
            outputs=['matmul_out'],
            name='sparse_encoding_matmul'
        )
        
        # Node 2: True soft thresholding implementation
        # soft_thresh(x, lambda) = sign(x) * max(|x| - lambda, 0)
        
        # Extract threshold parameter (lambda) - default L1 penalty strength
        threshold_value = getattr(learner, 'lam', 0.1)  # Default threshold
        if hasattr(learner, '_penalty') and hasattr(learner._penalty, 'lam'):
            threshold_value = learner._penalty.lam
        
        # Create threshold constant
        threshold_tensor = helper.make_tensor(
            name='threshold',
            data_type=onnx.TensorProto.FLOAT,
            dims=[],
            vals=[float(threshold_value)]
        )
        
        # Soft thresholding nodes
        # Step 1: Compute absolute value |x|
        abs_node = helper.make_node(
            'Abs',
            inputs=['matmul_out'],
            outputs=['abs_out'],
            name='absolute_value'
        )
        
        # Step 2: Subtract threshold: |x| - lambda
        sub_node = helper.make_node(
            'Sub',
            inputs=['abs_out', 'threshold'],
            outputs=['sub_out'],
            name='subtract_threshold'
        )
        
        # Step 3: Apply max(|x| - lambda, 0)
        zero_tensor = helper.make_tensor(
            name='zero',
            data_type=onnx.TensorProto.FLOAT,
            dims=[],
            vals=[0.0]
        )
        
        max_node = helper.make_node(
            'Max',
            inputs=['sub_out', 'zero'],
            outputs=['max_out'],
            name='max_with_zero'
        )
        
        # Step 4: Compute sign(x)
        sign_node = helper.make_node(
            'Sign',
            inputs=['matmul_out'],
            outputs=['sign_out'],
            name='compute_sign'
        )
        
        # Step 5: Final multiplication: sign(x) * max(|x| - lambda, 0)
        final_mul_node = helper.make_node(
            'Mul',
            inputs=['sign_out', 'max_out'],
            outputs=['output'],
            name='soft_thresholding'
        )
        
        # Create graph with all soft-thresholding nodes
        graph = helper.make_graph(
            [matmul_node, abs_node, sub_node, max_node, sign_node, final_mul_node],
            'sparse_coding_model',
            [input_tensor],
            [output_tensor],
            [dictionary_tensor, threshold_tensor, zero_tensor]
        )
        
        # Create model
        model = helper.make_model(graph)
        model.opset_import[0].version = opset_version
        
        # Add metadata
        model.metadata_props.append(
            helper.make_attribute("model_type", "sparse_coding")
        )
        model.metadata_props.append(
            helper.make_attribute("n_atoms", n_atoms)
        )
        model.metadata_props.append(
            helper.make_attribute("sparsity_param", float(lam))
        )
        
        # Validate and save
        onnx.checker.check_model(model)
        onnx.save(model, str(output_path))
        
        print(f"✅ ONNX model exported to {output_path}")
        print(f"   Input shape: {input_shape}")
        print(f"   Dictionary: {dictionary.shape}")
        print(f"   Sparsity: {lam}")
        
        return True
        
    except Exception as e:
        warnings.warn(f"ONNX export failed: {e}")
        return False


def test_onnx_model(
    onnx_path: Union[str, Path],
    test_input: ArrayLike,
    original_learner=None
) -> Dict[str, Any]:
    """
    Test exported ONNX model and compare with original.
    
    Parameters
    ----------
    onnx_path : str or Path
        Path to ONNX model file
    test_input : ArrayLike, shape (batch_size, n_features)
        Test input data
    original_learner : object, optional
        Original learner for comparison
        
    Returns
    -------
    results : dict
        Test results with outputs and comparisons
    """
    
    # Load ONNX model
    session = ort.InferenceSession(str(onnx_path))
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    test_input_array = ensure_array(test_input).astype(np.float32)
    onnx_output = session.run([output_name], {input_name: test_input_array})[0]
    
    results = {
        'onnx_output': onnx_output,
        'input_shape': test_input_array.shape,
        'output_shape': onnx_output.shape
    }
    
    # Compare with original if available
    if original_learner is not None:
        try:
            original_output = original_learner.encode(test_input_array.T).T
            mse = np.mean((onnx_output - original_output)**2)
            results['comparison'] = {
                'original_output': original_output,
                'mse_difference': float(mse),
                'max_abs_difference': float(np.max(np.abs(onnx_output - original_output)))
            }
        except Exception as e:
            results['comparison_error'] = str(e)
    
    return results


class SparseCodingSklearnWrapper(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper for sparse coding models.
    
    Provides fit/transform interface compatible with sklearn pipelines
    while maintaining sparse coding functionality.
    """
    
    def __init__(self, sparse_coder=None, **kwargs):
        self.sparse_coder = sparse_coder
        self.kwargs = kwargs
        self.fitted_ = False
        
    def fit(self, X, y=None):
        """
        Fit sparse coding model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency
            
        Returns
        -------
        self : SparseCodingSklearnWrapper
            Fitted transformer
        """
        if self.sparse_coder is None:
            from ..sparse_coder import SparseCoder
            self.sparse_coder = SparseCoder(**self.kwargs)
        
        # SparseCoder expects (n_features, n_samples)
        X_array = ensure_array(X)
        self.sparse_coder.fit(X_array.T)
        self.fitted_ = True
        
        return self
    
    def transform(self, X):
        """
        Transform data to sparse codes.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_sparse : array-like, shape (n_samples, n_atoms)
            Sparse codes
        """
        if not self.fitted_:
            raise ValueError("Must call fit() before transform()")
        
        X_array = ensure_array(X)
        # SparseCoder encode expects (n_features, n_samples) and returns (n_atoms, n_samples)
        codes = self.sparse_coder.encode(X_array.T)
        return codes.T  # Return (n_samples, n_atoms)
    
    def inverse_transform(self, X_sparse):
        """
        Reconstruct data from sparse codes.
        
        Parameters
        ----------
        X_sparse : array-like, shape (n_samples, n_atoms)
            Sparse codes
            
        Returns
        -------
        X_reconstructed : array-like, shape (n_samples, n_features)
            Reconstructed data
        """
        if not self.fitted_:
            raise ValueError("Must call fit() before inverse_transform()")
        
        X_sparse_array = ensure_array(X_sparse)
        # Reconstruction: D @ codes
        reconstruction = self.sparse_coder.D @ X_sparse_array.T
        return reconstruction.T
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if not self.fitted_:
            raise ValueError("Must call fit() before get_feature_names_out()")
        
        n_atoms = self.sparse_coder.D.shape[1]
        return [f"sparse_code_{i}" for i in range(n_atoms)]
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {'sparse_coder': self.sparse_coder}
        params.update(self.kwargs)
        return params
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            if parameter == 'sparse_coder':
                self.sparse_coder = value
            else:
                self.kwargs[parameter] = value
        return self


def export_to_sklearn_pipeline(
    learner,
    include_preprocessing: bool = True,
    include_postprocessing: bool = False
) -> Pipeline:
    """
    Export sparse coding model as sklearn pipeline.
    
    Creates sklearn Pipeline with preprocessing steps and sparse coding
    transformation for easy integration with sklearn workflows.
    
    Parameters
    ----------
    learner : object
        Fitted sparse coding learner
    include_preprocessing : bool, default=True
        Whether to include standard preprocessing (scaling)
    include_postprocessing : bool, default=False
        Whether to include postprocessing steps
        
    Returns
    -------
    pipeline : sklearn.Pipeline
        Complete processing pipeline
        
    Examples
    --------
    >>> from sparse_coding import SparseCoder
    >>> from sparse_coding.serialization import export_to_sklearn_pipeline
    >>> 
    >>> # Train model
    >>> learner = SparseCoder(n_atoms=64)
    >>> learner.fit(X)
    >>> 
    >>> # Export as pipeline
    >>> pipeline = export_to_sklearn_pipeline(learner)
    >>> 
    >>> # Use in sklearn workflow
    >>> sparse_codes = pipeline.transform(X_test)
    """
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Create wrapper
    wrapper = SparseCodingSklearnWrapper(sparse_coder=learner)
    wrapper.fitted_ = True  # Already fitted
    
    steps = []
    
    # Add preprocessing
    if include_preprocessing:
        steps.append(('scaler', StandardScaler()))
    
    # Add sparse coding
    steps.append(('sparse_coding', wrapper))
    
    # Add postprocessing
    if include_postprocessing:
        # Optional: PCA on sparse codes for further dimensionality reduction
        steps.append(('pca', PCA(n_components=0.95)))
    
    pipeline = Pipeline(steps)
    
    print(f"✅ Sklearn pipeline created with {len(steps)} steps:")
    for name, step in steps:
        print(f"   - {name}: {type(step).__name__}")
    
    return pipeline


def export_dictionary_as_numpy(
    learner,
    output_path: Union[str, Path]
) -> bool:
    """
    Export dictionary as simple numpy array.
    
    Parameters
    ----------
    learner : object
        Fitted sparse coding learner
    output_path : str or Path
        Output path for .npy file
        
    Returns
    -------
    success : bool
        True if export succeeded
    """
    try:
        # Get dictionary
        if hasattr(learner, 'D') and learner.D is not None:
            dictionary = learner.D
        elif hasattr(learner, 'dictionary') and learner.dictionary is not None:
            dictionary = learner.dictionary
        else:
            raise ValueError("Learner has no fitted dictionary")
        
        dictionary_array = ensure_array(dictionary)
        
        # Save with metadata
        np.save(str(output_path), dictionary_array)
        
        print(f"✅ Dictionary exported to {output_path}")
        print(f"   Shape: {dictionary_array.shape}")
        print(f"   Dtype: {dictionary_array.dtype}")
        
        return True
        
    except Exception as e:
        warnings.warn(f"Dictionary export failed: {e}")
        return False


def create_deployment_package(
    learner,
    package_dir: Union[str, Path],
    formats: list = ['numpy', 'sklearn'],
    include_examples: bool = True
) -> Dict[str, bool]:
    """
    Create complete deployment package with multiple formats.
    
    Parameters
    ----------
    learner : object
        Fitted sparse coding learner
    package_dir : str or Path
        Directory for deployment package
    formats : list, default=['numpy', 'sklearn']
        Export formats to include ('numpy', 'sklearn', 'onnx')
    include_examples : bool, default=True
        Whether to include usage examples
        
    Returns
    -------
    results : dict
        Success status for each export format
    """
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Export dictionary as numpy
    if 'numpy' in formats:
        numpy_path = package_dir / 'dictionary.npy'
        results['numpy'] = export_dictionary_as_numpy(learner, numpy_path)
    
    # Export sklearn pipeline
    if 'sklearn' in formats:
        try:
            pipeline = export_to_sklearn_pipeline(learner)
            
            # Save using joblib
            import joblib
            pipeline_path = package_dir / 'sklearn_pipeline.joblib'
            joblib.dump(pipeline, pipeline_path)
            results['sklearn'] = True
            print(f"✅ Sklearn pipeline saved to {pipeline_path}")
                
        except Exception as e:
            results['sklearn'] = False
            print(f"❌ Sklearn export failed: {e}")
    
    # Export ONNX model
    if 'onnx' in formats:
        onnx_path = package_dir / 'sparse_coder.onnx'
        results['onnx'] = export_to_onnx(learner, onnx_path)
    
    # Create README
    if include_examples:
        readme_content = _create_deployment_readme(learner, results)
        readme_path = package_dir / 'README.md'
        readme_path.write_text(readme_content)
        print(f"✅ README created at {readme_path}")
    
    return results


def _create_deployment_readme(learner, export_results: Dict[str, bool]) -> str:
    """Create README for deployment package."""
    
    # Get model info
    if hasattr(learner, 'D') and learner.D is not None:
        dictionary_shape = learner.D.shape
    elif hasattr(learner, 'dictionary') and learner.dictionary is not None:
        dictionary_shape = learner.dictionary.shape
    else:
        dictionary_shape = "Unknown"
    
    readme = f"""# Sparse Coding Model Deployment Package

This package contains a trained sparse coding model exported in multiple formats for deployment.

## Model Information

- **Dictionary Shape**: {dictionary_shape}
- **Model Type**: {type(learner).__name__}
- **Library**: sparse_coding v2.5.0

## Available Formats

"""
    
    if export_results.get('numpy'):
        readme += """### NumPy Dictionary (`dictionary.npy`)

Pure NumPy array containing the learned dictionary.

```python
import numpy as np

# Load dictionary
D = np.load('dictionary.npy')

# Sparse encoding (simplified FISTA)
def sparse_encode(x, D, lam=0.1, max_iter=100):
    # Implement FISTA algorithm
    # This is a simplified version
    codes = np.linalg.lstsq(D, x, rcond=None)[0]
    return np.sign(codes) * np.maximum(np.abs(codes) - lam, 0)

# Usage
codes = sparse_encode(test_signal, D)
reconstruction = D @ codes
```

"""
    
    if export_results.get('sklearn'):
        readme += """### Scikit-learn Pipeline (`sklearn_pipeline.joblib`)

Complete sklearn pipeline with preprocessing and sparse coding.

```python
import joblib

# Load pipeline
pipeline = joblib.load('sklearn_pipeline.joblib')

# Transform data
sparse_codes = pipeline.transform(X_test)

# Inverse transform (reconstruction)
reconstructed = pipeline.named_steps['sparse_coding'].inverse_transform(sparse_codes)
```

"""
    
    if export_results.get('onnx'):
        readme += """### ONNX Model (`sparse_coder.onnx`)

ONNX format for cross-platform deployment.

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('sparse_coder.onnx')

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Prepare input (batch_size, n_features)
test_input = np.random.randn(1, n_features).astype(np.float32)

# Get sparse codes
sparse_codes = session.run([output_name], {input_name: test_input})[0]
```

"""
    
    readme += """## Performance Considerations

- **Memory**: Dictionary size determines memory usage
- **Computation**: Encoding complexity is O(n_features * n_atoms * n_iterations)
- **Sparsity**: Higher sparsity parameters reduce computation but may affect quality

## Contact

For questions about this model or deployment issues, please contact the model authors.
"""
    
    return readme