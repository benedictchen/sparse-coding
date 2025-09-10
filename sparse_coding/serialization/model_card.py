"""
Model cards for sparse coding models.

Provides standardized documentation and metadata for trained models,
following ML model documentation best practices.
"""

import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
import warnings

try:
    from .. import __version__
except ImportError:
    __version__ = "2.5.0"


@dataclass
class ModelCard:
    """
    Standardized model documentation following model card best practices.
    
    Provides comprehensive metadata about model training, intended use,
    limitations, and performance characteristics.
    
    Attributes
    ----------
    model_name : str
        Name/identifier for the model
    model_version : str
        Version of the model
    library_version : str
        Version of sparse_coding library used
    created : str
        ISO timestamp of creation
    description : str
        Human-readable description of the model
    intended_use : dict
        Information about intended use cases and applications
    training_data : dict
        Information about training dataset
    model_architecture : dict
        Technical details about model architecture
    training_procedure : dict
        Details about training procedure and hyperparameters  
    evaluation : dict
        Performance metrics and evaluation results
    limitations : dict
        Known limitations and failure modes
    ethical_considerations : dict
        Ethical considerations and potential biases
    references : list
        Related papers, datasets, or other models
    contact : dict
        Contact information for model authors
    """
    model_name: str
    model_version: str = "1.0.0"
    library_version: str = __version__
    created: str = ""
    description: str = ""
    intended_use: Dict[str, Any] = None
    training_data: Dict[str, Any] = None
    model_architecture: Dict[str, Any] = None
    training_procedure: Dict[str, Any] = None
    evaluation: Dict[str, Any] = None
    limitations: Dict[str, Any] = None
    ethical_considerations: Dict[str, Any] = None
    references: List[str] = None
    contact: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if not self.created:
            self.created = datetime.now().isoformat()
        
        if self.intended_use is None:
            self.intended_use = {}
        
        if self.training_data is None:
            self.training_data = {}
        
        if self.model_architecture is None:
            self.model_architecture = {}
        
        if self.training_procedure is None:
            self.training_procedure = {}
        
        if self.evaluation is None:
            self.evaluation = {}
        
        if self.limitations is None:
            self.limitations = {}
        
        if self.ethical_considerations is None:
            self.ethical_considerations = {}
        
        if self.references is None:
            self.references = []
        
        if self.contact is None:
            self.contact = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model card to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert model card to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_yaml(self) -> str:
        """Convert model card to YAML string."""
        try:
            return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        except ImportError:
            warnings.warn("PyYAML not available. Using JSON format.")
            return self.to_json()
    
    def save(self, path: Union[str, Path], format: str = 'auto'):
        """
        Save model card to file.
        
        Parameters
        ----------
        path : str or Path
            Output file path
        format : str, default='auto'
            Output format ('json', 'yaml', 'auto')
        """
        path = Path(path)
        
        if format == 'auto':
            if path.suffix.lower() in ['.yaml', '.yml']:
                format = 'yaml'
            else:
                format = 'json'
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            if format == 'yaml':
                f.write(self.to_yaml())
            else:
                f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelCard':
        """Create model card from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ModelCard':
        """Create model card from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'ModelCard':
        """Create model card from YAML string."""
        try:
            data = yaml.safe_load(yaml_str)
            return cls.from_dict(data)
        except ImportError:
            raise ImportError("PyYAML required for YAML support")
    
    def add_training_info(self, config: Dict[str, Any], stats: Optional[Dict[str, Any]] = None):
        """
        Add training information to model card.
        
        Parameters
        ----------
        config : dict
            Training configuration (hyperparameters, etc.)
        stats : dict, optional
            Training statistics and metrics
        """
        self.training_procedure.update({
            'hyperparameters': config,
            'training_date': datetime.now().isoformat()
        })
        
        if stats:
            self.training_procedure['training_stats'] = stats
    
    def add_evaluation_results(self, metrics: Dict[str, Any], test_data_info: Optional[Dict[str, Any]] = None):
        """
        Add evaluation results to model card.
        
        Parameters
        ----------
        metrics : dict
            Evaluation metrics
        test_data_info : dict, optional
            Information about test dataset
        """
        self.evaluation.update({
            'metrics': metrics,
            'evaluation_date': datetime.now().isoformat()
        })
        
        if test_data_info:
            self.evaluation['test_data'] = test_data_info
    
    def add_model_architecture_info(self, learner, dataset_info: Optional[Dict[str, Any]] = None):
        """
        Add model architecture information from fitted learner.
        
        Parameters
        ----------
        learner : object
            Fitted sparse coding learner
        dataset_info : dict, optional
            Information about training dataset
        """
        # Extract architecture info
        arch_info = {
            'model_type': type(learner).__name__,
            'library': 'sparse_coding'
        }
        
        # Get learner-specific info
        if hasattr(learner, 'D') and learner.D is not None:
            arch_info['dictionary_shape'] = learner.D.shape
        
        if hasattr(learner, 'get_config'):
            arch_info.update(learner.get_config())
        
        self.model_architecture.update(arch_info)
        
        if dataset_info:
            self.training_data.update(dataset_info)


def create_model_card(
    model_name: str,
    learner,
    description: str = "",
    dataset_info: Optional[Dict[str, Any]] = None,
    evaluation_metrics: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ModelCard:
    """
    Create model card for sparse coding model.
    
    Parameters
    ----------
    model_name : str
        Name for the model
    learner : object
        Fitted sparse coding learner
    description : str, default=""
        Model description
    dataset_info : dict, optional
        Training dataset information
    evaluation_metrics : dict, optional
        Evaluation results
    **kwargs
        Additional model card fields
        
    Returns
    -------
    card : ModelCard
        Populated model card
        
    Examples
    --------
    >>> from sparse_coding import SparseCoder
    >>> from sparse_coding.serialization import create_model_card
    >>> 
    >>> # Train model
    >>> learner = SparseCoder(n_atoms=64)
    >>> learner.fit(X)
    >>> 
    >>> # Create model card
    >>> card = create_model_card(
    ...     "image_patches_64atoms",
    ...     learner,
    ...     description="Sparse dictionary for 8x8 image patches",
    ...     dataset_info={"source": "natural_images", "n_samples": 10000},
    ...     evaluation_metrics={"reconstruction_mse": 0.05, "sparsity": 0.85}
    ... )
    >>> 
    >>> # Save model card
    >>> card.save("model_card.yaml")
    """
    # Create base model card
    card = ModelCard(
        model_name=model_name,
        description=description,
        **kwargs
    )
    
    # Add architecture information
    card.add_model_architecture_info(learner, dataset_info)
    
    # Add evaluation results
    if evaluation_metrics:
        card.add_evaluation_results(evaluation_metrics)
    
    # Add default intended use and limitations
    if not card.intended_use:
        card.intended_use = {
            "primary_use": "Sparse representation learning and feature extraction",
            "out_of_scope_use": "This model should not be used for applications requiring real-time inference without proper validation",
            "applications": ["Feature learning", "Dimensionality reduction", "Signal denoising", "Representation learning"]
        }
    
    if not card.limitations:
        card.limitations = {
            "general": "Model performance depends heavily on training data distribution and hyperparameters",
            "computational": "Dictionary learning can be computationally expensive for large datasets",
            "data_requirements": "Requires sufficient data diversity for learning meaningful sparse representations"
        }
    
    # Add sparse coding references
    if not card.references:
        card.references = [
            "Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.",
            "Mairal, J., Bach, F., Ponce, J., & Sapiro, G. (2010). Online dictionary learning for sparse coding. ICML."
        ]
    
    return card


def load_model_card(path: Union[str, Path]) -> ModelCard:
    """
    Load model card from file.
    
    Parameters
    ----------
    path : str or Path
        Path to model card file
        
    Returns
    -------
    card : ModelCard
        Loaded model card
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model card not found: {path}")
    
    with open(path, 'r') as f:
        content = f.read()
    
    # Detect format
    if path.suffix.lower() in ['.yaml', '.yml']:
        return ModelCard.from_yaml(content)
    else:
        return ModelCard.from_json(content)


def generate_model_card_template(save_path: Optional[Union[str, Path]] = None) -> str:
    """
    Generate model card template with guidance comments.
    
    Parameters
    ----------
    save_path : str or Path, optional
        If provided, saves template to file
        
    Returns
    -------
    template : str
        Model card template in YAML format
    """
    template = """# Model Card for [Model Name]

model_name: "your_model_name"
model_version: "1.0.0"
description: |
  Brief description of your sparse coding model.
  What does it do? What was it trained on?

# Intended Use
intended_use:
  primary_use: "Sparse representation learning for [specific domain]"
  applications:
    - "Feature extraction"
    - "Signal denoising" 
    - "Dimensionality reduction"
  out_of_scope_use: |
    Do not use for applications requiring [specific constraints].
    Not suitable for [specific scenarios].

# Training Data
training_data:
  dataset_name: "Name of training dataset"
  dataset_size: 
    n_samples: 10000
    n_features: 256
  data_source: "Description of data source"
  preprocessing: |
    Describe any data preprocessing steps:
    - Normalization
    - Filtering
    - Augmentation
  splits:
    train: 80%
    validation: 10%
    test: 10%

# Model Architecture
model_architecture:
  model_type: "SparseCoder"  # or DictionaryLearner, etc.
  dictionary_size: 
    n_features: 256
    n_atoms: 64
  sparsity_constraint: "L1 penalty with lambda=0.1"
  solver: "FISTA"
  
# Training Procedure
training_procedure:
  hyperparameters:
    learning_rate: 0.01
    max_iterations: 1000
    batch_size: 256
    sparsity_penalty: 0.1
  training_time: "Approximately X hours on Y hardware"
  convergence_criteria: "Reconstruction error < 1e-6"

# Evaluation
evaluation:
  metrics:
    reconstruction_mse: 0.05
    sparsity_level: 0.85
    dictionary_coherence: 0.3
  test_data:
    dataset_name: "Test dataset name"
    n_samples: 2000
  
# Limitations
limitations:
  data_requirements: |
    Requires diverse training data representative of target distribution.
  computational_cost: |
    Dictionary learning scales O(n_features * n_atoms * n_iterations).
  hyperparameter_sensitivity: |
    Performance sensitive to sparsity penalty and solver parameters.

# Ethical Considerations
ethical_considerations:
  bias_assessment: |
    Model may exhibit bias present in training data.
  fairness: |
    Evaluate fairness across different subgroups in your application.
  privacy: |
    Consider privacy implications of learned representations.

# References
references:
  - "Olshausen & Field (1996). Emergence of simple-cell receptive field properties..."
  - "Your relevant papers or datasets"

# Contact
contact:
  name: "Your Name"
  email: "your.email@example.com"
  organization: "Your Organization"
"""
    
    if save_path:
        Path(save_path).write_text(template)
    
    return template


def validate_model_card(card: ModelCard) -> List[str]:
    """
    Validate model card completeness and consistency.
    
    Parameters
    ----------
    card : ModelCard
        Model card to validate
        
    Returns
    -------
    warnings : list
        List of validation warnings/suggestions
    """
    warnings_list = []
    
    # Check required fields
    if not card.model_name:
        warnings_list.append("model_name is required")
    
    if not card.description:
        warnings_list.append("description should provide model overview")
    
    # Check intended use
    if not card.intended_use.get('primary_use'):
        warnings_list.append("intended_use.primary_use should be specified")
    
    # Check training info
    if not card.training_data:
        warnings_list.append("training_data information recommended")
    
    if not card.model_architecture:
        warnings_list.append("model_architecture details recommended")
    
    # Check evaluation
    if not card.evaluation.get('metrics'):
        warnings_list.append("evaluation.metrics recommended for model assessment")
    
    # Check limitations
    if not card.limitations:
        warnings_list.append("limitations section recommended for responsible use")
    
    return warnings_list