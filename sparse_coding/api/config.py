"""
Configuration management for sparse coding components.

Provides validation, loading, and saving of component configurations
with support for both JSON and YAML formats.
"""

import json
import warnings
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    warnings.warn("jsonschema not available. Config validation disabled.")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# Configuration schema for validation
CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "penalty": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "params": {"type": "object"}
            },
            "required": ["name"],
            "additionalProperties": False
        },
        "solver": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "params": {"type": "object"}
            },
            "required": ["name"],
            "additionalProperties": False
        },
        "dict_updater": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "params": {"type": "object"}
            },
            "required": ["name"],
            "additionalProperties": False
        },
        "learner": {
            "type": "object",
            "properties": {
                "n_atoms": {"type": "integer", "minimum": 1},
                "max_iterations": {"type": "integer", "minimum": 1},
                "tolerance": {"type": "number", "minimum": 0},
                "random_seed": {"type": ["integer", "null"]},
                "verbose": {"type": "boolean"}
            },
            "additionalProperties": True
        },
        "meta": {
            "type": "object",
            "properties": {
                "version": {"type": "string"},
                "created": {"type": "string"},
                "description": {"type": "string"}
            },
            "additionalProperties": True
        }
    },
    "additionalProperties": False
}


def validate_config(config: Dict[str, Any], strict: bool = True) -> Dict[str, List[str]]:
    """
    Validate configuration against schema.
    
    Args:
        config: Configuration to validate
        strict: Whether to raise on validation errors
        
    Returns:
        Dict with validation errors by section
        
    Raises:
        ValueError: If strict=True and validation fails
    """
    errors = {}
    
    if not HAS_JSONSCHEMA:
        if strict:
            warnings.warn("jsonschema not available. Skipping validation.")
        return errors
    
    try:
        jsonschema.validate(config, CONFIG_SCHEMA)
    except jsonschema.ValidationError as e:
        # Parse validation errors by section
        path_parts = list(e.absolute_path)
        section = path_parts[0] if path_parts else "root"
        
        if section not in errors:
            errors[section] = []
        errors[section].append(e.message)
        
        if strict:
            raise ValueError(f"Config validation failed: {e.message}") from e
    
    # Additional semantic validation
    _validate_component_references(config, errors, strict)
    
    return errors


def _validate_component_references(config: Dict[str, Any], errors: Dict[str, List[str]], strict: bool):
    """Validate that referenced components exist in registry."""
    from .registry import list_registered
    
    available = list_registered()
    
    for section in ['penalty', 'solver', 'dict_updater']:
        if section in config:
            component_name = config[section].get('name')
            if component_name and component_name not in available.get(section, []):
                error_msg = f"Unknown {section}: '{component_name}'. Available: {available.get(section, [])}"
                if section not in errors:
                    errors[section] = []
                errors[section].append(error_msg)
                
                if strict:
                    raise ValueError(error_msg)


def load_config(path: Union[str, Path], validate: bool = True) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        path: Path to config file (.json or .yaml/.yml)
        validate: Whether to validate loaded config
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If validation fails (when validate=True)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    # Determine format from extension
    suffix = path.suffix.lower()
    
    try:
        with open(path, 'r') as f:
            if suffix == '.json':
                config = json.load(f)
            elif suffix in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ImportError("PyYAML required for YAML config files")
                config = yaml.safe_load(f)
            else:
                # Try JSON first, then YAML
                content = f.read()
                f.seek(0)
                try:
                    config = json.loads(content)
                except json.JSONDecodeError:
                    if HAS_YAML:
                        config = yaml.safe_load(content)
                    else:
                        raise ValueError(f"Unknown config format: {suffix}")
    except Exception as e:
        raise ValueError(f"Failed to load config from {path}: {e}") from e
    
    if validate:
        validate_config(config, strict=True)
    
    return config


def save_config(config: Dict[str, Any], path: Union[str, Path], format: str = 'auto') -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        path: Output file path
        format: File format ('json', 'yaml', or 'auto' to infer from extension)
    """
    path = Path(path)
    
    # Infer format if auto
    if format == 'auto':
        suffix = path.suffix.lower()
        if suffix == '.json':
            format = 'json'
        elif suffix in ['.yaml', '.yml']:
            format = 'yaml'
        else:
            format = 'json'  # default
    
    # Validate before saving
    validate_config(config, strict=True)
    
    # Create directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w') as f:
            if format == 'json':
                json.dump(config, f, indent=2, sort_keys=True)
            elif format == 'yaml':
                if not HAS_YAML:
                    raise ImportError("PyYAML required for YAML output")
                yaml.dump(config, f, default_flow_style=False, sort_keys=True)
            else:
                raise ValueError(f"Unknown format: {format}")
    except Exception as e:
        raise ValueError(f"Failed to save config to {path}: {e}") from e


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration.
    
    Returns:
        Default config dictionary
    """
    from datetime import datetime
    try:
        from .. import __version__
    except ImportError:
        __version__ = "2.5.0"
    
    return {
        "penalty": {
            "name": "l1",
            "params": {"lam": 0.1}
        },
        "solver": {
            "name": "fista", 
            "params": {"max_iter": 200, "tol": 1e-6}
        },
        "dict_updater": {
            "name": "mod",
            "params": {"eps": 1e-6}
        },
        "learner": {
            "n_atoms": 144,
            "max_iterations": 30,
            "tolerance": 1e-6,
            "random_seed": None,
            "verbose": False
        },
        "meta": {
            "version": __version__,
            "created": datetime.now().isoformat(),
            "description": "Default sparse coding configuration"
        }
    }


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configurations with later ones taking precedence.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration
    """
    result = {}
    
    for config in configs:
        for section, values in config.items():
            if section not in result:
                result[section] = {}
            
            if isinstance(values, dict):
                result[section].update(values)
            else:
                result[section] = values
    
    return result


def config_diff(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two configurations and return differences.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        Dict showing differences
    """
    diff = {}
    
    # All keys from both configs
    all_keys = set(config1.keys()) | set(config2.keys())
    
    for key in all_keys:
        if key not in config1:
            diff[key] = {"status": "added", "value": config2[key]}
        elif key not in config2:
            diff[key] = {"status": "removed", "value": config1[key]}
        elif config1[key] != config2[key]:
            diff[key] = {
                "status": "changed",
                "old_value": config1[key],
                "new_value": config2[key]
            }
    
    return diff


def get_config_template(include_comments: bool = True) -> str:
    """
    Get configuration template with documentation.
    
    Args:
        include_comments: Whether to include explanatory comments
        
    Returns:
        Configuration template as string
    """
    template = create_default_config()
    
    if include_comments:
        # Add comments for documentation
        commented = {
            "# Penalty function configuration": None,
            "penalty": {
                "name": "l1",  # Available: l1, l2, elastic_net, cauchy
                "params": {
                    "lam": 0.1  # Regularization strength
                }
            },
            "\n# Sparse inference solver": None,
            "solver": {
                "name": "fista",  # Available: fista, ista, ncg
                "params": {
                    "max_iter": 200,  # Maximum iterations
                    "tol": 1e-6       # Convergence tolerance  
                }
            },
            "\n# Dictionary update method": None,
            "dict_updater": {
                "name": "mod",    # Available: mod, grad_d
                "params": {
                    "eps": 1e-6   # Regularization for matrix inversion
                }
            },
            "\n# Main learner parameters": None,
            "learner": template["learner"],
            "\n# Metadata": None,
            "meta": template["meta"]
        }
        
        # Convert to YAML-style with comments
        import yaml
        if HAS_YAML:
            return yaml.dump(commented, default_flow_style=False, allow_unicode=True)
    
    return json.dumps(template, indent=2)