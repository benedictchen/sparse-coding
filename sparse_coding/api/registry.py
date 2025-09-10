"""
Plugin registry system for extensible sparse coding components.

Enables registration and discovery of penalties, solvers, and dictionary updaters
without import dependencies. Supports both decorator and direct registration.
"""

from typing import Dict, Any, Type, Callable, Union, Optional, List
import inspect
import warnings
from functools import wraps

# Global registry storage
_REGISTRY: Dict[str, Dict[str, Any]] = {
    "penalty": {},
    "solver": {}, 
    "dict_updater": {},
    "learner": {}
}

# Component type validation
VALID_KINDS = {"penalty", "solver", "dict_updater", "learner"}


def register(kind: str, name: str, *, override: bool = False):
    """
    Register a component in the plugin system.
    
    Can be used as decorator or called directly.
    
    Args:
        kind: Component type ('penalty', 'solver', 'dict_updater', 'learner')
        name: Unique name within the kind
        override: Whether to allow overriding existing registrations
        
    Returns:
        Decorator function or original class/function
        
    Examples:
        @register("penalty", "l1")
        class L1Penalty: ...
        
        @register("solver", "fista")
        class FISTA: ...
        
        # Direct registration
        register("penalty", "custom", MyCustomPenalty)
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"Invalid kind '{kind}'. Must be one of: {VALID_KINDS}")
    
    def decorator(cls_or_fn: Union[Type, Callable]) -> Union[Type, Callable]:
        # Check for conflicts
        if name in _REGISTRY[kind] and not override:
            existing = _REGISTRY[kind][name]
            if existing is not cls_or_fn:  # Allow re-registration of same object
                warnings.warn(
                    f"Overriding existing {kind} '{name}': {existing} -> {cls_or_fn}. "
                    f"Use override=True to suppress this warning."
                )
        
        # Validate component interface
        _validate_component(cls_or_fn, kind)
        
        # Add metadata
        if hasattr(cls_or_fn, '__dict__'):
            cls_or_fn._sparse_coding_registry = {
                'kind': kind,
                'name': name,
                'module': cls_or_fn.__module__,
                'qualname': getattr(cls_or_fn, '__qualname__', str(cls_or_fn))
            }
        
        # Register
        _REGISTRY[kind][name] = cls_or_fn
        return cls_or_fn
    
    return decorator


def get_registry(kind: str, name: str) -> Any:
    """
    Get registered component by kind and name.
    
    Args:
        kind: Component type
        name: Component name
        
    Returns:
        Registered component class or function
        
    Raises:
        KeyError: If component not found
        ValueError: If kind invalid
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"Invalid kind '{kind}'. Must be one of: {VALID_KINDS}")
    
    if name not in _REGISTRY[kind]:
        available = list(_REGISTRY[kind].keys())
        raise KeyError(
            f"No {kind} named '{name}' found. Available: {available}"
        )
    
    return _REGISTRY[kind][name]


def list_registered(kind: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
    """
    List all registered components.
    
    Args:
        kind: Specific kind to list, or None for all
        
    Returns:
        Dict mapping kinds to component names, or list of names for specific kind
    """
    if kind is None:
        return {k: list(v.keys()) for k, v in _REGISTRY.items()}
    
    if kind not in VALID_KINDS:
        raise ValueError(f"Invalid kind '{kind}'. Must be one of: {VALID_KINDS}")
    
    return list(_REGISTRY[kind].keys())


def create_from_config(config: Dict[str, Any]) -> Any:
    """
    Create component instance from configuration.
    
    Args:
        config: Configuration dict with 'kind', 'name', and optional 'params'
        
    Returns:
        Instantiated component
        
    Example:
        config = {
            'kind': 'penalty', 
            'name': 'l1',
            'params': {'lam': 0.1}
        }
        penalty = create_from_config(config)
    """
    required_keys = {'kind', 'name'}
    if not required_keys.issubset(config.keys()):
        missing = required_keys - config.keys()
        raise ValueError(f"Config missing required keys: {missing}")
    
    kind = config['kind']
    name = config['name']
    params = config.get('params', {})
    
    # Get component class
    component_cls = get_registry(kind, name)
    
    # Instantiate with parameters
    try:
        if inspect.isclass(component_cls):
            return component_cls(**params)
        else:
            # Function - call with params
            return component_cls(**params) if params else component_cls
    except TypeError as e:
        raise TypeError(
            f"Failed to instantiate {kind} '{name}' with params {params}: {e}"
        ) from e


def unregister(kind: str, name: str) -> bool:
    """
    Remove component from registry.
    
    Args:
        kind: Component type
        name: Component name
        
    Returns:
        True if removed, False if not found
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"Invalid kind '{kind}'. Must be one of: {VALID_KINDS}")
    
    if name in _REGISTRY[kind]:
        del _REGISTRY[kind][name]
        return True
    return False


def clear_registry(kind: Optional[str] = None) -> None:
    """
    Clear registry entries.
    
    Args:
        kind: Specific kind to clear, or None for all
    """
    if kind is None:
        for k in _REGISTRY:
            _REGISTRY[k].clear()
    else:
        if kind not in VALID_KINDS:
            raise ValueError(f"Invalid kind '{kind}'. Must be one of: {VALID_KINDS}")
        _REGISTRY[kind].clear()


def get_registry_info(kind: str, name: str) -> Dict[str, Any]:
    """
    Get metadata about registered component.
    
    Args:
        kind: Component type
        name: Component name
        
    Returns:
        Dict with component metadata
    """
    component = get_registry(kind, name)
    
    info = {
        'kind': kind,
        'name': name,
        'type': type(component).__name__,
        'module': getattr(component, '__module__', 'unknown'),
        'qualname': getattr(component, '__qualname__', str(component)),
        'is_class': inspect.isclass(component),
        'is_function': inspect.isfunction(component),
    }
    
    # Add custom registry metadata if available
    if hasattr(component, '_sparse_coding_registry'):
        info.update(component._sparse_coding_registry)
    
    # Add signature info
    try:
        if inspect.isclass(component):
            sig = inspect.signature(component.__init__)
            # Remove 'self' parameter
            params = {k: v for k, v in sig.parameters.items() if k != 'self'}
        else:
            sig = inspect.signature(component)
            params = dict(sig.parameters)
        
        info['parameters'] = {
            name: {
                'kind': str(param.kind),
                'default': param.default if param.default != param.empty else None,
                'annotation': str(param.annotation) if param.annotation != param.empty else None
            }
            for name, param in params.items()
        }
    except (ValueError, TypeError):
        info['parameters'] = {}
    
    return info


def _validate_component(component: Any, kind: str) -> None:
    """
    Validate component implements expected interface.
    
    Args:
        component: Component to validate
        kind: Expected component type
    """
    # Import protocols here to avoid circular imports
    from ..core.interfaces import Penalty, InferenceSolver, DictUpdater, Learner
    
    protocol_map = {
        'penalty': Penalty,
        'solver': InferenceSolver, 
        'dict_updater': DictUpdater,
        'learner': Learner
    }
    
    if kind not in protocol_map:
        return  # Skip validation for unknown kinds
    
    protocol = protocol_map[kind]
    
    # Check required methods exist (basic duck typing validation)
    required_methods = [
        name for name, obj in inspect.getmembers(protocol)
        if not name.startswith('_') and callable(obj)
    ]
    
    missing_methods = []
    for method_name in required_methods:
        if not hasattr(component, method_name):
            # Check if it's a class that might implement the method
            if inspect.isclass(component):
                # Look for method in class definition
                if not any(hasattr(cls, method_name) for cls in component.__mro__):
                    missing_methods.append(method_name)
            else:
                missing_methods.append(method_name)
    
    if missing_methods:
        warnings.warn(
            f"{kind.title()} '{component}' may not implement required methods: {missing_methods}. "
            f"This may cause runtime errors."
        )


# Convenience aliases
get = get_registry
create = create_from_config
list_all = list_registered