"""
Register default components in the plugin system.
"""

from ..api.registry import register


def register_default_components():
    """Register all default components."""
    
    # Import components locally to avoid circular imports
    from .penalties import L1Penalty, L2Penalty, ElasticNetPenalty
    from .solvers import FISTASolver, ISTASolver
    from .dict_updaters import MODUpdater, GradDUpdater
    
    # Register penalties
    register("penalty", "l1")(L1Penalty)
    register("penalty", "l2")(L2Penalty)
    register("penalty", "elastic_net")(ElasticNetPenalty)
    
    # Register solvers
    register("solver", "fista")(FISTASolver)
    register("solver", "ista")(ISTASolver)
    
    # Register dictionary updaters
    register("dict_updater", "mod")(MODUpdater)
    register("dict_updater", "grad_d")(GradDUpdater)