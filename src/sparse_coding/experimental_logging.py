"""
Structured JSON logging for sparse coding experiments.

Provides machine-readable logging for experimental tracking, debugging,
and performance analysis. Outputs structured JSON records suitable for
log analysis tools and experimental databases.

Format:
    Each log entry contains timestamp, event name, and arbitrary fields:
    {"ts": 1640995200.0, "event": "training_start", "n_atoms": 100, "lambda": 0.1}

Usage in Research:
    Essential for reproducible experiments and systematic parameter studies.
    Enables automated analysis of large-scale sparse coding experiments.
"""

import json, sys, time

def log(event: str, **fields):
    """
    Log a structured JSON event with timestamp and arbitrary fields.
    
    Args:
        event: Event name/type (e.g., "training_start", "convergence", "error")
        **fields: Additional key-value pairs to include in log record
        
    Output:
        Writes JSON record to stdout with automatic flushing for real-time logging
        
    Example:
        >>> log("training_start", n_atoms=100, lambda=0.1, dataset="natural_images")
        {"ts": 1640995200.0, "event": "training_start", "n_atoms": 100, "lambda": 0.1, "dataset": "natural_images"}
    """
    rec = {"ts": time.time(), "event": event}
    rec.update(fields)
    sys.stdout.write(json.dumps(rec) + "\n")
    sys.stdout.flush()
