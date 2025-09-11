"""
TensorBoard Dashboard and Logging Support for Sparse Coding

Provides TensorBoard integration for monitoring training progress,
visualizing dictionary atoms, and tracking convergence metrics.
"""

import time
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path


class TB:
    """TensorBoard logging wrapper"""
    
    def __init__(self, log_dir: Optional[str] = None, logdir: Optional[str] = None):
        """Initialize TensorBoard writer
        
        Args:
            log_dir: Directory for TensorBoard logs (preferred parameter name)
            logdir: Directory for TensorBoard logs (legacy parameter name)
        """
        # Support both parameter names for backwards compatibility
        log_directory = log_dir or logdir
        
        if log_directory:
            # Import tensorboard - requires explicit installation
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_directory)
            print(f"TensorBoard logging to: {log_directory}")
        else:
            self.writer = None
        
    def add_scalar(self, tag: str, val: float, step: int):
        """Log scalar value"""
        if self.writer:
            self.writer.add_scalar(tag, val, step)
    
    def log_scalar(self, tag: str, val: float, step: int):
        """Log scalar value (alias for add_scalar)"""
        self.add_scalar(tag, val, step)
    
    def add_image(self, tag: str, img: np.ndarray, step: int):
        """Log image"""
        if self.writer:
            self.writer.add_image(tag, img, step)
    
    def add_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log histogram of values"""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log histogram of values (alias for add_histogram)"""
        self.add_histogram(tag, values, step)
    
    def close(self):
        """Close writer"""
        if self.writer:
            self.writer.close()


class CSVDump:
    """CSV logging for metrics"""
    
    def __init__(self, path: Optional[str] = None):
        """Initialize CSV logger
        
        Args:
            path: Path to CSV file
        """
        self.path = path
        self.headers_written = False
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, step: Optional[int] = None, metric: Optional[str] = None, value: Optional[float] = None, **kwargs):
        """Log metric value(s)
        
        Supports both individual logging: log(step, metric, value)
        And batch logging: log(**metrics_dict)
        """
        if not self.path:
            return
            
        # Handle batch logging with **kwargs
        if step is None and metric is None and value is None and kwargs:
            # Write headers if not done yet
            if not self.headers_written:
                with open(self.path, "w", encoding="utf-8") as f:
                    headers = ['timestamp'] + sorted(kwargs.keys())
                    f.write(",".join(headers) + "\n")
                self.headers_written = True
            
            # Write metrics row
            with open(self.path, "a", encoding="utf-8") as f:
                timestamp = time.time()
                values = [str(timestamp)] + [str(kwargs[key]) for key in sorted(kwargs.keys())]
                f.write(",".join(values) + "\n")
        
        # Handle individual metric logging
        elif step is not None and metric is not None and value is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"{time.time()},{step},{metric},{value}\n")
        else:
            raise ValueError("Either provide (step, metric, value) or **kwargs for batch logging")
    
    def close(self):
        """Close CSV logger (for compatibility with test interface)"""
        pass


class DashboardLogger:
    """Combined dashboard logging for dictionary learning"""
    
    def __init__(self, 
                 tensorboard_dir: Optional[str] = None,
                 csv_path: Optional[str] = None,
                 tb_log_dir: Optional[str] = None):
        """Initialize dashboard logger
        
        Args:
            tensorboard_dir: TensorBoard log directory (legacy parameter)
            csv_path: CSV log file path
            tb_log_dir: TensorBoard log directory (preferred parameter)
        """
        # Support both parameter names for backwards compatibility
        tb_dir = tb_log_dir or tensorboard_dir
        self.tb = TB(log_dir=tb_dir)
        self.csv = CSVDump(csv_path)
        self.step = 0
    
    def log_training_metrics(self, metrics: Dict[str, float]):
        """Log training metrics"""
        for metric, value in metrics.items():
            self.tb.add_scalar(f"training/{metric}", value, self.step)
            self.csv.log(self.step, metric, value)
    
    def log_dictionary_atoms(self, dictionary: np.ndarray, patch_size: tuple):
        """Visualize dictionary atoms as images"""
        if not self.tb.writer:
            return
            
        n_atoms = dictionary.shape[1]
        atoms = dictionary.T.reshape(n_atoms, *patch_size)
        
        # Normalize atoms for visualization
        atoms_normalized = []
        for atom in atoms:
            atom_norm = (atom - atom.min()) / (atom.max() - atom.min() + 1e-8)
            atoms_normalized.append(atom_norm)
        
        # Create grid of atoms
        grid_size = int(np.ceil(np.sqrt(n_atoms)))
        atom_grid = np.zeros((grid_size * patch_size[0], grid_size * patch_size[1]))
        
        for i, atom in enumerate(atoms_normalized):
            row = i // grid_size
            col = i % grid_size
            if row < grid_size and col < grid_size:
                start_row = row * patch_size[0]
                end_row = start_row + patch_size[0]
                start_col = col * patch_size[1]
                end_col = start_col + patch_size[1]
                atom_grid[start_row:end_row, start_col:end_col] = atom
        
        # Log to TensorBoard
        self.tb.add_image("dictionary/atoms", atom_grid[None, :, :], self.step)
    
    def log_sparsity_histogram(self, codes: np.ndarray):
        """Log histogram of sparse code magnitudes"""
        if self.tb.writer:
            nonzero_codes = codes[np.abs(codes) > 1e-6]
            if len(nonzero_codes) > 0:
                self.tb.add_histogram("codes/nonzero_magnitudes", nonzero_codes, self.step)
    
    def log_convergence_analysis(self, history: Dict[str, list]):
        """Log convergence analysis"""
        for metric, values in history.items():
            if len(values) > 1:
                # Compute convergence rate
                recent_values = values[-10:]  # Last 10 values
                if len(recent_values) > 1:
                    slope = (recent_values[-1] - recent_values[0]) / len(recent_values)
                    self.tb.add_scalar(f"convergence/{metric}_slope", slope, self.step)
    
    def step_forward(self):
        """Advance step counter"""
        self.step += 1
    
    def log_training_step(self, step: int, sparse_codes: Optional[np.ndarray] = None, 
                         dictionary_atoms: Optional[np.ndarray] = None, **metrics):
        """Log comprehensive training step with metrics and visualizations
        
        Args:
            step: Training step number
            sparse_codes: Sparse codes for histogram logging
            dictionary_atoms: Dictionary atoms for visualization
            **metrics: Additional scalar metrics to log
        """
        # Log scalar metrics to both TB and CSV
        scalar_metrics = {k: v for k, v in metrics.items() 
                         if isinstance(v, (int, float)) and k not in ['sparse_codes', 'dictionary_atoms']}
        
        # Log to TensorBoard
        for metric, value in scalar_metrics.items():
            self.tb.add_scalar(f"training/{metric}", value, step)
        
        # Log to CSV
        if scalar_metrics:
            self.csv.log(**scalar_metrics)
        
        # Log sparse codes histogram if provided
        if sparse_codes is not None:
            self.log_sparsity_histogram(sparse_codes)
        
        # Log dictionary atoms if provided
        if dictionary_atoms is not None:
            # Assume square patches for visualization
            patch_size = int(np.sqrt(dictionary_atoms.shape[0]))
            if patch_size * patch_size == dictionary_atoms.shape[0]:
                self.log_dictionary_atoms(dictionary_atoms, (patch_size, patch_size))
    
    def close(self):
        """Close all loggers"""
        self.tb.close()