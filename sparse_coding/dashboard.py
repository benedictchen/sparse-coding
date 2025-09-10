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
    
    def __init__(self, logdir: Optional[str] = None):
        """Initialize TensorBoard writer
        
        Args:
            logdir: Directory for TensorBoard logs
        """
        if logdir:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(logdir)
            print(f"TensorBoard logging to: {logdir}")
        else:
            self.writer = None
        
    def add_scalar(self, tag: str, val: float, step: int):
        """Log scalar value"""
        if self.writer:
            self.writer.add_scalar(tag, val, step)
    
    def add_image(self, tag: str, img: np.ndarray, step: int):
        """Log image"""
        if self.writer:
            self.writer.add_image(tag, img, step)
    
    def add_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log histogram of values"""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
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
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("time,step,metric,value\n")
    
    def log(self, step: int, metric: str, value: float):
        """Log metric value"""
        if not self.path:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{time.time()},{step},{metric},{value}\n")


class DashboardLogger:
    """Combined dashboard logging for dictionary learning"""
    
    def __init__(self, 
                 tensorboard_dir: Optional[str] = None,
                 csv_path: Optional[str] = None):
        """Initialize dashboard logger
        
        Args:
            tensorboard_dir: TensorBoard log directory
            csv_path: CSV log file path
        """
        self.tb = TB(tensorboard_dir)
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
    
    def close(self):
        """Close all loggers"""
        self.tb.close()