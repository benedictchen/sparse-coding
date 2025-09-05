"""
Command-line interface for sparse coding.
"""

import argparse
import os
import sys
import glob
import json
import numpy as np
from PIL import Image

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

from . import SparseCoder
from .whitening import zero_phase_whiten

def _load_config(path):
    """Load configuration from YAML or JSON file."""
    if not path:
        return {}
    
    with open(path, "r") as f:
        if path.endswith((".yml", ".yaml")):
            if not HAS_YAML:
                raise SystemExit("pyyaml not installed. Run: pip install pyyaml")
            return yaml.safe_load(f)
        return json.load(f)

def _load_images(folder):
    """Load all images from a folder as grayscale arrays."""
    pattern = os.path.join(folder, "*")
    files = sorted([p for p in glob.glob(pattern) if os.path.isfile(p)])
    
    if not files:
        raise SystemExit(f"No images found in: {folder}")
    
    images = []
    for f in files:
        try:
            img = Image.open(f).convert("L")  # Grayscale
            arr = np.asarray(img, dtype=float) / 255.0
            images.append(arr)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
    
    if not images:
        raise SystemExit("Failed to load any valid images.")
    
    return images

def _sample_patches(images, patch_size, n_samples, rng):
    """Sample random patches from images."""
    if not images:
        raise ValueError("No images provided")
    
    H, W = images[0].shape
    if H < patch_size or W < patch_size:
        raise ValueError(f"Images too small for {patch_size}x{patch_size} patches")
    
    X = np.zeros((patch_size * patch_size, n_samples))
    
    for i in range(n_samples):
        # Random image
        img = images[rng.integers(0, len(images))]
        
        # Random location
        y = rng.integers(0, H - patch_size + 1)
        x = rng.integers(0, W - patch_size + 1)
        
        # Extract patch
        patch = img[y:y+patch_size, x:x+patch_size]
        X[:, i] = patch.reshape(-1)
    
    return X

def cmd_train(args):
    """Train sparse coding dictionary from images."""
    print(f"Training sparse coder (mode={args.mode}, seed={args.seed})")
    
    # Set random seed
    rng = np.random.default_rng(args.seed)
    
    # Load configuration
    cfg = _load_config(args.config)
    patch_size = int(cfg.get("patch_size", 16))
    n_atoms = int(cfg.get("n_atoms", 144))
    n_steps = int(cfg.get("n_steps", 50))
    lr = float(cfg.get("lr", 0.1))
    f0 = float(cfg.get("f0", 200.0))
    samples = int(cfg.get("samples", 50000))
    
    print(f"Config: {patch_size}x{patch_size} patches, {n_atoms} atoms, {samples} samples")
    
    # Load and whiten images
    images = _load_images(args.images)
    print(f"Loaded {len(images)} images")
    
    # Apply image-level whitening
    images_whitened = [zero_phase_whiten(img, f0=f0) for img in images]
    
    # Sample patches
    X = _sample_patches(images_whitened, patch_size, samples, rng)
    
    # Remove DC component from patches
    X = X - np.mean(X, axis=0, keepdims=True)
    
    print(f"Extracted patches: {X.shape}")
    
    # Train sparse coder
    coder = SparseCoder(n_atoms=n_atoms, mode=args.mode, seed=args.seed)
    coder.fit(X, n_steps=n_steps, lr=lr)
    
    # Save results
    os.makedirs(args.out, exist_ok=True)
    D_path = os.path.join(args.out, "D.npy")
    A_path = os.path.join(args.out, "A.npy")
    
    np.save(D_path, coder.D)
    
    # Encode all patches and save
    A = coder.encode(X)
    np.save(A_path, A)
    
    print(f"Saved dictionary: {D_path}")
    print(f"Saved codes: {A_path}")
    
    return 0

def cmd_encode(args):
    """Encode patches using existing dictionary."""
    print("Encoding patches...")
    
    # Load inputs
    X = np.load(args.patches)
    D = np.load(args.dictionary)
    
    print(f"Patches: {X.shape}, Dictionary: {D.shape}")
    
    # Create coder and set dictionary
    coder = SparseCoder(n_atoms=D.shape[1], mode=args.mode)
    coder.D = D.astype(float)
    
    # Encode
    A = coder.encode(X.astype(float))
    
    # Save codes
    np.save(args.out, A)
    print(f"Saved codes: {args.out}")
    
    return 0

def cmd_reconstruct(args):
    """Reconstruct patches from codes and dictionary.""" 
    print("Reconstructing patches...")
    
    # Load inputs
    A = np.load(args.codes)
    D = np.load(args.dictionary)
    
    print(f"Codes: {A.shape}, Dictionary: {D.shape}")
    
    # Reconstruct
    X_hat = D @ A
    
    # Save reconstruction
    np.save(args.out, X_hat)
    print(f"Saved reconstruction: {args.out}")
    
    return 0

def main():
    """Main CLI entry point."""
    ap = argparse.ArgumentParser(
        prog="sparse-coding",
        description="Research-faithful sparse coding with O&F paper-exact mode"
    )
    
    subparsers = ap.add_subparsers(dest="cmd", required=True, help="Commands")
    
    # Train command
    ap_train = subparsers.add_parser("train", help="Train dictionary from image folder")
    ap_train.add_argument("--images", required=True, help="Folder containing images")
    ap_train.add_argument("--config", help="YAML/JSON config file")
    ap_train.add_argument("--out", default="out", help="Output directory")
    ap_train.add_argument("--mode", default="paper", choices=["paper", "l1"], 
                         help="Sparse coding mode")
    ap_train.add_argument("--seed", type=int, default=0, help="Random seed")
    ap_train.set_defaults(func=cmd_train)
    
    # Encode command  
    ap_encode = subparsers.add_parser("encode", help="Encode patches with dictionary")
    ap_encode.add_argument("--dictionary", required=True, help="Dictionary file (D.npy)")
    ap_encode.add_argument("--patches", required=True, help="Patches file (X.npy, shape p×N)")
    ap_encode.add_argument("--out", default="A.npy", help="Output codes file")
    ap_encode.add_argument("--mode", default="paper", choices=["paper", "l1"])
    ap_encode.set_defaults(func=cmd_encode)
    
    # Reconstruct command
    ap_recon = subparsers.add_parser("reconstruct", help="Reconstruct from codes+dictionary")
    ap_recon.add_argument("--dictionary", required=True, help="Dictionary file (D.npy)")
    ap_recon.add_argument("--codes", required=True, help="Codes file (A.npy, shape K×N)")
    ap_recon.add_argument("--out", default="X_hat.npy", help="Output reconstruction file")
    ap_recon.set_defaults(func=cmd_reconstruct)
    
    # Parse and dispatch
    args = ap.parse_args()
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())