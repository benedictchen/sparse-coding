import os, json, argparse, numpy as np, yaml
from pathlib import Path
from PIL import Image
from .sparse_coder import SparseCoder
from .data_preprocessing_whitening import zero_phase_whiten
from .reproducible_sparse_coding import set_deterministic
from .experimental_logging import log
from .sparse_coding_configuration import TrainingConfig, make_metadata
from .streaming_sparse_coding import encode_stream

def _load_cfg(path):
    if not path: return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

def _load_images(img_dir, f0=200.0):
    imgs = []
    for root, _, files in os.walk(img_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                im = Image.open(Path(root) / f).convert("L")
                arr = np.asarray(im, dtype=float)
                arr = zero_phase_whiten(arr, f0=f0)
                imgs.append(arr)
    if not imgs:
        raise ValueError("No images found in directory")
    return imgs

def _sample_patches(images, patch_size, n):
    H, W = images[0].shape
    ps = patch_size
    X = np.zeros((ps*ps, n))
    rng = np.random.default_rng(0)
    for i in range(n):
        im = images[rng.integers(0, len(images))]
        y = rng.integers(0, H-ps+1); x = rng.integers(0, W-ps+1)
        X[:, i] = im[y:y+ps, x:x+ps].reshape(-1)
    X -= X.mean(axis=0, keepdims=True)
    return X

def cmd_train(args):
    if args.deterministic: set_deterministic(args.seed)
    raw = _load_cfg(args.config)
    cfg = TrainingConfig(**raw) if raw else TrainingConfig()
    images = _load_images(args.images, f0=cfg.f0)
    X = _sample_patches(images, cfg.patch_size, cfg.samples)
    lam = None
    if cfg.lam_sigma is not None:
        lam = float(cfg.lam_sigma * np.std(X))
    coder = SparseCoder(n_atoms=cfg.n_atoms, mode=args.mode, seed=args.seed, lam=lam, max_iter=200, tol=1e-6)
    log("train_start", images=args.images, mode=args.mode, seed=args.seed)
    coder.fit(X, n_steps=cfg.steps, lr=cfg.lr)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "D.npy", coder.D)
    A = coder.encode(X); np.save(output_dir / "A.npy", A)
    meta = make_metadata(cfg, coder.D.shape, A.shape, {"mode": args.mode})
    with open(output_dir / "METADATA.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log("train_done", out=args.out)

def cmd_train_patches(args):
    if args.deterministic: set_deterministic(args.seed)
    raw = _load_cfg(args.config)
    cfg = TrainingConfig(**raw) if raw else TrainingConfig()
    X = np.load(args.patches).astype(float)
    X -= X.mean(axis=0, keepdims=True)
    lam = None
    if cfg.lam_sigma is not None:
        lam = float(cfg.lam_sigma * np.std(X))
    coder = SparseCoder(n_atoms=cfg.n_atoms, mode=args.mode, seed=args.seed, lam=lam, max_iter=200, tol=1e-6)
    log("train_patches_start", patches=args.patches, mode=args.mode, seed=args.seed)
    coder.fit(X, n_steps=cfg.steps, lr=cfg.lr)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "D.npy", coder.D)
    A = coder.encode(X); np.save(output_dir / "A.npy", A)
    meta = make_metadata(cfg, coder.D.shape, A.shape, {"mode": args.mode})
    with open(output_dir / "METADATA.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log("train_patches_done", out=args.out)

def cmd_encode(args):
    D = np.load(args.dictionary).astype(float)
    X = np.load(args.patches).astype(float)
    lam = float(args.lam) if args.lam is not None else 0.1 * np.median(np.abs(D.T @ X))
    from .fista_batch import fista_batch
    A = fista_batch(D, X, lam)
    np.save(args.out, A); log("encode_done", out=args.out)

def cmd_encode_stream(args):
    out = encode_stream(np.load(args.dictionary).astype(float), args.patches, lam=float(args.lam), batch=int(args.batch), out_path=args.out)
    log("encode_stream_done", out=out)

def cmd_reconstruct(args):
    D = np.load(args.dictionary).astype(float)
    A = np.load(args.codes).astype(float)
    X_hat = D @ A
    np.save(args.out, X_hat); log("reconstruct_done", out=args.out)

def main(argv=None):
    ap = argparse.ArgumentParser("sparse-coding")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_tr = sub.add_parser("train", help="Train from images folder")
    ap_tr.add_argument("--images", required=True)
    ap_tr.add_argument("--out", default="out")
    ap_tr.add_argument("--config")
    ap_tr.add_argument("--mode", choices=["paper", "l1"], default="paper")
    ap_tr.add_argument("--seed", type=int, default=0)
    ap_tr.add_argument("--deterministic", action="store_true")
    ap_tr.set_defaults(func=cmd_train)

    ap_trp = sub.add_parser("train-patches", help="Train from X.npy (p x N)")
    ap_trp.add_argument("--patches", required=True)
    ap_trp.add_argument("--out", default="out")
    ap_trp.add_argument("--config")
    ap_trp.add_argument("--mode", choices=["paper", "l1"], default="paper")
    ap_trp.add_argument("--seed", type=int, default=0)
    ap_trp.add_argument("--deterministic", action="store_true")
    ap_trp.set_defaults(func=cmd_train_patches)

    ap_en = sub.add_parser("encode", help="Encode patches with a dictionary")
    ap_en.add_argument("--dictionary", required=True)
    ap_en.add_argument("--patches", required=True)
    ap_en.add_argument("--out", required=True)
    ap_en.add_argument("--lam", type=float)
    ap_en.set_defaults(func=cmd_encode)

    ap_es = sub.add_parser("encode-stream", help="Encode a huge X.npy in batches")
    ap_es.add_argument("--dictionary", required=True)
    ap_es.add_argument("--patches", required=True)
    ap_es.add_argument("--lam", type=float, required=True)
    ap_es.add_argument("--batch", type=int, default=10000)
    ap_es.add_argument("--out")
    ap_es.set_defaults(func=cmd_encode_stream)

    ap_rc = sub.add_parser("reconstruct", help="Reconstruct from codes and dictionary")
    ap_rc.add_argument("--dictionary", required=True)
    ap_rc.add_argument("--codes", required=True)
    ap_rc.add_argument("--out", required=True)
    ap_rc.set_defaults(func=cmd_reconstruct)

    args = ap.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
