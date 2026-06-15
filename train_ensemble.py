"""
train_ensemble.py — Train EnsembleModel on synthetic maps and save weights.

Two data modes
--------------
1. Ground-truth maps + random masking (default):
       --dataset dataset/dataset_POINTWISE/data_maps.npz
   Each map generates ``--samples-per-map`` random partial-observation pairs
   with coverage sampled in [``--coverage-min``, ``--coverage-max``].

2. Pre-built augmented dataset (obs_mask + obs_map already computed):
       --augmented-dataset dataset/augmented_dataset.npz
   The .npz must contain keys: obs_masks, obs_maps, ground_truths
   (as produced by generate_mask_dataset.py).

Usage
-----
    uv run python train_ensemble.py
    uv run python train_ensemble.py --augmented-dataset dataset/augmented_dataset.npz \\
        --weights-out weights/ensemble.pt --epochs 50

All arguments have sensible defaults; run with --help for the full list.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from scenario.models.MC_ensemble_model import EnsembleModel


# ── Device selection ──────────────────────────────────────────────────────────

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Data preparation ───────────────────────────────────────────────────────────

def load_maps(path: Path) -> np.ndarray:
    data = np.load(path)
    key = "maps" if "maps" in data else list(data.keys())[0]
    maps = data[key].astype(np.float32)   # (N, H, W)
    return maps


def load_augmented_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a pre-built augmented dataset produced by generate_mask_dataset.py.

    Returns
    -------
    X : float32  (N, 2, H, W) — channel 0: obs_mask, channel 1: obs_map
    y : float32  (N, 1, H, W) — ground truth
    """
    data = np.load(path)
    obs_masks     = data["obs_masks"].astype(np.float32)      # (N, H, W)
    obs_maps      = data["obs_maps"].astype(np.float32)       # (N, H, W)
    ground_truths = data["ground_truths"].astype(np.float32)  # (N, H, W)
    X = np.stack([obs_masks, obs_maps], axis=1)               # (N, 2, H, W)
    y = ground_truths[:, np.newaxis]                          # (N, 1, H, W)
    return X, y


def make_pairs(
    maps: np.ndarray,
    samples_per_map: int,
    coverage_min: float,
    coverage_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (X, y) training pairs from a set of ground-truth maps.

    For each map, ``samples_per_map`` random binary masks are drawn with
    coverage in [coverage_min, coverage_max].

    Returns
    -------
    X : float32  (N*samples_per_map, 2, H, W)  — [mask, observed_values]
    y : float32  (N*samples_per_map, 1, H, W)  — full ground-truth map
    """
    N, H, W = maps.shape
    total = N * samples_per_map
    X = np.empty((total, 2, H, W), dtype=np.float32)
    y = np.empty((total, 1, H, W), dtype=np.float32)

    idx = 0
    for m in maps:
        for _ in range(samples_per_map):
            coverage = float(rng.uniform(coverage_min, coverage_max))
            mask = (rng.random((H, W)) < coverage).astype(np.float32)
            obs  = m * mask                  # 0 at unobserved locations
            X[idx, 0] = mask
            X[idx, 1] = obs
            y[idx, 0] = m
            idx += 1

    return X, y


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Deep Ensemble on synthetic oil-spill maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data — mutually exclusive modes
    data_group = p.add_mutually_exclusive_group()
    data_group.add_argument(
        "--dataset", default="dataset/dataset_POINTWISE/data_maps.npz",
        help="Path to raw ground-truth maps .npz (random masking will be applied)")
    data_group.add_argument(
        "--augmented-dataset", default=None,
        help="Path to pre-built augmented dataset .npz "
             "(obs_masks, obs_maps, ground_truths — skips random masking)")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="Fraction of samples reserved for validation")
    p.add_argument("--samples-per-map", type=int, default=5,
                   help="Random observation pairs generated per map (ignored with --augmented-dataset)")
    p.add_argument("--coverage-min", type=float, default=0.02,
                   help="Minimum observation coverage fraction (ignored with --augmented-dataset)")
    p.add_argument("--coverage-max", type=float, default=0.30,
                   help="Maximum observation coverage fraction (ignored with --augmented-dataset)")

    # Model
    p.add_argument("--n-members",     type=int,   default=5)
    p.add_argument("--base-channels", type=int,   default=32)
    p.add_argument("--depth",         type=int,   default=4)

    # Training
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--seed",        type=int,   default=42)

    # Output
    p.add_argument("--weights-out", default="weights/ensemble.pt",
                   help="Path to save the trained weights (.pt)")
    p.add_argument("--results-out", default="weights/training_results.csv",
                   help="Path to save the per-epoch training results (.csv)")
    p.add_argument("--plot",        action="store_true",
                   help="Show loss curves after training")

    return p.parse_args()


# ── Training ───────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = select_device()
    rng    = np.random.default_rng(args.seed)

    # ── Determine dataset source and name ─────────────────────────────────────
    using_augmented = args.augmented_dataset is not None
    dataset_path    = Path(args.augmented_dataset if using_augmented else args.dataset)
    dataset_name    = dataset_path.stem

    print(f"Device        : {device}")
    print(f"Dataset       : {dataset_path}  ({'augmented' if using_augmented else 'raw maps'})")
    print(f"Weights out   : {args.weights_out}")
    print(f"Results out   : {args.results_out}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    if using_augmented:
        X_all, y_all = load_augmented_pairs(dataset_path)
        N = len(X_all)
        print(f"Samples loaded: {N}  shape {X_all.shape[2]}×{X_all.shape[3]} px")

        idx       = rng.permutation(N)
        n_val     = max(1, int(N * args.val_fraction))
        X_train   = X_all[idx[n_val:]]
        y_train   = y_all[idx[n_val:]]
        X_val     = X_all[idx[:n_val]]
        y_val     = y_all[idx[:n_val]]
    else:
        maps = load_maps(dataset_path)
        N    = len(maps)
        print(f"Maps loaded   : {N}  shape {maps.shape[1]}×{maps.shape[2]} px")

        idx        = rng.permutation(N)
        n_val      = max(1, int(N * args.val_fraction))
        train_maps = maps[idx[n_val:]]
        val_maps   = maps[idx[:n_val]]
        print(f"Train maps    : {len(train_maps)}   Val maps : {len(val_maps)}")

        print(f"\nGenerating training pairs  ({args.samples_per_map} per map, "
              f"coverage {args.coverage_min:.0%}–{args.coverage_max:.0%}) …")
        X_train, y_train = make_pairs(train_maps, args.samples_per_map,
                                      args.coverage_min, args.coverage_max, rng)
        X_val,   y_val   = make_pairs(val_maps,   args.samples_per_map,
                                      args.coverage_min, args.coverage_max, rng)

    print(f"Train pairs   : {len(X_train):,}   Val pairs : {len(X_val):,}")

    # ── Build model ────────────────────────────────────────────────────────────
    model = EnsembleModel(
        n_members=args.n_members,
        device=device,
        seed=args.seed,
        in_channels=2,
        base_channels=args.base_channels,
        depth=args.depth,
    )
    total_params = model.count_parameters()
    print(f"\nEnsemble      : {args.n_members} members  ×  "
          f"{total_params // args.n_members:,} params  =  {total_params:,} total")

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\nTraining  {args.epochs} epochs  |  batch {args.batch_size}  |  lr {args.lr}\n")
    t0 = time.time()

    results = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f"\nTraining time : {elapsed:.1f} s")
    print(f"Final val RMSE: {results['epoch_rmse'][-1]:.6f}")
    print(f"Final val NLL : {results['epoch_val_nll'][-1]:.6f}")

    # ── Save weights ──────────────────────────────────────────────────────────
    out_path = Path(args.weights_out)
    model.save_weights(out_path)
    print(f"\nWeights saved → {out_path}")

    # ── Build and save results DataFrame ─────────────────────────────────────
    df = pd.DataFrame({
        "epoch":        range(1, args.epochs + 1),
        "nll":          results["epoch_nll"],
        "rmse":         results["epoch_rmse"],
        "val_nll":      results["epoch_val_nll"],
        "dataset_name": dataset_name,
    })
    results_path = Path(args.results_out)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"Results saved → {results_path}")

    # ── Loss curves ───────────────────────────────────────────────────────────
    if args.plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Ensemble training — {dataset_name}", fontsize=11)

        for k, losses in enumerate(results["member_nll"]):
            axes[0].plot(losses, label=f"member {k+1}", alpha=0.8)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Train NLL")
        axes[0].set_title("Per-member train NLL")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df["epoch"], df["rmse"],    label="Val RMSE")
        axes[1].plot(df["epoch"], df["val_nll"], label="Val NLL", linestyle="--")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Validation metrics")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
