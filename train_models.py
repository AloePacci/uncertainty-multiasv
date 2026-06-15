"""
train_models.py — Uncertainty Estimation CAEPIA 2026
=====================================================
Entrena todos los modelos de estimación de incertidumbre sobre cada dataset
encontrado en Datasets/ y guarda los pesos en Weights/.

Pipeline por dataset
--------------------
1. Load .npz  →  ground_truth (N,H,W), observed_map (N,H,W), observed_mask (N,H,W)
2. Build inputs  X = [observed_mask, observed_map]  (N, 2, H, W)
                 y = ground_truth[:, np.newaxis]     (N, 1, H, W)
3. Random 80/20 train/test split (seed fijo para reproducibilidad)
4. For each model:
     a. model.train(X_train, y_train, ...)    [no-op para GP: no tiene pesos]
     b. model.save_weights(...)               [solo modelos DL]

Para evaluar los modelos entrenados, usa evaluate_models.py.

Model configs
-------------
All deep learning models use:
    in_channels=2, base_channels=32, depth=3
    epochs=50, batch_size=16, lr=1e-3

Run
---
    ~/miniconda3/python.exe train_models.py
    ~/miniconda3/python.exe train_models.py --datasets Datasets/dataset_NADIR.npz --epochs 10
"""

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from gaussian_process_model import GaussianProcessModel
from MC_dropout_model import MCDropoutModel
from MC_ensemble_model import EnsembleModel
from EDL_model import EDLModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("Datasets")
RESULTS_DIR  = Path("Results")
WEIGHTS_DIR  = Path("Weights")

SPLIT_SEED   = 42
TEST_SIZE    = 0.20

# Neural network architecture (shared across DL models)
NET_KWARGS = dict(in_channels=2, base_channels=32, depth=4)

# Training hyper-parameters (overrideable via CLI)
DEFAULT_EPOCHS     = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR         = 3e-4

# GP config
GP_KWARGS = dict(n_restarts_optimizer=0)

# Ensemble: number of members
ENSEMBLE_MEMBERS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(path: Path):
    """
    Load a dataset .npz and build model inputs / targets.

    Returns
    -------
    X      : np.ndarray  (N, 2, H, W)  float32  [mask | observed_map]
    y      : np.ndarray  (N, 1, H, W)  float32  ground truth
    gt     : np.ndarray  (N, H, W)     float32  ground truth (raw, for evaluation)
    meta   : dict        dataset metadata (from JSON bytes in the .npz)
    """
    data = np.load(path, allow_pickle=False)
    gt   = data["ground_truth"].astype(np.float32)   # (N, H, W)
    obs  = data["observed_map"].astype(np.float32)   # (N, H, W)
    mask = data["observed_mask"].astype(np.float32)  # (N, H, W)

    X = np.stack([mask, obs], axis=1)   # (N, 2, H, W)
    y = gt[:, np.newaxis]               # (N, 1, H, W)

    meta = {}
    if "metadata" in data.files:
        try:
            meta = json.loads(bytes(data["metadata"]).decode("utf-8"))
        except Exception:
            pass

    return X, y, gt, meta


def build_models(epochs: int, batch_size: int, lr: float):
    """
    Instantiate all models. Returns a list of (name, model, train_kwargs).

    train_kwargs are forwarded to model.train() (empty dict for GP).
    """
    models = [
        (
            "GP",
            GaussianProcessModel(**GP_KWARGS),
            {},
        ),
        (
            "MC_Dropout",
            MCDropoutModel(**NET_KWARGS, dropout_p=0.2, n_samples=30),
            dict(epochs=epochs, batch_size=batch_size, lr=lr),
        ),
        (
            "Ensemble",
            EnsembleModel(n_members=ENSEMBLE_MEMBERS, seed=SPLIT_SEED, **NET_KWARGS),
            dict(epochs=epochs, batch_size=batch_size, lr=lr),
        ),
        (
            "EDL",
            EDLModel(**NET_KWARGS),
            # lr reducido a 5e-4: NIG loss es más inestable que Gaussian NLL.
            # max_grad_norm=1.0 evita explosión de gradientes en épocas iniciales.
            dict(epochs=epochs, batch_size=batch_size,
                 lr=min(lr, 5e-4), lam=1e-3, max_grad_norm=1.0),
        ),
    ]
    return models


# ---------------------------------------------------------------------------
# Main training + evaluation loop
# ---------------------------------------------------------------------------

def weights_path(ds_name: str, algo_name: str) -> Path:
    """Ruta canónica del fichero de pesos para un dataset + algoritmo."""
    return WEIGHTS_DIR / f"{ds_name}_{algo_name}.pt"


def run_training(
    dataset_path: Path,
    epochs: int      = DEFAULT_EPOCHS,
    batch_size: int  = DEFAULT_BATCH_SIZE,
    lr: float        = DEFAULT_LR,
    skip_gp: bool    = False,
    verbose: bool    = True,
    force_retrain: bool = False,
):
    """
    Entrena todos los modelos sobre un dataset y guarda los pesos en Weights/.

    El GP no tiene pesos persistentes y se omite en la fase de entrenamiento.
    Para evaluar usa evaluate_models.py.

    Parameters
    ----------
    dataset_path : Path
        Ruta al fichero .npz del dataset.
    epochs, batch_size, lr : Hiper-parámetros de entrenamiento DL.
    skip_gp : bool
        No tiene efecto (el GP no entrena aquí), pero se mantiene por
        compatibilidad con la CLI.
    verbose : bool
        Muestra progreso.
    force_retrain : bool
        Si True, reentrena aunque existan pesos guardados.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ds_name   = dataset_path.stem
    obs_model = ds_name.replace("dataset_", "").upper()

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_path.name}")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────────────
    X, y, gt, meta = load_dataset(dataset_path)
    N, _, H, W = X.shape
    print(f"Samples : {N}  |  Grid : {H}x{W}")

    # ── 80/20 split ───────────────────────────────────────────────────
    idx = np.arange(N)
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SPLIT_SEED
    )
    X_train = X[idx_train]
    y_train = y[idx_train]
    print(f"Train   : {len(idx_train)} samples")
    print(f"Test    : {len(idx_test)} samples (reservado para evaluate_models.py)")

    # ── Models ────────────────────────────────────────────────────────
    models = build_models(epochs, batch_size, lr)

    for algo_name, model, train_kwargs in models:
        if algo_name == "GP":
            print(f"\n>>> [{obs_model}] GP: sin pesos persistentes, omitido en entrenamiento.")
            print("    (El GP se ajusta durante la evaluación en evaluate_models.py)")
            continue

        w_path = weights_path(ds_name, algo_name)
        has_save_load = hasattr(model, "save_weights")

        if has_save_load and w_path.exists() and not force_retrain:
            print(f"\n>>> [{obs_model}] {algo_name}: pesos ya existentes en {w_path} (--force-retrain para reentrenar)")
            continue

        t0 = time.time()
        print(f"\n>>> [{obs_model}] Training {algo_name} ...")
        model.train(X_train, y_train, **train_kwargs)
        elapsed = time.time() - t0
        print(f"    Training done in {elapsed:.1f}s")

        if has_save_load:
            model.save_weights(w_path)
            print(f"    Pesos guardados en {w_path}")

    print(f"\nEntrenamiento completado. Pesos en: {WEIGHTS_DIR}/")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena todos los modelos de incertidumbre sobre cada dataset."
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Paths to specific .npz dataset files. "
             "Defaults to all .npz files in Datasets/."
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Training epochs for DL models (default: {DEFAULT_EPOCHS})."
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})."
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR})."
    )
    parser.add_argument(
        "--skip-gp", action="store_true",
        help="Skip Gaussian Process (can be slow on large grids)."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-epoch output."
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Ignorar pesos guardados y reentrenar desde cero."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve dataset paths
    if args.datasets:
        dataset_paths = [Path(p) for p in args.datasets]
    else:
        dataset_paths = sorted(DATASETS_DIR.glob("*.npz"))

    if not dataset_paths:
        print(f"No .npz files found in {DATASETS_DIR}/")
        raise SystemExit(1)

    print(f"Datasets found : {[p.name for p in dataset_paths]}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Learning rate  : {args.lr}")
    print(f"Skip GP        : {args.skip_gp}")

    total_t0 = time.time()
    for ds_path in dataset_paths:
        run_training(
            dataset_path  = ds_path,
            epochs        = args.epochs,
            batch_size    = args.batch_size,
            lr            = args.lr,
            skip_gp       = args.skip_gp,
            verbose       = not args.quiet,
            force_retrain = args.force_retrain,
        )

    total_elapsed = time.time() - total_t0
    print(f"\nTodos los entrenamientos completados en {total_elapsed/60:.1f} min.")
    print(f"Pesos guardados en: {WEIGHTS_DIR}/")
    print(f"Para evaluar, ejecuta: python evaluate_models.py")
