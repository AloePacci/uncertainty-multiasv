"""
evaluate_models.py — Uncertainty Estimation CAEPIA 2026
========================================================
Evalúa todos los modelos entrenados sobre el conjunto de test reservado
(20% fijo, mismo seed que train_models.py) y guarda los resultados en
Results/ como un DataFrame pickle comprimido (.pkl.gz).

Cada fila del DataFrame resultante contiene:
  - Metadatos: algorithm, observation_model, map_shape, percentage_covered
  - Métricas escalares: rmse, r2, nll_*, ece_*
  - Mapas espaciales como arrays NumPy: ground_truth, predicted_mean,
    epistemic_std, aleatoric_std, observed_map, observed_mask

El GP no tiene pesos persistentes: se ajusta sobre X_train en cada ejecución.
Los modelos DL cargan pesos desde Weights/ (generados por train_models.py).

Pipeline por dataset
--------------------
1. Cargar .npz  →  ground_truth (N,H,W), observed_map (N,H,W), observed_mask (N,H,W)
2. Split 80/20 con el mismo seed que train_models.py
3. Para cada modelo:
     a. GP: model.train(X_train, y_train) → model.predict(X_test) batch completo
     b. DL: model.load_weights(Weights/{ds}_{algo}.pt) → model.predict(Xi) por muestra
4. store.add_record(...)  →  acumula resultados en DataFrame
5. store.save()  →  Results/eval_{OBS_MODEL}.pkl.gz

Run
---
    ~/miniconda3/python.exe evaluate_models.py
    ~/miniconda3/python.exe evaluate_models.py --datasets Datasets/dataset_NADIR.npz
    ~/miniconda3/python.exe evaluate_models.py --skip-gp
"""

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from train_models import (
    load_dataset,
    build_models,
    weights_path,
    DATASETS_DIR,
    RESULTS_DIR,
    WEIGHTS_DIR,
    SPLIT_SEED,
    TEST_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
)
from evaluation_store import EvaluationStore


# ---------------------------------------------------------------------------
# Evaluación de un dataset
# ---------------------------------------------------------------------------

def run_evaluation(
    dataset_path: Path,
    skip_gp: bool = False,
) -> EvaluationStore:
    """
    Evalúa todos los modelos sobre el 20% de test de un dataset y guarda
    los resultados en un EvaluationStore.

    Parameters
    ----------
    dataset_path : Path
        Ruta al fichero .npz del dataset.
    skip_gp : bool
        Si True, omite el GP (útil cuando la rejilla es grande y el GP es lento).

    Returns
    -------
    EvaluationStore con todos los registros guardados en disco.
    """
    ds_name   = dataset_path.stem
    obs_model = ds_name.replace("dataset_", "").upper()
    store_name = f"eval_{obs_model}"

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_path.name}")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────────────
    X, y, gt, meta = load_dataset(dataset_path)
    N, _, H, W = X.shape
    print(f"Samples : {N}  |  Grid : {H}x{W}")

    # ── 80/20 split (mismo seed que train_models.py) ───────────────────
    idx = np.arange(N)
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SPLIT_SEED
    )
    X_train = X[idx_train]
    y_train = y[idx_train]
    X_test  = X[idx_test]
    gt_test = gt[idx_test]
    n_test  = len(idx_test)

    print(f"Train   : {len(idx_train)} samples (no se usan, solo para GP)")
    print(f"Test    : {n_test} samples")

    # ── EvaluationStore ───────────────────────────────────────────────
    store = EvaluationStore(store_name, output_dir=str(RESULTS_DIR))

    # ── Models ────────────────────────────────────────────────────────
    models = build_models(DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR)

    for algo_name, model, train_kwargs in models:
        if skip_gp and algo_name == "GP":
            continue

        print(f"\n>>> [{obs_model}] {algo_name}")

        # ── Preparar modelo ───────────────────────────────────────────
        w_path = weights_path(ds_name, algo_name)
        has_save_load = hasattr(model, "save_weights")

        if algo_name == "GP":
            pass # El GP no tiene pesos persistentes: 
                #se ajusta sobre X_train en cada ejecución.
        elif has_save_load and w_path.exists():
            t0 = time.time()
            model.load_weights(w_path)
            print(f"    Pesos cargados desde {w_path} ({time.time() - t0:.1f}s)")
        else:
            print(f"    No se encontraron pesos en {w_path} — omitido.")
            print(f"    Ejecuta primero: python train_models.py")
            continue

        # ── Predicción sobre test set ──────────────────────────────────
        t0 = time.time()

        if algo_name == "GP":
            print(f"    Prediciendo {n_test} muestras con GP (batch)...")
            t_gp = time.perf_counter()
            gp_preds = model.predict(X_test)
            gp_time_per_sample_ms = (time.perf_counter() - t_gp) * 1000 / n_test
        else:
            gp_preds = None

        for i in tqdm(range(n_test),
                      desc=f"  [{obs_model}] {algo_name} eval",
                      unit="sample", leave=False):
            Xi  = X_test[i : i + 1]   # (1, 2, H, W)
            gti = gt_test[i]           # (H, W)

            if gp_preds is not None:
                pred = {k: v[i : i + 1] for k, v in gp_preds.items()}
                inference_time_ms = gp_time_per_sample_ms
            else:
                t_pred = time.perf_counter()
                pred = model.predict(Xi)
                inference_time_ms = (time.perf_counter() - t_pred) * 1000

            pred["observed_mask"] = Xi[:, 0:1]   # (1, 1, H, W)
            pred["observed_map"]  = Xi[:, 1:2]   # (1, 1, H, W)

            store.add_record(
                prediction         = pred,
                ground_truth       = gti,
                algorithm          = algo_name,
                observation_model  = obs_model,
                extra              = {"dataset": ds_name, "test_idx": int(idx_test[i])},
                inference_time_ms  = inference_time_ms,
            )

        print(f"    Evaluación completada en {time.time() - t0:.1f}s  ({n_test} samples)")

    # ── Guardar ───────────────────────────────────────────────────────
    print(f"\nGuardando resultados ...")
    store.save()

    # ── Resumen rápido ────────────────────────────────────────────────
    print(f"\n=== Resumen — {obs_model} ===")
    try:
        summary = store.summary()[["rmse", "r2", "nll_total", "uce_total"]]
        print(summary.to_string())
    except Exception as e:
        print(f"  (summary error: {e})")

    return store


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalúa todos los modelos de incertidumbre sobre el test set."
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Rutas a ficheros .npz específicos. "
             "Por defecto todos los .npz en Datasets/."
    )
    parser.add_argument(
        "--skip-gp", action="store_true",
        help="Omitir el Gaussian Process (lento en rejillas grandes)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.datasets:
        dataset_paths = [Path(p) for p in args.datasets]
    else:
        dataset_paths = sorted(DATASETS_DIR.glob("*.npz"))

    if not dataset_paths:
        print(f"No se encontraron ficheros .npz en {DATASETS_DIR}/")
        raise SystemExit(1)

    print(f"Datasets encontrados : {[p.name for p in dataset_paths]}")
    print(f"Skip GP              : {args.skip_gp}")

    total_t0 = time.time()
    for ds_path in dataset_paths:
        run_evaluation(
            dataset_path = ds_path,
            skip_gp      = args.skip_gp,
        )

    total_elapsed = time.time() - total_t0
    print(f"\nTodas las evaluaciones completadas en {total_elapsed/60:.1f} min.")
    print(f"Resultados en: {RESULTS_DIR}/")
