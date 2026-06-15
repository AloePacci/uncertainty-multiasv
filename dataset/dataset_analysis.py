"""
dataset_analysis.py -- Uncertainty Estimation CAEPIA 2026
==========================================================
Genera figuras de analisis exploratorio para cada .npz en Datasets/.

Figuras producidas (guardadas en Datasets/plots/):
  1.  coverage_dist        -- Distribucion de cobertura de observacion (% pixeles observados)
  2.  spill_extent_dist    -- Distribucion del area del spill (% pixeles GT > umbral)
  3.  spill_coverage_ratio -- Ratio: pixeles observados DENTRO del spill / total spill
  4.  obs_error_dist       -- Distribucion del error de observacion (obs - GT) en pixeles visitados
  5.  gt_value_dist        -- Distribucion de valores GT en pixeles no nulos
  6.  visit_heatmap        -- Mapa de calor de pixeles mas visitados en todo el dataset
  7.  gt_mean_heatmap      -- Media espacial del GT en todo el dataset
  8.  coverage_vs_extent   -- Scatter: cobertura vs extension del spill
  9.  coverage_vs_overlap  -- Scatter: cobertura vs ratio de spill cubierto
  10. obs_vs_gt_scatter    -- Scatter de valores observados vs GT (muestra aleatoria)
  11. sample_gallery       -- Galeria de N ejemplos aleatorios (GT / obs_map / obs_mask)

Uso:
    ~/miniconda3/python.exe dataset_analysis.py
    ~/miniconda3/python.exe dataset_analysis.py --dataset Datasets/dataset_NADIR.npz
    ~/miniconda3/python.exe dataset_analysis.py --n-examples 6 --spill-threshold 0.02
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS_DIR  = Path("Datasets")
PLOTS_SUBDIR  = "plots"          # subdirectory inside Datasets/
SPILL_THRESH  = 0.01             # GT value considered "active spill"
N_EXAMPLES    = 6                # samples shown in the gallery
SCATTER_MAX   = 20_000           # max points in scatter plots (random subsample)
DPI           = 150
CMAP_GT       = "hot"
CMAP_OBS      = "hot"
CMAP_MASK     = "grey"
CMAP_HEAT     = "hot"


plt.style.use("seaborn-v0_8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    gt   = data["ground_truth"].astype(np.float32)    # (N, H, W)
    obs  = data["observed_map"].astype(np.float32)    # (N, H, W)
    mask = data["observed_mask"].astype(np.float32)   # (N, H, W)  binary
    meta = {}
    if "metadata" in data.files:
        try:
            meta = json.loads(bytes(data["metadata"]).decode("utf-8"))
        except Exception:
            pass
    return gt, obs, mask, meta


def savefig(fig: plt.Figure, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"{name}.png"
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {fpath}")


# ---------------------------------------------------------------------------
# Individual figure functions
# ---------------------------------------------------------------------------

def fig_coverage_dist(gt, obs, mask, meta, out_dir, ds_stem):
    """Histograma de cobertura: % de pixeles observados por muestra."""
    coverage = mask.mean(axis=(1, 2)) * 100   # (N,) en %

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(coverage, bins=40, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(coverage.mean(), color="crimson", lw=1.5, ls="--",
               label=f"media = {coverage.mean():.1f}%")
    ax.axvline(np.median(coverage), color="orange", lw=1.5, ls=":",
               label=f"mediana = {np.median(coverage):.1f}%")
    ax.set_xlabel("Cobertura de observacion (%)")
    ax.set_ylabel("Numero de muestras")
    ax.set_title(f"{ds_stem} — Distribucion de cobertura de observacion\n"
                 f"N={len(coverage)}  modelo={meta.get('observation_model_type','?')}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, out_dir, f"{ds_stem}_01_coverage_dist")


def fig_spill_extent_dist(gt, obs, mask, meta, out_dir, ds_stem, thresh):
    """Histograma de extension del spill: % pixeles GT > thresh por muestra."""
    extent = (gt > thresh).mean(axis=(1, 2)) * 100   # (N,) en %

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(extent, bins=40, color="tomato", edgecolor="white", linewidth=0.4)
    ax.axvline(extent.mean(), color="navy", lw=1.5, ls="--",
               label=f"media = {extent.mean():.1f}%")
    ax.axvline(np.median(extent), color="gold", lw=1.5, ls=":",
               label=f"mediana = {np.median(extent):.1f}%")
    ax.set_xlabel(f"Extension del spill (% pixeles GT > {thresh})")
    ax.set_ylabel("Numero de muestras")
    ax.set_title(f"{ds_stem} — Distribucion de extension del spill\n"
                 f"umbral={thresh}  N={len(extent)}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, out_dir, f"{ds_stem}_02_spill_extent_dist")


def fig_spill_coverage_ratio(gt, obs, mask, meta, out_dir, ds_stem, thresh):
    """Histograma de ratio: pixeles observados dentro del spill / total spill."""
    spill = (gt > thresh)                    # (N, H, W) bool
    spill_size = spill.sum(axis=(1, 2))      # (N,)  numero de pixeles de spill
    observed_in_spill = (mask * spill).sum(axis=(1, 2))  # (N,)

    # Evitar division por cero en muestras sin spill
    valid = spill_size > 0
    ratio = np.full(len(gt), np.nan)
    ratio[valid] = observed_in_spill[valid] / spill_size[valid] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ratio[valid], bins=40, color="mediumseagreen", edgecolor="white", linewidth=0.4)
    ax.axvline(np.nanmean(ratio), color="crimson", lw=1.5, ls="--",
               label=f"media = {np.nanmean(ratio):.1f}%")
    ax.set_xlabel("% del area del spill cubierta por la trayectoria")
    ax.set_ylabel("Numero de muestras")
    ax.set_title(f"{ds_stem} — Cobertura sobre el area del spill\n"
                 f"umbral={thresh}  N={valid.sum()} muestras con spill")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, out_dir, f"{ds_stem}_03_spill_coverage_ratio")


def fig_obs_error_dist(gt, obs, mask, meta, out_dir, ds_stem):
    """Distribucion del error de observacion (obs - GT) en pixeles visitados."""
    visited = mask.astype(bool)
    error = (obs[visited] - gt[visited]).ravel()

    # subsample si hay demasiados puntos
    rng = np.random.default_rng(0)
    if len(error) > SCATTER_MAX:
        error = rng.choice(error, SCATTER_MAX, replace=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(error, bins=60, color="mediumpurple", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", lw=1.0, ls="-")
    ax.axvline(error.mean(), color="crimson", lw=1.5, ls="--",
               label=f"media = {error.mean():.4f}")
    ax.axvline(np.median(error), color="orange", lw=1.5, ls=":",
               label=f"mediana = {np.median(error):.4f}")
    ax.set_xlabel("Error de observacion  (obs - GT)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"{ds_stem} — Distribucion del error de observacion\n"
                 f"std={error.std():.4f}  n_puntos={len(error):,}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, out_dir, f"{ds_stem}_04_obs_error_dist")


def fig_gt_value_dist(gt, obs, mask, meta, out_dir, ds_stem, thresh):
    """Distribucion de valores GT en pixeles activos (GT > thresh)."""
    active_vals = gt[gt > thresh].ravel()

    rng = np.random.default_rng(0)
    if len(active_vals) > SCATTER_MAX * 5:
        active_vals = rng.choice(active_vals, SCATTER_MAX * 5, replace=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(active_vals, bins=60, color="darkorange", edgecolor="white", linewidth=0.3)
    ax.axvline(active_vals.mean(), color="navy", lw=1.5, ls="--",
               label=f"media = {active_vals.mean():.4f}")
    ax.set_xlabel(f"Valor GT  (solo pixeles > {thresh})")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"{ds_stem} — Distribucion de valores del ground truth\n"
                 f"n_pixeles_activos={len(active_vals):,}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, out_dir, f"{ds_stem}_05_gt_value_dist")


def fig_visit_heatmap(gt, obs, mask, meta, out_dir, ds_stem):
    """Mapa de calor acumulado de visitas en todo el dataset."""
    visit_map = mask.sum(axis=0)   # (H, W)  numero de veces visitado

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(visit_map, cmap=CMAP_HEAT, origin="upper")
    ax.set_title(f"{ds_stem} — Mapa de visitas acumuladas\n(N={len(mask)} muestras)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="N visitas")
    fig.tight_layout()
    savefig(fig, out_dir, f"{ds_stem}_06_visit_heatmap")


def fig_gt_mean_heatmap(gt, obs, mask, meta, out_dir, ds_stem):
    """Media espacial del GT en todo el dataset."""
    gt_mean = gt.mean(axis=0)   # (H, W)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(gt_mean, cmap=CMAP_GT, origin="upper")
    ax.set_title(f"{ds_stem} — GT medio espacial\n(N={len(gt)} muestras)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Densidad media")
    fig.tight_layout()
    savefig(fig, out_dir, f"{ds_stem}_07_gt_mean_heatmap")


def fig_coverage_vs_extent(gt, obs, mask, meta, out_dir, ds_stem, thresh):
    """Scatter: cobertura de observacion vs extension del spill."""
    coverage = mask.mean(axis=(1, 2)) * 100
    extent   = (gt > thresh).mean(axis=(1, 2)) * 100

    rng = np.random.default_rng(0)
    idx = rng.choice(len(coverage), min(SCATTER_MAX, len(coverage)), replace=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(extent[idx], coverage[idx], alpha=0.3, s=8, color="steelblue",
               edgecolors="none")
    # Tendencia (regresion lineal simple)
    m, b = np.polyfit(extent, coverage, 1)
    x_line = np.linspace(extent.min(), extent.max(), 100)
    ax.plot(x_line, m * x_line + b, color="crimson", lw=1.5,
            label=f"y={m:.2f}x+{b:.2f}")
    ax.set_xlabel(f"Extension del spill (% GT > {thresh})")
    ax.set_ylabel("Cobertura de observacion (%)")
    ax.set_title(f"{ds_stem} — Cobertura vs Extension del spill")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, out_dir, f"{ds_stem}_08_coverage_vs_extent")


def fig_coverage_vs_overlap(gt, obs, mask, meta, out_dir, ds_stem, thresh):
    """Scatter: cobertura total vs ratio de spill cubierto."""
    coverage = mask.mean(axis=(1, 2)) * 100
    spill    = (gt > thresh)
    spill_size = spill.sum(axis=(1, 2)).astype(float)
    obs_in_spill = (mask * spill).sum(axis=(1, 2))
    valid = spill_size > 0
    overlap = np.full(len(gt), np.nan)
    overlap[valid] = obs_in_spill[valid] / spill_size[valid] * 100

    rng = np.random.default_rng(0)
    sel = valid
    idx = np.where(sel)[0]
    if len(idx) > SCATTER_MAX:
        idx = rng.choice(idx, SCATTER_MAX, replace=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coverage[idx], overlap[idx], alpha=0.3, s=8, color="mediumseagreen",
               edgecolors="none")
    m, b = np.polyfit(coverage[idx], overlap[idx], 1)
    x_line = np.linspace(coverage[idx].min(), coverage[idx].max(), 100)
    ax.plot(x_line, m * x_line + b, color="crimson", lw=1.5,
            label=f"y={m:.2f}x+{b:.2f}")
    ax.set_xlabel("Cobertura de observacion (%)")
    ax.set_ylabel(f"% del spill cubierto (GT > {thresh})")
    ax.set_title(f"{ds_stem} — Cobertura total vs Ratio de spill cubierto")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, out_dir, f"{ds_stem}_09_coverage_vs_overlap")


def fig_obs_vs_gt_scatter(gt, obs, mask, meta, out_dir, ds_stem):
    """Scatter de valores observados vs GT en pixeles visitados."""
    visited = mask.astype(bool)
    gt_vals  = gt[visited].ravel()
    obs_vals = obs[visited].ravel()

    rng = np.random.default_rng(0)
    if len(gt_vals) > SCATTER_MAX:
        idx = rng.choice(len(gt_vals), SCATTER_MAX, replace=False)
        gt_vals  = gt_vals[idx]
        obs_vals = obs_vals[idx]

    vmin = min(gt_vals.min(), obs_vals.min())
    vmax = max(gt_vals.max(), obs_vals.max())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gt_vals, obs_vals, alpha=0.2, s=5, color="darkorchid", edgecolors="none")
    ax.plot([vmin, vmax], [vmin, vmax], color="crimson", lw=1.5, ls="--",
            label="obs = GT (ideal)")
    ax.set_xlabel("Valor GT")
    ax.set_ylabel("Valor observado (con ruido)")
    ax.set_title(f"{ds_stem} — Observado vs Ground Truth\n"
                 f"n={len(gt_vals):,} puntos")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, out_dir, f"{ds_stem}_10_obs_vs_gt_scatter")


def fig_sample_gallery(gt, obs, mask, meta, out_dir, ds_stem, n_examples):
    """Galeria de N muestras aleatorias: GT / mapa observado / mascara."""
    rng = np.random.default_rng(42)
    n   = min(n_examples, len(gt))
    idx = rng.choice(len(gt), n, replace=False)

    ncols = n
    fig, axes = plt.subplots(3, ncols, figsize=(3 * ncols, 8))
    if ncols == 1:
        axes = axes[:, np.newaxis]

    row_labels = ["Ground Truth", "Mapa observado", "Mascara observacion"]
    cmaps      = [CMAP_GT, CMAP_OBS, CMAP_MASK]

    for col, i in enumerate(idx):
        maps = [gt[i], obs[i], mask[i]]
        for row in range(3):
            ax = axes[row, col]
            im = ax.imshow(maps[row], cmap=cmaps[row], origin="upper",
                           vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            if row == 0:
                cov = mask[i].mean() * 100
                ext = (gt[i] > SPILL_THRESH).mean() * 100
                ax.set_title(f"#{i}\ncov={cov:.1f}%  ext={ext:.1f}%", fontsize=8)

            if row == 0:
                # Add a colorbar next to the GT (first row)
                fig.colorbar(im, label="Valor", ticks=[0, 0.5, 1])

    fig.suptitle(f"{ds_stem} — Galeria de {n} muestras aleatorias\n"
                 f"modelo={meta.get('observation_model_type','?')}  "
                 f"N_total={len(gt)}", fontsize=11)
    fig.tight_layout()
    savefig(fig, out_dir, f"{ds_stem}_11_sample_gallery")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def write_summary(gt, obs, mask, meta, out_dir, ds_stem, thresh):
    """Escribe un resumen estadistico en texto."""
    coverage     = mask.mean(axis=(1, 2)) * 100
    extent       = (gt > thresh).mean(axis=(1, 2)) * 100
    spill        = (gt > thresh)
    spill_size   = spill.sum(axis=(1, 2)).astype(float)
    obs_in_spill = (mask * spill).sum(axis=(1, 2))
    valid        = spill_size > 0
    overlap      = np.where(valid, obs_in_spill / spill_size * 100, np.nan)

    visited      = mask.astype(bool)
    error        = (obs[visited] - gt[visited]).ravel()
    active_gt    = gt[gt > thresh].ravel()

    lines = [
        f"Dataset : {ds_stem}",
        f"Modelo de observacion : {meta.get('observation_model_type', '?')}",
        f"N muestras            : {len(gt)}",
        f"Grid                  : {gt.shape[1]}x{gt.shape[2]}",
        f"timestamps_per_gt     : {meta.get('timestamps_per_groundtruth', 1)}",
        "",
        "--- Cobertura de observacion (% pixeles observados) ---",
        f"  Media   : {coverage.mean():.2f}%",
        f"  Mediana : {np.median(coverage):.2f}%",
        f"  Std     : {coverage.std():.2f}%",
        f"  Min/Max : {coverage.min():.2f}% / {coverage.max():.2f}%",
        "",
        f"--- Extension del spill (% pixeles GT > {thresh}) ---",
        f"  Media   : {extent.mean():.2f}%",
        f"  Mediana : {np.median(extent):.2f}%",
        f"  Std     : {extent.std():.2f}%",
        f"  Min/Max : {extent.min():.2f}% / {extent.max():.2f}%",
        "",
        "--- Ratio spill cubierto (pixeles obs dentro del spill / total spill) ---",
        f"  Media   : {np.nanmean(overlap):.2f}%",
        f"  Mediana : {np.nanmedian(overlap):.2f}%",
        f"  Std     : {np.nanstd(overlap):.2f}%",
        "",
        "--- Error de observacion (obs - GT, solo pixeles visitados) ---",
        f"  Media   : {error.mean():.5f}",
        f"  Std     : {error.std():.5f}",
        f"  |error| media : {np.abs(error).mean():.5f}",
        f"  RMSE    : {np.sqrt((error**2).mean()):.5f}",
        f"  N puntos: {len(error):,}",
        "",
        f"--- Valores GT activos (GT > {thresh}) ---",
        f"  Media   : {active_gt.mean():.5f}",
        f"  Mediana : {np.median(active_gt):.5f}",
        f"  Std     : {active_gt.std():.5f}",
        f"  Max     : {active_gt.max():.5f}",
        f"  N pixeles activos: {len(active_gt):,}",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"{ds_stem}_summary.txt"
    fpath.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Resumen: {fpath}")
    # Imprimir tambien en consola
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Full analysis for one dataset
# ---------------------------------------------------------------------------

def analyse_dataset(path: Path, thresh: float, n_examples: int):
    print(f"\n{'='*60}")
    print(f"Analizando: {path.name}")
    print(f"{'='*60}")

    gt, obs, mask, meta = load_dataset(path)
    ds_stem = path.stem
    out_dir = path.parent / PLOTS_SUBDIR

    fig_coverage_dist      (gt, obs, mask, meta, out_dir, ds_stem)
    fig_spill_extent_dist  (gt, obs, mask, meta, out_dir, ds_stem, thresh)
    fig_spill_coverage_ratio(gt, obs, mask, meta, out_dir, ds_stem, thresh)
    fig_obs_error_dist     (gt, obs, mask, meta, out_dir, ds_stem)
    fig_gt_value_dist      (gt, obs, mask, meta, out_dir, ds_stem, thresh)
    fig_visit_heatmap      (gt, obs, mask, meta, out_dir, ds_stem)
    fig_gt_mean_heatmap    (gt, obs, mask, meta, out_dir, ds_stem)
    fig_coverage_vs_extent (gt, obs, mask, meta, out_dir, ds_stem, thresh)
    fig_coverage_vs_overlap(gt, obs, mask, meta, out_dir, ds_stem, thresh)
    fig_obs_vs_gt_scatter  (gt, obs, mask, meta, out_dir, ds_stem)
    fig_sample_gallery     (gt, obs, mask, meta, out_dir, ds_stem, n_examples)
    write_summary          (gt, obs, mask, meta, out_dir, ds_stem, thresh)

    print(f"\nFiguras guardadas en: {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analisis exploratorio de datasets de oil-spill.")
    p.add_argument("--dataset", nargs="*", default=None,
                   help="Rutas a .npz especificos. Por defecto todos los de Datasets/.")
    p.add_argument("--spill-threshold", type=float, default=SPILL_THRESH,
                   help=f"Umbral de valor GT para considerar pixel como spill (default {SPILL_THRESH}).")
    p.add_argument("--n-examples", type=int, default=N_EXAMPLES,
                   help=f"Numero de ejemplos en la galeria (default {N_EXAMPLES}).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset:
        paths = [Path(p) for p in args.dataset]
    else:
        paths = sorted(DATASETS_DIR.glob("*.npz"))

    if not paths:
        print(f"No se encontraron .npz en {DATASETS_DIR}/")
        sys.exit(1)

    print(f"Datasets a analizar: {[p.name for p in paths]}")
    print(f"Umbral spill       : {args.spill_threshold}")
    print(f"Ejemplos galeria   : {args.n_examples}")

    for path in paths:
        analyse_dataset(path, thresh=args.spill_threshold,
                        n_examples=args.n_examples)

    print("\nAnalisis completado.")
