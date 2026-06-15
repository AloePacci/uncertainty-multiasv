"""
plot_results.py — Uncertainty Estimation CAEPIA 2026
=====================================================
Genera las gráficas de resultados a partir de los ficheros .pkl.gz
producidos por evaluate_models.py.

Run
---
    python plot_results.py
    python plot_results.py --results-dir Results --output-dir Datasets/plots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 12})

# New names for algorithms and observation models (for prettier plots)
ALGO_NAMES = {
    "GP": "GP",
    "MC_Dropout": "MC Dropout",
    "Ensemble": "Ensemble",
    "EDL": "EDL",
}

OBS_MODEL_NAMES = {
    "CONIC": "Conic FOV",
    "POINTWISE": "Pointwise sensor",
    "NADIR": "Nadir camera",
}


RESULTS_DIR = Path("Results")
OUTPUT_DIR  = Path("Results/plots")

# Orden canónico de los algoritmos (nombres ya traducidos por ALGO_NAMES)
ALGO_ORDER = [ALGO_NAMES[a] for a in ["GP", "MC_Dropout", "Ensemble", "EDL"]]

sns.set_style("darkgrid")


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> pd.DataFrame:
    """
    Carga todos los .pkl.gz del directorio y devuelve un DataFrame escalar
    combinado (sin columnas de arrays espaciales).
    """
    files = sorted(results_dir.glob("eval_*.pkl.gz"))
    if not files:
        raise FileNotFoundError(f"No se encontraron eval_*.pkl.gz en {results_dir}/")

    frames = []
    for f in files:
        df = pd.read_pickle(str(f))
        # Descartar columnas de arrays espaciales si las hubiera
        array_cols = [c for c in df.columns if df[c].dtype == object
                      and df[c].apply(lambda x: hasattr(x, "shape")).any()]
        frames.append(df.drop(columns=array_cols, errors="ignore"))

    combined = pd.concat(frames, ignore_index=True)
    print(f"Cargados {len(files)} fichero(s) — {len(combined)} registros totales.")
    
    # Renombrar algoritmos y modelos de observación para las gráficas
    combined["algorithm"] = combined["algorithm"].map(ALGO_NAMES).fillna(combined["algorithm"])
    combined["observation_model"] = combined["observation_model"].map(OBS_MODEL_NAMES).fillna(combined["observation_model"])
    
    return combined


def load_results_with_maps(results_dir: Path) -> pd.DataFrame:
    """
    Como load_results pero conservando las columnas de arrays espaciales
    (ground_truth, predicted_mean, epistemic_std, aleatoric_std, observed_mask).
    Necesario para calcular curvas de calibración por píxel.
    """
    files = sorted(results_dir.glob("eval_*.pkl.gz"))
    if not files:
        raise FileNotFoundError(f"No se encontraron eval_*.pkl.gz en {results_dir}/")

    frames = []
    for f in files:
        frames.append(pd.read_pickle(str(f)))

    combined = pd.concat(frames, ignore_index=True)
    combined["algorithm"] = combined["algorithm"].map(ALGO_NAMES).fillna(combined["algorithm"])
    combined["observation_model"] = combined["observation_model"].map(OBS_MODEL_NAMES).fillna(combined["observation_model"])
    return combined


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def plot_rmse_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Box plot del RMSE por modelo de entrenamiento (x) y modelo de observación
    (hue), todos en la misma figura.
    """
    # Orden de algoritmos presentes en los datos
    algo_order = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]

    fig, ax = plt.subplots(figsize=(8, 3))

    sns.boxplot(
        data      = df,
        x         = "algorithm",
        y         = "rmse",
        hue       = "observation_model",
        order     = algo_order,
        palette   = "Set2",
        ax        = ax,
        width     = 0.6,
        linewidth = 1.2,
        showfliers  = False, # para evitar que los outliers distorsionen la escala del eje y
    )

    ax.set_xlabel("Estimation model")
    ax.set_ylabel(r"RMSE = $\sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$")
    # 1column for the legend
    ax.legend(title="Observation model", ncol=3, loc="upper right", frameon=True)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rmse_boxplot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Guardado: {out_path}")
    plt.show()

    # También imprimir los valores numéricos de RMSE por consola
    print("\nRMSE por algoritmo y modelo de observación:")
    print(
        df.groupby(["algorithm", "observation_model"])["rmse"]
        .agg(["mean", "std"])
        .unstack()
        .round(4).to_string()
    )


def plot_nll_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Box plot del NLL total por modelo de estimación (x) y modelo de observación
    (hue), todos en la misma figura.
    """
    algo_order = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]

    fig, ax = plt.subplots(figsize=(8, 3))

    sns.boxplot(
        data       = df,
        x          = "algorithm",
        y          = "nll_total",
        hue        = "observation_model",
        order      = algo_order,
        palette    = "Set2",
        ax         = ax,
        width      = 0.6,
        linewidth  = 1.2,
        showfliers = False,
    )

    ax.set_xlabel("Estimation model")
    ax.set_ylabel(
        r"NLL $= \frac{1}{N}\sum_{i=1}^N \left["
        r"\frac{(y_i-\hat{\mu}_i)^2}{2\hat{\sigma}_i^2}"
        r"+ \frac{1}{2}\log\hat{\sigma}_i^2\right]$"
    )
    ax.legend(title="Observation model", ncol=3, loc="upper right", frameon=True)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "nll_boxplot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Guardado: {out_path}")
    plt.show()

    
    # También imprimir los valores numéricos de UCE por consola
    print("NLL total por algoritmo y modelo de observación:")
    print(
        df.groupby(["algorithm", "observation_model"])[["rmse", "nll_total"]]
        .agg(["mean", "std"])
        .unstack()
        .round(4).to_string()
    )


def plot_uce_barplot(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Diagrama de barras del UCE total (media ± IC 95%) por modelo de estimación
    (x) y modelo de observación (hue).
    """
    if "uce_total" not in df.columns:
        print("Aviso: columna 'uce_total' no encontrada — re-ejecuta evaluate_models.py.")
        return

    algo_order = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]

    fig, ax = plt.subplots(figsize=(8, 3))

    sns.barplot(
        data      = df,
        x         = "algorithm",
        y         = "uce_total",
        hue       = "observation_model",
        order     = algo_order,
        palette   = "Set2",
        ax        = ax,
        width     = 0.6,
        capsize   = 0.05,
        err_kws   = {"linewidth": 1.2},
    )

    ax.set_xlabel("Estimation model")
    ax.set_ylabel(
        r"UCE $= \sum_j \frac{|B_j|}{N}\,\left|\,\mathrm{err}(B_j) - \mathrm{uncert}(B_j)\right|$"
    )
    ax.legend(title="Observation model", ncol=3, loc="upper right", frameon=True)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "uce_barplot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Guardado: {out_path}")
    plt.show()

    # También imprimir los valores numéricos de UCE por consola
    print("\nUCE total por algoritmo y modelo de observación:")
    print(
        df.groupby(["algorithm", "observation_model"])["uce_total"]
        .agg(["mean", "std"])
        .unstack()
        .round(4).to_string()
    )




def plot_calibration_curve(results_dir: Path, output_dir: Path, n_bins: int = 10) -> None:
    """
    Curva de calibración espacial normalizada.

    Eje X → σ̃  (incertidumbre total normalizada min-max por muestra)
    Eje Y → error absoluto medio normalizado min-max por muestra

    Normalizar por muestra elimina diferencias de escala global y deja solo
    la correspondencia de forma: ¿asigna el modelo más σ donde más se equivoca?

    Binning por percentiles de σ̃ (bins equipoblados). La curva de cada
    algoritmo es la media entre muestras ± error estándar (banda sombreada).

    Tres subplots (uno por modelo de observación), una línea por algoritmo.
    """
    def _minmax(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    df_full  = load_results_with_maps(results_dir)
    obs_list = [v for v in OBS_MODEL_NAMES.values() if v in df_full["observation_model"].unique()]
    palette  = dict(zip(ALGO_ORDER, sns.color_palette("Set2", len(ALGO_ORDER))))

    sigma_grid = np.linspace(0, 1, n_bins)   # rejilla común fija en [0,1]

    fig, axes = plt.subplots(1, len(obs_list), figsize=(5 * len(obs_list), 4.5), sharey=True)
    if len(obs_list) == 1:
        axes = [axes]

    for ax, obs_name in zip(axes, obs_list):
        df_obs = df_full[df_full["observation_model"] == obs_name]

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.6, label="Ideal (y = x)", zorder=1)
        ax.fill_between([0, 1], [0, 1], [1, 1],
                        alpha=0.05, color="tomato",   label="Overconfident (error > σ)")
        ax.fill_between([0, 1], [0, 0], [0, 1],
                        alpha=0.05, color="steelblue", label="Underconfident (error < σ)")

        for algo in ALGO_ORDER:
            df_algo = df_obs[df_obs["algorithm"] == algo]
            if df_algo.empty:
                continue

            bin_sigmas_list, bin_errs_list = [], []

            for _, row in df_algo.iterrows():
                eval_mask = (1.0 - row["observed_mask"]).astype(bool)
                if not eval_mask.any():
                    eval_mask = np.ones_like(row["observed_mask"], dtype=bool)

                epi   = row["epistemic_std"].ravel()
                ale   = row["aleatoric_std"].ravel()
                sigma = np.sqrt(epi ** 2 + ale ** 2).clip(min=1e-6)
                abs_err = np.abs(row["ground_truth"].ravel() - row["predicted_mean"].ravel())

                sigma   = sigma[eval_mask.ravel()]
                abs_err = abs_err[eval_mask.ravel()]

                # Normalización min-max por muestra
                sigma_n = _minmax(sigma)
                err_n   = _minmax(abs_err)

                if sigma_n.max() - sigma_n.min() < 1e-8 or err_n.max() - err_n.min() < 1e-8:
                    continue   # mapa constante, no informativo

                # Binning por percentiles de σ̃ (equipoblado)
                bin_edges = np.percentile(sigma_n, np.linspace(0, 100, n_bins + 1))
                bin_edges[-1] += 1e-8
                bin_idx = np.clip(np.digitize(sigma_n, bin_edges) - 1, 0, n_bins - 1)

                bsm = np.full(n_bins, np.nan)
                bem = np.full(n_bins, np.nan)
                for b in range(n_bins):
                    mask_b = bin_idx == b
                    if mask_b.sum() == 0:
                        continue
                    bsm[b] = sigma_n[mask_b].mean()
                    bem[b] = err_n[mask_b].mean()

                valid = ~np.isnan(bsm) & ~np.isnan(bem)
                if valid.sum() < 2:
                    continue

                # Normalizar medias de bin a [0,1] para alinear la rejilla
                bsm_v, bem_v = bsm[valid], bem[valid]
                bsm_v = _minmax(bsm_v)
                bem_v = _minmax(bem_v)

                bin_sigmas_list.append(bsm_v)
                bin_errs_list.append(bem_v)

            if not bin_sigmas_list:
                continue

            # Interpolar a rejilla común y promediar entre muestras
            interp_errs = np.array([
                np.interp(sigma_grid, s, e)
                for s, e in zip(bin_sigmas_list, bin_errs_list)
            ])
            mean_err = interp_errs.mean(axis=0)
            se_err   = interp_errs.std(axis=0) / np.sqrt(len(interp_errs))

            color = palette.get(algo)
            ax.plot(sigma_grid, mean_err,
                    color=color, linewidth=2.0, marker="o", markersize=4,
                    label=algo, zorder=3)
            ax.fill_between(sigma_grid,
                            mean_err - se_err, mean_err + se_err,
                            color=color, alpha=0.2, zorder=2)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(obs_name)
        ax.set_xlabel("σ̃  (normalized uncertainty per sample)")
        ax.get_legend().remove() if ax.get_legend() else None

    axes[0].set_ylabel("Mean absolute error (normalized per sample)")

    # Leyenda compartida extraída del último subplot con datos
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(handles), fontsize=8, frameon=True,
               bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "calibration_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Guardado: {out_path}")
    plt.show()

    # También imprimir la diferencia media de cada punto de la curva respecto a la diagonal ideal por consola
    print("\nCalibración (mean absolute deviation from ideal):")
    for obs_name in obs_list:
        df_obs = df_full[df_full["observation_model"] == obs_name]
        print(f"\n{obs_name}:")
        for algo in ALGO_ORDER:
            df_algo = df_obs[df_obs["algorithm"] == algo]
            if df_algo.empty:
                continue

            deviations = []
            for _, row in df_algo.iterrows():
                eval_mask = (1.0 - row["observed_mask"]).astype(bool)
                if not eval_mask.any():
                    eval_mask = np.ones_like(row["observed_mask"], dtype=bool)

                epi   = row["epistemic_std"].ravel()
                ale   = row["aleatoric_std"].ravel()
                sigma = np.sqrt(epi ** 2 + ale ** 2).clip(min=1e-6)
                abs_err = np.abs(row["ground_truth"].ravel() - row["predicted_mean"].ravel())

                sigma   = sigma[eval_mask.ravel()]
                abs_err = abs_err[eval_mask.ravel()]

                sigma_n = _minmax(sigma)
                err_n   = _minmax(abs_err)

                if sigma_n.max() - sigma_n.min() < 1e-8 or err_n.max() - err_n.min() < 1e-8:
                    continue

                deviation = np.abs(err_n - sigma_n).mean()
                deviations.append(deviation)

            if deviations:
                mean_dev = np.mean(deviations)
                print(f"  {algo}: {mean_dev:.4f}")

    

def plot_sample_maps(
    results_dir: Path,
    output_dir: Path,
    obs_model: str = "Pointwise sensor",
    sample_idx: int = 0,
) -> None:
    """
    Grid de mapas para un único sample: una fila por algoritmo, columnas
    [Ground truth | Observed | Predicted mean | Epistemic std | Aleatoric std].

    GT y Observed son iguales en todas las filas (mismo sample).
    Los algoritmos se muestran en el orden canónico ALGO_ORDER.

    Parameters
    ----------
    obs_model  : nombre del modelo de observación (ya traducido, e.g. "Pointwise sensor")
    sample_idx : índice dentro del test set (posición ordinal dentro del DataFrame
                 filtrado por obs_model, no el test_idx del split).
    """
    df_full = load_results_with_maps(results_dir)
    df_obs  = df_full[df_full["observation_model"] == obs_model]

    if df_obs.empty:
        print(f"No hay datos para observation_model='{obs_model}'.")
        return

    # Tomar el sample_idx-ésimo test_idx que aparezca en todos los algoritmos
    algos_present = [a for a in ALGO_ORDER if a in df_obs["algorithm"].unique()]

    # Buscar test_idx común a todos los algoritmos presentes
    def _test_idx(row):
        try:
            return row["extra"]["test_idx"]
        except (TypeError, KeyError):
            return None

    df_obs = df_obs.copy()
    df_obs["_test_idx"] = df_obs.apply(_test_idx, axis=1)

    sets = [set(df_obs[df_obs["algorithm"] == a]["_test_idx"].dropna()) for a in algos_present]
    common = sorted(sets[0].intersection(*sets[1:]))
    if not common:
        print("No hay test_idx común a todos los algoritmos, usando posición ordinal.")
        common = sorted(df_obs["_test_idx"].dropna().unique())

    if sample_idx >= len(common):
        print(f"sample_idx={sample_idx} fuera de rango ({len(common)} disponibles), usando 0.")
        sample_idx = 0
    chosen_test_idx = common[sample_idx]

    COL_NAMES = ["Ground truth", "Observed", "Predicted mean", "Epistemic std", "Aleatoric std"]
    n_rows = len(algos_present)
    n_cols = len(COL_NAMES)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 3 * n_rows),
                             squeeze=False)

    for r, algo in enumerate(algos_present):
        row = df_obs[(df_obs["algorithm"] == algo) &
                     (df_obs["_test_idx"] == chosen_test_idx)]
        if row.empty:
            for ax in axes[r]:
                ax.axis("off")
            continue
        row = row.iloc[0]

        maps = {
            "Ground truth":  row["ground_truth"],
            "Observed":      row["observed_map"],
            "Predicted mean": row["predicted_mean"],
            "Epistemic std": row["epistemic_std"],
            "Aleatoric std": row["aleatoric_std"],
        }

        # Colormaps: densidad en viridis, incertidumbre en plasma
        cmaps = {
            "Ground truth":   "viridis",
            "Observed":       "viridis",
            "Predicted mean": "viridis",
            "Epistemic std":  "plasma",
            "Aleatoric std":  "plasma",
        }

        # Rango compartido de densidad para las tres primeras columnas
        vmin_d = min(maps["Ground truth"].min(), maps["Predicted mean"].min())
        vmax_d = max(maps["Ground truth"].max(), maps["Predicted mean"].max())

        for c, col_name in enumerate(COL_NAMES):
            ax  = axes[r][c]
            arr = maps[col_name]

            if col_name in ("Ground truth", "Observed", "Predicted mean"):
                im = ax.imshow(arr, cmap=cmaps[col_name],
                               vmin=vmin_d, vmax=vmax_d, origin="upper")
            else:
                im = ax.imshow(arr, cmap=cmaps[col_name], origin="upper")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")

            if r == 0:
                ax.set_title(col_name, fontsize=10, pad=4)

        axes[r][0].set_ylabel(algo, fontsize=10)
        # ylabel en imshow no se muestra bien con axis("off"), usar text
        axes[r][0].text(-0.12, 0.5, algo,
                        transform=axes[r][0].transAxes,
                        va="center", ha="right", fontsize=10, rotation=90)

    fig.suptitle(f"{obs_model}  —  sample {chosen_test_idx}", fontsize=11, y=1.01)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sample_maps_{obs_model.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Guardado: {out_path}")
    plt.show()


def plot_uncertainty_stacked_bars(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Barras acumuladas de incertidumbre epistémica (abajo) y aleatoria (arriba)
    agrupadas por algoritmo (x) y modelo de observación (hue).
    """
    algo_order = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]
    obs_order  = [v for v in OBS_MODEL_NAMES.values() if v in df["observation_model"].unique()]
    palette    = sns.color_palette("Set2", len(obs_order))

    grp   = df.groupby(["algorithm", "observation_model"])[["epistemic_mean", "aleatoric_mean"]]
    means = grp.mean().reset_index()
    stds  = grp.std(ddof=1).reset_index()

    n_algos = len(algo_order)
    n_obs   = len(obs_order)
    width   = 0.8 / n_obs
    x_base  = np.arange(n_algos)
    err_kw  = dict(ecolor="black", capsize=3, capthick=0.8, linewidth=0.8, zorder=4)

    fig, ax = plt.subplots(figsize=(8, 4))

    for k, (obs_name, color) in enumerate(zip(obs_order, palette)):
        offsets = x_base + (k - (n_obs - 1) / 2) * width

        sub_m = means[means["observation_model"] == obs_name].set_index("algorithm")
        sub_s = stds[stds["observation_model"]   == obs_name].set_index("algorithm")

        def _get(sub, col, algo):
            return float(sub.loc[algo, col]) if algo in sub.index else 0.0

        epi_vals = np.array([_get(sub_m, "epistemic_mean", a) for a in algo_order])
        ale_vals = np.array([_get(sub_m, "aleatoric_mean", a) for a in algo_order])
        epi_stds = np.array([_get(sub_s, "epistemic_mean", a) for a in algo_order])
        ale_stds = np.array([_get(sub_s, "aleatoric_mean", a) for a in algo_order])

        ax.bar(offsets, epi_vals, width=width, color=color,
               yerr=epi_stds, error_kw=err_kw, label=obs_name, zorder=3)
        ax.bar(offsets, ale_vals, width=width, bottom=epi_vals,
               color=color, alpha=0.45,
               yerr=ale_stds, error_kw=err_kw, zorder=3)

    # Leyenda: modelos de observación (color sólido) + patrón epistémica/aleatoria
    from matplotlib.patches import Patch
    obs_handles = [Patch(facecolor=c, label=o) for c, o in zip(palette, obs_order)]
    type_handles = [
        Patch(facecolor="gray",          label="Epistemic"),
        Patch(facecolor="gray", alpha=0.45, label="Aleatoric"),
    ]
    ax.legend(handles=obs_handles + type_handles,
              ncol=2, loc="upper right", frameon=True, fontsize=8)

    ax.set_xticks(x_base)
    ax.set_xticklabels(algo_order)
    ax.set_xlabel("Estimation model")
    ax.set_ylabel("Mean uncertainty (std)")

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "uncertainty_stacked_bars.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Guardado: {out_path}")
    plt.show()


def save_inference_time_table(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Genera una tabla con el tiempo medio de inferencia (ms) ± desviación
    típica por algoritmo y modelo de observación.
    """
    algo_order = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]
    obs_order  = [v for v in OBS_MODEL_NAMES.values()
                  if v in df["observation_model"].unique()]

    stats = (
        df.groupby(["algorithm", "observation_model"])["inference_time_ms"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Formatear como tabla md
    table = "| Estimation model | Observation model | Inference time (ms) |\n"
    table += "|------------------|-------------------|---------------------|\n"
    for algo in algo_order:
        for obs in obs_order:
            row = stats[(stats["algorithm"] == algo) &
                        (stats["observation_model"] == obs)]
            if not row.empty:
                mean = row["mean"].values[0]
                std  = row["std"].values[0]
                table += f"| {algo} | {obs} | {mean:.1f} ± {std:.1f} |\n"

    # Convertir tabla md a LaTeX usando pandas

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "inference_time_table.md"
    out_path.write_text(table)
    print(f"Guardado: {out_path}")

    print("\nInference time (ms) por algoritmo y modelo de observación:")
    print(table)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera gráficas de resultados de evaluación."
    )
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR),
        help=f"Directorio con los .pkl.gz (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help=f"Directorio de salida para las imágenes (default: {OUTPUT_DIR})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    df    = load_results(Path(args.results_dir))
    outdir = Path(args.output_dir)

    plot_rmse_boxplot(df, outdir)
    plot_nll_boxplot(df, outdir)
    plot_uce_barplot(df, outdir)
    plot_calibration_curve(Path(args.results_dir), outdir)
    plot_sample_maps(Path(args.results_dir), outdir)
    plot_uncertainty_stacked_bars(df, outdir)
    save_inference_time_table(df, outdir)
