"""
evaluation_store.py — Uncertainty Estimation CAEPIA 2026
=========================================================
Sistema de almacenamiento y recuperación de resultados de evaluación.

Cada fila del DataFrame representa una muestra evaluada. Las columnas incluyen:
  · Metadatos (algorithm, observation_model, map_shape, percentage_covered, extra)
  · Estadísticas de los mapas de incertidumbre
  · Métricas escalares (rmse, r2, nll_*, ence_*)
  · Mapas espaciales (H×W) almacenados como arrays NumPy en cada celda

El DataFrame completo se persiste en un único fichero pickle comprimido (.pkl.gz).
Al cargarlo con pandas se obtiene directamente el DataFrame completo.

Uso típico
----------
    store = EvaluationStore("experimento_1")

    idx = store.add_record(
        prediction        = model.predict(X),   # salida directa del modelo (B=1)
        ground_truth      = gt_map,             # (H, W)
        algorithm         = "MC_Dropout",
        observation_model = "NadirFOV",
    )

    store.save()   # escribe Results/experimento_1.pkl.gz

    # --- análisis posterior ---
    store2 = EvaluationStore.load("Results/experimento_1.pkl.gz")
    df = store2.to_dataframe()                    # DataFrame completo (con arrays)
    df_s = store2.to_scalar_dataframe()           # sin columnas de arrays
    print(df_s.groupby("algorithm")[["rmse", "ence_total"]].mean())
    maps = store2.get_maps(idx)   # dict con predicted_mean, epistemic_std, …
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import rmse, r2, nll, ence, uce


# ---------------------------------------------------------------------------
# Columnas del DataFrame
# ---------------------------------------------------------------------------

# Columnas que contienen arrays espaciales (H×W)
MAP_COLUMNS = [
    "ground_truth",
    "predicted_mean",
    "epistemic_std",
    "aleatoric_std",
    "observed_map",
    "observed_mask",
]

# Columnas de métricas escalares
METRIC_COLUMNS = [
    "rmse", "r2",
    "nll_total", "nll_epistemic", "nll_aleatoric",
    "ence_total", "ence_epistemic", "ence_aleatoric",
    "uce_total", "uce_epistemic", "uce_aleatoric",
]

# Columnas de estadísticas de los mapas de incertidumbre
STATS_COLUMNS = [
    "epistemic_mean", "epistemic_min", "epistemic_max",
    "aleatoric_mean", "aleatoric_min", "aleatoric_max",
    "predicted_mean_mean", "predicted_mean_min", "predicted_mean_max",
]

# Columnas de metadatos escalares
META_COLUMNS = [
    "algorithm",
    "observation_model",
    "map_shape",
    "percentage_covered",
    "extra",
]


# ---------------------------------------------------------------------------
# EvaluationStore
# ---------------------------------------------------------------------------

class EvaluationStore:
    """
    Almacén de resultados de evaluación para experimentos de estimación de
    incertidumbre.

    Parameters
    ----------
    name : str
        Nombre del experimento; define los nombres de los ficheros de salida.
    output_dir : str
        Directorio donde se guardarán los ficheros (se crea si no existe).
    """

    def __init__(self, name: str, output_dir: str = "Results"):
        self.name       = name
        self.output_dir = Path(output_dir)
        self._records: list[dict] = []

    # ------------------------------------------------------------------
    # Añadir registro
    # ------------------------------------------------------------------

    def add_record(
        self,
        prediction: dict,
        ground_truth: np.ndarray,
        algorithm: str,
        observation_model: str,
        mask_eval: np.ndarray | None = None,
        extra: dict | None = None,
        inference_time_ms: float | None = None,
    ) -> int:
        """
        Procesa la salida de un modelo, calcula todas las métricas y almacena
        el registro completo (mapas + escalares) en el DataFrame interno.

        Parameters
        ----------
        prediction : dict
            Salida de ``model.predict(X)`` para B=1. Claves esperadas:
            ``'predicted_mean'``, ``'predicted_std_epistemic'`` y
            ``'predicted_std_aleatoric'``, cada una con forma (1, 1, H, W).
            Opcionalmente puede incluir ``'observed_map'`` y ``'observed_mask'``.
        ground_truth : np.ndarray  (H, W)
            Mapa de verdad contra el que se evalúa la predicción.
        algorithm : str
            Nombre del algoritmo (e.g. 'GP', 'MC_Dropout', 'EDL').
        observation_model : str
            Nombre del modelo de observación (e.g. 'Nadir', 'Conic').
        mask_eval : np.ndarray or None  (H, W)
            Máscara binaria (1=evaluar). Si es None se evalúa sobre los
            píxeles no observados (1 - observed_mask).
        extra : dict or None
            Metadatos adicionales arbitrarios.

        Returns
        -------
        int
            Índice de la fila añadida en el DataFrame interno.
        """

        # ── Extraer y aplanar mapas a (H, W) ─────────────────────────────
        def _squeeze(arr):
            """(B,1,H,W) o (H,W) → (H,W) float32."""
            a = np.asarray(arr, dtype=np.float32)
            while a.ndim > 2:
                a = a.squeeze(0)
            return a

        pred_mean = _squeeze(prediction["predicted_mean"])
        epi_std   = _squeeze(prediction["predicted_std_epistemic"])
        ale_std   = _squeeze(prediction["predicted_std_aleatoric"])

        # observed_map / observed_mask — opcionales en el dict de predicción
        obs_map  = _squeeze(prediction["observed_map"])  \
                   if "observed_map"  in prediction else np.zeros_like(pred_mean)
        obs_mask = _squeeze(prediction["observed_mask"]) \
                   if "observed_mask" in prediction else np.zeros_like(pred_mean)

        gt = np.asarray(ground_truth, dtype=np.float32)
        if gt.ndim != 2 or gt.shape != pred_mean.shape:
            raise ValueError(
                f"ground_truth shape {gt.shape} does not match "
                f"predicted_mean shape {pred_mean.shape}."
            )

        # ── Máscara de evaluación ─────────────────────────────────────────
        if mask_eval is not None:
            eval_mask = np.asarray(mask_eval, dtype=np.float32)
        else:
            # Evaluar sobre píxeles NO observados (generalización)
            eval_mask = 1.0 - obs_mask
            if eval_mask.sum() == 0:
                eval_mask = None   # si toda la máscara es 0, evaluar todo

        # ── Porcentaje de cobertura ───────────────────────────────────────
        pct_covered = float(obs_mask.mean()) * 100.0   # en %

        # ── Estadísticas de los mapas de incertidumbre ────────────────────
        def _stats(arr, name):
            return {
                f"{name}_mean": float(arr.mean()),
                f"{name}_min":  float(arr.min()),
                f"{name}_max":  float(arr.max()),
            }

        stats = {}
        stats.update(_stats(epi_std,   "epistemic"))
        stats.update(_stats(ale_std,   "aleatoric"))
        stats.update(_stats(pred_mean, "predicted_mean"))

        # ── Std total como combinación cuadrática ─────────────────────────
        total_std = np.sqrt(epi_std ** 2 + ale_std ** 2).clip(min=1e-6)

        # ── Métricas ──────────────────────────────────────────────────────
        metrics_dict = {
            "rmse": rmse(gt, pred_mean, mask=eval_mask),
            "r2":   r2(  gt, pred_mean, mask=eval_mask),
            # NLL
            "nll_total":     nll(gt, pred_mean, total_std,  mask=eval_mask),
            "nll_epistemic": nll(gt, pred_mean, epi_std.clip(min=1e-6),
                                 mask=eval_mask),
            "nll_aleatoric": nll(gt, pred_mean, ale_std.clip(min=1e-6),
                                 mask=eval_mask),
            # ence
            "ence_total":     ence(gt, pred_mean, total_std,  mask=eval_mask),
            "ence_epistemic": ence(gt, pred_mean, epi_std.clip(min=1e-6),
                                 mask=eval_mask),
            "ence_aleatoric": ence(gt, pred_mean, ale_std.clip(min=1e-6),
                                 mask=eval_mask),
            # uce
            "uce_total":     uce(gt, pred_mean, total_std,  mask=eval_mask),
            "uce_epistemic": uce(gt, pred_mean, epi_std.clip(min=1e-6),
                                 mask=eval_mask),
            "uce_aleatoric": uce(gt, pred_mean, ale_std.clip(min=1e-6),
                                 mask=eval_mask),
        }

        # ── Registro completo (escalares + mapas) ────────────────────────
        record = {
            # Metadatos
            "algorithm":          algorithm,
            "observation_model":  observation_model,
            "map_shape":          pred_mean.shape,
            "percentage_covered": pct_covered,
            "inference_time_ms":  inference_time_ms,
            "extra":              extra,
            # Estadísticas
            **stats,
            # Métricas
            **metrics_dict,
            # Mapas espaciales (arrays NumPy en cada celda)
            "ground_truth":   gt,
            "predicted_mean": pred_mean,
            "epistemic_std":  epi_std,
            "aleatoric_std":  ale_std,
            "observed_map":   obs_map,
            "observed_mask":  obs_mask,
        }
        self._records.append(record)
        return len(self._records) - 1

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def _file_path(self, path: str | None) -> Path:
        if path is not None:
            return Path(path)
        return self.output_dir / f"{self.name}.pkl.gz"

    def save(self, path: str | None = None) -> Path:
        """
        Persiste el store en un único fichero pickle comprimido (.pkl.gz).

        El fichero puede cargarse directamente con ``pd.read_pickle()``
        para obtener el DataFrame completo con mapas y métricas.

        Parameters
        ----------
        path : str or None
            Ruta completa del fichero de salida.
            Por defecto ``{output_dir}/{name}.pkl.gz``.

        Returns
        -------
        Path
            Ruta del fichero guardado.
        """
        fpath = self._file_path(path)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_pickle(str(fpath))
        print(f"Saved {len(self._records)} records → {fpath}")
        return fpath

    @classmethod
    def load(cls, path: str) -> "EvaluationStore":
        """
        Carga un ``EvaluationStore`` desde disco.

        Parameters
        ----------
        path : str
            Ruta al fichero ``.pkl.gz`` generado por ``save()``.

        Returns
        -------
        EvaluationStore
        """
        fpath = Path(path)
        df = pd.read_pickle(str(fpath))
        store = cls.__new__(cls)
        store.name       = fpath.name.replace(".pkl.gz", "").replace(".pkl", "")
        store.output_dir = fpath.parent
        store._records   = df.to_dict(orient="records")
        return store

    # ------------------------------------------------------------------
    # Acceso a datos
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Devuelve el DataFrame completo, incluyendo las columnas de arrays
        espaciales (``ground_truth``, ``predicted_mean``, etc.).
        """
        return pd.DataFrame(self._records)

    def to_scalar_dataframe(self) -> pd.DataFrame:
        """
        Devuelve el DataFrame sin las columnas de arrays espaciales.
        Útil para análisis numérico rápido y exportación ligera.
        """
        df = self.to_dataframe()
        drop = [c for c in MAP_COLUMNS if c in df.columns]
        return df.drop(columns=drop)

    def get_maps(self, index: int) -> dict[str, np.ndarray]:
        """
        Devuelve los mapas espaciales de la fila ``index``.

        Parameters
        ----------
        index : int
            Índice entero devuelto por ``add_record()``.

        Returns
        -------
        dict con claves:
            ``ground_truth``, ``predicted_mean``, ``epistemic_std``,
            ``aleatoric_std``, ``observed_map``, ``observed_mask``
        """
        record = self._records[index]
        return {k: record[k] for k in MAP_COLUMNS if k in record}

    def summary(self) -> pd.DataFrame:
        """
        Resumen por algoritmo: media y desviación estándar de todas las
        métricas y estadísticas numéricas.

        Returns
        -------
        pd.DataFrame con MultiIndex de columnas (métrica, estadístico).

        Ejemplo de salida::

            algorithm        rmse            ence_total
                        mean   std       mean    std
            EDL         0.12   0.03      0.08    0.02
            MC_Dropout  0.15   0.04      0.10    0.03
        """
        df = self.to_scalar_dataframe()
        available = [c for c in METRIC_COLUMNS + STATS_COLUMNS + ["percentage_covered"]
                     if c in df.columns]
        return df.groupby("algorithm")[available].agg(["mean", "std"])

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"EvaluationStore(name={self.name!r}, "
            f"records={len(self._records)}, "
            f"output_dir={str(self.output_dir)!r})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, W = 64, 64

    def _synth_gt():
        cy, cx = H // 2, W // 2
        yy, xx = np.ogrid[:H, :W]
        return np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (H / 5) ** 2)
        ).astype(np.float32)

    def _synth_prediction(gt, coverage=0.15, noise=0.05, epi_scale=0.1, ale_scale=0.05):
        mask = (rng.random((H, W)) < coverage).astype(np.float32)
        obs  = (gt + rng.normal(0, noise, (H, W)).astype(np.float32)) * mask
        pred_mean = gt + rng.normal(0, noise, (H, W)).astype(np.float32)
        epi_std   = (np.abs(rng.normal(epi_scale, epi_scale * 0.3, (H, W)))).astype(np.float32)
        ale_std   = (np.abs(rng.normal(ale_scale, ale_scale * 0.3, (H, W)))).astype(np.float32)
        return {
            "predicted_mean":          pred_mean[None, None],   # (1,1,H,W)
            "predicted_std_epistemic": epi_std[None, None],
            "predicted_std_aleatoric": ale_std[None, None],
            "observed_map":            obs[None, None],
            "observed_mask":           mask[None, None],
        }, mask

    # ── Crear store y añadir registros ────────────────────────────────────
    store = EvaluationStore("test_eval", output_dir="Results")
    gt = _synth_gt()

    indices = []
    configs = [
        ("MC_Dropout", "Nadir",  dict(coverage=0.10, epi_scale=0.12)),
        ("MC_Dropout", "Conic",  dict(coverage=0.20, epi_scale=0.09)),
        ("EDL",        "Nadir",  dict(coverage=0.10, ale_scale=0.04)),
        ("EDL",        "Conic",  dict(coverage=0.20, ale_scale=0.03)),
        ("GP",         "Nadir",  dict(coverage=0.15, epi_scale=0.08, ale_scale=0.02)),
    ]
    for algo, obs_model, kwargs in configs:
        pred, _ = _synth_prediction(gt, **kwargs)
        idx = store.add_record(
            prediction        = pred,
            ground_truth      = gt,
            algorithm         = algo,
            observation_model = obs_model,
            extra             = {"note": f"synth test — {algo}"},
        )
        indices.append(idx)
        print(f"Added record: idx={idx}  algo={algo}  obs={obs_model}")

    print(f"\nStore: {store}\n")

    # ── Guardar ───────────────────────────────────────────────────────────
    saved_path = store.save()

    # ── Recargar y verificar integridad ───────────────────────────────────
    store2 = EvaluationStore.load(str(saved_path))
    assert len(store2) == len(store), "Record count mismatch after reload!"
    for i in indices:
        m1 = store.get_maps(i)
        m2 = store2.get_maps(i)
        for key in MAP_COLUMNS:
            assert np.allclose(m1[key], m2[key]), f"Map mismatch for idx={i} / {key}"
    print("Reload integrity check: OK\n")

    # ── Vista rápida del DataFrame escalar ───────────────────────────────
    df_s = store2.to_scalar_dataframe()
    print("Scalar DataFrame (sin arrays):")
    print(df_s[["algorithm", "observation_model", "rmse", "ence_total"]].to_string())
    print("\nResumen por algoritmo:")
    print(store2.summary()[["rmse", "ence_total"]].to_string())

    # ── DataFrame con métricas clave ─────────────────────────────────────
    df = store2.to_dataframe()
    print("=== Metrics DataFrame (key columns) ===")
    print(df[["algorithm", "observation_model", "percentage_covered",
              "rmse", "r2", "nll_total", "ence_total"]].to_string(index=False))

    # ── Resumen por algoritmo ─────────────────────────────────────────────
    print("\n=== Summary by algorithm (mean +- std) ===")
    print(store2.summary()[["rmse", "nll_total", "ence_total"]].to_string())
