"""
experiment_logger.py — Records per-step metrics across IPP experiments.

Usage
-----
    logger = ExperimentLogger()

    # Inside the episode loop:
    logger.log_step(
        step=step,
        info=info,          # dict from ExtendedScenario.step
        obs=obs,            # dict from ExtendedScenario.step
        position=env.position,
        policy_name="OrienteeringPolicy",
        map_idx=args.map_idx,
        dataset_name="dataset_v1",
        model_name="EnsembleModel-5",
    )

    # After all experiments:
    df = logger.to_dataframe()
    logger.save("results/experiments.csv")
    logger.save("results/experiments.parquet", fmt="parquet")

DataFrame columns
-----------------
    step              int     — step index within the episode (1-based)
    distance          float   — accumulated Euclidean distance so far (px)
    budget            float   — total distance budget for the episode (px)
    policy            str     — name of the policy class / label
    rmse              float   — sqrt(MSE) between prediction and GT at unobserved cells
    iou               float   — IoU of oil-spill masks (threshold 0.05) between
                                predicted_mean and ground truth; NaN if no prediction
    mean_uncertainty  float   — mean predicted uncertainty over the whole map
    coverage          float   — % of map cells covered by the observation mask
    map_idx           int     — index of the ground-truth map used
    pos_x             int     — agent column position after the step
    pos_y             int     — agent row position after the step
    dataset_name      str     — identifier of the dataset used
    model_name        str     — identifier of the estimator model used
    experiment_id     int     — auto-incremented episode counter
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


class ExperimentLogger:
    """
    Accumulates per-step metrics from ExtendedScenario episodes and
    serialises them to a pandas DataFrame.

    Parameters
    ----------
    experiment_id : int
        Starting value for the episode counter.  Useful when resuming a
        logging session across multiple script runs.
    """

    _COLUMNS = [
        "experiment_id",
        "step",
        "distance",
        "budget",
        "policy",
        "rmse",
        "iou",
        "mean_uncertainty",
        "coverage",
        "map_idx",
        "position",
        "dataset_name",
        "model_name",
        "mean_time",
    ]

    _SPILL_THRESHOLD: float = 0.05

    def __init__(self, experiment_id: int = 0) -> None:
        self._records: list[dict] = []
        self.experiment_id: int = experiment_id

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def new_episode(self) -> None:
        """Increment the episode counter.  Call once per env.reset()."""
        self.experiment_id += 1

    def log_step(
        self,
        *,
        step: int,
        info: dict,
        obs: dict,
        ground_truth: np.ndarray,
        position: tuple[tuple[int, int], ...],
        policy_name: str,
        map_idx: int,
        dataset_name: str,
        model_name: str,
        mean_time: float | None = None,
    ) -> None:
        """
        Record one step.

        Parameters
        ----------
        step         : 1-based step index within the current episode.
        info         : dict returned by ExtendedScenario.step
                       (must contain ``mse``, ``distance``, ``budget``).
        obs          : dict returned by ExtendedScenario.step
                       (must contain ``obs_mask``, ``predicted_uncertainty``,
                       ``predicted_mean``).
        ground_truth : (H, W) ground-truth map for the current episode
                       (available as ``env.ground_truth`` after reset).
        position     : ((row, col), ...) agent positions *after* the step.
        policy_name  : free-form label identifying the policy.
        map_idx      : ground-truth map index used for this episode.
        dataset_name : identifier of the dataset (e.g. file stem or config key).
        model_name   : identifier of the estimator model.
        """
        obs_mask: np.ndarray    = obs["obs_mask"]
        uncertainty: np.ndarray = obs["predicted_uncertainty"]
        predicted_mean: np.ndarray = obs["predicted_mean"]

        mse  = float(info["mse"])
        rmse = math.sqrt(mse) if mse >= 0 else float("nan")
        iou  = self._compute_iou(predicted_mean, ground_truth)

        self._records.append({
            "experiment_id":    self.experiment_id,
            "step":             int(step),
            "distance":         info["distance"].copy(),
            "budget":           float(info["budget"]),
            "policy":           policy_name,
            "rmse":             rmse,
            "iou":              iou,
            "mean_uncertainty": float(uncertainty.mean()),
            "coverage":         float(obs_mask.mean()) * 100.0,
            "map_idx":          int(map_idx),
            "position":         position.copy(),
            "dataset_name":     dataset_name,
            "model_name":       model_name,
            "mean_time":        mean_time,
        })

    def _compute_iou(
        self,
        predicted_mean: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """
        IoU between predicted and ground-truth oil-spill binary masks.

        A cell is considered part of the spill when its value > _SPILL_THRESHOLD.
        If both masks are empty (union == 0) the agreement is perfect → 1.0.
        """
        pred_mask = predicted_mean > self._SPILL_THRESHOLD
        gt_mask   = ground_truth   > self._SPILL_THRESHOLD
        intersection = float((pred_mask & gt_mask).sum())
        union        = float((pred_mask | gt_mask).sum())
        return intersection / union if union > 0 else 1.0

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        """Return all recorded rows as a DataFrame with canonical column order."""
        if not self._records:
            return pd.DataFrame(columns=self._COLUMNS)
        return pd.DataFrame(self._records)[self._COLUMNS]

    def save(self, path: str | Path, fmt: str = "csv") -> Path:
        """
        Persist the DataFrame to disk.

        Parameters
        ----------
        path : destination file path.
        fmt  : ``"csv"`` (default) or ``"parquet"``.

        Returns
        -------
        The resolved Path that was written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()

        if fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

        return path

    def clear(self) -> None:
        """Discard all recorded rows (episode counter is preserved)."""
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"ExperimentLogger("
            f"episodes={self.experiment_id}, "
            f"rows={len(self._records)})"
        )
