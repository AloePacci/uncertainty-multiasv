"""
extended_scenario.py — Extended observation scenario with step budget and model.

Extends ObservationScenario so that:
  - The episode ends after `budget` steps (done=True).
  - A prediction model is called on every step; its outputs (mean + uncertainty)
    are returned alongside the observation state.
  - An info dict is returned with the MSE evaluated only on unobserved cells.

Model interface
---------------
The model passed to ExtendedScenario must implement:

    model.predict(obs_map: np.ndarray, obs_mask: np.ndarray)
        -> tuple[np.ndarray, np.ndarray]   # (mean (H,W), uncertainty (H,W))

A ready-made adapter for EnsembleModel is provided at the bottom of this file.

Usage
-----
    from scenario.extended_scenario import ExtendedScenario, EnsembleAdapter
    from MC_ensemble_model import EnsembleModel

    ensemble = EnsembleModel(...)
    model    = EnsembleAdapter(ensemble)
    env      = ExtendedScenario("scenario/scenario_config.yaml", model=model)

    gt = env.reset()
    while True:
        obs, done, info = env.step((row, col))
        env.render()
        if done:
            break
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to sys.path

from scenario import ObservationScenario


# ---------------------------------------------------------------------------
# Extended scenario
# ---------------------------------------------------------------------------

class ExtendedScenario(ObservationScenario):
    """
    Observation scenario extended with a distance budget and a prediction model.

    Parameters
    ----------
    config_path : str | Path
        Path to scenario_config.yaml (must contain a ``budget`` key in pixels).
    model : object
        Any object with a ``predict(obs_map, obs_mask) -> (mean, uncertainty)``
        method.  Both outputs must be np.ndarray of shape (H, W).
    budget : float or None
        Maximum total Euclidean distance (pixels) per episode.
        Overrides the value in the config file when provided explicitly.

    Step return value
    -----------------
    obs  : dict
        ``obs_map``        — (H, W) concentration at visited cells, 0 elsewhere
        ``obs_mask``       — (H, W) 1 at visited cells, 0 elsewhere
        ``predicted_mean`` — (H, W) model's predicted ground-truth map
        ``predicted_uncertainty`` — (H, W) model's uncertainty map
    done : bool
        True when the accumulated Euclidean distance reaches the budget.
    info : dict
        ``mse``              — float, MSE between predicted_mean and ground truth
                               evaluated only at *unobserved* cells.
        ``distance``         — float, total distance travelled so far (pixels).
        ``budget``           — float, total distance budget (pixels).
    """

    def __init__(self, config_path: str | Path, model, budget: float | None = None):
        super().__init__(config_path)

        config_path = Path(config_path)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.budget: float = float(budget if budget is not None else cfg.get("budget", 500))
        self.model = model

        self._distance: list[float] = [0.0] * len(self.initial_position)

        # Last model outputs (for render)
        self._predicted_mean: np.ndarray | None = None
        self._predicted_uncertainty: np.ndarray | None = None

    # ------------------------------------------------------------------
    def reset(self, map_idx: int | None = None) -> np.ndarray:
        self._distance = [0.0] * len(self.initial_position)
        self._predicted_mean = None
        self._predicted_uncertainty = None
        return super().reset(map_idx)

    # ------------------------------------------------------------------
    def step(self, action: tuple[int, int]) -> tuple[dict, bool, dict]:
        """
        Move to *action*, record observations, run the model, compute MSE.

        Parameters
        ----------
        action : (row, col)

        Returns
        -------
        obs  : dict  — observation + model prediction
        done : bool  — True when budget is exhausted
        info : dict  — evaluation metrics
        """
        if self.ground_truth is None:
            raise RuntimeError("Call reset() before step().")

        pos_before = self.position.copy()   # position before the move
        obs_map, obs_mask = super().step(action)
        pos_after = self.position.copy()   # position after the move
        for pos_index in range(len(pos_before)):
            r0, c0 = pos_before[pos_index]
            r1, c1 = pos_after[pos_index]
            self._distance[pos_index] += float(np.hypot(r1 - r0, c1 - c0))

        # ── Model prediction ─────────────────────────────────────────────
        output = self.model.predict({"obs_map": obs_map, "obs_mask": obs_mask})

        # Ensure outputs are (H, W) numpy arrays
        mean = output["predicted_mean"]
        uncertainty = np.sqrt(output["predicted_std_epistemic"]** 2 + output["predicted_std_aleatoric"]**2)
        
        if mean.shape != (self.H, self.W) or uncertainty.shape != (self.H, self.W):
            mean = mean.reshape(self.H, self.W)
            uncertainty = uncertainty.reshape(self.H, self.W)
            
        self._predicted_mean        = mean
        self._predicted_uncertainty = uncertainty

        # ── Evaluation: MSE at unobserved cells only ─────────────────────
        unobserved = obs_mask == 0
        if unobserved.any():
            mse = float(np.mean((mean[unobserved] - self.ground_truth[unobserved]) ** 2))
        else:
            mse = 0.0

        # ── Evaluation: IoU of oil-spill masks (threshold 0.05) ──────────
        _SPILL_THRESHOLD = 0.05
        pred_mask = mean          > _SPILL_THRESHOLD
        gt_mask   = self.ground_truth > _SPILL_THRESHOLD
        intersection = float((pred_mask & gt_mask).sum())
        union        = float((pred_mask | gt_mask).sum())
        iou = intersection / union if union > 0 else 1.0

        done = any(distance >= self.budget for distance in self._distance)

        obs = {
            "obs_map":               obs_map,
            "obs_mask":              obs_mask,
            "predicted_mean":        mean,
            "predicted_uncertainty": uncertainty,
        }
        info = {
            "mse":      mse,
            "iou":      iou,
            "distance": self._distance,
            "budget":   self.budget,
        }

        return obs, done, info

    # ------------------------------------------------------------------
    def render(self) -> None:
        """
        Six-panel interactive figure:
            Ground Truth | Obs Map | Obs Mask | Trajectory |
            Predicted Mean | Predicted Uncertainty
        """
        if self.ground_truth is None:
            raise RuntimeError("Call reset() before render().")

        has_prediction = self._predicted_mean is not None

        if self._fig is None:
            self._fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            self._fig.suptitle("Extended Observation Scenario", fontsize=12)
            axes = axes.flat
            self._axes = list(axes)

            kw = dict(vmin=0, vmax=1, interpolation="nearest")

            self._im_gt   = self._axes[0].imshow(self.ground_truth,                           cmap="hot",    **kw)
            self._im_obs  = self._axes[1].imshow(self.obs_map,                                cmap="hot",    **kw)
            self._im_mask = self._axes[2].imshow(self.obs_mask,                               cmap="gray",   **kw)
            self._im_traj = self._axes[3].imshow(self.ground_truth,                           cmap="hot",    **kw)
            blank = np.zeros((self.H, self.W), dtype=np.float32)
            self._im_pred = self._axes[4].imshow(self._predicted_mean if self._predicted_mean is not None else blank,               cmap="hot",    **kw)
            self._im_unc  = self._axes[5].imshow(self._predicted_uncertainty if self._predicted_uncertainty is not None else blank, cmap="plasma", vmin=0, interpolation="nearest")

            titles = ["Ground Truth", "Observation Map", "Observation Mask",
                      "Trajectory", "Predicted Mean", "Uncertainty"]
            for ax, title in zip(self._axes, titles):
                ax.set_title(title, fontsize=10)
                ax.axis("off")

            self._traj_line = [[]]*len(self.trajectory)
            self._traj_dot = [[]]*len(self.trajectory)

            
            for index, traj in enumerate(self.trajectory):
                # print(f"Trajectory {index}: {traj} on trajline {self._traj_line} and trajdot {self._traj_dot}")
                traj = np.array(traj)
                self._traj_line[index], = axes[3].plot(
                    traj[:, 1], traj[:, 0],
                    color="cyan", linewidth=1.5, alpha=0.8,
                )
                self._traj_dot[index], = axes[3].plot(
                    [self.position[index][1]], [self.position[index][0]],
                    "o", color="lime", markersize=6,
                )

            for ax, im in zip(self._axes[:4], [self._im_gt, self._im_obs, self._im_mask, self._im_traj]):
                self._fig.colorbar(im, ax=ax, fraction=0.046)
            self._fig.colorbar(self._im_pred, ax=self._axes[4], fraction=0.046)
            self._fig.colorbar(self._im_unc,  ax=self._axes[5], fraction=0.046)

            plt.tight_layout()
            plt.ion()
            plt.show()

        else:
            self._im_gt.set_data(self.ground_truth)
            self._im_obs.set_data(self.obs_map)
            self._im_mask.set_data(self.obs_mask)
            self._im_traj.set_data(self.ground_truth)

            if has_prediction:
                self._im_pred.set_data(self._predicted_mean)
                unc = self._predicted_uncertainty
                self._im_unc.set_data(unc)
                self._im_unc.set_clim(vmin=0, vmax=max(float(unc.max()), 1e-6))

            for index, traj in enumerate(self.trajectory):
                traj = np.array(traj)
                self._traj_line[index].set_xdata(traj[:, 1])
                self._traj_line[index].set_ydata(traj[:, 0])
                self._traj_dot[index].set_xdata([self.position[index][1]])
                self._traj_dot[index].set_ydata([self.position[index][0]])

            mse_str = ""
            if has_prediction:
                unobs = self.obs_mask == 0
                if unobs.any():
                    mse = float(np.mean(
                        (self._predicted_mean[unobs] - self.ground_truth[unobs]) ** 2
                    ))
                    mse_str = f"  |  MSE (unobs): {mse:.4f}"
            self._fig.suptitle(
                f"Extended Scenario  —  dist {self._distance}/{self.budget:.0f} px{mse_str}",
                fontsize=11,
            )

            self._fig.canvas.draw()
            self._fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# ModelAdapter 
# ---------------------------------------------------------------------------

class ModelAdapter:
    """
    Thin wrapper that delegates to any model implementing the dict-based
    predict interface.  Kept for backward compatibility with code that
    instantiates models outside of ExtendedScenario.
    """

    def __init__(self, model):
        self._model = model

    def predict(self, obs: dict) -> dict:
        return self._model.predict(obs)


# ---------------------------------------------------------------------------
# Quick demo (GaussianProcessModel)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.append(str(Path(__file__)))  # Add parent directory to sys.path
    from models.gaussian_process_model import GaussianProcessModel

    cfg   = Path(__file__).parent / "scenario_config.yaml"
    model = ModelAdapter(GaussianProcessModel())
    env   = ExtendedScenario(cfg, model=model)

    gt = env.reset(map_idx=0)
    print(f"Ground truth shape : {gt.shape},  budget : {env.budget}")

    env.render()

    rng = np.random.default_rng(0)
    done = False
    while not done:
        wp = [(np.random.randint(0, env.H), np.random.randint(0, env.W)) for _ in range(len(env.position))] 
        obs, done, info = env.step(wp)
        # print(f"  dist {info['distance']}/{info['budget']:.0f} px  "
        #       f"→ {wp}  |  MSE(unobs)={info['mse']:.4f}  |  done={done}")
        env.render()
        plt.pause(0.3)

    print("Episode finished.")
    plt.ioff()
    plt.show()
