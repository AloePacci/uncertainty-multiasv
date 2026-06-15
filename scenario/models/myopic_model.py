"""
myopic_model.py — Myopic (inverse-distance-weighted) interpolation model.

Predicts the scalar field at unobserved cells as a weighted average of all
observed values, where weights are inversely proportional to distance
(classic IDW — Shepard's method).  No offline training required.

Interface (identical to GaussianProcessModel and EnsembleModel):

    model.predict(obs)  →  dict with keys:
        'predicted_mean'           : np.ndarray  (B, 1, H, W)  float32
        'predicted_std_epistemic'  : np.ndarray  (B, 1, H, W)  float32
        'predicted_std_aleatoric'  : np.ndarray  (B, 1, H, W)  float32

Input obs : dict
    'obs_map'  : array-like (H, W) or (B, H, W) — observed values (0 at unobserved)
    'obs_mask' : array-like (H, W) or (B, H, W) — binary observation mask

Uncertainty
-----------
Epistemic σ(x) : distance to the nearest observed cell, normalised so that
                 the farthest unobserved point has σ = 1.  Zero at observed cells.
Aleatoric σ(x) : zeros (the model does not estimate sensor noise).

IDW power parameter
-------------------
The ``power`` parameter controls how quickly influence drops with distance:
    w_i = 1 / d(x, x_i)^power
Higher power → more local (only nearby points matter).
power=1 → linear distance weighting; power=2 → classic Shepard's method.
"""

import numpy as np
from scipy.spatial import cKDTree


class MyopicModel:
    """
    Inverse-distance-weighted (IDW) field interpolator.

    Parameters
    ----------
    power : float
        IDW exponent p.  Weight of observation i at query point x is
        proportional to 1 / dist(x, x_i)^p.  Default: 1 (linear).
    k_neighbors : int
        Number of nearest observations to use for each query point.
        Using all observations causes the result to converge toward the
        global mean; limiting to k nearby points keeps the model local
        ("myopic").  Default: 5.  Use k=1 for pure nearest-neighbor.
    epsilon : float
        Small value added to distances to avoid division by zero when a
        query point coincides exactly with an observation.
    """

    def __init__(self, power: float = 1.0, k_neighbors: int = 5, epsilon: float = 1e-6):
        self.power       = float(power)
        self.k_neighbors = k_neighbors
        self.epsilon     = float(epsilon)

    # ------------------------------------------------------------------
    def train(self, *args, **kwargs) -> list:
        """No-op: IDW requires no offline training."""
        return []

    # ------------------------------------------------------------------
    def predict(self, obs: dict) -> dict:
        """
        Parameters
        ----------
        obs : dict
            'obs_map'  : (H, W) or (B, H, W)
            'obs_mask' : (H, W) or (B, H, W)

        Returns
        -------
        dict with 'predicted_mean', 'predicted_std_epistemic',
        'predicted_std_aleatoric' — each (B, 1, H, W) float32.
        """
        obs_map  = np.asarray(obs["obs_map"],  dtype=np.float32)
        obs_mask = np.asarray(obs["obs_mask"], dtype=np.float32)
        if obs_map.ndim == 2:
            obs_map  = obs_map[np.newaxis]
            obs_mask = obs_mask[np.newaxis]

        B, H, W = obs_map.shape
        means   = np.zeros((B, 1, H, W), dtype=np.float32)
        epi_std = np.zeros((B, 1, H, W), dtype=np.float32)
        ale_std = np.zeros((B, 1, H, W), dtype=np.float32)

        for b in range(B):
            means[b, 0], epi_std[b, 0] = self._predict_single(
                obs_map[b], obs_mask[b], H, W
            )

        return {
            "predicted_mean":          means,
            "predicted_std_epistemic": epi_std,
            "predicted_std_aleatoric": ale_std,
        }

    # ------------------------------------------------------------------
    def _predict_single(
        self,
        obs_map:  np.ndarray,
        obs_mask: np.ndarray,
        H: int,
        W: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """IDW interpolation and uncertainty for one sample."""

        obs_rows, obs_cols = np.where(obs_mask > 0.5)

        # ── Full grid coordinates ─────────────────────────────────────────
        grid_r, grid_c = np.mgrid[0:H, 0:W]
        grid_coords = np.column_stack([grid_r.ravel(), grid_c.ravel()])  # (H*W, 2)

        # ── No observations: return zeros + maximum uncertainty ───────────
        if len(obs_rows) == 0:
            mean    = np.zeros((H, W), dtype=np.float32)
            epi_std = np.ones((H, W),  dtype=np.float32)
            return mean, epi_std

        obs_coords = np.column_stack([obs_rows, obs_cols])  # (N_obs, 2)
        obs_values = obs_map[obs_rows, obs_cols]             # (N_obs,)

        tree = cKDTree(obs_coords)
        k = min(self.k_neighbors, len(obs_rows)) if self.k_neighbors is not None else len(obs_rows)

        # ── IDW mean (k-nearest neighbours only) ─────────────────────────
        dists, idxs = tree.query(grid_coords, k=k, workers=-1)
        # dists: (H*W, k)  — distances to the k nearest observations
        # idxs:  (H*W, k)  — indices into obs_values

        if k == 1:
            # Scalar output from query when k=1 — reshape for uniformity
            dists = dists[:, np.newaxis]
            idxs  = idxs[:, np.newaxis]

        dists   = np.maximum(dists, self.epsilon)           # avoid /0
        weights = 1.0 / dists ** self.power                 # (H*W, k)
        weights_sum = weights.sum(axis=1, keepdims=True)    # (H*W, 1)
        neighbour_values = obs_values[idxs]                 # (H*W, k)
        predicted = (weights * neighbour_values).sum(axis=1) / weights_sum.squeeze(1)
        mean = predicted.reshape(H, W).astype(np.float32)

        # At exactly observed cells, replace IDW estimate with the true reading
        mean[obs_rows, obs_cols] = obs_values

        # ── Epistemic uncertainty: normalised distance to nearest obs ─────
        nearest_dist, _ = tree.query(grid_coords, k=1, workers=-1)  # (H*W,)
        nearest_dist = nearest_dist.reshape(H, W).astype(np.float32)

        # Observed cells have 0 uncertainty by definition
        nearest_dist[obs_rows, obs_cols] = 0.0

        # Normalise so the farthest unobserved cell has σ = 1
        max_dist = nearest_dist.max()
        if max_dist > 0:
            epi_std = nearest_dist / max_dist
        else:
            epi_std = nearest_dist  # all cells observed → all zeros

        return mean, epi_std


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    H, W = 100, 100

    # Synthetic ground truth: 2D Gaussian bump
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    gt = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (H / 5) ** 2)).astype(np.float32)

    # Sparse observation mask (~10% of pixels)
    mask = (rng.random((H, W)) < 0.10).astype(np.float32)
    obs  = gt * mask

    model = MyopicModel(power=1, k_neighbors=10)  # less aggressive than classic IDW (power=2)
    out   = model.predict({"obs_map": obs, "obs_mask": mask})

    mean = out["predicted_mean"][0, 0]
    epi  = out["predicted_std_epistemic"][0, 0]

    print(f"predicted_mean  shape={mean.shape}  range=[{mean.min():.3f}, {mean.max():.3f}]")
    print(f"epistemic_std   shape={epi.shape}   range=[{epi.min():.3f}, {epi.max():.3f}]")
    assert epi[mask > 0.5].max() == 0.0, "Uncertainty must be 0 at observed cells"
    assert epi.max() == 1.0,             "Max uncertainty must be 1"
    print("All checks passed.")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("MyopicModel — IDW interpolation", fontsize=12)

    panels = [
        (gt,   "Ground Truth",        "viridis"),
        (mask, "Observation Mask",    "gray"),
        (mean, "Predicted Mean (IDW)","viridis"),
        (epi,  "Epistemic Std",       "plasma"),
    ]
    for ax, (data, title, cmap) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
