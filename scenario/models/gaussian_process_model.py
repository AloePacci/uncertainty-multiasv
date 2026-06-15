"""
gaussian_process_model.py — Uncertainty Estimation CAEPIA 2026
===========================================================
Classes that implement the uncertainty estimation methods compared in the paper.

Each class exposes a common interface:

    model.predict(X)  →  dict with keys:
        'predicted_mean'           : (B, 1, H, W)  float32   — field reconstruction
        'predicted_std_epistemic'  : (B, 1, H, W)  float32   — epistemic std  σ_epi
        'predicted_std_aleatoric'  : (B, 1, H, W)  float32   — aleatoric std  σ_ale

Input X : array-like or torch.Tensor  (B, 2, H, W)
    Channel 0 — observation mask  M ∈ {0, 1}^{H×W}
    Channel 1 — observed values   V ∈ R^{H×W}  (0 at unobserved locations)

Classes
-------
GaussianProcessModel
    Kernel: C(·) · RBF(·) + White(·).
    Epistemic σ : posterior std of the signal kernel (decreases with coverage).
    Aleatoric σ : sqrt of the fitted WhiteKernel noise_level (spatially uniform).
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from multiprocessing import Pool, cpu_count

import warnings

# Supress warning

import warnings
from sklearn.exceptions import ConvergenceWarning

# Opción global (para todo el script)
warnings.filterwarnings("ignore", category=ConvergenceWarning)



# ---------------------------------------------------------------------------
# Gaussian Process
# ---------------------------------------------------------------------------

class GaussianProcessModel:
    """
    Scalar field reconstruction and uncertainty decomposition via Gaussian
    Processes (scikit-learn backend).

    Kernel
    ------
    k(x, x') = C · exp(−‖x−x'‖² / (2ℓ²))  +  White(noise_level)
               └──────────────────────────┘    └────────────────────┘
                      signal (C · RBF)             homoscedastic noise

    Uncertainty decomposition
    -------------------------
    Epistemic σ(x) : posterior std of the *signal* kernel, returned by
                     sklearn's predict(return_std=True). Decreases where the
                     vehicle has collected observations.
    Aleatoric σ(x) : sqrt(fitted noise_level) · y_train_std — constant over
                     the domain, reflects irreducible sensor noise as estimated
                     by marginal log-likelihood optimisation.

    Parameters
    ----------
    constant_value : float
        Initial signal variance C₀.
    constant_value_bounds : (float, float)
        Optimisation bounds for C.
    length_scale : float
        Initial RBF length-scale ℓ₀ (in normalised [0,1] coordinate space).
    length_scale_bounds : (float, float)
        Optimisation bounds for ℓ.
    noise_level : float
        Initial noise variance σ²_n₀.
    noise_level_bounds : (float, float)
        Optimisation bounds for σ²_n.
    n_restarts_optimizer : int
        Number of random restarts for kernel hyperparameter optimisation.
    """

    def __init__(
        self,
        constant_value: float         = 1.0,
        constant_value_bounds: tuple  = (1e-2, 1e2),
        length_scale: float           = 10,
        length_scale_bounds: tuple    = (1e-2, 1e2),
        noise_level: float            = 0.1,
        noise_level_bounds: tuple     = (1e-5, 1e1),
        n_restarts_optimizer: int     = 0,
        normalize_y: bool              = True,
    ):
        self.constant_value        = constant_value
        self.constant_value_bounds = constant_value_bounds
        self.length_scale          = length_scale
        self.length_scale_bounds   = length_scale_bounds
        self.noise_level           = noise_level
        self.noise_level_bounds    = noise_level_bounds
        self.n_restarts_optimizer  = n_restarts_optimizer
        self.normalize_y           = normalize_y
    # ------------------------------------------------------------------
    def _build_gp(self) -> GaussianProcessRegressor:
        kernel = (
            ConstantKernel(self.constant_value, self.constant_value_bounds)
            * RBF(self.length_scale, self.length_scale_bounds)
            + WhiteKernel(self.noise_level, self.noise_level_bounds)
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,       # centres and scales y for numerical stability
            alpha=1e-5,                      # small nugget for numerical stability (added to diagonal of kernel matrix)
        )


    # ------------------------------------------------------------------
    def _predict_single(
        self,
        gp: GaussianProcessRegressor,
        obs_map: np.ndarray,
        obs_mask: np.ndarray,
    ):
        """
        Fit the GP on observed points and predict over the full H×W grid.

        Coordinates are normalised to [0, 1]² so that the length-scale
        hyperparameter is grid-size agnostic.

        Parameters
        ----------
        obs_map  : (H, W) float32  — observed values (0 at unobserved cells)
        obs_mask : (H, W) float32  — binary observation mask

        Returns
        -------
        mean     : (H, W) float32
        epi_std  : (H, W) float32  — epistemic standard deviation
        ale_std  : (H, W) float32  — aleatoric standard deviation (uniform map)
        """
        H, W = obs_map.shape

        # ── Extract observed samples ──────────────────────────────────────
        rows, cols = np.where(obs_mask > 0.5)

        if len(rows) == 0:
            # No observations: return prior (zero mean, prior signal std)
            prior_signal_std = np.sqrt(self.constant_value)
            prior_noise_std  = np.sqrt(self.noise_level)
            return (
                np.zeros((H, W), dtype=np.float32),
                np.full((H, W), prior_signal_std, dtype=np.float32),
                np.full((H, W), prior_noise_std,  dtype=np.float32),
            )

        # Normalise coordinates to [0, 1]
        row_norm = rows / max(H - 1, 1)
        col_norm = cols / max(W - 1, 1)
        X_obs = np.column_stack([row_norm, col_norm])   # (N_obs, 2)
        y_obs = obs_map[rows, cols].astype(np.float64)  # (N_obs,)

        # ── Fit ──────────────────────────────────────────────────────────
        gp.fit(X_obs, y_obs)

        # ── Full-grid coordinates ─────────────────────────────────────────
        grid_r, grid_c = np.mgrid[0:H, 0:W]
        X_grid = np.column_stack([
            grid_r.ravel() / max(H - 1, 1),
            grid_c.ravel() / max(W - 1, 1),
        ])  # (H*W, 2)

        # ── Predict ───────────────────────────────────────────────────────
        # sklearn returns posterior std of the signal kernel (epistemic).
        # With normalize_y=True the result is already in the original y scale.
        mean_flat, epi_std_flat = gp.predict(X_grid, return_std=True)
        epi_std_flat = np.maximum(epi_std_flat, 0.0)   # numerical safety

        # ── Aleatoric: WhiteKernel noise in original y scale ──────────────
        # kernel_ structure after fitting:  (ConstantKernel * RBF)  +  WhiteKernel
        #   gp.kernel_.k1  →  ConstantKernel * RBF
        #   gp.kernel_.k2  →  WhiteKernel
        # noise_level is the variance in the *normalised* space; de-normalise:
        #   ale_std_original = sqrt(noise_level_normalised) * y_train_std
        noise_var_norm = gp.kernel_.k2.noise_level          # scalar, normalised
        y_std_raw = getattr(gp, "_y_train_std", 1.0)
        y_std = float(np.squeeze(y_std_raw)) if hasattr(y_std_raw, "__len__") else float(y_std_raw)
        ale_std_val = float(np.sqrt(noise_var_norm) * y_std)
        ale_std_val = max(ale_std_val, 1e-6)                # safety floor

        mean    = mean_flat.reshape(H, W).astype(np.float32)
        epi_std = epi_std_flat.reshape(H, W).astype(np.float32)
        ale_std = np.full((H, W), ale_std_val, dtype=np.float32)

        return mean, epi_std, ale_std

    # ------------------------------------------------------------------
    def train(self, X_train=None, y_train=None, **kwargs) -> list:
        """
        No-op: the GP is fitted per-sample inside predict() using the
        observations contained in each input X. No offline training phase
        is required or possible for this model.

        Returns an empty list (compatible with the common train interface).
        """
        return []


    # ------------------------------------------------------------------
    def predict(self, obs: dict) -> dict:
        """
        Fit and predict for each sample in a batch.

        Parameters
        ----------
        obs : dict with keys:
            'obs_map'  : array-like (H, W) or (B, H, W) — observed values
            'obs_mask' : array-like (H, W) or (B, H, W) — binary observation mask

        Returns
        -------
        dict with:
            'predicted_mean'           : np.ndarray  (B, 1, H, W)
            'predicted_std_epistemic'  : np.ndarray  (B, 1, H, W)
            'predicted_std_aleatoric'  : np.ndarray  (B, 1, H, W)
        """
        obs_map  = np.asarray(obs["obs_map"],  dtype=np.float32)
        obs_mask = np.asarray(obs["obs_mask"], dtype=np.float32)
        if obs_map.ndim == 2:
            obs_map  = obs_map[np.newaxis]
            obs_mask = obs_mask[np.newaxis]
        # Build (B, 2, H, W): channel 0 = mask, channel 1 = values
        X = np.stack([obs_mask, obs_map], axis=1)

        B, C, H, W = X.shape
        means   = np.zeros((B, 1, H, W), dtype=np.float32)
        epi_std = np.zeros((B, 1, H, W), dtype=np.float32)
        ale_std = np.zeros((B, 1, H, W), dtype=np.float32)

  
        gps = self._build_gp()
        for b in range(B):
            obs_mask = X[b, 0]   # (H, W)
            obs_map  = X[b, 1]   # (H, W)
            m, e, a  = self._predict_single(gps, obs_map, obs_mask)
            means[b, 0]   = m
            epi_std[b, 0] = e
            ale_std[b, 0] = a


        return {
            "predicted_mean":          means,
            "predicted_std_epistemic": epi_std,
            "predicted_std_aleatoric": ale_std,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    H, W = 100, 100

    # ── Synthetic ground truth: 2D Gaussian bump ─────────────────────────
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    gt = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (0.2 * (H / 5)**2)).astype(np.float32)

    # ── Simulate a sparse observation mask (20% of grid) ──────────────────
    mask = (rng.random((H, W)) < 0.05).astype(np.float32)
    obs  = (gt + rng.normal(0, 0.05, (H, W)).astype(np.float32)) * mask

    # ── Build batch (B=2): same sample twice for a quick check ───────────
    obs_dict = {
        "obs_map":  np.stack([obs, obs], axis=0),    # (2, H, W)
        "obs_mask": np.stack([mask, mask], axis=0),  # (2, H, W)
    }

    print(f"obs_map shape  : {obs_dict['obs_map'].shape}")

    model = GaussianProcessModel(n_restarts_optimizer=0)
    out   = model.predict(obs_dict)

    mean  = out["predicted_mean"]
    epi   = out["predicted_std_epistemic"]
    ale   = out["predicted_std_aleatoric"]

    print(f"predicted_mean           : {mean.shape}  "
          f"range [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"predicted_std_epistemic  : {epi.shape}   "
          f"range [{epi.min():.4f}, {epi.max():.4f}]")
    print(f"predicted_std_aleatoric  : {ale.shape}   "
          f"range [{ale.min():.4f}, {ale.max():.4f}]  "
          f"(uniform: {np.allclose(ale[0, 0], ale[0, 0, 0, 0])})")

    # Epistemic should be lower near observed cells than in unobserved cells
    obs_locs   = mask > 0.5
    unobs_locs = ~obs_locs
    epi_map    = epi[0, 0]
    print(f"\nEpistemic std  @ observed   cells : {epi_map[obs_locs].mean():.4f}")
    print(f"Epistemic std  @ unobserved cells : {epi_map[unobs_locs].mean():.4f}")
    assert epi_map[obs_locs].mean() < epi_map[unobs_locs].mean(), \
        "Epistemic uncertainty should be lower at observed locations!"

    print("\nAll checks passed.")

    # ── Matplotlib visualisation ──────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("GaussianProcessModel — output maps (sample 0)", fontsize=13)

    panels = [
        (gt,         "Ground Truth",            "viridis",  False),
        (mask,       "Observation Mask",         "gray",     False),
        (obs,        "Observed Values",          "viridis",  False),
        (mean[0, 0], "Predicted Mean",           "viridis",  False),
        (epi[0, 0],  "Epistemic Std (C*RBF)",   "plasma",   False),
        (ale[0, 0],  "Aleatoric Std (White)",   "plasma",   False),
    ]

    for ax, (data, title, cmap, _) in zip(axes.flat, panels):
        im = ax.imshow(data, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
