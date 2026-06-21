"""
MC_dropout_model.py — Uncertainty Estimation CAEPIA 2026
=========================================================
Monte Carlo Dropout model for scalar field reconstruction and uncertainty
decomposition (Gal & Ghahramani, 2016).

Interface (identical to GaussianProcessModel and EnsembleModel):
    model.predict(X, n_samples=50)  ->  dict with keys:
        'predicted_mean'           : np.ndarray  (B, 1, H, W) float32
        'predicted_std_epistemic'  : np.ndarray  (B, 1, H, W) float32
        'predicted_std_aleatoric'  : np.ndarray  (B, 1, H, W) float32

Input X : array-like or torch.Tensor  (B, 2, H, W)
    Channel 0 — observation mask  M ∈ {0,1}^{H×W}
    Channel 1 — observed values   V ∈ R^{H×W}  (0 at unobserved locations)

Uncertainty decomposition
-------------------------
T stochastic forward passes are performed with dropout active (model kept in
train() mode so Dropout2d layers sample independent masks each pass):

    Epistemic σ_epi(x) = std  over T samples of predicted_mean(x)
    Aleatoric σ_ale(x) = mean over T samples of predicted_std(x)

Architecture
------------
Uses UNet from models.py with dropout_p > 0, which inserts Dropout2d after
each activation in every convolutional block.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__)))  # allow imports from scenario/
from unet_model import UNet


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def _gaussian_nll_loss(
    y_true: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL: 0.5·[(y-μ)²/σ² + log σ²]."""
    var = std.pow(2).clamp(min=1e-6)
    return (0.5 * ((y_true - mean).pow(2) / var + var.log())).mean()


# ---------------------------------------------------------------------------
# MCDropoutModel
# ---------------------------------------------------------------------------

class MCDropoutModel:
    """
    Monte Carlo Dropout uncertainty estimator.

    Parameters
    ----------
    net : UNet or None
        Pre-built / pre-trained UNet with dropout_p > 0.
        If None, a UNet is instantiated from **unet_kwargs.
    n_samples : int
        Default number of stochastic forward passes (T).
    device : torch.device or None
        Inference device; auto-detected (CUDA > CPU) if None.
    **unet_kwargs
        Forwarded to UNet when net=None (e.g. in_channels, base_channels,
        depth, dropout_p).  dropout_p defaults to 0.2 if not specified.
    """

    def __init__(
        self,
        net=None,
        n_samples: int = 50,
        device=None,
        **unet_kwargs,
    ):
        self.n_samples = n_samples
        self.device    = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
        if net is None:
            unet_kwargs.setdefault("dropout_p", 0.2)
            self.net = UNet(**unet_kwargs).to(self.device)
        else:
            self.net = net.to(self.device)

    # ------------------------------------------------------------------
    def train(
        self,
        X_train,
        y_train,
        epochs: int    = 50,
        batch_size: int = 16,
        lr: float      = 1e-3,
        verbose: bool  = True,
    ) -> list[float]:
        """
        Train the UNet backbone with heteroscedastic Gaussian NLL loss.

        The model is kept in train() mode during training so Dropout2d is
        active, providing implicit regularisation.

        Parameters
        ----------
        X_train : array-like  (N, 2, H, W)
            Stacked [observation_mask, observed_values] input.
        y_train : array-like  (N, 1, H, W)
            Ground-truth scalar fields.
        epochs : int
            Number of full passes over the training set.
        batch_size : int
            Mini-batch size.
        lr : float
            Adam learning rate.
        verbose : bool
            Print loss every 10 epochs if True.

        Returns
        -------
        list[float]
            Mean loss per epoch (for loss-curve plotting).
        """
        X_t = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y_train), dtype=torch.float32)

        loader    = DataLoader(TensorDataset(X_t, y_t),
                               batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        epoch_losses = []
        bar = tqdm(range(1, epochs + 1), desc="MC_Dropout train",
                   unit="epoch", disable=not verbose, leave=True)
        for ep in bar:
            self.net.train()
            batch_losses = []
            for X_b, y_b in loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device)
                out  = self.net(X_b)
                loss = _gaussian_nll_loss(y_b, out["predicted_mean"],
                                          out["predicted_std"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            ep_loss = float(np.mean(batch_losses))
            epoch_losses.append(ep_loss)
            bar.set_postfix(loss=f"{ep_loss:.4f}")

        return epoch_losses

    # ------------------------------------------------------------------
    def predict(self, X, n_samples: int = None) -> dict:
        """
        Run MC Dropout inference.

        Parameters
        ----------
        X : array-like or torch.Tensor  (B, 2, H, W)
        n_samples : int, optional
            Override the default number of stochastic passes.

        Returns
        -------
        dict with:
            'predicted_mean'           : np.ndarray  (B, 1, H, W)
            'predicted_std_epistemic'  : np.ndarray  (B, 1, H, W)
            'predicted_std_aleatoric'  : np.ndarray  (B, 1, H, W)
        """
        T = n_samples if n_samples is not None else self.n_samples

        try:
            X_t = X.detach().float()
        except AttributeError:
            X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        X_t = X_t.to(self.device)

        # Keep train() mode so Dropout2d samples independently each pass
        self.net.train()

        mean_samples = []
        std_samples  = []
        with torch.no_grad():
            for _ in range(T):
                out = self.net(X_t)
                mean_samples.append(out["predicted_mean"])
                std_samples.append(out["predicted_std"])

        # Stack: (T, B, 1, H, W)
        mean_stack = torch.stack(mean_samples, dim=0)
        std_stack  = torch.stack(std_samples,  dim=0)

        predicted_mean = mean_stack.mean(dim=0).cpu().numpy()
        epistemic_std  = mean_stack.std(dim=0).cpu().numpy()
        aleatoric_std  = std_stack.mean(dim=0).cpu().numpy()

        return {
            "predicted_mean":          predicted_mean.astype(np.float32),
            "predicted_std_epistemic": epistemic_std.astype(np.float32),
            "predicted_std_aleatoric": aleatoric_std.astype(np.float32),
        }

    # ------------------------------------------------------------------
    def save_weights(self, path: str | Path) -> None:
        """
        Guarda los pesos del modelo en disco.

        Parameters
        ----------
        path : str | Path
            Ruta del fichero .pt donde se guardan los pesos
            (p. ej. ``Weights/dataset_NADIR_MC_Dropout.pt``).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    # ------------------------------------------------------------------
    def load_weights(self, path: str | Path) -> None:
        """
        Carga pesos previamente guardados con ``save_weights``.

        Parameters
        ----------
        path : str | Path
            Ruta del fichero .pt generado por ``save_weights``.
        """
        state = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state)
        self.net.to(self.device)


# ---------------------------------------------------------------------------
# Smoke test + visualisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    H, W = 64, 64

    # ── Synthetic ground truth: 2D Gaussian bump ─────────────────────────
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    gt = np.exp(
        -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (H / 5) ** 2)
    ).astype(np.float32)

    # ── Sparse observation mask (15% of pixels) ───────────────────────────
    mask = (rng.random((H, W)) < 0.15).astype(np.float32)
    obs  = (gt + rng.normal(0, 0.05, (H, W)).astype(np.float32)) * mask

    # ── Batch (B=1) ───────────────────────────────────────────────────────
    X = np.stack([np.stack([mask, obs], axis=0)], axis=0)  # (1, 2, H, W)

    # ── Model ─────────────────────────────────────────────────────────────
    model = MCDropoutModel(
        in_channels=2, base_channels=16, depth=3,
        dropout_p=0.2, n_samples=30,
    )
    print(f"Parameters : {model.net.count_parameters():,}")
    print(f"Input shape: {X.shape}")

    out = model.predict(X)

    mean = out["predicted_mean"]
    epi  = out["predicted_std_epistemic"]
    ale  = out["predicted_std_aleatoric"]

    print(f"predicted_mean           : {mean.shape}  "
          f"range [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"predicted_std_epistemic  : {epi.shape}   "
          f"range [{epi.min():.4f}, {epi.max():.4f}]")
    print(f"predicted_std_aleatoric  : {ale.shape}   "
          f"range [{ale.min():.4f}, {ale.max():.4f}]")
    print("Note: network is untrained — outputs are random but shapes are correct.")

    # ── Matplotlib visualisation ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("MCDropoutModel — output maps (sample 0, untrained network)",
                 fontsize=13)

    panels = [
        (gt,         "Ground Truth",                  "viridis"),
        (mask,       "Observation Mask",               "gray"),
        (obs,        "Observed Values",                "viridis"),
        (mean[0, 0], "Predicted Mean",                 "viridis"),
        (epi[0, 0],  "Epistemic Std (MC Dropout)",     "plasma"),
        (ale[0, 0],  "Aleatoric Std (network head)",   "plasma"),
    ]

    for ax, (data, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(data, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
