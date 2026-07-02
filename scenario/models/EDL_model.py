"""
EDL_model.py — Uncertainty Estimation CAEPIA 2026
==================================================
Evidential Deep Learning (EDL) model for scalar field reconstruction and
uncertainty decomposition (Amini et al., 2020 — "Deep Evidential Regression").

Interface (identical to MCDropoutModel and EnsembleModel):
    model.predict(X)  ->  dict with keys:
        'predicted_mean'           : np.ndarray  (B, 1, H, W) float32
        'predicted_std_epistemic'  : np.ndarray  (B, 1, H, W) float32
        'predicted_std_aleatoric'  : np.ndarray  (B, 1, H, W) float32

Input X : array-like or torch.Tensor  (B, 2, H, W)
    Channel 0 — observation mask  M ∈ {0,1}^{H×W}
    Channel 1 — observed values   V ∈ R^{H×W}  (0 at unobserved locations)

Uncertainty decomposition — Normal-Inverse-Gamma (NIG) prior
-------------------------------------------------------------
The network outputs 4 parameter maps (γ, ν, α, β) that parameterise a
Normal-Inverse-Gamma distribution p(μ, σ² | γ, ν, α, β).

Parameter constraints:
    γ  ∈ R           — predicted mean (no constraint)
    ν  > 0           — virtual observation count (Softplus)
    α  > 1           — shape of inverse-gamma (Softplus + 1)
    β  > 0           — scale of inverse-gamma (Softplus)

Derived uncertainty quantities:
    Aleatoric  σ_ale = sqrt( β / (α - 1) )           — inherent noise
    Epistemic  σ_epi = sqrt( β / (ν · (α - 1)) )    — model ignorance

Architecture
------------
Reuses UNet.forward_features() (shared encoder-bottleneck-decoder backbone)
from models.py and attaches four 1×1 convolutional heads for (γ, ν, α, β).
No encoder/decoder blocks are duplicated.

Training loss (NIG-NLL + regularisation):
    L = NIG_NLL(y; γ, ν, α, β) + λ · |y - γ| · (2ν + α)
    where NIG_NLL is the negative log-likelihood of the NIG distribution.
    See nig_loss() below.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from unet_model import UNet


# ---------------------------------------------------------------------------
# EDLNet — backbone + NIG heads
# ---------------------------------------------------------------------------

class EDLNet(nn.Module):
    """
    Evidential U-Net: shared UNet backbone + four NIG parameter heads.

    Parameters
    ----------
    in_channels : int
        Number of input channels passed to UNet.
    base_channels : int
        Base feature channels (doubles each encoder stage).
    depth : int
        Number of encoder/decoder stages.
    kernel_size : int
        Convolutional kernel size (must be odd).
    dropout_p : float
        Dropout probability in UNet backbone (0 = disabled).
    """

    def __init__(
        self,
        in_channels: int   = 2,
        base_channels: int = 32,
        depth: int         = 4,
        kernel_size: int   = 3,
        dropout_p: float   = 0.0,
    ):
        super().__init__()

        # ── Shared backbone (encoder + bottleneck + decoder) ──────────────────
        self.backbone = UNet(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
        )
        feat_ch = self.backbone.out_features_channels  # channels before heads

        # ── Four NIG parameter heads (1×1 convolutions) ───────────────────────
        self.head_gamma = nn.Conv2d(feat_ch, 1, kernel_size=1)  # mean
        self.head_nu    = nn.Conv2d(feat_ch, 1, kernel_size=1)  # virtual obs count
        self.head_alpha = nn.Conv2d(feat_ch, 1, kernel_size=1)  # IG shape
        self.head_beta  = nn.Conv2d(feat_ch, 1, kernel_size=1)  # IG scale

        self.softplus = nn.Softplus()
        self._init_heads()

    # ------------------------------------------------------------------
    def _init_heads(self):
        # γ head: zero-init weights + bias so initial mean ≈ 0
        nn.init.zeros_(self.head_gamma.weight)
        nn.init.zeros_(self.head_gamma.bias)

        # ν, α, β heads: small weights + negative bias so that
        # softplus(output) starts small, avoiding exploding uncertainties.
        #   softplus(-2) ≈ 0.13  →  ν ≈ 0.13,  (α-1) ≈ 0.13,  β ≈ 0.13
        # This keeps the initial aleatoric std = sqrt(β/(α-1)) ≈ 1.0.
        for head in (self.head_nu, self.head_alpha, self.head_beta):
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.constant_(head.bias, -2.0)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, in_channels, H, W)

        Returns
        -------
        dict with keys 'gamma', 'nu', 'alpha', 'beta', each (B, 1, H, W).
        """
        features = self.backbone.forward_features(x)  # shared backbone

        gamma = self.head_gamma(features)                               # ∈ R
        nu    = self.softplus(self.head_nu(features))    + 1e-6        # > 0
        alpha = self.softplus(self.head_alpha(features)) + 1.0 + 1e-6  # > 1
        beta  = self.softplus(self.head_beta(features))  + 1e-6        # > 0

        return {"gamma": gamma, "nu": nu, "alpha": alpha, "beta": beta}

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# NIG training loss
# ---------------------------------------------------------------------------

def nig_loss(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    lam: float = 1e-3,
) -> torch.Tensor:
    """
    Normal-Inverse-Gamma NLL loss + evidential regularisation.

    Loss = NIG_NLL(y; γ, ν, α, β) + λ · |y − γ| · (2ν + α)

    Parameters
    ----------
    y     : ground-truth tensor  (B, 1, H, W)
    gamma, nu, alpha, beta : NIG parameter tensors  (B, 1, H, W)
    lam   : regularisation weight λ  (default 1e-2)

    Returns
    -------
    Scalar loss tensor.
    """
    import math

    two_beta = 2.0 * beta

    # NIG NLL (Eq. 4 in Amini et al., 2020)
    nll = (
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(two_beta)
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + two_beta)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    # Evidential regulariser: penalises high evidence on wrong predictions
    reg = torch.abs(y - gamma) * (2.0 * nu + alpha)

    return (nll + lam * reg).mean()


# ---------------------------------------------------------------------------
# EDLModel — sklearn-like wrapper
# ---------------------------------------------------------------------------

class EDLModel:
    """
    Evidential Deep Learning uncertainty estimator.

    Parameters
    ----------
    net : EDLNet or None
        Pre-built / pre-trained EDLNet.
        If None, an EDLNet is instantiated from **edlnet_kwargs.
    device : torch.device or None
        Inference device; auto-detected (CUDA > CPU) if None.
    **edlnet_kwargs
        Forwarded to EDLNet when net=None
        (e.g. in_channels, base_channels, depth, dropout_p).
    """

    def __init__(self, net=None, device=None, **edlnet_kwargs):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
        if net is None:
            self.net = EDLNet(**edlnet_kwargs).to(self.device)
        else:
            self.net = net.to(self.device)

    # ------------------------------------------------------------------
    def train(
        self,
        X_train,
        y_train,
        epochs: int     = 50,
        batch_size: int = 16,
        lr: float       = 5e-4,
        lam: float      = 1e-2,
        max_grad_norm: float = 1.0,
        verbose: bool   = True,
    ) -> list[float]:
        """
        Train the EDLNet with NIG-NLL loss + evidential regularisation.

        Parameters
        ----------
        X_train : array-like  (N, 2, H, W)
        y_train : array-like  (N, 1, H, W)
        epochs, batch_size, lr : standard training hyper-parameters.
        lam : float
            Evidential regularisation weight λ (see nig_loss()).
        max_grad_norm : float
            Gradient clipping norm (default 1.0). Prevents exploding
            gradients during the initial high-loss epochs of NIG training.
        verbose : bool

        Returns
        -------
        list[float]
            Mean NIG loss per epoch.
        """
        X_t = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y_train), dtype=torch.float32)

        loader    = DataLoader(TensorDataset(X_t, y_t),
                               batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        epoch_losses = []
        bar = tqdm(range(1, epochs + 1), desc="EDL train",
                   unit="epoch", disable=not verbose, leave=True)
        for ep in bar:
            self.net.train()
            batch_losses = []
            for X_b, y_b in loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device)
                out  = self.net(X_b)
                loss = nig_loss(y_b, out["gamma"], out["nu"],
                                out["alpha"], out["beta"], lam=lam)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
                optimizer.step()
                batch_losses.append(loss.item())
            ep_loss = float(np.mean(batch_losses))
            epoch_losses.append(ep_loss)
            bar.set_postfix(loss=f"{ep_loss:.4f}")

        return epoch_losses

    # ------------------------------------------------------------------
    def predict(self, X) -> dict:
        """
        Run a single forward pass and derive uncertainty maps.

        Parameters
        ----------
        X : array-like or torch.Tensor  (B, 2, H, W)

        Returns
        -------
        dict with:
            'predicted_mean'           : np.ndarray  (B, 1, H, W) float32 — γ
            'predicted_std_epistemic'  : np.ndarray  (B, 1, H, W) float32
            'predicted_std_aleatoric'  : np.ndarray  (B, 1, H, W) float32
            '_nig_params'              : dict of np.ndarrays (gamma, nu, alpha, beta)
                                         for advanced inspection / loss computation
        """
        if type(X) is dict:
            obs_map  = np.asarray(X["obs_map"],  dtype=np.float32)
            obs_mask = np.asarray(X["obs_mask"], dtype=np.float32)
            if obs_map.ndim == 2:
                obs_map  = obs_map[np.newaxis]
                obs_mask = obs_mask[np.newaxis]
            # Build (B, 2, H, W): channel 0 = mask, channel 1 = values
            X = np.stack([obs_mask, obs_map], axis=1)
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            try:
                X_t = X.detach().float()
            except AttributeError:
                X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
            X_t = X_t.to(self.device)

        self.net.eval()
        with torch.no_grad():
            out = self.net(X_t)

        gamma = out["gamma"]
        nu    = out["nu"]
        alpha = out["alpha"]
        beta  = out["beta"]

        # Uncertainty derivation from NIG moments
        # E[σ²] = β / (α - 1)   →   aleatoric std = sqrt(E[σ²])
        # Var[μ] = β / (ν·(α-1)) →   epistemic std = sqrt(Var[μ])
        #
        # Numerical stability: clamp (α-1) and (ν·(α-1)) from below so
        # that a poorly-initialised or undertrained network cannot produce
        # infinite uncertainties.  We also clamp β from above for the same
        # reason.  The clamps are intentionally loose (1e-3 / 1e3) so they
        # only catch genuine numerical pathologies, not learned values.
        alpha_m1    = (alpha - 1.0).clamp(min=1e-3)           # α - 1 > 0
        beta_c      = beta.clamp(max=1e3)                      # β bounded
        var_aleatoric = beta_c / alpha_m1                      # (B,1,H,W)
        var_epistemic = beta_c / (nu.clamp(min=1e-3) * alpha_m1)  # (B,1,H,W)

        to_np = lambda t: t.cpu().numpy().astype(np.float32)

        return {
            "predicted_mean":          to_np(gamma),
            "predicted_std_epistemic": to_np(var_epistemic.sqrt()),
            "predicted_std_aleatoric": to_np(var_aleatoric.sqrt()),
            "_nig_params": {
                "gamma": to_np(gamma),
                "nu":    to_np(nu),
                "alpha": to_np(alpha),
                "beta":  to_np(beta),
            },
        }

    # ------------------------------------------------------------------
    def save_weights(self, path: str | Path) -> None:
        """
        Guarda los pesos del EDLNet en disco.

        Parameters
        ----------
        path : str | Path
            Ruta del fichero .pt de salida
            (p. ej. ``Weights/dataset_NADIR_EDL.pt``).
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

    rng = np.random.default_rng(42)
    H, W = 64, 64

    # ── Synthetic ground truth: 2D Gaussian bump ──────────────────────────
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    gt = np.exp(
        -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (H / 5) ** 2)
    ).astype(np.float32)

    # ── Sparse observations (20 % of pixels, + noise) ─────────────────────
    mask = (rng.random((H, W)) < 0.20).astype(np.float32)
    obs  = (gt + rng.normal(0, 0.05, (H, W)).astype(np.float32)) * mask

    # ── Batch (B=1) ───────────────────────────────────────────────────────
    X = np.stack([np.stack([mask, obs], axis=0)], axis=0)   # (1, 2, H, W)

    # ── Model ─────────────────────────────────────────────────────────────
    model = EDLModel(in_channels=2, base_channels=16, depth=3)
    print(f"Backbone parameters : {model.net.backbone.count_parameters():,}")
    print(f"Total  parameters   : {model.net.count_parameters():,}")
    print(f"Input shape         : {X.shape}")

    out = model.predict(X)
    mean = out["predicted_mean"]
    epi  = out["predicted_std_epistemic"]
    ale  = out["predicted_std_aleatoric"]
    nig  = out["_nig_params"]

    print(f"predicted_mean           : {mean.shape}  "
          f"range [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"predicted_std_epistemic  : {epi.shape}   "
          f"range [{epi.min():.4f}, {epi.max():.4f}]")
    print(f"predicted_std_aleatoric  : {ale.shape}   "
          f"range [{ale.min():.4f}, {ale.max():.4f}]")
    for k, v in nig.items():
        print(f"  NIG {k:5s}: range [{v.min():.4f}, {v.max():.4f}]")
    print("Note: network is untrained — outputs are random but shapes are correct.")

    # ── Matplotlib visualisation ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "EDLModel — NIG parameter maps + uncertainty (sample 0, untrained network)",
        fontsize=12,
    )

    panels = [
        # Row 0: inputs & prediction
        (gt,         "Ground Truth",                      "viridis"),
        (mask,       "Observation Mask",                  "gray"),
        (obs,        "Observed Values",                   "viridis"),
        (mean[0, 0], "Predicted Mean (gamma)",            "viridis"),
        # Row 1: NIG params & uncertainties
        (nig["nu"][0, 0],    "nu  (virtual obs count)",  "cividis"),
        (nig["alpha"][0, 0], "alpha  (IG shape > 1)",    "cividis"),
        (ale[0, 0],          "Aleatoric Std",             "plasma"),
        (epi[0, 0],          "Epistemic Std",             "plasma"),
    ]

    for ax, (data, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(data, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
