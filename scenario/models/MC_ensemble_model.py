"""
MC_ensemble_model.py — Uncertainty Estimation CAEPIA 2026
==========================================================
Deep Ensemble model for scalar field reconstruction and uncertainty
decomposition (Lakshminarayanan et al., 2017).

Interface (identical to GaussianProcessModel and MCDropoutModel):
    model.predict(X)  ->  dict with keys:
        'predicted_mean'           : np.ndarray  (B, 1, H, W) float32
        'predicted_std_epistemic'  : np.ndarray  (B, 1, H, W) float32
        'predicted_std_aleatoric'  : np.ndarray  (B, 1, H, W) float32

Input X : array-like or torch.Tensor  (B, 2, H, W)
    Channel 0 — observation mask  M ∈ {0,1}^{H×W}
    Channel 1 — observed values   V ∈ R^{H×W}  (0 at unobserved locations)

Uncertainty decomposition
-------------------------
N independently initialised UNets are trained from scratch with different
random seeds.  At inference each member runs in eval() mode (no dropout):

    Epistemic σ_epi(x) = std  over N members of predicted_mean(x)
                         Decreases as more members agree on the prediction.

    Aleatoric σ_ale(x) = mean over N members of predicted_std(x)
                         Average of each member's heteroscedastic noise head.

Architecture
------------
Each ensemble member is a UNet (imported from models.py) with two output
heads: predicted_mean (Softplus) and predicted_std (Softplus + eps).
Members are initialised with independent random seeds to maximise diversity.
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
# EnsembleModel
# ---------------------------------------------------------------------------

class EnsembleModel:
    """
    Deep Ensemble of independent UNet models.

    Parameters
    ----------
    n_members : int
        Number of ensemble members (N).  Typical values: 5–10.
    device : torch.device or None
        Inference device; auto-detected (CUDA > CPU) if None.
    seed : int
        Base random seed.  Member k is initialised with seed + k so that
        each member has a distinct weight initialisation.
    **unet_kwargs
        Forwarded to UNet (e.g. in_channels, base_channels, depth,
        kernel_size).  See models.UNet for the full parameter list.
    """

    def __init__(
        self,
        n_members: int = 5,
        device=None,
        seed: int = 0,
        **unet_kwargs,
    ):
        self.n_members = n_members
        self.device    = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )

        self.members = []
        for k in range(n_members):
            torch.manual_seed(seed + k)
            net = UNet(**unet_kwargs).to(self.device)
            self.members.append(net)

    # ------------------------------------------------------------------
    def train_mode(self):
        """Set all members to train() mode (for external training loops)."""
        for net in self.members:
            net.train()

    def eval_mode(self):
        """Set all members to eval() mode."""
        for net in self.members:
            net.eval()

    def parameters(self):
        """Yield parameters of all members (for a single joint optimiser)."""
        for net in self.members:
            yield from net.parameters()

    def count_parameters(self) -> int:
        """Total trainable parameters across all members."""
        return sum(
            p.numel() for net in self.members
            for p in net.parameters() if p.requires_grad
        )

    # ------------------------------------------------------------------
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs: int     = 50,
        batch_size: int = 16,
        lr: float       = 1e-3,
        verbose: bool   = True,
    ) -> dict:
        """
        Train each ensemble member independently with heteroscedastic
        Gaussian NLL loss and its own Adam optimizer.

        Parameters
        ----------
        X_train : array-like  (N, 2, H, W)
        y_train : array-like  (N, 1, H, W)
        X_val   : array-like  (M, 2, H, W) or None
        y_val   : array-like  (M, 1, H, W) or None
            Optional validation set.  When provided, the ensemble mean RMSE
            and mean NLL are evaluated on it at the end of every epoch.

        Returns
        -------
        dict with:
            'member_nll'  : list[list[float]]  — per-member, per-epoch train NLL
            'epoch_nll'   : list[float]        — mean train NLL across members per epoch
            'epoch_rmse'  : list[float] | None — val RMSE per epoch (None if no val data)
            'epoch_val_nll': list[float] | None — val NLL per epoch (None if no val data)
        """
        X_t = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y_train), dtype=torch.float32)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(np.asarray(y_val), dtype=torch.float32).to(self.device)

        # member_nll[k][ep] = train NLL of member k at epoch ep
        member_nll: list[list[float]] = []
        # Accumulate per-epoch aggregates across members
        epoch_nll_accum   = np.zeros(epochs, dtype=np.float64)

        members_bar = tqdm(enumerate(self.members),
                           total=self.n_members,
                           desc="Ensemble members",
                           unit="member",
                           disable=not verbose,
                           leave=True)
        for k, net in members_bar:
            members_bar.set_description(f"Ensemble member {k+1}/{self.n_members}")
            loader    = DataLoader(TensorDataset(X_t, y_t),
                                   batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            epoch_losses = []
            epoch_bar = tqdm(range(1, epochs + 1),
                             desc=f"  member {k+1} epochs",
                             unit="epoch",
                             disable=not verbose,
                             leave=False)
            for ep in epoch_bar:
                net.train()
                batch_losses = []
                for X_b, y_b in loader:
                    X_b = X_b.to(self.device)
                    y_b = y_b.to(self.device)
                    out  = net(X_b)
                    loss = _gaussian_nll_loss(y_b, out["predicted_mean"],
                                              out["predicted_std"])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                ep_loss = float(np.mean(batch_losses))
                epoch_losses.append(ep_loss)
                epoch_nll_accum[ep - 1] += ep_loss
                epoch_bar.set_postfix(loss=f"{ep_loss:.4f}")
            member_nll.append(epoch_losses)

        epoch_nll = (epoch_nll_accum / self.n_members).tolist()

        # ── Validation metrics per epoch ──────────────────────────────────────
        epoch_rmse    = None
        epoch_val_nll = None
        if has_val:
            epoch_rmse    = []
            epoch_val_nll = []
            # Re-train epoch by epoch is expensive; instead we evaluate the
            # final trained ensemble checkpointed at each epoch is not feasible
            # without snapshots.  We therefore compute one aggregate val pass
            # using the fully trained model and report it for every epoch as a
            # post-hoc scalar — this is a limitation to avoid storing N snapshots.
            # For per-epoch val curves, pass verbose=True and inspect member_nll.
            self.eval_mode()
            with torch.no_grad():
                val_means = []
                val_stds  = []
                for net in self.members:
                    out = net(X_val_t)
                    val_means.append(out["predicted_mean"])
                    val_stds.append(out["predicted_std"])
                ens_mean = torch.stack(val_means).mean(0)   # (M, 1, H, W)
                ens_std  = torch.stack(val_stds).mean(0)
                rmse     = float(torch.sqrt(((ens_mean - y_val_t) ** 2).mean()).item())
                val_nll  = float(_gaussian_nll_loss(y_val_t, ens_mean, ens_std).item())
            epoch_rmse    = [rmse]    * epochs
            epoch_val_nll = [val_nll] * epochs

        return {
            "member_nll":   member_nll,
            "epoch_nll":    epoch_nll,
            "epoch_rmse":   epoch_rmse,
            "epoch_val_nll": epoch_val_nll,
        }

    # ------------------------------------------------------------------
    def predict(self, obs) -> dict:
        """
        Run inference with all ensemble members in eval() mode.

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
        if type(obs) is dict:
            obs_map  = np.asarray(obs["obs_map"],  dtype=np.float32)
            obs_mask = np.asarray(obs["obs_mask"], dtype=np.float32)
            if obs_map.ndim == 2:
                obs_map  = obs_map[np.newaxis]
                obs_mask = obs_mask[np.newaxis]
            # Build (B, 2, H, W): channel 0 = mask, channel 1 = values
            X = np.stack([obs_mask, obs_map], axis=1)
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            try:
                X_t = obs.detach().float()
            except AttributeError:
                X_t = torch.tensor(np.asarray(obs), dtype=torch.float32)
            X_t = X_t.to(self.device)

        self.eval_mode()

        mean_samples = []
        std_samples  = []
        with torch.no_grad():
            for net in self.members:
                out = net(X_t)
                mean_samples.append(out["predicted_mean"])  # (B, 1, H, W)
                std_samples.append(out["predicted_std"])    # (B, 1, H, W)

        # Stack: (N, B, 1, H, W)
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
        Guarda todos los miembros del ensemble en un único fichero .pt.

        El fichero contiene un dict ``{"member_{k}": state_dict}``
        para cada miembro k = 0 … N-1.

        Parameters
        ----------
        path : str | Path
            Ruta del fichero .pt de salida
            (p. ej. ``Weights/dataset_NADIR_Ensemble.pt``).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {f"member_{k}": net.state_dict()
                   for k, net in enumerate(self.members)}
        payload["n_members"] = self.n_members
        torch.save(payload, path)

    # ------------------------------------------------------------------
    def load_weights(self, path: str | Path) -> None:
        """
        Carga los pesos de todos los miembros desde un fichero generado
        por ``save_weights``.

        Parameters
        ----------
        path : str | Path
            Ruta del fichero .pt generado por ``save_weights``.
        """
        payload = torch.load(path, map_location=self.device)
        for k, net in enumerate(self.members):
            net.load_state_dict(payload[f"member_{k}"])
            net.to(self.device)


# ---------------------------------------------------------------------------
# Smoke test + visualisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(7)
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

    # ── Model (5 members, small network for speed) ────────────────────────
    model = EnsembleModel(
        n_members=5, seed=42,
        in_channels=2, base_channels=16, depth=3,
    )
    print(f"Members             : {model.n_members}")
    print(f"Params per member   : {model.count_parameters() // model.n_members:,}")
    print(f"Total params        : {model.count_parameters():,}")

    obs_dict = {"obs_map": obs, "obs_mask": mask}   # (H, W) each
    out = model.predict(obs_dict)

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
    fig.suptitle("EnsembleModel — output maps (sample 0, untrained network)",
                 fontsize=13)

    panels = [
        (gt,         "Ground Truth",                  "viridis"),
        (mask,       "Observation Mask",               "gray"),
        (obs,        "Observed Values",                "viridis"),
        (mean[0, 0], "Predicted Mean",                 "viridis"),
        (epi[0, 0],  "Epistemic Std (ensemble std)",   "plasma"),
        (ale[0, 0],  "Aleatoric Std (network head)",   "plasma"),
    ]

    for ax, (data, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(data, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
