"""
models.py — Uncertainty Estimation CAEPIA 2026
================================================
U-Net backbone for scalar field reconstruction with two output heads:
  - predicted_mean : reconstructed scalar field  ŷ ∈ R^{H×W}
  - predicted_std  : predictive std (aleatoric)  σ ∈ R_{>0}^{H×W}

Architecture follows Section 3.1 of the paper:
  • Encoder  : N_enc blocks of [Conv→BN→ReLU] × 2  +  MaxPool(2)
               channels double at each stage: C_0, 2C_0, 4C_0, …
  • Bottleneck: [Conv→BN→ReLU] × 2
  • Decoder  : TransposedConv(upsample) + skip-concat + [Conv→BN→ReLU] × 2
  • Heads    : two independent 1×1 convolutions on the final feature map

Input tensor  : X = [M, V] ∈ R^{in_channels × H × W}
                  M — binary observation mask
                  V — observed values (0 at unobserved locations)
Output        : dict with keys 'predicted_mean' and 'predicted_std'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive (Conv → BN → ReLU [→ Dropout2d]) layers — basic U-Net building block."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dropout_p: float = 0.0):
        super().__init__()
        pad = kernel_size // 2          # 'same' padding for odd kernels
        layers = [
            nn.Conv2d(in_channels,  out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv followed by MaxPool(2) — one encoder stage."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dropout_p: float = 0.0):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, dropout_p)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)     # feature map passed as skip connection
        pooled = self.pool(skip)
        return pooled, skip


class DecoderBlock(nn.Module):
    """TransposedConv upsample + skip-connection concat + DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 kernel_size: int = 3, dropout_p: float = 0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels,
                               kernel_size, dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle spatial size mismatches (when H or W is not a power of 2)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net backbone
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net backbone with two output heads: predicted_mean and predicted_std.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 2: observation mask + value map).
    base_channels : int
        Number of feature channels in the first encoder block (C_0 in the paper).
        Doubles at each subsequent encoder stage.
    depth : int
        Number of encoder/decoder stages (N_enc in the paper).
    kernel_size : int
        Convolutional kernel size (k in the paper); must be odd.
    dropout_p : float
        Dropout probability inserted after each activation (0.0 = disabled).
        Set > 0 to enable Monte Carlo Dropout at inference time.

    Outputs (forward pass)
    ------
    dict with:
        'predicted_mean' : torch.Tensor  (B, 1, H, W)  — unbounded reconstruction
        'predicted_std'  : torch.Tensor  (B, 1, H, W)  — strictly positive via Softplus
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

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial dimensions.")
        if depth < 1:
            raise ValueError("depth must be >= 1.")

        self.depth = depth

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        ch_in = in_channels
        for i in range(depth):
            ch_out = base_channels * (2 ** i)
            self.encoders.append(EncoderBlock(ch_in, ch_out, kernel_size, dropout_p))
            ch_in = ch_out

        # ── Bottleneck ────────────────────────────────────────────────────────
        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = DoubleConv(ch_in, bottleneck_ch, kernel_size, dropout_p)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        ch_in = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            skip_ch = base_channels * (2 ** i)
            ch_out  = skip_ch
            self.decoders.append(DecoderBlock(ch_in, skip_ch, ch_out, kernel_size, dropout_p))
            ch_in = ch_out

        # ── Output heads (1×1 convolutions) ──────────────────────────────────
        self.out_features_channels = ch_in   # exposed for external head attachment
        self.head_mean = nn.Conv2d(ch_in, 1, kernel_size=1)
        self.head_std  = nn.Conv2d(ch_in, 1, kernel_size=1)
        self.softplus  = nn.Softplus()          # ensures σ > 0+
        self.softplus_2 = nn.Softplus()

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Kaiming initialization for Conv layers; bias=0 for output heads.

         He, Kaiming et al. (2015). "Delving Deep into Rectifiers:
         Surpassing Human-Level Performance on ImageNet Classification".
         arXiv:1502.01852.
        """

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run encoder + bottleneck + decoder and return the final feature map
        (B, out_features_channels, H, W) **before** the output heads.

        Useful for subclasses / wrapper models (e.g. EDL) that attach their
        own heads on top of the shared backbone.
        """
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return x

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, in_channels, H, W)

        Returns
        -------
        dict with keys 'predicted_mean' and 'predicted_std', each (B, 1, H, W).
        """
        features = self.forward_features(x)

        predicted_mean = self.softplus_2(self.head_mean(features))      # ŷ ∈ R
        predicted_std  = self.softplus(self.head_std(features)) + 1e-6  # σ > 0

        return {
            "predicted_mean": predicted_mean,
            "predicted_std":  predicted_std,
        }

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_unet(in_channels: int = 2, **kwargs) -> UNet:
    """
    Instantiate a UNet with the given input channels and optional overrides.

    Example
    -------
    # >>> model = build_unet(in_channels=2, base_channels=32, depth=4)
    """
    return UNet(in_channels=in_channels, **kwargs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}\n")

    # Default config: 2 channels (mask + values), 64×64 grid
    B, C, H, W = 4, 2, 64, 64
    model = build_unet(in_channels=C, base_channels=32, depth=4).to(device)
    print(f"Trainable parameters: {model.count_parameters():,}")

    # Print architecture summary (channel sizes per stage)
    print(f"\nEncoder channel progression:")
    ch = 32
    for i in range(4):
        print(f"  Stage {i+1}: {ch * (2**i):4d} channels -> MaxPool(2)")
    print(f"  Bottleneck: {32 * (2**4):4d} channels")

    # Forward pass
    x = torch.randn(B, C, H, W, device=device)
    with torch.no_grad():
        out = model(x)

    mean = out["predicted_mean"]
    std  = out["predicted_std"]

    print(f"\nInput shape       : {tuple(x.shape)}")
    print(f"predicted_mean    : {tuple(mean.shape)}  "
          f"range [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"predicted_std     : {tuple(std.shape)}   "
          f"range [{std.min():.4f}, {std.max():.4f}]  (all > 0: {(std > 0).all().item()})")

    # Quick sanity: non-square input
    x2 = torch.randn(2, C, 50, 80, device=device)
    with torch.no_grad():
        out2 = model(x2)
    print(f"\nNon-square input  : {tuple(x2.shape)}")
    print(f"predicted_mean    : {tuple(out2['predicted_mean'].shape)}")
    print(f"predicted_std     : {tuple(out2['predicted_std'].shape)}")
    print("\nAll checks passed.")
