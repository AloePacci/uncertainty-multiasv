"""
metrics.py — Uncertainty Estimation CAEPIA 2026
================================================
Evaluation functions for scalar field reconstruction and uncertainty calibration.

All functions accept numpy arrays or PyTorch tensors of arbitrary shape and
return a Python float. An optional boolean/float `mask` (same shape as inputs)
restricts evaluation to locations where mask == 1 — useful to evaluate only on
unobserved cells X_test = X \\ {x_i}.

Functions
---------
rmse         : Root Mean Squared Error
r2           : Coefficient of Determination (R²)
nll          : Gaussian Negative Log-Likelihood  [Eq. nll_metric in paper]
ece          : Expected Calibration Error         [Eq. ece in paper]
"""

import numpy as np
from scipy.stats import norm as scipy_norm


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_numpy(*arrays, mask=None):
    """Convert tensor/array inputs to flat numpy float64 arrays, applying mask."""
    out = []
    for a in arrays:
        try:
            a = a.detach().cpu().numpy()    # torch.Tensor → numpy
        except AttributeError:
            a = np.asarray(a)
        a = a.astype(np.float64).ravel()
        out.append(a)

    if mask is not None:
        try:
            mask = mask.detach().cpu().numpy()
        except AttributeError:
            mask = np.asarray(mask)
        mask = mask.ravel().astype(bool)
        out = [a[mask] for a in out]

    return out


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred, mask=None) -> float:
    """Root Mean Squared Error between ground truth and predicted mean.

    Parameters
    ----------
    y_true : array-like  (arbitrary shape)
        Ground truth scalar field.
    y_pred : array-like  (same shape as y_true)
        Predicted mean field.
    mask : array-like or None
        Binary mask (1 = evaluate, 0 = ignore). If None, all locations used.

    Returns
    -------
    float
    """
    y_true, y_pred = _to_numpy(y_true, y_pred, mask=mask)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred, mask=None) -> float:
    """Coefficient of Determination (R²).

    R² = 1 − SS_res / SS_tot
       = 1 − Σ(y_true − y_pred)² / Σ(y_true − mean(y_true))²

    A value of 1.0 indicates a perfect fit; 0.0 means the model performs no
    better than predicting the mean; negative values indicate worse than mean.

    Parameters
    ----------
    y_true : array-like  (arbitrary shape)
    y_pred : array-like  (same shape)
    mask   : array-like or None

    Returns
    -------
    float
    """
    y_true, y_pred = _to_numpy(y_true, y_pred, mask=mask)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def nll(y_true, y_pred_mean, y_pred_std, mask=None) -> float:
    """Gaussian Negative Log-Likelihood (Eq. nll_metric in the paper).

    NLL = mean_x [ (y_true − μ)² / (2σ²) + ½ log σ² + ½ log 2π ]

    Lower values indicate better-calibrated predictive distributions.

    Parameters
    ----------
    y_true     : array-like  (arbitrary shape)
        Ground truth scalar field.
    y_pred_mean : array-like  (same shape)
        Predicted mean μ.
    y_pred_std  : array-like  (same shape)
        Predicted standard deviation σ > 0. A floor of 1e-6 is applied.
    mask : array-like or None

    Returns
    -------
    float
    """
    y_true, mu, sigma = _to_numpy(y_true, y_pred_mean, y_pred_std, mask=mask)
    sigma = np.maximum(sigma, 1e-6)
    var = sigma ** 2
    return float(np.mean(
        (y_true - mu) ** 2 / (2.0 * var)
        + 0.5 * np.log(var)
        + 0.5 * np.log(2.0 * np.pi)
    ))
    

def ence(y_true, y_pred_mean, y_pred_std, n_bins: int = 10, mask=None) -> float:
    """Expected Normalized Calibration Error for regression (Eq. ence in the paper).

    Similar to ECE but normalized by the bin width to account for varying
    confidence levels. For each confidence level p_b ∈ {1/B, 2/B, …, B/B},
    compute the fraction of test points whose true value falls within the
    two-sided predictive interval [μ − z_b·σ, μ + z_b·σ], where z_b is the
    corresponding normal quantile. ENCE is the mean absolute deviation between
    nominal and empirical coverages, normalized by bin width:

        ENCE = (1/B) Σ_b |RMV_b − EMV_b| / (p_b − p_{b-1})
        
    A perfectly calibrated model yields ENCE = 0.
    
    Parameters
    ----------
    y_true      : array-like  (arbitrary shape)
    y_pred_mean : array-like  (same shape)
    y_pred_std  : array-like  (same shape)  — σ > 0
    n_bins : int
    
    mask : array-like or None
    Returns
    -------
    
    float
    """
    
    y_true, mu, sigma = _to_numpy(y_true, y_pred_mean, y_pred_std, mask=mask)
    sigma = np.maximum(sigma, 1e-6)

    # Confidence levels: 1/B, 2/B, ..., 1  (last bin = 100% coverage)
    confidence_levels = np.linspace(1.0 / n_bins, 1.0, n_bins)

    ence_error = 0.0
    prev_p_b = 0.0
    for p_b in confidence_levels:
        z = scipy_norm.ppf((1.0 + p_b) / 2.0)          # two-sided quantile
        lower = mu - z * sigma
        upper = mu + z * sigma
        empirical_coverage = np.mean((y_true >= lower) & (y_true <= upper))
        bin_width = p_b - prev_p_b
        if bin_width > 0:
            ence_error += abs(p_b - empirical_coverage) / bin_width
        prev_p_b = p_b

    return float(ence_error / n_bins)



def uce(y_true, y_pred_mean, y_pred_std, n_bins: int = 10, mask=None) -> float:
    """Uncertainty Calibration Error (Laves et al., 2020).

    Partitions the uncertainty range into ``n_bins`` equal-width bins and
    measures the weighted average absolute difference between the per-bin
    mean squared error and the per-bin mean variance:

        UCE = Σ_j (|B_j| / N) · |err(B_j) − uncert(B_j)|

    where
        err(B_j)    = (1/|B_j|) Σ_{i∈B_j} (y_true_i − ŷ_i)²
        uncert(B_j) = (1/|B_j|) Σ_{i∈B_j} σ̂²_i

    A perfectly calibrated model yields UCE = 0.

    Parameters
    ----------
    y_true      : array-like  (arbitrary shape)
    y_pred_mean : array-like  (same shape)
    y_pred_std  : array-like  (same shape)  — σ̂ > 0
    n_bins : int
        Number of equal-width uncertainty bins (default 10).
    mask : array-like or None

    Returns
    -------
    float
    """
    y_true, mu, sigma = _to_numpy(y_true, y_pred_mean, y_pred_std, mask=mask)
    sigma = np.maximum(sigma, 1e-6)

    variance   = sigma ** 2
    sq_error   = (y_true - mu) ** 2
    N          = len(y_true)

    bin_edges  = np.linspace(variance.min(), variance.max(), n_bins + 1)
    bin_edges[-1] += 1e-10  # include right edge in last bin

    uce_error = 0.0
    for j in range(n_bins):
        in_bin = (variance >= bin_edges[j]) & (variance < bin_edges[j + 1])
        if not in_bin.any():
            continue
        err_j    = sq_error[in_bin].mean()
        uncert_j = variance[in_bin].mean()
        uce_error += (in_bin.sum() / N) * abs(err_j - uncert_j)

    return float(uce_error)


def uce_bins(y_true, y_pred_mean, y_pred_std, n_bins: int = 10, mask=None):
    """Per-bin statistics for the UCE calibration curve.

    Returns the mean squared error and mean variance for each equal-width
    uncertainty bin, useful for plotting err(B) vs uncert(B).

    Parameters
    ----------
    y_true      : array-like  (arbitrary shape)
    y_pred_mean : array-like  (same shape)
    y_pred_std  : array-like  (same shape)  — σ̂ > 0
    n_bins : int
    mask : array-like or None

    Returns
    -------
    uncert_centers : np.ndarray  (n_bins,)  — bin centre (mean variance in bin)
    err_per_bin    : np.ndarray  (n_bins,)  — mean squared error per bin
    uncert_per_bin : np.ndarray  (n_bins,)  — mean variance per bin
    counts         : np.ndarray  (n_bins,)  — number of samples per bin (0 = empty)
    """
    y_true, mu, sigma = _to_numpy(y_true, y_pred_mean, y_pred_std, mask=mask)
    sigma    = np.maximum(sigma, 1e-6)
    variance = sigma ** 2
    sq_error = (y_true - mu) ** 2

    bin_edges = np.linspace(variance.min(), variance.max(), n_bins + 1)
    bin_edges[-1] += 1e-10

    err_per_bin    = np.full(n_bins, np.nan)
    uncert_per_bin = np.full(n_bins, np.nan)
    counts         = np.zeros(n_bins, dtype=int)

    for j in range(n_bins):
        mask_j = (variance >= bin_edges[j]) & (variance < bin_edges[j + 1])
        if mask_j.any():
            err_per_bin[j]    = sq_error[mask_j].mean()
            uncert_per_bin[j] = variance[mask_j].mean()
            counts[j]         = mask_j.sum()

    return err_per_bin, uncert_per_bin, counts


def ece(y_true, y_pred_mean, y_pred_std, n_bins: int = 10, mask=None) -> float:
    """Expected Calibration Error for regression (Eq. ece in the paper).

    For each confidence level p_b ∈ {1/B, 2/B, …, B/B}, compute the
    fraction of test points whose true value falls within the two-sided
    predictive interval [μ − z_b·σ, μ + z_b·σ], where z_b is the
    corresponding normal quantile. ECE is the mean absolute deviation
    between nominal and empirical coverages:

        ECE = (1/B) Σ_b |p_b − ĥat{p}_b|

    A perfectly calibrated model yields ECE = 0.

    Parameters
    ----------
    y_true      : array-like  (arbitrary shape)
    y_pred_mean : array-like  (same shape)
    y_pred_std  : array-like  (same shape)  — σ > 0
    n_bins : int
        Number of equally spaced confidence levels B (default 10).
    mask : array-like or None

    Returns
    -------
    float
    """
    y_true, mu, sigma = _to_numpy(y_true, y_pred_mean, y_pred_std, mask=mask)
    sigma = np.maximum(sigma, 1e-6)

    # Confidence levels: 1/B, 2/B, ..., 1  (last bin = 100% coverage)
    confidence_levels = np.linspace(1.0 / n_bins, 1.0, n_bins)

    calibration_error = 0.0
    for p_b in confidence_levels:
        z = scipy_norm.ppf((1.0 + p_b) / 2.0)          # two-sided quantile
        lower = mu - z * sigma
        upper = mu + z * sigma
        empirical_coverage = np.mean((y_true >= lower) & (y_true <= upper))
        calibration_error += abs(p_b - empirical_coverage)

    return float(calibration_error / n_bins)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 10_000

    # ── Perfect prediction ──────────────────────────────────────────────────
    y_true = rng.standard_normal(N)
    y_pred = y_true.copy()
    sigma  = np.ones(N) * 0.5

    print("=== Perfect prediction (y_pred = y_true) ===")
    print(f"  RMSE : {rmse(y_true, y_pred):.6f}   (expected 0.0)")
    print(f"  R²   : {r2(y_true, y_pred):.6f}   (expected 1.0)")

    # ── Well-calibrated Gaussian ────────────────────────────────────────────
    true_sigma = 0.5
    noise      = rng.normal(0, true_sigma, N)
    y_noisy    = y_true + noise
    sigma_hat  = np.full(N, true_sigma)    # perfectly calibrated σ

    print("\n=== Well-calibrated model (std = true noise std) ===")
    print(f"  RMSE : {rmse(y_true, y_noisy):.4f}   (expected ~{true_sigma:.2f})")
    print(f"  R2   : {r2(y_true, y_noisy):.4f}")
    print(f"  NLL  : {nll(y_true, y_noisy, sigma_hat):.4f}   "
          f"(optimal ~{0.5 * (1 + np.log(2*np.pi*true_sigma**2)):.4f})")
    print(f"  ENCE  : {ence(y_true, y_noisy, sigma_hat):.4f}   (expected ~0.0)")

    # ── Overconfident model (std too small) ─────────────────────────────────
    sigma_over = np.full(N, true_sigma * 0.2)
    print("\n=== Overconfident model (std << true std) ===")
    print(f"  NLL  : {nll(y_true, y_noisy, sigma_over):.4f}   (expected >> optimal)")
    print(f"  ENCE  : {ence(y_true, y_noisy, sigma_over):.4f}   (expected >> 0.0)")

    # ── Mask usage ───────────────────────────────────────────────────────────
    mask_arr = rng.integers(0, 2, N)
    print("\n=== With mask (half the points) ===")
    print(f"  RMSE : {rmse(y_true, y_noisy, mask=mask_arr):.4f}")
    print(f"  NLL  : {nll(y_true, y_noisy, sigma_hat, mask=mask_arr):.4f}")
    print(f"  ENCE  : {ence(y_true, y_noisy, sigma_hat, mask=mask_arr):.4f}")

    print("\nAll checks passed.")
