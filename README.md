# Uncertainty Estimation for Oil Spill Field Reconstruction

This repository contains the code and experiments for the paper submitted to , comparing four uncertainty-aware methods for reconstructing scalar density fields from sparse, noisy sensor observations in an oil spill monitoring scenario.

## Table of Contents

1. [Problem Setup](#problem-setup)
2. [Datasets](#datasets)
3. [Models](#models)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Repository Structure](#repository-structure)
8. [Requirements](#requirements)

---

## Problem Setup

A fleet of mobile sensors (AUV or drone) traverse a region containing an oil spill and collects sparse, noisy observations along their trajectory. The goal is to **reconstruct the full 2D density field** from these partial measurements and, crucially, to **quantify prediction uncertainty** — decomposed into:

- **Epistemic uncertainty** — model/knowledge uncertainty, reducible with more data.
- **Aleatoric uncertainty** — irreducible observation noise inherent to the sensor.

Each model receives a 2-channel input `(observation_mask, observed_values)` on a 100×100 grid and outputs a predicted mean field alongside both uncertainty components.

---

## Datasets

Three synthetic datasets are generated using a physics-based oil spill particle simulator ([DatasetGeneration.py](DatasetGeneration.py), [ground_truths.py](ground_truths.py)). Each dataset contains **2000 simulations** on a **100×100 grid**, differing only in their **observation (sensor) model**:

| Dataset | Sensor Model | Avg. Coverage |
|---|---|---|
| **CONIC** | Conic field-of-view (60° angle, radius 10 px) | 14.4% |
| **NADIR** | Nadir downward-looking camera (radius 1 px) | low |
| **POINTWISE** | Single waypoint measurements | lowest |

Each sample is stored in a `.npz` file with three arrays:
- `ground_truth`: (2000, 100, 100) — true spill density field
- `observed_map`: (2000, 100, 100) — noisy observations (0 where unobserved)
- `observed_mask`: (2000, 100, 100) — binary mask of observed pixels

**Simulation parameters**: 2000 particles, spill radius 3.0, wind/tide speed 1.0, 50–100 steps per simulation, observation noise σ ∈ [0.01, 0.08].

### Sample Visualisation

The figure below shows six random samples from the CONIC dataset. Each row shows the ground truth density field, the noisy observation map, and the binary observation mask.

![CONIC sample gallery](Datasets/plots/dataset_CONIC_11_sample_gallery.png)

### Observation Coverage Distribution

Coverage varies substantially across simulations — even with the widest sensor (CONIC), large portions of the field remain unobserved, making reconstruction a genuinely ill-posed problem.

![Coverage distribution](Datasets/plots/dataset_CONIC_01_coverage_dist.png)

### Observation vs. Ground Truth

The scatter below shows observed pixel values against their true density. The spread captures sensor noise and illustrates the signal-to-noise conditions that uncertainty models must handle.

![Observed vs GT scatter](Datasets/plots/dataset_CONIC_10_obs_vs_gt_scatter.png)

---

## Models

All models share a common interface:

```python
predictions = model.predict(X)
# returns dict with keys:
#   'predicted_mean'           (B, 1, H, W)
#   'predicted_std_epistemic'  (B, 1, H, W)
#   'predicted_std_aleatoric'  (B, 1, H, W)
```

### 1. Monte Carlo Dropout ([MC_dropout_model.py](MC_dropout_model.py))

Standard U-Net with **Dropout2d (p=0.2)** inserted after every ReLU. At inference, dropout remains active and **T=30 stochastic forward passes** are drawn:

- **Epistemic σ** = std of predicted means across T passes
- **Aleatoric σ** = mean of predicted stds across T passes

Loss: heteroscedastic Gaussian NLL — `0.5 * [(y−μ)²/σ² + log σ²]`

### 2. Deep Ensemble ([MC_ensemble_model.py](MC_ensemble_model.py))

**5 independently trained U-Nets**, each with a distinct random seed and no weight sharing. At inference all members run in deterministic eval mode:

- **Epistemic σ** = std of predicted means across members
- **Aleatoric σ** = mean of predicted stds across members

Ensembles exploit diversity through different weight initialisation and SGD stochasticity.

### 3. Evidential Deep Learning — EDL ([EDL_model.py](EDL_model.py))

A single U-Net backbone with **four output heads** parametrising a **Normal-Inverse-Gamma (NIG)** distribution `p(μ, σ² | γ, ν, α, β)`:

| Head | Parameter | Constraint | Meaning |
|---|---|---|---|
| γ | mean | unconstrained | predicted field value |
| ν | virtual obs. count | Softplus | epistemic evidence |
| α | IG shape | Softplus + 1 | aleatoric shape |
| β | IG scale | Softplus | aleatoric scale |

Uncertainty decomposition:
- **Aleatoric σ** = √(β / (α − 1))
- **Epistemic σ** = √(β / (ν · (α − 1)))

Loss: NIG-NLL plus regularisation `λ · |y − γ| · (2ν + α)`, with λ=1e-3 and gradient clipping (max_norm=1.0) to stabilise training.

### 4. Gaussian Process ([gaussian_process_model.py](gaussian_process_model.py))

**No offline training** — a GP is fitted per test sample directly on the observed pixels. Kernel: `C · RBF(ℓ) + WhiteKernel(σ_n)`, optimised by L-BFGS.

- **Epistemic σ** = posterior predictive std of the signal kernel (zero at observation sites)
- **Aleatoric σ** = √(fitted noise_level) × y_std (spatially uniform)

Provides a strong non-parametric baseline but scales poorly to dense observations.

### U-Net Backbone ([models.py](models.py))

Shared by all deep learning models:

- **Encoder**: 4 blocks of `(Conv→BN→ReLU)×2 + MaxPool(2)`, channels: 32→64→128→256→512
- **Bottleneck**: 512→1024
- **Decoder**: 4 transposed-convolution blocks with skip connections
- **Output heads**: 1×1 convolutions with Softplus activations

---

## Training

Training is orchestrated by [train_models.py](train_models.py). The dataset is split **80% / 20%** (seed=42) — the same split is reused at evaluation.

| Hyperparameter | MC Dropout | Ensemble | EDL |
|---|---|---|---|
| Epochs | 50 | 50 | 50 |
| Batch size | 16 | 16 | 16 |
| Learning rate | 3e-4 | 3e-4 | 5e-4 |
| Optimizer | Adam | Adam | Adam |
| Gradient clipping | — | — | max_norm=1.0 |
| Loss | Gaussian NLL | Gaussian NLL | NIG-NLL + reg |

Ensemble training runs each of the 5 members independently. GP models require no weight training and are fitted at inference time. Trained weights are saved to `Weights/` as `{dataset}_{model}.pt`.

```bash
python train_models.py     # trains all DL models on all three datasets
python evaluate_models.py  # runs inference and saves Results/eval_{dataset}.pkl.gz
python plot_results.py     # generates all figures in Results/plots/
python dataset_analysis.py # generates exploratory figures in Datasets/plots/
```

---

## Evaluation

Metrics are computed in [metrics.py](metrics.py) on the 20% held-out test set (400 samples per dataset):

| Metric | Measures |
|---|---|
| **RMSE** | Reconstruction accuracy |
| **R²** | Explained variance |
| **NLL** | Calibration quality (penalises over/under-confidence) |
| **ECE** | Mean coverage deviation across confidence levels |
| **UCE** | Uncertainty vs. empirical variance across uncertainty bins |

---

## Results

All result figures are in [Results/plots/](Results/plots/).

### Reconstruction Quality — RMSE

Box plots of RMSE across all 400 test samples and three datasets. **EDL achieves the best reconstruction** across all sensor models, far ahead of all other methods.

![RMSE boxplot](Results/plots/rmse_boxplot.png)

**RMSE by algorithm and observation model (mean ± std):**

|           | Conic FOV | Nadir camera | Pointwise sensor |
|-----------|:---------:|:------------:|:----------------:|
| **EDL**       | 0.0274 ± 0.0234 | 0.0293 ± 0.0209 | 0.0272 ± 0.0204 |
| **Ensemble**  | 0.0506 ± 0.0238 | 0.0570 ± 0.0200 | 0.0586 ± 0.0188 |
| **MC Dropout**| 0.0878 ± 0.0222 | 0.0915 ± 0.0160 | 0.0992 ± 0.0167 |
| **GP**        | 0.1146 ± 0.0348 | 0.1193 ± 0.0648 | 0.1146 ± 0.0321 |

### Calibration — Negative Log-Likelihood

Lower NLL indicates better-calibrated uncertainty. **Ensemble achieves the best NLL**, followed by EDL. MC Dropout shows high variance, suggesting unstable uncertainty estimates.

![NLL boxplot](Results/plots/nll_boxplot.png)

**NLL by algorithm and observation model (mean ± std):**

|           | Conic FOV | Nadir camera | Pointwise sensor |
|-----------|:---------:|:------------:|:----------------:|
| **Ensemble**  | −4.6572 ± 0.5717 | −4.0291 ± 0.4421 | −4.1654 ± 0.3356 |
| **EDL**       | −1.9935 ± 16.7829 | −2.6260 ± 12.4711 | −3.3552 ± 5.4053 |
| **MC Dropout**| 1.5204 ± 61.1348 | −2.9536 ± 10.1969 | −3.4230 ± 1.1549 |
| **GP**        | −0.6332 ± 0.5984 | 2.8468 ± 10.6790 | 2.6945 ± 10.0616 |

### Calibration Curves

Normalized error vs. normalized uncertainty across normalized confidence levels (0.0 to 1.0). A perfectly calibrated model follows the diagonal. **EDL and Ensemble are best calibrated**.

![Calibration curves](Results/plots/calibration_curve.png)

**Mean absolute deviation from the ideal calibration diagonal:**

| Algorithm | Conic FOV | Nadir camera | Pointwise sensor |
|-----------|:---------:|:------------:|:----------------:|
| **EDL**       | **0.0380** | **0.0378** | 0.0401 |
| **Ensemble**  | 0.0502 | 0.0722 | **0.0425** |
| **MC Dropout**| 0.1254 | 0.1277 | 0.0698 |
| **GP**        | 0.5281 | 0.6149 | 0.5993 |

### Uncertainty Calibration Error (UCE)

UCE measures whether predicted uncertainty magnitudes are consistent with actual errors. **EDL achieves the lowest UCE** across all sensor models.

![UCE barplot](Results/plots/uce_barplot.png)

**UCE by algorithm and observation model (mean ± std):**

|           | Conic FOV | Nadir camera | Pointwise sensor |
|-----------|:---------:|:------------:|:----------------:|
| **EDL**       | **0.0018 ± 0.0031** | **0.0016 ± 0.0025** | **0.0013 ± 0.0022** |
| **Ensemble**  | 0.0026 ± 0.0016 | 0.0040 ± 0.0016 | 0.0040 ± 0.0017 |
| **MC Dropout**| 0.0084 ± 0.0026 | 0.0063 ± 0.0018 | 0.0110 ± 0.0029 |
| **GP**        | 0.0110 ± 0.0158 | 0.0241 ± 0.2443 | 0.0096 ± 0.0093 |

### Epistemic vs. Aleatoric Decomposition

Stacked bars show the mean epistemic and aleatoric contribution to total predictive uncertainty per method. **EDL balances both** (48% epistemic), while MC Dropout is dominated by aleatoric uncertainty (71%).

![Uncertainty decomposition](Results/plots/uncertainty_stacked_bars.png)

### Inference Time

**EDL is the fastest method** (~7 ms), while GP is the slowest due to per-sample fitting (up to 2247 ms for Conic FOV).

| Algorithm | Conic FOV | Nadir camera | Pointwise sensor |
|-----------|:---------:|:------------:|:----------------:|
| **EDL**       | 7.1 ± 1.5 ms | 7.2 ± 1.4 ms | 7.4 ± 1.3 ms |
| **Ensemble**  | 20.3 ± 2.5 ms | 20.6 ± 0.5 ms | 20.9 ± 0.6 ms |
| **MC Dropout**| 202.4 ± 12.4 ms | 202.0 ± 2.7 ms | 207.1 ± 8.3 ms |
| **GP**        | 2247.1 ± 0.0 ms | 531.8 ± 0.0 ms | 81.2 ± 0.0 ms |

### Qualitative Comparison — POINTWISE Dataset

Side-by-side prediction maps for a representative test sample: ground truth, predicted mean, epistemic uncertainty, and aleatoric uncertainty for each method.

![Sample prediction maps](Results/plots/sample_maps_Pointwise_sensor.png)

### Summary

**Key takeaways**:
- **EDL** dominates on reconstruction quality (RMSE), uncertainty calibration (UCE, calibration curves), and inference speed, making it the best overall method.
- **Ensemble** achieves the best NLL calibration but is ~3× slower than EDL at inference.
- **GP** offers principled probabilistic construction but scales poorly: it has the worst RMSE and is orders of magnitude slower than deep learning methods.
- **MC Dropout** produces the most uncertain and least accurate predictions — instability stems from dropout remaining active at inference, creating noisy aleatoric estimates.

---

## Repository Structure

```
.
├── DatasetGeneration.py         # Physics-based oil spill dataset generator
├── ground_truths.py             # Particle-based spill simulator
├── ObservationModels.py         # Pointwise / Nadir / Conic sensor models
├── models.py                    # Shared U-Net backbone
├── MC_dropout_model.py          # Monte Carlo Dropout model
├── MC_ensemble_model.py         # Deep Ensemble model
├── EDL_model.py                 # Evidential Deep Learning (NIG) model
├── gaussian_process_model.py    # Gaussian Process baseline
├── train_models.py              # Training pipeline for all DL models
├── evaluate_models.py           # Evaluation pipeline (inference + metrics)
├── metrics.py                   # RMSE, R², NLL, ECE, UCE implementations
├── evaluation_store.py          # Results serialisation (DataFrame ↔ pkl.gz)
├── plot_results.py              # Generate all result figures
├── dataset_analysis.py          # Exploratory dataset plots
├── utils.py                     # Shared visualisation helpers
├── Datasets/
│   ├── dataset_CONIC.npz        # 2000 samples, Conic FOV sensor
│   ├── dataset_NADIR.npz        # 2000 samples, Nadir camera
│   ├── dataset_POINTWISE.npz    # 2000 samples, Pointwise sensor
│   ├── dataset_config_*.yaml    # Simulation parameters per dataset
│   └── plots/                   # 11 exploratory figures × 3 datasets
├── Weights/
│   └── {dataset}_{model}.pt     # Trained model weights
└── Results/
    ├── eval_{dataset}.pkl.gz    # Evaluation DataFrames (400 samples × metrics)
    └── plots/                   # Result figures (boxplots, calibration, maps)
```

---

## Requirements

```
torch
numpy
scikit-learn
scipy
matplotlib
seaborn
pyyaml
tqdm
```

Install with:

```bash
pip install -r requirements.txt
```

To regenerate datasets from scratch:

```bash
python DatasetGeneration.py
```

To retrain all models:

```bash
python train_models.py
```

To reproduce all results and figures:

```bash
python evaluate_models.py
python plot_results.py
python dataset_analysis.py
```
