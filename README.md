# Uncertainty-Aware Informative Path Planning for Multi-ASV Oil Spill Monitoring

Code and experiments for the paper *Uncertainty-Aware Informative Path Planning for Multi-ASV Oil Spill Monitoring* by Alejandro Casado-Pérez, Samuel Yanes, Marija Popović, Sergio L. Toral and Daniel Gutiérrez-Reina.

A fleet of four Autonomous Surface Vehicles (ASVs) maps an unknown oil spill from sparse, noisy, pointwise measurements under a distance budget. The ASVs share one deep reconstruction model, which returns the spill field together with its uncertainty, and use the uncertainty map to coordinate where to sample next. The study compares three deep uncertainty quantification methods (Monte Carlo Dropout, Deep Ensembles and Evidential Deep Learning) against a Gaussian Process baseline. Each model runs in closed loop with five planners: Value-Greedy, Uncertainty-Greedy, ε-Greedy, receding-horizon Orienteering and multi-agent Monte Carlo Tree Search.

**Main finding:** calibrated uncertainty is the best signal for guiding a fleet. Evidential Deep Learning (EDL) has the best-calibrated uncertainty, the lowest reconstruction error (mean RMSE 0.001847) and the highest spill IoU (84.9 %), at 3.9 ms per inference. Look-ahead planners amplify the effect of calibration: MCTS gets near-best reconstruction with EDL and the worst of all configurations with Deep Ensembles.

**Framework overview** ([PDF](Results/plots/framework_figure.pdf)): *(a) four ASVs explore an unknown spill; (b) the measured nodes are the only model input; (c) reconstruction μ and (d) uncertainty σ returned by EDL; MCTS uses σ to choose each agent's next waypoint.*

---

## Table of Contents

1. [Problem Setup](#problem-setup)
2. [Sensing–Learning–Planning Loop](#sensinglearningplanning-loop)
3. [Reconstruction Models](#reconstruction-models)
4. [Planners](#planners)
5. [Datasets](#datasets)
6. [Results](#results)
7. [Repository Structure](#repository-structure)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Acknowledgments](#acknowledgments)

---

## Problem Setup

| Element | Value in the code |
|---|---|
| Map | 100 × 100 grid. An 8-pixel border is non-navigable (masked out of observations, predictions and planning). |
| Fleet | N = 4 homogeneous ASVs, starting at `(50,50)`, `(60,50)`, `(50,60)`, `(60,60)` ([scenario_config.yaml](scenario/scenario_config.yaml)). |
| Field | Static oil-spill concentration f, normalised to [0, 1]. |
| Sensor | Pointwise contact sensor. Each agent moves in a straight line to its waypoint and measures every cell it crosses, with additive Gaussian noise (`noise_std` in the config). |
| Budget | D = 300 px of Euclidean distance per vehicle. The episode ends when any vehicle exhausts its budget. |
| Objective | Minimise the RMSE between f and the reconstruction f̂ at the end of the mission. |

Constraints and assumptions follow the paper: the occupancy map and initial positions are known, localisation is exact, and communication is instantaneous and lossless, so all agents share one model f̂.

### Uncertainty decomposition

Every model returns a predictive mean and two standard deviations:

σ²<sub>total</sub>(x) = σ²<sub>ale</sub>(x) (irreducible sensor noise) + σ²<sub>epi</sub>(x) (reducible lack of knowledge)

The environment ([extended_scenario.py](scenario/extended_scenario.py)) gives the policies `predicted_uncertainty = sqrt(σ²_epi + σ²_ale)`, the total predictive standard deviation, masked to the navigable area.

---

## Sensing–Learning–Planning Loop

All planners run inside the same receding-horizon loop ([run_experiments.py](scenario/run_experiments.py)):

1. The agents move to their waypoints and measure along the way (`ObservationScenario.step`).
2. The reconstruction model is queried on the observation mask and the observed values. It returns the predicted mean and uncertainty maps.
3. The policy receives the observation dictionary and the agent positions, and returns one waypoint per agent. Only the first step of any longer plan is executed.
4. The loop repeats until the budget runs out.

Every step logs RMSE (unobserved cells), IoU (threshold τ = 0.05), per-agent distance, all maps, positions, and model and policy wall-clock time with [sylegendarium](https://pypi.org/project/sylegendarium/) into `experiments/`.

---

## Reconstruction Models

All models share the interface used by the scenario:

```python
out = model.predict({"obs_map": obs_map, "obs_mask": obs_mask})
# out["predicted_mean"], out["predicted_std_epistemic"], out["predicted_std_aleatoric"]
```

### Shared U-Net backbone ([unet_model.py](scenario/models/unet_model.py))

- **Input:** `X = [U, V]` ∈ ℝ<sup>2×H×W</sup>. U is the binary observation mask and V holds the sensor readings (zero elsewhere). Encoding the mask explicitly lets the network tell an unobserved cell from a zero reading.
- **Encoder:** 4 blocks of `(Conv → BN → ReLU) × 2 + MaxPool`, channels 32 → 64 → 128 → 256 → 512.
- **Bottleneck:** 1024 channels.
- **Decoder:** transposed convolutions with skip connections.
- **Head:** 1×1 convolution producing the mean and a variance head (Softplus).
- **Loss:** heteroscedastic Gaussian NLL, `½ (y − μ)²/σ² + ½ log σ²`. This keeps the predicted variance from collapsing to zero, which is what happens when training with MSE alone.

### Monte Carlo Dropout ([MC_dropout_model.py](scenario/models/MC_dropout_model.py))

Dropout2d with p = 0.2 after each convolutional block, kept active at inference. S = 30 stochastic passes:

- f̂ = mean of the predicted means
- σ²<sub>epi</sub> = variance of the predicted means
- σ²<sub>ale</sub> = mean of the predicted variances

### Deep Ensemble ([MC_ensemble_model.py](scenario/models/MC_ensemble_model.py))

M = 5 independently initialised U-Nets (seed + k), each trained with the Gaussian NLL on independently shuffled data. At inference all members run deterministically:

- σ²<sub>epi</sub> = disagreement (variance) between the member means
- σ²<sub>ale</sub> = mean of the member variances

### Evidential Deep Learning ([EDL_model.py](scenario/models/EDL_model.py))

The U-Net outputs the four Normal-Inverse-Gamma parameters (γ, ν, α, β) per pixel, with ν > 0, α > 1 and β > 0 enforced through Softplus. A single forward pass gives:

| Quantity | Closed form |
|---|---|
| f̂ | γ |
| σ²<sub>ale</sub> | β / (α − 1) |
| σ²<sub>epi</sub> | β / (ν (α − 1)) |

Loss: Student-t NLL of the marginalised NIG plus the evidence regulariser `λ |f − γ| (2ν + α)`, with λ = 10⁻³ and gradient clipping at max-norm 1.0.

### Gaussian Process baseline ([gaussian_process_model.py](scenario/models/gaussian_process_model.py))

There is no offline training. At every step a GP is refitted on the samples collected so far, with kernel `C · RBF(ℓ) + WhiteKernel(σ_n²)` and `normalize_y=True`. Hyperparameters are fitted by maximising the marginal log-likelihood with L-BFGS (2 restarts in the experiments).

- σ²<sub>epi</sub>: posterior variance. It contracts at sampled locations.
- σ²<sub>ale</sub>: the fitted noise level, spatially uniform.

The posterior variance depends on where samples were taken, not on their values, so a GP-driven planner degenerates into coverage.

[myopic_model.py](scenario/models/myopic_model.py) also implements an inverse-distance-weighting interpolator, which is not part of the paper's benchmark.

---

## Planners

Policies live in [scenario/policies/](scenario/policies/) and implement `Policy.act(obs, positions) -> waypoints`.

**Multi-agent coordination.** Agents choose one after another within a decision step. Each agent sees the targets already committed by the previous ones (they are marked as visited), so the fleet spreads out without any explicit distance constraint.

| Paper name | Experiment key | Class | Behaviour in the code |
|---|---|---|---|
| Value-Greedy | `myopic_greedy` | `MaxGreedyMiopic` | Picks the unobserved cell within 30 px that maximises `predicted_mean × uncertainty`. |
| Uncertainty-Greedy | `uncertainty_greedy` | `MaxUncertaintyPolicy` | Picks the unobserved cell within 30 px with the highest uncertainty. |
| ε-Greedy | `epsilon_greedy` | `EpsilonGreedy` | ε decays linearly from 0.1 to 0.01 over 100 steps. Chooses between a random unobserved cell within 30 px and the highest-mean candidate. |
| RH Orienteering | `orienteering` | `OrienteeringPolicy` | Orienteering problem on a sub-grid of candidate waypoints (step 2 px). Edge reward is the sum of uncertainty along the segment, edge cost is the Euclidean length. Planning horizon 30 px, replans every 5 px. Solved per agent with multi-start greedy construction followed by remove/insert hill-climbing. |
| MCTS | `mcts` | `MAMCTSPolicy` | Multi-agent UCT ([multiagent_mcts.py](scenario/policies/algorithms/multiagent_mcts.py)) on [multiagent_max_informative_path_waypoints.py](scenario/multiagent_max_informative_path_waypoints.py). Settings: 500 simulations, depth 50, γ = 0.8, c = √2, tree reuse. Each tree level is one agent's action, so a joint action spans N levels, and agents are ordered by remaining budget. Branching is uncertainty-adaptive: the 8-neighbour step shrinks from 4 px to 1 px as local information grows. The reward map is the min-max-normalised uncertainty on unobserved cells. |

`MCTSPolicy` (single-agent MCTS) and the forward-search, branch-and-bound and sparse-sampling planners in [scenario/policies/algorithms/](scenario/policies/algorithms/) are not part of the paper's benchmark. [algorithms.md](scenario/policies/algorithms/algorithms.md) and [scenarios.md](scenario/scenarios.md) document them (in Spanish).

---

## Datasets

The synthetic oil spills come from a physics-based particle simulator ([ground_truths.py](dataset/ground_truths.py)) with random spill origin, wind and current. Each ground-truth map is Gaussian-smoothed and min-max normalised to [0, 1].

[DatasetGeneration.py](dataset/DatasetGeneration.py) builds 2000 simulations per dataset on a 100 × 100 grid (seed 42). Each simulation is paired with a sparse observation path produced by an observation model from [ObservationModels.py](dataset/ObservationModels.py):

| Config | Sensor model |
|---|---|
| [dataset_config_POINTWISE.yaml](dataset/dataset_config_POINTWISE.yaml) | Pointwise contact sensor along random waypoint paths (used by the paper) |
| [dataset_config_NADIR.yaml](dataset/dataset_config_NADIR.yaml) | Nadir downward-looking camera |
| [dataset_config_CONIC.yaml](dataset/dataset_config_CONIC.yaml) | Conic field of view (60°, radius 10 px) |

Each `.npz` stores `ground_truth`, `observed_map` and `observed_mask`, each of shape `(2000, 100, 100)`, plus JSON metadata. The deep models are trained on the POINTWISE dataset with an 80/20 split (seed 42). The planning scenario reads its ground-truth maps from `dataset_path` in [scenario_config.yaml](scenario/scenario_config.yaml).

Datasets (`*.npz`) and weights (`*.pt`) are git-ignored. Generate them locally (see [Usage](#usage)).

---

## Results

The numbers below are taken from the paper (100 × 100 grid, 4 ASVs, D = 300 px, sensor noise 𝒩(0, 0.1²)).

### Uncertainty calibration — UCE (×10³, lower is better)

| Policy | GP | MC Dropout | Ensemble | EDL |
|---|:-:|:-:|:-:|:-:|
| Value Greedy | 12.6 (7.8) | 8.5 (2.1) | 14.3 (8.9) | **1.0 (1.7)** |
| Uncertainty Greedy | 9.4 (4.0) | 8.6 (2.9) | 14.5 (6.6) | **3.9 (7.5)** |
| ε-Greedy | 11.2 (3.5) | 11.8 (5.8) | 13.4 (8.5) | **1.9 (3.4)** |
| RH Orienteering | 9.1 (5.2) | 9.3 (2.9) | 30.0 (19.5) | **1.0 (1.2)** |
| MCTS | 19.4 (8.3) | 10.5 (3.1) | 71.6 (40.0) | **0.7 (0.2)** |

EDL is the best-calibrated model under every policy. MC Dropout's calibration barely depends on the planner. The ensemble degrades sharply under the look-ahead planners. Calibration curves: [calibration_by_model_ieee.pdf](Results/plots/calibration_by_model_ieee.pdf).

### Reconstruction (mean over planners)

| Model | RMSE | IoU (τ = 0.05) | Comment |
|---|:-:|:-:|---|
| GP | 0.011060 | 52.75 % | Homoscedastic noise smooths the spill boundary. Value-Greedy can stall on noise-induced local maxima. |
| MC Dropout | 0.006845 | 51.1 % | High-frequency prediction noise exceeds τ in clean water. |
| Deep Ensemble | 0.011820 | 73.2 % | Averaging the members suppresses activations outside the spill. |
| **EDL** | **0.001847** | **84.9 %** | Lowest error, and the least variation across planners. |

Box plots: [RMSE](Results/normalized_rmse_boxplot.pdf), [MSE](Results/mse_boxplot.pdf), [IoU](Results/iou_boxplot.pdf).

### Computation time per decision step

| Model | EDL | Ensemble | MC Dropout | GP |
|---|:-:|:-:|:-:|:-:|
| Time | 3.9 ms | 13.0 ms | 101.3 ms | 94.2 s |

| Policy | ε-Greedy | Value Greedy | Uncertainty Greedy | Orienteering | MCTS |
|---|:-:|:-:|:-:|:-:|:-:|
| Time | 1.1 ms | 1.8 ms | 1.7 ms | 1.29 s | 2.51 s |

All planners fit within the 20–40 s an ASV needs to reach a waypoint. The GP, refitted at every step, is about 24,000× slower than EDL.

### Takeaways

- How well a planner works depends on the calibration of the uncertainty map it is given. MCTS + EDL reaches the lowest RMSE of the study, while MCTS + Ensemble reaches the highest.
- RH Orienteering is the most robust to poor calibration: integrating reward along edges biases it towards coverage.
- With an accurate model (EDL), even the uncertainty-agnostic Value-Greedy planner stays competitive.

---

## Repository Structure

```
.
├── dataset/
│   ├── DatasetGeneration.py          # Builds the .npz datasets from a YAML config
│   ├── ground_truths.py              # Particle-based oil spill simulator
│   ├── ObservationModels.py          # Pointwise / Nadir / Conic sensor + path generator
│   ├── dataset_analysis.py           # Exploratory dataset figures
│   └── dataset_config_*.yaml         # Simulation + sensor parameters per dataset
├── scenario/
│   ├── scenario.py                   # ObservationScenario: multi-agent movement + noisy sensing
│   ├── extended_scenario.py          # Adds distance budget, model inference, RMSE/IoU
│   ├── scenario_config.yaml          # Dataset path, start positions, noise, budget
│   ├── run_experiments.py            # Benchmark: every (model × policy × map) episode
│   ├── multiagent_max_informative_path_waypoints.py  # Multi-agent IPP problem for MCTS
│   ├── max_informative_path*.py      # Single-agent IPP problems
│   ├── grid_world.py                 # Toy problem for the planning algorithms
│   ├── generate_mask_dataset.py      # Policy-driven observation-mask dataset generator
│   ├── test_policy.py                # Quick visual test of a single policy
│   ├── models/
│   │   ├── unet_model.py             # Shared U-Net backbone
│   │   ├── MC_dropout_model.py       # Monte Carlo Dropout
│   │   ├── MC_ensemble_model.py      # Deep Ensemble
│   │   ├── EDL_model.py              # Evidential Deep Learning (NIG)
│   │   ├── gaussian_process_model.py # GP baseline
│   │   ├── myopic_model.py           # IDW interpolation (not in the paper)
│   │   ├── train_models.py           # Offline training of the deep models
│   │   ├── evaluate_models.py        # Offline (static-dataset) evaluation
│   │   ├── metrics.py                # RMSE, R², NLL, ECE, UCE
│   │   ├── evaluation_store.py       # Results (de)serialisation
│   │   └── plot_results.py           # Offline-evaluation figures
│   ├── policies/
│   │   ├── base.py                   # Policy interface
│   │   ├── myopic_greedy.py          # Value-Greedy
│   │   ├── uncertainty_greedy.py     # Uncertainty-Greedy
│   │   ├── epsilon_greedy.py         # ε-Greedy
│   │   ├── orienteering_policy.py    # Receding-horizon Orienteering
│   │   ├── mamcts_policy.py          # Multi-agent MCTS (paper)
│   │   ├── mcts_policy.py            # Single-agent MCTS
│   │   └── algorithms/               # MCTS, MAMCTS, forward search, B&B, sparse sampling
│   └── results/plot_results.py       # Figures/report from CSV experiment logs
├── viewer.ipynb                      # Paper analysis: UCE, calibration curves, box plots, timing, figures
├── train_ensemble.py                 # Stand-alone ensemble trainer (random-mask augmentation)
├── experiments/                      # sylegendarium logs (*.meta.yaml + *.metrics.tar via Git LFS)
├── Results/                          # Paper figures (PDF)
├── Weights/                          # Trained weights: dataset_{POINTWISE,NADIR,CONIC}_{EDL,Ensemble,MC_Dropout}.pt
└── onlineplanning/                   # Notes on the classic online-planning algorithms (Kochenderfer)
```

---

## Installation

Python ≥ 3.10 with:

```bash
pip install torch numpy scipy scikit-learn matplotlib seaborn pandas pyyaml tqdm sylegendarium
```

A CUDA GPU is optional but recommended for training. The experiment logs in `experiments/` are stored with Git LFS (`git lfs pull`).

---

## Usage

Run every command from the repository root.

### 1. Generate the datasets

```bash
python dataset/DatasetGeneration.py --dataset dataset/dataset_config_POINTWISE.yaml
```

The output folder is set by `output_dir` in the YAML. Move the resulting `.npz` into `dataset/`, which is where the training and scenario scripts look for it. Run `python dataset/dataset_analysis.py` for exploratory plots.

### 2. Train the deep models

```bash
python scenario/models/train_models.py --datasets dataset/dataset_POINTWISE.npz
```

This trains MC Dropout, the Ensemble and EDL for 50 epochs with batch size 16 and Adam, and writes `Weights/{dataset}_{model}.pt`. The default learning rate is 3·10⁻⁴; EDL is capped at 5·10⁻⁴ and uses λ = 10⁻³ with gradient clipping at 1.0. Override with `--epochs`, `--batch-size` and `--lr`. The GP has no weights.

Optional offline evaluation on the static dataset (RMSE, NLL, ECE, UCE):

```bash
python scenario/models/evaluate_models.py
```

### 3. Run the closed-loop benchmark

```bash
python scenario/run_experiments.py --models edl,ensemble,mcdropout,gaussian_process --policies mcts --n-maps 10
```

| Flag | Meaning |
|---|---|
| `--models` | Any of `edl`, `ensemble`, `mcdropout`, `gaussian_process` |
| `--policies` | Keys enabled in `POLICY_CATALOGUE` in [run_experiments.py](scenario/run_experiments.py). Only `mcts` is active by default; uncomment the other entries to run `myopic_greedy`, `uncertainty_greedy`, `epsilon_greedy` and `orienteering`. |
| `--n-maps`, `--map-start` | Which ground-truth maps to evaluate |
| `--budget` | Overrides the config budget (px) |
| `--dataset` | Alternative `.npz` of ground-truth maps |
| `--weights` | Weights folder (default `Weights/`). The `dataset_POINTWISE_*.pt` files are loaded. |
| `--render` | Live six-panel visualisation |

Each run writes a timestamped log to `experiments/`.

### 4. Analyse the results

Open [viewer.ipynb](viewer.ipynb). It loads every log in `experiments/` with `sylegendarium.load_experiments`. From those logs it produces the UCE table, the calibration curves, the RMSE and IoU box plots, the timing tables, the trajectory plots and the framework figure, saving them under `Results/plots/`.

---
