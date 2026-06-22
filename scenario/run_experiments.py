"""
run_experiments.py — Benchmark IPP policies against estimation models.

Evaluates all combinations of (policy × model) over N ground-truth maps and
records per-step metrics via ExperimentLogger.

Usage
-----
    uv run python scenario/run_experiments.py [options]

Options
-------
    --n-maps    INT     Number of maps to evaluate per combination (default: 10)
    --map-start INT     First map index (default: 0)
    --budget    FLOAT   Distance budget per episode in px (default: from config)
    --weights   PATH    Path to ensemble weights (default: weights/ensemble.pt)
    --output    PATH    Output CSV path (default: results/experiments.csv)
    --policies  LIST    Subset of policies to run, comma-separated
                        (default: all)
    --models    LIST    Subset of models to run, comma-separated
                        (default: all)
    --no-render         Disable matplotlib rendering

Available policy names  : epsilon_greedy, myopic_greedy, uncertainty_greedy,
                          orienteering, mcts
Available model names   : gaussian_process, ensemble
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

from scenario.models.train_models import weights_path
matplotlib.use("Agg")          # headless-safe default; overridden below if rendering
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))

from extended_scenario import ExtendedScenario
from experiment_logger import ExperimentLogger
from policies import (
    EpsilonGreedy,
    MaxGreedyMiopic,
    MaxUncertaintyPolicy,
    OrienteeringPolicy,
    MCTSPolicy,
)
from models.gaussian_process_model import GaussianProcessModel
from models.MC_ensemble_model import EnsembleModel
from models.myopic_model import MyopicModel
from models.EDL_model import EDLModel
from models.MC_dropout_model import MCDropoutModel


# ── Policy catalogue ──────────────────────────────────────────────────────────

def _make_epsilon_greedy() -> EpsilonGreedy:
    return EpsilonGreedy(max_radius=30, eps_interval=(0.1, 0.01), eps_decay_steps=100)

def _make_myopic_greedy() -> MaxGreedyMiopic:
    return MaxGreedyMiopic(max_radius=30)

def _make_uncertainty_greedy() -> MaxUncertaintyPolicy:
    return MaxUncertaintyPolicy(max_radius=30)

def _make_orienteering() -> OrienteeringPolicy:
    return OrienteeringPolicy(h_plan=30, h_act=5, grid_step=2, n_starts=1)

def _make_mcts(budget: float) -> MCTSPolicy:
    return MCTSPolicy(
        budget=budget,
        depth=budget,
        min_resolution=1,
        max_resolution=4,
        n_simulations=2000,
        anneal_radius=1,
        control_horizon=0,
        reuse_tree=True,
        gamma=0.8,
        exploration_c=1.414,
        uncertainty_change_threshold=1e-4,
    )


POLICY_CATALOGUE: dict[str, callable] = {
    "epsilon_greedy":     _make_epsilon_greedy,
    "myopic_greedy":      _make_myopic_greedy,
    "uncertainty_greedy": _make_uncertainty_greedy,
    "orienteering":       _make_orienteering,
    "mcts":                _make_mcts
}


# ── Model catalogue ───────────────────────────────────────────────────────────

def _make_gaussian_process() -> GaussianProcessModel:
    return GaussianProcessModel(n_restarts_optimizer=2, normalize_y=True,
                                constant_value_bounds=(1e-3, 1e3), noise_level_bounds=(1e-5, 1e-1), length_scale_bounds=(1e-3, 1e4))

def _make_ensemble(weights_path: Path) -> EnsembleModel:
    model = EnsembleModel(n_members=5)
    model.load_weights(weights_path)
    return model

def _make_edl(weights_path: Path) -> EDLModel:
    model = EDLModel(in_channels=2, base_channels=16, depth=3)
    model.load_weights(weights_path)
    return model

def _make_mcdropout(weights_path: Path) -> MCDropoutModel:
    model = MCDropoutModel(in_channels=2, base_channels=16, depth=3, dropout_p=0.2, n_samples=30)
    model.load_weights(weights_path)
    return model

def _make_myopic() -> MyopicModel:
    return MyopicModel(power=2, k_neighbors=10)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark IPP policies across models and maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",   type=Path,  default=None,
                   help="Path to a .npz file with evaluation maps. "
                        "Overrides the dataset_path in scenario_config.yaml.")
    p.add_argument("--n-maps",    type=int,   default=10,
                   help="Number of maps to evaluate per (policy, model) pair")
    p.add_argument("--map-start", type=int,   default=0,
                   help="First map index")
    p.add_argument("--budget",    type=float, default=None,
                   help="Distance budget per episode (px). None = use config value")
    p.add_argument("--weights",   type=Path,
                   default=ROOT / "models" / "weights" / "ensemble.pt",
                   help="Path to ensemble model weights")
    p.add_argument("--output",    type=Path,
                   default=ROOT / "results" / "experiments.csv",
                   help="Output CSV path")
    p.add_argument("--policies",  type=str, default=None,
                   help="Comma-separated subset of policies to run")
    p.add_argument("--models",    type=str, default=None,
                   help="Comma-separated subset of models to run")
    p.add_argument("--render", action="store_true",
                   help="Disable matplotlib rendering")
    p.add_argument("--nagents", type=int, default=4,
                   help="Number of agents in the environment (default: 4)")
    return p.parse_args()


# ── Dataset loader ────────────────────────────────────────────────────────────

def _load_maps(npz_path: Path) -> np.ndarray:
    """Load the maps array from an NPZ file (any key, shape (N, H, W))."""
    data = np.load(npz_path)
    key = "maps" if "maps" in data else list(data.keys())[0]
    maps = data[key].astype(np.float32)
    if maps.ndim != 3:
        raise ValueError(f"Expected shape (N, H, W) in {npz_path}, got {maps.shape}")
    return maps


def _inject_dataset(env: ExtendedScenario, maps: np.ndarray) -> None:
    """Replace the environment's map array in-place."""
    env.maps = maps
    env.H, env.W = maps.shape[1], maps.shape[2]


# ── Bootstrap obs dict (before first step) ────────────────────────────────────

def _bootstrap_obs(env: ExtendedScenario) -> dict:
    return {
        "obs_map":               env.obs_map,
        "obs_mask":              env.obs_mask,
        "predicted_mean":        np.zeros((env.H, env.W), dtype=np.float32),
        "predicted_uncertainty": np.ones( (env.H, env.W), dtype=np.float32),
    }


# ── Single episode ────────────────────────────────────────────────────────────

def run_episode(
    env: ExtendedScenario,
    policy,
    logger: ExperimentLogger,
    map_idx: int,
    policy_name: str,
    model_name: str,
    dataset_name: str,
    render: bool,
) -> dict:
    """
    Run one episode and return a summary dict.
    """
    policy.reset()
    logger.new_episode()

    gt = env.reset(map_idx=map_idx)
    action = policy.act(_bootstrap_obs(env), env.position)

    done = False
    step = 0

    while not done:
        obs, done, info = env.step(action)
        step += 1

        logger.log_step(
            step=step,
            info=info,
            obs=obs,
            ground_truth=env.ground_truth,
            position=env.position,
            policy_name=policy_name,
            map_idx=map_idx,
            dataset_name=dataset_name,
            model_name=model_name,
        )

        if render:
            env.render()
            plt.pause(0.02)

        if not done:
            action = policy.act(obs, env.position)

    return {
        "steps":    step,
        "distance": info["distance"],
        "rmse":     info["mse"] ** 0.5,
        "coverage": float(env.obs_mask.mean()) * 100,
        "iou":      info["iou"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.render:
        matplotlib.use("TkAgg")   # interactive backend when rendering

    # ── Resolve which policies and models to run ───────────────────────────
    all_policy_names = list(POLICY_CATALOGUE)
    policy_names = (
        [p.strip() for p in args.policies.split(",")]
        if args.policies else all_policy_names
    )
    model_names_req = (
        [m.strip() for m in args.models.split(",")]
        if args.models else ["gaussian_process", "ensemble"]
    )

    unknown_p = set(policy_names) - set(all_policy_names)
    if unknown_p:
        print(f"[ERROR] Unknown policies: {unknown_p}")
        print(f"        Available: {all_policy_names}")
        sys.exit(1)

    unknown_m = set(model_names_req) - {"gaussian_process", "ensemble", "myopic"}
    if unknown_m:
        print(f"[ERROR] Unknown models: {unknown_m}")
        sys.exit(1)

    # ── Build model instances ──────────────────────────────────────────────
    model_instances: dict[str, object] = {}

    if "gaussian_process" in model_names_req:
        print("Loading GaussianProcessModel …")
        model_instances["gaussian_process"] = _make_gaussian_process()

    if "ensemble" in model_names_req:
        if not args.weights.exists():
            print(f"[ERROR] Ensemble weights not found: {args.weights}")
            sys.exit(1)
        print(f"Loading EnsembleModel from {args.weights} …")
        model_instances["ensemble"] = _make_ensemble(args.weights)

    if "myopic" in model_names_req:
        print("Loading MyopicModel …")
        model_instances["myopic"] = _make_myopic()

    if "edl" in model_names_req:
        print("Loading EDLModel …")
        model_instances["edl"] = _make_edl(args.weights)

    if "mcdropout" in model_names_req:
        print("Loading MCDropoutModel …")
        model_instances["mcdropout"] = _make_mcdropout(args.weights)

    # ── Setup ──────────────────────────────────────────────────────────────
    cfg = ROOT / "scenario_config.yaml"

    # Optional external dataset
    ext_maps: np.ndarray | None = None
    if args.dataset is not None:
        if not args.dataset.exists():
            print(f"[ERROR] Dataset not found: {args.dataset}")
            sys.exit(1)
        ext_maps = _load_maps(args.dataset)
        dataset_name = args.dataset.stem
        n_available = ext_maps.shape[0]
        if args.map_start + args.n_maps > n_available:
            print(
                f"[ERROR] Requested maps {args.map_start}…"
                f"{args.map_start + args.n_maps - 1} but dataset only has "
                f"{n_available} maps."
            )
            sys.exit(1)
        print(f"Dataset  : {args.dataset}  ({n_available} maps, shape {ext_maps.shape[1:]})")
    else:
        dataset_name = Path(cfg).stem

    map_indices = list(range(args.map_start, args.map_start + args.n_maps))
    logger = ExperimentLogger()

    print(f"\nPolicies : {policy_names}")
    print(f"Models   : {list(model_instances)}")
    print(f"Maps     : {map_indices[0]} … {map_indices[-1]}  ({args.n_maps} total)")
    print(f"Output   : {args.output}\n")

    total = len(model_instances) * len(policy_names) * args.n_maps
    done_count = 0

    # ── Experiment loop ────────────────────────────────────────────────────
    for model_name, model in model_instances.items():
        env = ExtendedScenario(cfg, model=model, budget=args.budget)
        if ext_maps is not None:
            _inject_dataset(env, ext_maps)
        active_catalogue = {**POLICY_CATALOGUE, "mcts": lambda: _make_mcts(env.budget)}
        for policy_name in policy_names:
            policy = active_catalogue[policy_name]()

            print(f"┌─ model={model_name}  policy={policy_name}")

            for map_idx in map_indices:
                summary = run_episode(
                    env=env,
                    policy=policy,
                    logger=logger,
                    map_idx=map_idx,
                    policy_name=policy_name,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    render=args.render,
                )
                done_count += 1
                print(
                    f"│  map {map_idx:3d}  "
                    f"steps={summary['steps']:3d}  "
                    f"dist={summary['distance']:6.1f}  "
                    f"rmse={summary['rmse']:.4f}  "
                    f"cov={summary['coverage']:5.1f}%  "
                    f"iou={summary['iou']:.4f}  "
                    f"[{done_count}/{total}]"
                )

            print(f"└─ done\n")

    # ── Save results ───────────────────────────────────────────────────────
    df = logger.to_dataframe()
    saved = logger.save(args.output)

    print("─" * 60)
    print(f"Total rows recorded : {len(df)}")
    print(f"Results saved to    : {saved}")
    print("─" * 60)

    # Quick per-(model, policy) summary
    summary = (
        df.groupby(["model_name", "policy"])
        .agg(
            episodes=("experiment_id", "nunique"),
            mean_rmse=("rmse", "mean"),
            final_rmse=("rmse", lambda s: s.groupby(
                df.loc[s.index, "experiment_id"]).last().mean()),
            mean_coverage=("coverage", lambda s: s.groupby(
                df.loc[s.index, "experiment_id"]).last().mean()),
            mean_iou=("iou", "mean"),
        )
        .reset_index()
    )
    print("\nPer-(model, policy) summary (mean over episodes):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
