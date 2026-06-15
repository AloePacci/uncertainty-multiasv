"""
test_policy.py — Run one episode of ExtendedScenario driven by MaxUncertaintyPolicy.

Usage
-----
    uv run python scenario/test_policy.py [--map-idx N] [--radius R] [--budget B] [--no-render]

Options
-------
    --map-idx   Index of the ground-truth map to use (default: 0)
    --radius    Max-radius for MaxUncertaintyPolicy in pixels (default: 30)
    --budget    Distance budget in pixels (overrides scenario_config.yaml)
    --no-render Disable matplotlib rendering (useful in headless environments)
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from extended_scenario import ExtendedScenario
from policies import EpsilonGreedy, MaxGreedyMiopic, MaxUncertaintyPolicy, OrienteeringPolicy, MCTSPolicy
from models.MC_ensemble_model import EnsembleModel

ROOT = Path(__file__).parent

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test MaxUncertaintyPolicy on ExtendedScenario.")
    p.add_argument("--map-idx",   type=int,   default=10,    help="Ground-truth map index")
    p.add_argument("--radius",    type=float, default=30, help="MaxUncertaintyPolicy radius (px)")
    p.add_argument("--budget",    type=float, default=300, help="Distance budget in pixels")
    p.add_argument("--no-render", action="store_true",      help="Disable rendering")
    p.add_argument("--weights",   type=Path,  default=ROOT / "models" / "weights" / "ensemble.pt", help="Path to model weights")
    return p.parse_args()


def print_step(step: int, action: tuple, info: dict, done: bool) -> None:
    print(
        f"  step {step:3d}  →  {action}  |"
        f"  dist {info['distance']:6.1f}/{info['budget']:.0f} px  |"
        f"  MSE(unobs) {info['mse']:.4f}  |"
        f"  IoU {info['iou']:.3f}  |"
        f"  done={done}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    
    
    args = parse_args()

    cfg    = ROOT / "scenario_config.yaml"
    # model  = ModelAdapter(GaussianProcessModel(n_restarts_optimizer=5))
    base_model = EnsembleModel(n_members=5)
    base_model.load_weights(args.weights)
    env    = ExtendedScenario(cfg, model=base_model, budget=args.budget)
    policy = MCTSPolicy(
        budget=args.budget,
        depth=args.budget,
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
    
    print(f"Config   : {cfg}")
    print(f"Budget   : {env.budget:.0f} px")
    print(f"Radius   : {args.radius:.0f} px")
    print(f"Map idx  : {args.map_idx}")
    print()

    # ── Episode ───────────────────────────────────────────────────────────────
    gt = env.reset(map_idx=args.map_idx)
    print(f"Ground truth  shape={gt.shape}  max={gt.max():.3f}")

    # First render requires a dummy obs (use zeros before any step)
    # Instead we step once to bootstrap, then enter the loop
    done = False
    step = 0
    mse_history: list[float] = []

    # Bootstrap: first action is a step from initial position
    action = policy.act(
        {
            "obs_map":               env.obs_map,
            "obs_mask":              env.obs_mask,
            "predicted_mean":        np.zeros((env.H, env.W), dtype=np.float32),
            "predicted_uncertainty": np.ones( (env.H, env.W), dtype=np.float32),
        },
        env.position,
    )

    while not done:
        obs, done, info = env.step(action)
        step += 1
        mse_history.append(info["mse"])
        print_step(step, action, info, done)
        
        if not args.no_render:
            env.render()
        
                
            plt.pause(0.05)

        if not done:
            action = policy.act(obs, env.position)

    # ── Summary ───────────────────────────────────────────────────────────────
    coverage = float(env.obs_mask.mean()) * 100
    print()
    print("─" * 60)
    print(f"Episode finished after {step} steps.")
    print(f"Total distance : {info['distance']:.1f} / {info['budget']:.0f} px")
    print(f"Coverage       : {coverage:.1f} % of cells observed")
    print(f"Final MSE      : {info['mse']:.4f}")
    print(f"Initial MSE    : {mse_history[0]:.4f}")
    print(f"MSE reduction  : {(mse_history[0] - info['mse']) / max(mse_history[0], 1e-9) * 100:.1f} %")
    print(f"IoU            : {info['iou']:.3f}")
    print("─" * 60)

    if not args.no_render:
        print("\nClose the figure window to exit.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
