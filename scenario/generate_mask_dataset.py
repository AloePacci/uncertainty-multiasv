"""
generate_mask_dataset.py — Generate observation mask datasets using multiple policies.

A MaskDatasetGenerator runs episodes of ObservationScenario, sampling a policy
from a weighted list for each episode. The resulting (obs_map, obs_mask) pairs
are saved to a .npz file for downstream model training.

Usage
-----
    from scenario.generate_mask_dataset import MaskDatasetGenerator
    from scenario.policies import MaxUncertaintyPolicy
    from scenario.policies.greedy_uncertainty import MaxGreedyMiopic

    policies = [MaxUncertaintyPolicy(max_radius=30), MaxGreedyMiopic(max_radius=30)]
    probs    = [0.6, 0.4]

    gen = MaskDatasetGenerator(
        config_path="scenario/scenario_config.yaml",
        policies=policies,
        probabilities=probs,
        budget=500,
    )
    gen.generate(n_episodes=1000, output_path="dataset/masks.npz")
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from tqdm import trange

from scenario.scenario import ObservationScenario
from scenario.policies.base import Policy

ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MaskDatasetGenerator:
    """
    Generate a mask dataset by running episodes with a mixture of policies.

    Each episode uses a single policy drawn randomly according to *probabilities*.
    The agent runs until the distance budget is exhausted; the final obs_map and
    obs_mask are recorded as one dataset sample.

    Parameters
    ----------
    config_path : str | Path
        Path to scenario_config.yaml.
    policies : sequence of Policy
        Policy instances to sample from.
    probabilities : sequence of float or None
        Sampling probability for each policy (must sum to 1).
        Defaults to uniform if None.
    model : object or None
        Optional model with a ``predict(obs_th) -> dict`` method
        (same interface as ExtendedScenario). Required for policies that rely
        on ``predicted_uncertainty`` or ``predicted_mean`` in the obs dict.
        If None, those keys are filled with zeros/ones as placeholders.
    budget : float or None
        Maximum total Euclidean distance per episode (pixels).
        Overrides the value in the config file when provided.
    """

    def __init__(
        self,
        config_path: str | Path,
        policies: Sequence[Policy],
        probabilities: Sequence[float] | None = None,
        model=None,
        budget: float | None = None,
    ) -> None:
        if len(policies) == 0:
            raise ValueError("At least one policy must be provided.")

        if probabilities is None:
            probabilities = [1.0 / len(policies)] * len(policies)

        if len(probabilities) != len(policies):
            raise ValueError("len(probabilities) must equal len(policies).")

        probs = np.asarray(probabilities, dtype=float)
        if not np.isclose(probs.sum(), 1.0):
            raise ValueError(f"probabilities must sum to 1, got {probs.sum():.4f}.")

        self.policies      = list(policies)
        self.probabilities = probs
        self.model         = model

        config_path = Path(config_path)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.budget: float = float(budget if budget is not None else cfg.get("budget", 500))

        self._env = ObservationScenario(config_path)

    # ------------------------------------------------------------------
    def generate(
        self,
        n_episodes: int,
        output_path: str | Path,
        seed: int | None = None,
        visualize: bool = False
    ) -> None:
        """
        Run *n_episodes* episodes and save the dataset to *output_path*.

        The saved .npz contains:
            obs_maps    : (N, H, W) float32 — observation maps
            obs_masks   : (N, H, W) float32 — observation masks
            ground_truths : (N, H, W) float32 — ground truth maps
            policy_ids  : (N,) int           — index of the policy used

        Parameters
        ----------
        n_episodes : int
            Number of episodes to generate.
        output_path : str | Path
            Destination .npz file path.
        seed : int or None
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        env = self._env
        H, W = env.H, env.W

        obs_maps      = np.zeros((n_episodes, H, W), dtype=np.float32)
        obs_masks     = np.zeros((n_episodes, H, W), dtype=np.float32)
        ground_truths = np.zeros((n_episodes, H, W), dtype=np.float32)
        policy_ids    = np.zeros(n_episodes, dtype=np.int32)

        n_maps = len(env.maps)

        for ep in trange(n_episodes):
            # ── Sample policy ─────────────────────────────────────────────
            policy_idx = int(rng.choice(len(self.policies), p=self.probabilities))
            policy     = self.policies[policy_idx]

            # ── Reset environment ─────────────────────────────────────────
            map_idx   = int(rng.integers(n_maps))
            gt        = env.reset(map_idx=map_idx)
            distance  = 0.0

            # Bootstrap obs dict (no model output yet)
            obs = self._build_obs(env.obs_map, env.obs_mask, H, W)

            # ── Run episode until budget exhausted ────────────────────────
            while distance < self.budget:
                action     = policy.act(obs, env.position)
                r0, c0     = env.position
                obs_map, obs_mask = env.step(action)
                r1, c1     = env.position
                distance  += float(np.hypot(r1 - r0, c1 - c0))

                obs = self._build_obs(obs_map, obs_mask, H, W)

            # ── Store final state ─────────────────────────────────────────
            obs_maps[ep]      = env.obs_map
            obs_masks[ep]     = env.obs_mask
            ground_truths[ep] = gt
            policy_ids[ep]    = policy_idx

            if (ep + 1) % max(1, n_episodes // 10) == 0:
                print(f"  [{ep + 1}/{n_episodes}] episodes generated "
                      f"(last policy: {policy_idx}, "
                      f"coverage: {env.obs_mask.mean() * 100:.1f}%)")
                
                
            import matplotlib.pyplot as plt
            
            # Optional: visualize the episode's obs_map and obs_mask
            
            if visualize:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title(f"Episode {ep + 1} - Policy {policy_idx}")
                plt.imshow(obs_maps[ep], cmap="viridis")
                plt.colorbar(label="Observed Value")
                plt.subplot(1, 2, 2)
                plt.title(f"Episode {ep + 1} - Observation Mask")
                plt.imshow(obs_masks[ep], cmap="gray")
                plt.colorbar(label="Observed (1) vs Unobserved (0)")
                plt.tight_layout()
                plt.show()
            

        # ── Save ──────────────────────────────────────────────────────────
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            obs_maps=obs_maps,
            obs_masks=obs_masks,
            ground_truths=ground_truths,
            policy_ids=policy_ids,
        )
        print(f"\nDataset saved to {output_path}  ({n_episodes} samples, {H}x{W})")

    # ------------------------------------------------------------------
    def _build_obs(
        self,
        obs_map: np.ndarray,
        obs_mask: np.ndarray,
        H: int,
        W: int,
    ) -> dict:
        """
        Build the obs dict expected by Policy.act().

        If a model is available it is called to fill predicted_mean and
        predicted_uncertainty.  Otherwise these are filled with placeholders (obs_map for mean, binary mask for uncertainty).
        """
        if self.model is not None:
            output      = self.model.predict({"obs_map": obs_map, "obs_mask": obs_mask})
            mean        = output["predicted_mean"].reshape(H, W).astype(np.float32)
            uncertainty = output["predicted_std_epistemic"].reshape(H, W).astype(np.float32)
        else:
            # Si no hay modelo, se toma el mean como el obs_map (valor observado) y la incertidumbre como 1 para celdas no observadas, 0 para observadas
            mean        = obs_map.astype(np.float32)
            uncertainty = np.where(obs_mask == 0, 1.0, 0.0).astype(np.float32)

        return {
            "obs_map":               obs_map,
            "obs_mask":              obs_mask,
            "predicted_mean":        mean,
            "predicted_uncertainty": uncertainty,
        }


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate a mask dataset.")
    p.add_argument("--n-episodes", type=int,   default=2000,
                   help="Number of episodes to generate")
    p.add_argument("--output",     type=str,   default="dataset/augmented_dataset.npz",
                   help="Output .npz path")
    p.add_argument("--budget",     type=float, default=500,
                   help="Distance budget per episode (pixels)")
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    from scenario.policies.epsilon_greedy import EpsilonGreedy
    from scenario.policies.myopic_greedy import MaxGreedyMiopic
    from scenario.models.myopic_model import MyopicModel

    cfg = ROOT / "scenario_config.yaml"

    policies = [
        MaxGreedyMiopic(max_radius=30),
        EpsilonGreedy(max_radius=30, eps_interval=(1.0, 0.1), eps_decay_steps=10),
    ]
    probs = [0.5, 0.5]

    gen = MaskDatasetGenerator(
        config_path=cfg,
        policies=policies,
        probabilities=probs,
        budget=args.budget,
        model=MyopicModel(power=1, k_neighbors=10),  # less aggressive than classic IDW (power=2)
    )
    
    gen.generate(n_episodes=args.n_episodes, output_path=args.output, seed=args.seed, visualize=False)
