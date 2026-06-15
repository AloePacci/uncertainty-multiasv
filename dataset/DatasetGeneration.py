import json
import os
import yaml
import numpy as np
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import argparse

from ground_truths import OilSpillGroundTruth, DEFAULT_OIL_SPILL_PARAMS
from ObservationModels import (
    ObservationConfig,
    PathGeneratorConfig,
    RandomSplineGenerator,
    PointwiseObservationModel,
    NadirObservationModel,
    ConicFOVObservationModel,
)

_OBSERVATION_MODELS = {
    "pointwise": PointwiseObservationModel,
    "nadir":     NadirObservationModel,
    "conic":     ConicFOVObservationModel,
}


def _build_sim_params(user_sim_params: dict | None, n_steps: int, sim_seed: int | None) -> dict:
    params = dict(DEFAULT_OIL_SPILL_PARAMS)
    # MAX_FUEL is not in DEFAULT_OIL_SPILL_PARAMS but is required by the simulator
    params["MAX_FUEL"] = params["N_PARTICLES"] * 10
    if user_sim_params:
        params.update(user_sim_params)
    # Recalculate MAX_FUEL if user changed N_PARTICLES but not MAX_FUEL
    if user_sim_params and "N_PARTICLES" in user_sim_params and "MAX_FUEL" not in user_sim_params:
        params["MAX_FUEL"] = params["N_PARTICLES"] * 10
    params["N_STEPS"] = n_steps
    params["SEED"] = sim_seed
    return params


def _build_observation_model(
    model_type: str,
    water_mask: np.ndarray,
    observation_length: tuple,
    obs_overrides: dict | None,
    pg_overrides: dict | None,
    sim_seed: int | None,
):
    pg_config = PathGeneratorConfig()
    pg_config.base_map = water_mask
    pg_config.number_of_waypoints_interval = observation_length

    if pg_overrides:
        for k, v in pg_overrides.items():
            setattr(pg_config, k, v)

    path_generator = RandomSplineGenerator(pg_config)

    obs_config = ObservationConfig()
    obs_config.base_map = water_mask
    obs_config.path_generator = path_generator
    obs_config.seed = sim_seed if sim_seed is not None else 0

    if obs_overrides:
        for k, v in obs_overrides.items():
            setattr(obs_config, k, v)

    return _OBSERVATION_MODELS[model_type](obs_config)


def _run_single_simulation(
    water_mask: np.ndarray,
    source_mask: np.ndarray,
    sim_params_dict: dict,
    model_type: str,
    observation_length: tuple,
    obs_overrides: dict | None,
    pg_overrides: dict | None,
    sim_seed: int | None,
    timestamps_per_groundtruth: int = 1,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Runs one simulation and returns a list of (gt, obs_map, obs_mask) tuples.

    If timestamps_per_groundtruth == 1, returns the last accumulated observation
    (original behaviour). If > 1, collects all intermediate observation frames
    and samples timestamps_per_groundtruth of them at random. Each sampled frame
    represents a different coverage level for the same ground truth.
    """
    gt_gen = OilSpillGroundTruth(water_mask, sim_params_dict, source_mask)
    ground_truth = deque(gt_gen, maxlen=1).pop()

    obs_model = _build_observation_model(
        model_type, water_mask, observation_length, obs_overrides, pg_overrides, sim_seed
    )
    obs_gen = obs_model.observation(ground_truth)

    if timestamps_per_groundtruth == 1:
        # Fast path: only materialise the last frame (original behaviour)
        observed_map, observed_mask = deque(obs_gen, maxlen=1).pop()
        return [(ground_truth, observed_map, observed_mask)]

    # Collect every intermediate frame.
    # Frame 0 yields an empty subpath (no observations), so we skip it.
    all_frames = list(obs_gen)                           # list of (obs_map, obs_mask)
    valid_frames = all_frames[1:] if len(all_frames) > 1 else all_frames

    n_available = len(valid_frames)
    rng = np.random.default_rng(sim_seed)
    replace = timestamps_per_groundtruth > n_available
    indices = sorted(
        rng.choice(n_available, size=timestamps_per_groundtruth, replace=replace).tolist()
    )
    return [(ground_truth, valid_frames[i][0], valid_frames[i][1]) for i in indices]


def generate_dataset(
    water_mask: np.ndarray,
    source_mask: np.ndarray,
    n_simulations: int,
    sim_params: dict | None = None,
    steps_range: tuple[int, int] = (100, 300),
    observation_model_type: str = "nadir",
    observation_length: tuple = (5, 15),
    timestamps_per_groundtruth: int = 1,
    observation_config_overrides: dict | None = None,
    path_generator_config_overrides: dict | None = None,
    output_dir: str | Path = "Datasets",
    dataset_name: str | None = None,
    seed: int | None = None,
    n_workers: int = 1,
) -> Path:
    if observation_model_type not in _OBSERVATION_MODELS:
        raise ValueError(
            f"observation_model_type must be one of {list(_OBSERVATION_MODELS)}, "
            f"got '{observation_model_type}'"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name is None:
        dataset_name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if timestamps_per_groundtruth < 1:
        raise ValueError(f"timestamps_per_groundtruth must be >= 1, got {timestamps_per_groundtruth}")

    H, W = water_mask.shape
    N_total = n_simulations * timestamps_per_groundtruth
    ground_truths = np.zeros((N_total, H, W), dtype=np.float32)
    observed_maps = np.zeros((N_total, H, W), dtype=np.float32)
    observed_masks = np.zeros((N_total, H, W), dtype=np.float32)

    # --- Resolve effective worker count ---
    # n_workers=-1 → all logical CPUs; n_workers≥1 → exact count.
    if n_workers == -1:
        effective_workers = os.cpu_count() or 1
    elif n_workers >= 1:
        effective_workers = n_workers
    else:
        raise ValueError(f"n_workers must be -1 (all CPUs) or a positive integer, got {n_workers}")
    effective_workers = min(effective_workers, n_simulations)

    # --- Pre-build per-simulation arguments (keeps RNG logic in the main process) ---
    # ── Dataset-level shared parameters ──────────────────────────────────
    # Wind angle and tide speed are drawn ONCE per dataset so that all
    # simulations share the same environmental conditions.  Only the spill
    # origin (controlled by sim_seed inside OilSpillGroundTruth) varies.
    dataset_rng  = np.random.default_rng(seed)          # reproducible per dataset
    shared_wind_angle = float(dataset_rng.uniform(0.0, 2.0 * np.pi))
    print(f"Dataset wind angle (rad): {shared_wind_angle:.4f}  "
          f"({np.degrees(shared_wind_angle):.1f} deg)")

    sim_args_list = []
    for i in range(n_simulations):
        sim_seed = (seed + i) if seed is not None else None
        rng = np.random.default_rng(sim_seed)
        n_steps = int(rng.integers(steps_range[0], steps_range[1] + 1))
        params_dict = _build_sim_params(sim_params, n_steps, sim_seed)
        # Inject shared wind angle — overrides any per-sim random draw
        params_dict["WIND_ANGLE"] = shared_wind_angle
        sim_args_list.append((
            water_mask, source_mask, params_dict,
            observation_model_type, observation_length,
            observation_config_overrides, path_generator_config_overrides,
            sim_seed, timestamps_per_groundtruth,
        ))

    # --- Run simulations (sequential or parallel) ---
    def _store_results(i: int, samples: list):
        """Write the timestamps_per_groundtruth samples for simulation i."""
        base = i * timestamps_per_groundtruth
        for k, (gt, obs_map, obs_mask) in enumerate(samples):
            ground_truths[base + k] = gt
            observed_maps[base + k]  = obs_map
            observed_masks[base + k] = obs_mask

    if effective_workers == 1:
        for i, sim_args in enumerate(sim_args_list):
            print(f"  [{i + 1}/{n_simulations}] Running simulation {i + 1}...")
            samples = _run_single_simulation(*sim_args)
            _store_results(i, samples)
    else:
        print(f"Running {n_simulations} simulations with {effective_workers} parallel workers...")
        # NOTE: on Windows, ProcessPoolExecutor uses 'spawn' — callers must be
        # inside an `if __name__ == "__main__":` block to avoid recursive imports.
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_idx = {
                executor.submit(_run_single_simulation, *args): i
                for i, args in enumerate(sim_args_list)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                samples = future.result()
                _store_results(i, samples)
                completed += 1
                print(f"  [{completed}/{n_simulations}] Simulation {i + 1} done.")

    metadata = {
        "dataset_name":              dataset_name,
        "n_simulations":             n_simulations,
        "n_samples":                 N_total,
        "timestamps_per_groundtruth": timestamps_per_groundtruth,
        "sim_params":                sim_params or {},
        "steps_range":               list(steps_range),
        "observation_model_type":    observation_model_type,
        "observation_length":        observation_length,
        "seed":                      seed,
        "shared_wind_angle_rad":     round(shared_wind_angle, 6),
        "shared_wind_angle_deg":     round(float(np.degrees(shared_wind_angle)), 2),
        "timestamp":                 datetime.now().isoformat(),
    }
    metadata_bytes = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)

    filepath = output_dir / f"{dataset_name}.npz"
    np.savez_compressed(
        filepath,
        ground_truth=ground_truths,
        observed_map=observed_maps,
        observed_mask=observed_masks,
        metadata=metadata_bytes,
    )

    print(f"Dataset saved to: {filepath}")
    return filepath


def generate_dataset_from_yaml(
    yaml_path: str | Path,
    water_mask: np.ndarray,
    source_mask: np.ndarray,
) -> Path:
    """Load generation parameters from a YAML config file and run generate_dataset.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the YAML configuration file (e.g. dataset_config_CONIC.yaml).
    water_mask : np.ndarray
        (H, W) binary array: 1 = water, 0 = land.
    source_mask : np.ndarray
        (H, W) binary array: 1 = valid spill origin cells, 0 = excluded.

    Returns
    -------
    Path
        Path to the saved .npz dataset file.
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    steps_range = cfg.get("steps_range", [100, 300])

    return generate_dataset(
        water_mask=water_mask,
        source_mask=source_mask,
        n_simulations=cfg["n_simulations"],
        sim_params=cfg.get("sim_params") or None,
        steps_range=(steps_range[0], steps_range[1]),
        observation_model_type=cfg.get("observation_model_type", "nadir"),
        observation_length=cfg.get("observation_length", (5, 15)),
        timestamps_per_groundtruth=cfg.get("timestamps_per_groundtruth", 1),
        observation_config_overrides=cfg.get("observation_config") or None,
        path_generator_config_overrides=cfg.get("path_generator_config") or None,
        output_dir=cfg.get("output_dir", "Datasets"),
        dataset_name=cfg.get("dataset_name") or None,
        seed=cfg.get("seed"),
        n_workers=cfg.get("n_workers", 1),
    )



if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Generate a dataset of oil spill simulations.")
    argparser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to generate.",
        default="Datasets/dataset_config_NADIR.yaml",
    )
    args = argparser.parse_args()

    H, W = 100, 100
    water_mask = np.ones((H, W), dtype=np.uint8)
    water_mask[:8, :] = water_mask[-8:, :] = 0
    water_mask[:, :8] = water_mask[:, -8:] = 0

    source_mask = np.zeros((H, W), dtype=np.uint8)
    source_mask[np.arange(10, 90), np.arange(10, 90)] = 1

    path = generate_dataset_from_yaml(
        yaml_path=args.dataset,
        water_mask=water_mask,
        source_mask=source_mask,
    )

    # Verify the saved dataset
    data = np.load(path, allow_pickle=False)
    print(f"ground_truth shape: {data['ground_truth'].shape}")
    print(f"observed_map shape: {data['observed_map'].shape}")
    print(f"observed_mask shape: {data['observed_mask'].shape}")
    print(f"Metadata: {json.loads(bytes(data['metadata']).decode('utf-8'))}")