"""
scenario.py — Observation scenario for informative path planning.

An agent starts at a fixed position and moves in straight lines across
a ground-truth concentration map. At every visited cell it records a
(noisy) sensor reading, building an observation map and mask over time.

Usage
-----
    from scenario.scenario import ObservationScenario

    env = ObservationScenario("scenario/scenario_config.yaml")
    gt  = env.reset()                # (H, W) float32 ground-truth map
    obs_map, obs_mask = env.step((30, 70))   # move to row=30, col=70
    env.render()
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _line_pixels(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """
    Return all integer grid cells on the straight line from (r0,c0) to (r1,c1)
    using the parametric form with enough samples to hit every cell.
    The start cell is excluded (already visited); the end cell is included.
    """
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    n_steps = max(dr, dc, 1)

    rows = np.round(np.linspace(r0, r1, n_steps + 1)).astype(int)
    cols = np.round(np.linspace(c0, c1, n_steps + 1)).astype(int)

    # Deduplicate while preserving order, skip first point (already observed)
    seen: set[list[int, int]] = set()
    pixels: list[list[int, int]] = []
    for r, c in zip(rows[1:], cols[1:]):
        if (r, c) not in seen:
            seen.add((r, c))
            pixels.append([r, c])
    return pixels


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ObservationScenario:
    """
    Simulates an agent observing a contamination map along a path.

    Parameters
    ----------
    config_path : str | Path
        Path to scenario_config.yaml.

    Attributes (available after reset)
    ------------------------------------
    ground_truth : np.ndarray (H, W)  — current episode's ground-truth map
    obs_map      : np.ndarray (H, W)  — sensor readings (0 at unvisited cells)
    obs_mask     : np.ndarray (H, W)  — 1 at visited cells, 0 elsewhere
    position     : tuple[int, int]    — current agent position (row, col)
    trajectory   : list[tuple]        — ordered list of visited (row, col) cells
    """

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Resolve dataset path relative to config file location
        dataset_path = Path(cfg["dataset_path"])
        if not dataset_path.is_absolute():
            dataset_path = config_path.parent.parent / dataset_path

        data = np.load(dataset_path)
        key = "maps" if "maps" in data else list(data.keys())[0]
        self.maps: np.ndarray = data[key]          # (N, H, W) float32

        self.H, self.W = self.maps.shape[1], self.maps.shape[2]
        self.mask = np.ones((self.H, self.W), dtype=np.float32)
        self.mask[:8, :] = self.mask[-8:, :] = 0
        self.mask[:, :8] = self.mask[:, -8:] = 0
        self.initial_position: tuple[tuple[int, int], ...] = cfg["initial_position"]
        self.noise_std: float = float(cfg.get("noise_std", 0.0))

        self.n_agents = len(self.initial_position)

        self._rng = np.random.default_rng()

        # State (populated by reset)
        self.ground_truth: np.ndarray | None = None
        self.obs_map:  np.ndarray | None = None
        self.obs_mask: np.ndarray | None = None
        self.position: list[list[int, int]] | None = None
        self.trajectory: list[list[int]] = []
        self._map_idx: int | None = None

        # Matplotlib figure (created lazily by render)
        self._fig = None
        self._traj_line = []
        self._traj_dot = []

    # ------------------------------------------------------------------
    def reset(self, map_idx: int | None = None) -> np.ndarray:
        """
        Start a new episode.

        Parameters
        ----------
        map_idx : int or None
            Index of the ground-truth map to use.  If None, a random map
            is chosen each call.

        Returns
        -------
        ground_truth : np.ndarray (H, W) float32
        """
        if map_idx is None:
            map_idx = int(self._rng.integers(len(self.maps)))
        self._map_idx = map_idx
        self.ground_truth = self.maps[map_idx].copy()

        self.obs_map  = np.zeros((self.H, self.W), dtype=np.float32)
        self.obs_mask = np.zeros((self.H, self.W), dtype=np.float32)
        self.position = [pos.copy() for pos in self.initial_position]
        self.trajectory = [[position.copy()] for position in self.initial_position]
        # print(self.trajectory)

        # Observe the starting cell
        self._observe(self.initial_position)

        return self.ground_truth

    # ------------------------------------------------------------------
    def step(self, action: tuple[int, int] | tuple[tuple[int, int], ...]) -> tuple[np.ndarray, np.ndarray]:
        """
        Move the agent in a straight line to *action* and record observations.

        Parameters
        ----------
        action : (row, col) or [(row, col), ...]
            Target cell(s).  Clipped to map bounds if out of range.

        Returns
        -------
        obs_map  : np.ndarray (H, W) — concentration at visited cells, 0 elsewhere
        obs_mask : np.ndarray (H, W) — 1 at visited cells, 0 elsewhere
        """
        if self.ground_truth is None:
            raise RuntimeError("Call reset() before step().")
        
        for index, act in enumerate(action):

            r1 = int(np.clip(act[0], 0, self.H - 1))
            c1 = int(np.clip(act[1], 0, self.W - 1))

            r0, c0 = self.position[index]
            new_cells = _line_pixels(r0, c0, r1, c1)

            # print(f"Moving agent {index} from {(r0, c0)} to {(r1, c1)} through {len(new_cells)} new cells. accessing {self.trajectory}")
            self._observe(new_cells)
            self.trajectory[index].extend(new_cells)
            self.position[index] = (r1, c1)

        return self.obs_map.copy(), self.obs_mask.copy()

    # ------------------------------------------------------------------
    def _observe(self, cells: list[tuple[int, int]]) -> None:
        """Record noisy sensor readings at *cells*."""
        for r, c in cells:
            reading = float(self.ground_truth[r, c])
            if self.noise_std > 0:
                reading += float(self._rng.normal(0.0, self.noise_std))
            self.obs_map[r, c]  = np.clip(reading, 0.0, 1.0)
            self.obs_mask[r, c] = 1.0

    # ------------------------------------------------------------------
    def render(self) -> None:
        """
        Show (or update) an interactive matplotlib figure with four panels:
            1. Ground truth map
            2. Observation map (sensor readings)
            3. Observation mask
            4. Agent trajectory overlaid on the ground truth
        """
        if self.ground_truth is None:
            raise RuntimeError("Call reset() before render().")

        if self._fig is None:
            self._fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            self._fig.suptitle("Observation Scenario", fontsize=12)
            self._axes = axes

            kw = dict(vmin=0, vmax=1, interpolation="nearest")

            self._im_gt   = axes[0].imshow(self.ground_truth, cmap="hot",   **kw)
            self._im_obs  = axes[1].imshow(self.obs_map,      cmap="hot",   **kw)
            self._im_mask = axes[2].imshow(self.obs_mask,     cmap="gray",  **kw)
            self._im_traj = axes[3].imshow(self.ground_truth, cmap="hot",   **kw)

            axes[0].set_title("Ground Truth")
            axes[1].set_title("Observation Map")
            axes[2].set_title("Observation Mask")
            axes[3].set_title("Trajectory")
            
            for ax in axes:
                ax.axis("off")
            
            self._traj_line = [[]]*len(self.trajectory)
            self._traj_dot = [[]]*len(self.trajectory)

            for index, traj in enumerate(self.trajectory):
                # print(f"Trajectory {index}: {traj} on trajline {self._traj_line} and trajdot {self._traj_dot}")
                traj = np.array(traj)
                self._traj_line[index], = axes[3].plot(
                    traj[:, 1], traj[:, 0],
                    color="cyan", linewidth=1.5, alpha=0.8,
                )
                self._traj_dot[index], = axes[3].plot(
                    [traj[-1, 1]], [traj[-1, 0]],
                    "o", color="lime", markersize=6,
                )

            self._fig.colorbar(self._im_gt,   ax=axes[0], fraction=0.046)
            self._fig.colorbar(self._im_obs,  ax=axes[1], fraction=0.046)
            self._fig.colorbar(self._im_mask, ax=axes[2], fraction=0.046)

            plt.tight_layout()
            plt.ion()
            plt.show()

        else:
            self._im_gt.set_data(self.ground_truth)
            self._im_obs.set_data(self.obs_map)
            self._im_mask.set_data(self.obs_mask)
            self._im_traj.set_data(self.ground_truth)
            
            for index, traj in enumerate(self.trajectory):

                traj = np.array(traj)
                self._traj_line[index].set_xdata(traj[:, 1])
                self._traj_line[index].set_ydata(traj[:, 0])
                self._traj_dot[index].set_xdata([self.position[index][1]])
                self._traj_dot[index].set_ydata([self.position[index][0]])

            self._fig.canvas.draw()
            self._fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = Path(__file__).parent / "scenario_config.yaml"
    env = ObservationScenario(cfg)

    gt = env.reset(map_idx=0)
    print(f"Ground truth shape : {gt.shape},  max={gt.max():.3f}")

    env.render()

    waypoints = [
        [(np.random.randint(0, env.H), np.random.randint(0, env.W)) for _ in range(len(env.position))] for _ in range(30)
    ]
    for wp in waypoints:
        # print(f"Moving to {wp}...")
        obs_map, obs_mask = env.step(wp)
        coverage = obs_mask.mean() * 100
        # print(f"  → moved to {wp}  |  coverage {coverage:.1f}%")
        env.render()
        plt.pause(0.4)

    print("Done. Close the window to exit.")
    plt.ioff()
    plt.show()
