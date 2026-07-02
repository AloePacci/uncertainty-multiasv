"""
mcts_policy.py — MCTSPolicy: online MCTS-based informative path planning.

Builds an information map as ``predicted_uncertainty × (1 − obs_mask)`` and
uses MCTS with a ``MaxInformativePathWaypoints`` problem to select the next
waypoint.

When the model produces a new uncertainty estimate (detected by comparing
``predicted_uncertainty`` with the previous call), the planner is rebuilt and
all queued open-loop actions are discarded.  Between such events, the MCTS
``control_horizon`` parameter allows executing several pre-planned actions
without re-running simulations.

Coordinate conventions
----------------------
The Policy API uses ``(row, col)`` = ``(y, x)`` in image space.
``MaxInformativePathWaypoints`` uses ``(x, y)`` = ``(col, row)``.
All conversions are handled internally.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Policy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # allow imports from scenario/
from algorithms.multiagent_mcts import MAMCTS

sys.path.append(str(Path(__file__).parent.parent))  # allow imports from scenario/
from multiagent_max_informative_path_waypoints import (
    MultiagentMaxInformativePathWaypoints,
    _min_path_cost,
)
from scipy.ndimage import binary_dilation

class MCTSPolicy(Policy):
    """
    Online MCTS policy for informative path planning.

    On each call to ``act``, the information map ``uncertainty × (1 − mask)``
    is built and passed to a ``MaxInformativePathWaypoints`` problem.  MCTS
    with ``candidates_adaptive`` selects the next waypoint.

    Parameters
    ----------
    budget : float
        Total path-length budget (pixels) available to the planner.
    n_simulations : int
        Number of MCTS simulations run per replanning step.
    depth : int
        Maximum search-tree depth (and rollout horizon).
    control_horizon : int
        Number of steps to execute open-loop before replanning.
        Forwarded to MCTS so the planner pre-queues the next
        ``control_horizon − 1`` actions after each full tree search.
    min_resolution : int
        Minimum step size for ``candidates_adaptive``.
    max_resolution : int
        Maximum step size for ``candidates_adaptive``.
    anneal_radius : int
        Sensor footprint radius in cells (0 = single traversed cell).
    reuse_tree : bool
        Whether to keep the MCTS sub-tree between consecutive steps.
    gamma : float
        Discount factor.
    exploration_c : float
        UCB1 exploration constant c (default √2).
    uncertainty_change_threshold : float
        Mean-absolute-difference threshold above which a new uncertainty
        map triggers a full replanning (problem + planner rebuilt).
    """

    def __init__(
        self,
        budget: float,
        n_simulations: int = 500,
        depth: int = 50,
        control_horizon: int = 1,
        min_resolution: int = 2,
        max_resolution: int = 6,
        anneal_radius: int = 0,
        reuse_tree: bool = True,
        gamma: float = 1.0,
        exploration_c: float = math.sqrt(2),
        uncertainty_change_threshold: float = 1e-4,
        nagents: int = 3,
    ) -> None:
        self._budget = float(budget)
        self._n_simulations = int(n_simulations)
        self._depth = int(depth)
        self._control_horizon = int(control_horizon)
        self._min_resolution = int(min_resolution)
        self._max_resolution = int(max_resolution)
        self._anneal_radius = int(anneal_radius)
        self._reuse_tree = reuse_tree
        self._gamma = float(gamma)
        self._exploration_c = float(exploration_c)
        self._uncertainty_change_threshold = float(uncertainty_change_threshold)
        self._nagents = int(nagents)

        # Episode state
        self._remaining_budget: float = self._budget
        self._problem: MultiagentMaxInformativePathWaypoints | None = None
        self._mcts: MAMCTS | None = None
        self._last_uncertainty: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Policy API                                                           #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._remaining_budget = self._budget
        self._problem = None
        self._mcts = None
        self._last_uncertainty = None

    def act(
        self,
        obs: dict,
        position: tuple[tuple[int, int], ...],
    ) -> tuple[int, int]:
        """
        Select the next waypoint using MCTS.

        Parameters
        ----------
        obs      : dict — must contain ``predicted_uncertainty`` (H, W) and
                   ``obs_mask`` (H, W).
        position : ((row, col), ...) — current agent positions.

        Returns
        -------
        (row, col) — next waypoint.
        """
        uncertainty: np.ndarray = obs["predicted_uncertainty"]
        mask: np.ndarray = obs["obs_mask"]

        # Information map: uncertain AND unobserved cells are most valuable.
        
        # Dilate the mask
        dilated_mask = np.copy(mask)
        if self._anneal_radius > 0:
            structure = np.ones((2 * self._anneal_radius + 1,) * 2, dtype=bool)
            dilated_mask = binary_dilation(mask > 0, structure=structure).astype(mask.dtype)
            info_map = uncertainty * (1 - dilated_mask)
        else:
            info_map: np.ndarray = uncertainty * (1 - mask)
        
        # Minmax normalisation to [0, 1] so reward scale is independent of model output.
        info_min, info_max = float(info_map.min()), float(info_map.max())
        if info_max > info_min:
            info_map = (info_map - info_min) / (info_max - info_min)

        # Rebuild planner when the uncertainty map changed significantly.
        if self._uncertainty_changed(uncertainty):
            self._build_planner(info_map, position)
            self._last_uncertainty = uncertainty.copy()

        # Current MCTS state derived from the real environment.
        visited = _mask_to_visited(mask)
        state: dict = {
            "position": position,
            "budget": self._remaining_budget,
            "visited": visited,
        }

        # Query MCTS.
        action, _ = self._mcts.select_action(state)  # type: ignore[union-attr]

        if action is None:
            # Fallback: global information maximum.
            idx = int(np.argmax(info_map))
            r, c = np.unravel_index(idx, info_map.shape)
            return int(r), int(c)

        # Deduct movement cost from remaining budget.
        cost = _min_path_cost(position, action)
        self._remaining_budget = max(0.0, self._remaining_budget - cost)

        # Scenario (x=col, y=row) → policy (row, col).
        ax, ay = action
        return int(ay), int(ax)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _uncertainty_changed(self, uncertainty: np.ndarray) -> bool:
        """True when no planner exists yet or the uncertainty map differs."""
        if self._problem is None or self._mcts is None:
            return True
        if self._last_uncertainty is None:
            return True
        diff = float(np.mean(np.abs(uncertainty - self._last_uncertainty)))
        return diff > self._uncertainty_change_threshold

    def _build_planner(
        self,
        info_map: np.ndarray,
        initial_position: tuple[tuple[int, int], ...],
    ) -> None:
        """Construct a fresh problem and MCTS planner for the current info map."""
        problem = MultiagentMaxInformativePathWaypoints(
            info_map=info_map,
            max_budget=self._remaining_budget,
            candidate_fn=None,          # assigned after construction
            initial_position=initial_position,
            gamma=self._gamma,
            anneal_radius=self._anneal_radius,
            neg_reward=0.0,
        )
        problem.candidate_fn = problem.candidates_adaptive(
            min_resolution=self._min_resolution,
            max_resolution=self._max_resolution,
        )
        self._problem = problem
        self._mcts = MAMCTS(
            problem=self._problem,
            n_simulations=self._n_simulations,
            depth=self._depth,
            gamma=self._gamma,
            exploration_c=self._exploration_c,
            reuse_tree=self._reuse_tree,
            control_horizon=self._control_horizon,
        )


# ────────────────────────────────────────────────────────────────────────── #
# Module-level helper                                                        #
# ────────────────────────────────────────────────────────────────────────── #

def _mask_to_visited(mask: np.ndarray) -> frozenset[tuple[int, int]]:
    """
    Convert a binary observation mask (H, W) to a frozenset of visited
    scenario coordinates ``(x, y) = (col, row)``.
    """
    rows, cols = np.where(mask > 0)
    return frozenset((int(c), int(r)) for r, c in zip(rows, cols))
