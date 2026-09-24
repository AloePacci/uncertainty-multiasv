"""
mamcts_policy.py — Multi-agent MCTSPolicy: online MAMCTS-based informative path planning.

Builds an information map as ``predicted_uncertainty × (1 − obs_mask)`` and
uses MAMCTS with a ``MultiagentMaxInformativePathWaypoints`` problem to select the next
waypoint.

When the model produces a new uncertainty estimate (detected by comparing
``predicted_uncertainty`` with the previous call), the planner is rebuilt and
all queued open-loop actions are discarded.  Between such events, the MAMCTS
``control_horizon`` parameter allows executing several pre-planned actions
without re-running simulations.

Coordinate conventions
----------------------
The Policy API uses ``(row, col)`` = ``(y, x)`` in image space.
``MultiagentMaxInformativePathWaypoints`` uses ``(x, y)`` = ``(col, row)``.
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
    agent_priority_heuristic,
)
from scipy.ndimage import binary_dilation
from collections import defaultdict
from datetime import datetime
class MAMCTSPolicy(Policy):
    """
    Online MCTS policy for informative path planning.

    On each call to ``act``, the information map ``uncertainty × (1 − mask)``
    is built and passed to a ``MultiagentMaxInformativePathWaypoints`` problem.  MAMCTS
    with ``candidates_adaptive`` selects the next waypoint.

    Parameters
    ----------
    budget : float
        Total path-length budget (pixels) available to the planner.
    n_simulations : int
        Number of MAMCTS simulations run per replanning step.
    depth : int
        Maximum search-tree depth (and rollout horizon).
    control_horizon : int
        Number of steps to execute open-loop before replanning.
        Forwarded to MAMCTS so the planner pre-queues the next
        ``control_horizon − 1`` actions after each full tree search.
    min_resolution : int
        Minimum step size for ``candidates_adaptive``.
    max_resolution : int
        Maximum step size for ``candidates_adaptive``.
    anneal_radius : int
        Sensor footprint radius in cells (0 = single traversed cell).
    reuse_tree : bool
        Whether to keep the MAMCTS sub-tree between consecutive steps.
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
        self._remaining_budget: list[float] | None = None
        self._last_position: list[tuple[int, int]] | None = None   # scenario (x, y)
        self._problem: MultiagentMaxInformativePathWaypoints | None = None
        self._mcts: MAMCTS | None = None
        self._last_uncertainty: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Policy API                                                           #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._remaining_budget = None
        self._last_position = None
        self._problem = None
        self._mcts = None
        self._last_uncertainty = None

    def act(
        self,
        obs: dict,
        position: tuple[tuple[int, int], ...],
    ) -> tuple[int, int]:
        """
        Select the next waypoint using MAMCTS.

        Parameters
        ----------
        obs      : dict — must contain ``predicted_uncertainty`` (H, W) and
                   ``obs_mask`` (H, W).
        position : ((row, col), ...) — current agent positions.

        Returns
        -------
        ((row, col), ...) — next waypoint list.
        """
        uncertainty: np.ndarray = obs["predicted_uncertainty"]
        # Copy: obs["obs_mask"] may be the environment's own array.
        mask: np.ndarray = obs["obs_mask"].copy()
        # Borders carry no information: treat them as already visited.
        mask[:8, :] = mask[-8:, :] = 1
        mask[:, :8] = mask[:, -8:] = 1
        start_time = datetime.now().timestamp()

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

        # Policy (row, col) → scenario (x=col, y=row).
        scenario_pos = [(int(c), int(r)) for r, c in position]

        # Remaining budget: the environment charges the straight-line
        # Euclidean distance actually travelled by each agent.
        if self._remaining_budget is None:
            self._remaining_budget = [self._budget] * len(scenario_pos)
        elif self._last_position is not None:
            for i, (p0, p1) in enumerate(zip(self._last_position, scenario_pos)):
                step = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                self._remaining_budget[i] = max(0.0, self._remaining_budget[i] - step)
        last_position = self._last_position or scenario_pos
        self._last_position = scenario_pos

        # Rebuild planner when the uncertainty map changed significantly.
        if self._uncertainty_changed(uncertainty):
            self._build_planner(info_map, scenario_pos)
            self._last_uncertainty = uncertainty.copy()

        # Current MAMCTS state derived from the real environment.
        budget = list(self._remaining_budget)
        state = {
            "position": scenario_pos,
            "budget": budget,
            "visited": _mask_to_visited(mask),
            "last_position": list(last_position),
            "priority": agent_priority_heuristic({"budget": budget}),
        }

        # Query MAMCTS.
        action, _ = self._mcts.select_action(state)  # type: ignore[union-attr]
        if action is None:
            # Terminal for the planner (some agent has < 1 px left): take one
            # greedy unit step per agent so the episode ends with minimal
            # budget overshoot.
            H, W = info_map.shape
            action = []
            for r, c in position:
                r, c = int(r), int(c)
                neighbours = [
                    (r + dr, c + dc)
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if (dr, dc) != (0, 0) and 0 <= r + dr < H and 0 <= c + dc < W
                ]
                action.append(max(neighbours, key=lambda rc: info_map[rc]))
            return tuple(action)

        end_time = datetime.now().timestamp()
        print(f"Selected action: {action} \n from position {scenario_pos} \n with remaining budget {self._remaining_budget} \n time taken: {end_time - start_time} seconds \n  ------------------------------------------------------")

        # Scenario (x=col, y=row) → policy (row, col).
        return tuple((int(ay), int(ax)) for ax, ay in action)

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
            max_budget=list(self._remaining_budget),
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
