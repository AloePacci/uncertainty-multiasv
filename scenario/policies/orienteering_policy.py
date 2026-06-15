"""
orienteering_policy.py — Receding-horizon orienteering policy.

The planning problem is modelled as an Orienteering Problem (OP):

  - Candidate waypoints are sampled on a regular sub-grid of the map.
  - Edge reward  = sum of uncertainty values over the pixels of the
                   straight-line segment connecting two waypoints.
  - Edge cost    = Euclidean distance between the two waypoints.
  - Constraint   : total path cost ≤ H_PLAN.
  - Objective    : maximise total edge reward.

Solver
------
1. Multi-start greedy construction
   Each run shuffles the candidate list and builds a greedy path by
   always extending to the unvisited candidate with the highest
   reward/cost efficiency that still fits within the remaining budget.

2. Hill-climbing local search
   Starting from the best greedy solution, alternates between:
     - Remove: drop one interior waypoint if it increases reward.
     - Insert: add an unvisited candidate at the best position if it
               increases reward without exceeding the budget.
   Stops when no single move improves the current solution.

Receding horizon
----------------
The policy executes the planned path one waypoint at a time.
After H_ACT distance has been covered, the plan is discarded and
rebuilt using the latest uncertainty map.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.ndimage import binary_dilation

from .base import Policy

from tqdm import trange


class OrienteeringPolicy(Policy):
    """
    Receding-horizon orienteering policy.

    Parameters
    ----------
    h_plan : float
        Planning horizon: maximum total Euclidean path length (pixels).
    h_act : float
        Replan trigger: accumulated distance (pixels) to execute before
        rebuilding the plan with the updated uncertainty map.
    grid_step : int
        Spacing (pixels) between candidate waypoints on the sub-grid.
    n_starts : int
        Number of greedy restarts for multi-start construction.
    max_lc_iters : int
        Maximum hill-climbing iterations.
    rng_seed : int | None
        Optional seed for reproducibility.
    """

    def __init__(
        self,
        h_plan: float,
        h_act: float,
        grid_step: int = 10,
        n_starts: int = 5,
        max_lc_iters: int = 200,
        rng_seed: int | None = None,
    ) -> None:
        self.h_plan = float(h_plan)
        self.h_act = float(h_act)
        self.grid_step = int(grid_step)
        self.n_starts = int(n_starts)
        self.max_lc_iters = int(max_lc_iters)
        self._rng = np.random.default_rng(rng_seed)

        # Episode state
        self._plan: list[tuple[int, int]] = []
        self._plan_idx: int = 0
        self._dist_since_replan: float = 0.0
        self._last_position: tuple[int, int] | None = None


    # ------------------------------------------------------------------ #
    # Policy API                                                           #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._plan = []
        self._plan_idx = 0
        self._dist_since_replan = 0.0
        self._last_position = None

    def act(
        self,
        obs: dict,
        position: tuple[int, int],
    ) -> tuple[int, int]:
        """
        Return the next waypoint.

        Replans when no plan exists, the plan is exhausted, or H_ACT
        distance has accumulated since the last replan.
        """
        uncertainty: np.ndarray = obs["predicted_uncertainty"]
        obs_mask = obs["obs_mask"]
        position = (int(position[0]), int(position[1]))

        # Accumulate distance since last replan
        if self._last_position is not None:
            d = float(np.hypot(
                position[0] - self._last_position[0],
                position[1] - self._last_position[1],
            ))
            self._dist_since_replan += d
        self._last_position = position

        need_replan = (
            len(self._plan) == 0
            or self._plan_idx >= len(self._plan)
            or self._dist_since_replan >= self.h_act
        )

        if need_replan:
            self._plan = self._build_plan(uncertainty, obs_mask, position)
            self._plan_idx = 0
            self._dist_since_replan = 0.0

        # Skip waypoints already coinciding with current position
        while (
            self._plan_idx < len(self._plan)
            and self._plan[self._plan_idx] == position
        ):
            self._plan_idx += 1

        if self._plan_idx < len(self._plan):
            wp = self._plan[self._plan_idx]
            self._plan_idx += 1
            return wp

        # Fallback: global uncertainty maximum (plan was empty or exhausted)
        idx = int(np.argmax(uncertainty))
        r, c = np.unravel_index(idx, uncertainty.shape)
        return int(r), int(c)

    # ------------------------------------------------------------------ #
    # Candidate sampling                                                   #
    # ------------------------------------------------------------------ #

    def _candidates(
        self, H: int, W: int, start: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Regular sub-grid candidates, excluding the start position."""
        return [
            (r, c)
            for r in range(0, H, self.grid_step)
            for c in range(0, W, self.grid_step)
            if (r, c) != start
        ]

    # ------------------------------------------------------------------ #
    # Edge primitives                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _line_pixels(
        r0: int, c0: int, r1: int, c1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pixel indices along the straight segment (r0,c0) → (r1,c1)."""
        n = max(abs(r1 - r0), abs(c1 - c0)) + 1
        rows = np.round(np.linspace(r0, r1, n)).astype(int)
        cols = np.round(np.linspace(c0, c1, n)).astype(int)
        return rows, cols

    def _edge_reward(
        self,
        uncertainty: np.ndarray,
        r0: int, c0: int,
        r1: int, c1: int,
    ) -> float:
        rows, cols = self._line_pixels(r0, c0, r1, c1)
        return float(uncertainty[rows, cols].sum())

    @staticmethod
    def _edge_cost(r0: int, c0: int, r1: int, c1: int) -> float:
        return float(np.hypot(r1 - r0, c1 - c0))

    # ------------------------------------------------------------------ #
    # Path evaluation                                                      #
    # ------------------------------------------------------------------ #

    def _path_reward(
        self, uncertainty: np.ndarray, path: list[tuple[int, int]]
    ) -> float:
        if len(path) < 2:
            return 0.0
        return sum(
            self._edge_reward(uncertainty, *a, *b)
            for a, b in zip(path, path[1:])
        )

    def _path_cost(self, path: list[tuple[int, int]]) -> float:
        if len(path) < 2:
            return 0.0
        return sum(
            self._edge_cost(*a, *b)
            for a, b in zip(path, path[1:])
        )

    # ------------------------------------------------------------------ #
    # Greedy construction                                                  #
    # ------------------------------------------------------------------ #

    def _greedy_build(
        self,
        uncertainty: np.ndarray,
        start: tuple[int, int],
        candidates: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """
        Build a path greedily from *start* using the shuffled *candidates*.
        At each step, extend to the unvisited candidate that maximises
        reward/cost efficiency while the remaining budget allows it.
        """
        path = [start]
        remaining = self.h_plan
        unvisited = set(candidates)

        while unvisited:
            current = path[-1]
            best_pt: tuple[int, int] | None = None
            best_eff = -np.inf

            for pt in unvisited:
                cost = self._edge_cost(*current, *pt)
                if cost == 0 or cost > remaining:
                    continue
                
                
                reward = self._edge_reward(uncertainty, *current, *pt)
                eff = reward / cost
                if eff > best_eff:
                    best_eff = eff
                    best_pt = pt

            if best_pt is None:
                break  # no candidate fits within remaining budget

            remaining -= self._edge_cost(*current, *best_pt)
            path.append(best_pt)
            unvisited.discard(best_pt)

        return path

    # ------------------------------------------------------------------ #
    # Hill-climbing local search                                           #
    # ------------------------------------------------------------------ #

    def _local_search(
        self,
        uncertainty: np.ndarray,
        path: list[tuple[int, int]],
        candidates: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """
        Hill-climbing local search with remove and insert moves.

        Remove: drop one interior waypoint if the resulting path has
                higher reward (the two adjacent edges merge into one).
        Insert: add an unvisited candidate at the best insertion
                position if budget allows and reward improves.

        Restarts the scan after every accepted move.
        """
        best = list(path)
        best_reward = self._path_reward(uncertainty, best)
        in_path: set[tuple[int, int]] = set(best[1:])  # excludes start

        improved = True
        iters = 0

        while improved and iters < self.max_lc_iters:
            improved = False
            iters += 1

            # ── Remove move ────────────────────────────────────────────
            for i in range(1, len(best)):
                new_path = best[:i] + best[i + 1:]
                r = self._path_reward(uncertainty, new_path)
                if r > best_reward:
                    removed = best[i]
                    best = new_path
                    best_reward = r
                    in_path.discard(removed)
                    improved = True
                    break  # restart after first improvement

            if improved:
                continue

            # ── Insert move ────────────────────────────────────────────
            insertable = [p for p in candidates if p not in in_path]
            order = self._rng.permutation(len(insertable))

            for idx in order:
                pt = insertable[idx]
                best_ins_reward = best_reward
                best_ins_path: list[tuple[int, int]] | None = None

                for pos in range(1, len(best) + 1):
                    new_path = best[:pos] + [pt] + best[pos:]
                    if self._path_cost(new_path) > self.h_plan:
                        continue
                    r = self._path_reward(uncertainty, new_path)
                    if r > best_ins_reward:
                        best_ins_reward = r
                        best_ins_path = new_path

                if best_ins_path is not None:
                    best = best_ins_path
                    best_reward = best_ins_reward
                    in_path.add(pt)
                    improved = True
                    break  # restart after first improvement

        return best


    # ------------------------------------------------------------------ #
    # Top-level planner                                                    #
    # ------------------------------------------------------------------ #

    def _build_plan(
        self,
        uncertainty: np.ndarray,
        obs_mask: np.ndarray,
        position: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """
        Run multi-start greedy + hill-climbing and return the planned
        waypoints *after* the start (i.e. the sequence to execute).
        """
        H, W = uncertainty.shape

        # Zero out already-visited pixels so they contribute 0 reward
        
        dilated_mask = binary_dilation(obs_mask.astype(bool), structure=np.ones((3, 3)))     
        uncertainty = uncertainty * (dilated_mask == 0).astype(np.float32)

        candidates = [
            p for p in self._candidates(H, W, position)
            if dilated_mask[p] == 0
        ]

        if not candidates:
            return []

        best_path: list[tuple[int, int]] = []
        best_reward: float = -np.inf
        greedy_paths: list[tuple[list[tuple[int, int]], float]] = []

        for _ in range(self.n_starts):
            shuffled = list(candidates)
            self._rng.shuffle(shuffled)
            path = self._greedy_build(uncertainty, position, shuffled)
            r = self._path_reward(uncertainty, path)
            greedy_paths.append((path, r))
            if r > best_reward:
                best_reward = r
                best_path = path


        best_path = self._local_search(uncertainty, best_path, candidates)



        # Strip the start position — the caller already is there
        return best_path[1:] if len(best_path) > 1 else []
