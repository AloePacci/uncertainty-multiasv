"""
max_uncertainty.py — Greedy policy that moves to the highest-uncertainty cell
within a maximum Euclidean radius.
"""

import numpy as np
from scipy.ndimage import binary_erosion
from .base import Policy


class EpsilonGreedy(Policy):
    """
    Move to the unobserved cell with the highest predicted uncertainty
    within ``max_radius`` pixels of the current position.

    If no unobserved cell exists within the radius, the constraint is relaxed
    and the global uncertainty maximum is returned instead.

    Parameters
    ----------
    max_radius : float
        Maximum Euclidean distance (pixels) from the current position
        to the chosen waypoint.
    """

    def __init__(self, max_radius: float, eps_interval: tuple[float, float] = (0.1, 0.01), eps_decay_steps: int = 100):
        self.max_radius = float(max_radius)
        self.max_epsilon = float(eps_interval[0])
        self.min_epsilon = float(eps_interval[1])
        self.eps_decay_steps = int(eps_decay_steps)
        self.step_count = 0

    def reset(self) -> None:
        """Reset the step count at episode start."""
        self.step_count = 0
        
    def act(
        self,
        obs: dict,
        position: tuple[tuple[int, int], ...],
            ) -> tuple[tuple[int, int],...]:
        uncertainty: np.ndarray = obs["predicted_uncertainty"]
        value : np.ndarray = obs["predicted_mean"]
        obs_mask: np.ndarray    = obs["obs_mask"]
        obs_mask[:8, :] = obs_mask[-8:, :] = 1
        obs_mask[:, :8] = obs_mask[:, -8:] = 1
        H, W = uncertainty.shape
        destinations = []
        # Compute current epsilon based on decay schedule
        epsilon = max(
            self.min_epsilon,
            self.max_epsilon - (self.max_epsilon - self.min_epsilon) * (self.step_count / self.eps_decay_steps)
        )
        self.step_count += 1
        for pos in position:
            r0, c0 = pos
        
        
            rows, cols = np.mgrid[0:H, 0:W]
            dist = np.hypot(rows - r0, cols - c0)
            within_radius = dist <= self.max_radius
            candidates = within_radius & (obs_mask == 0)  # Only consider unobserved cells within radius
        
        
            if not candidates.any():
                # Relax radius: pick global uncertainty maximum
                candidates = obs_mask == 0  # All unobserved cells
            
            if np.random.rand() > epsilon:
                # Explore: choose greedy with respect to uncertainty, 
                candidate_indices = np.argwhere(candidates)
                if len(candidate_indices) == 0:
                    # If no candidates are available, fallback to global max
                    idx = np.argmax(uncertainty)
                    r1, c1 = np.unravel_index(idx, (H, W))
                else:
                    random_idx = np.random.choice(len(candidate_indices))
                    r1, c1 = candidate_indices[random_idx]
            else:
                # Exploit: choose candidate with highest value
                scores = value  # You can also try just uncertainty
                scores[~candidates] = -np.inf  # Exclude non-candidates
                idx = np.argmax(scores)
                r1, c1 = np.unravel_index(idx, (H, W))
            destinations.append((int(r1), int(c1)))
            obs_mask[r1, c1] = 1  # Mark the chosen cell as observed

        
        
        return tuple(destinations)
