"""
greedy_uncertainty.py — Greedy policy that moves to the highest-value cell that remains unvisited.
"""

import numpy as np
from scipy.ndimage import binary_erosion
from .base import Policy


class MaxGreedyMiopic(Policy):
    """
    Move to the unobserved cell with the highest predicted value
    within ``max_radius`` pixels of the current position.

    If no unobserved cell exists within the radius, the constraint is relaxed
    and the global value maximum is returned instead.

    Parameters
    ----------
    max_radius : float
        Maximum Euclidean distance (pixels) from the current position
        to the chosen waypoint.
    """

    def __init__(self, max_radius: float):
        self.max_radius = float(max_radius)


    def act(
        self,
        obs: dict,
        position: tuple[int, int],
    ) -> tuple[int, int]:
        model_value: np.ndarray = obs["predicted_mean"]
        uncertainty: np.ndarray = obs["predicted_uncertainty"]
        mask: np.ndarray = obs["obs_mask"]
        
        H, W = model_value.shape
        r0, c0 = position

        rows, cols = np.mgrid[0:H, 0:W]
        dist = np.hypot(rows - r0, cols - c0)

        within_radius = dist <= self.max_radius
        
        # Erode mask to ensure we only consider fully unobserved cells (not on the edge of observed area)
        unobserved = np.logical_not(mask)
        unobserved_eroded = binary_erosion(unobserved, structure=np.ones((3, 3)), border_value=0)
        
        candidates    = within_radius & (uncertainty != 0.0)  & unobserved_eroded # Only consider unobserved cells within radius
        
        if not candidates.any():
            # Relax radius: pick global uncertainty maximum
            candidates = np.ones((H, W), dtype=bool)
            
        # Select the candidate cell with the highest value*uncertainty (or just uncertainty if you prefer)
        scores = model_value * uncertainty  # You can also try just uncertainty
        scores[~candidates] = -np.inf  # Exclude non-candidates
        idx = np.argmax(scores)
        r1, c1 = np.unravel_index(idx, (H, W))
        
        return int(r1), int(c1)
