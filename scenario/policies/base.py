"""
base.py — Abstract base class for informative path planning policies.
"""

from abc import ABC, abstractmethod


class Policy(ABC):
    """
    Abstract policy for the ExtendedScenario.

    A policy receives the current observation dict (as returned by
    ``ExtendedScenario.step``) and the agent's current position, and
    returns the next waypoint.

    Observation dict keys
    ---------------------
    obs_map               : np.ndarray (H, W) — sensor readings at visited cells
    obs_mask              : np.ndarray (H, W) — 1 at visited cells, 0 elsewhere
    predicted_mean        : np.ndarray (H, W) — model's predicted ground-truth map
    predicted_uncertainty : np.ndarray (H, W) — model's uncertainty map
    """

    @abstractmethod
    def act(
        self,
        obs: dict,
        position: tuple[int, int],
    ) -> tuple[int, int]:
        """
        Select the next waypoint.

        Parameters
        ----------
        obs      : dict   — observation returned by ExtendedScenario.step
        position : (row, col) — current agent position

        Returns
        -------
        action : (row, col)
        """

    def reset(self) -> None:
        """
        Reset internal state at the start of a new episode.

        Stateless policies can leave this as the default no-op.
        Stateful policies (e.g. those that plan ahead) should clear
        any episode-specific memory here.
        """
