"""
Escenario de planificación de caminos informativos (Informative Path Planning).

Un agente explorador recorre una cuadrícula N×N maximizando la suma de valores
informativos de las celdas que visita por primera vez, sujeto a un presupuesto
de distancia euclidiana. El problema es una instancia del IPP (Informative Path
Planning), un problema NP-difícil en el caso general.

Estado:
    position : (x, y)      — celda actual (x=columna, y=fila, origen en (0,0)).
    budget   : float        — distancia euclidiana restante.
    visited  : frozenset    — celdas ya visitadas (su información fue recogida).

Acciones:
    8 direcciones cardinales e intercarddinales. El coste euclidiano es:
        · Ortogonales  (N, S, E, W)     → 1.0
        · Diagonales   (NE, NW, SE, SW) → √2 ≈ 1.4142

Recompensa:
    info_map[y, x]  al llegar por primera vez a (x, y).
    0.0             si la celda ya fue visitada.

Terminal:
    budget < 1.0  (insuficiente para el paso ortogonal mínimo).

Sistema de coordenadas:
    · Origen (0, 0) en la esquina superior-izquierda.
    · Eje x crece hacia la derecha; eje y crece hacia abajo.
    · Indexación del mapa: info_map[y, x]  (convención NumPy: fila, columna).
"""

import math
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # allow imports from scenario/
from policies.algorithms.problem import Problem

# ────────────────────────────────────────────────────────────────────────── #
# Constantes del dominio                                                     #
# ────────────────────────────────────────────────────────────────────────── #

_SQRT2: float = math.sqrt(2)

# Desplazamiento (Δx, Δy) de cada una de las 8 acciones.
_ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "N":  ( 0, -1),
    "S":  ( 0, +1),
    "E":  (+1,  0),
    "W":  (-1,  0),
    "NE": (+1, -1),
    "NW": (-1, -1),
    "SE": (+1, +1),
    "SW": (-1, +1),
}

# Coste euclidiano de cada acción (distancia en píxeles).
_ACTION_COSTS: dict[str, float] = {
    "N": 1.0, "S": 1.0, "E": 1.0, "W": 1.0,
    "NE": _SQRT2, "NW": _SQRT2, "SE": _SQRT2, "SW": _SQRT2,
}

# Coste mínimo de cualquier paso (usado para la condición de terminal).
_MIN_COST: float = 1.0


# ────────────────────────────────────────────────────────────────────────── #
# Generación del mapa informativo                                            #
# ────────────────────────────────────────────────────────────────────────── #

def generate_smooth_map(
    N: int,
    n_peaks: int = 6,
    seed: int | None = None,
) -> np.ndarray:
    """
    Genera un campo escalar 2D de forma N×N con una distribución suave en [0, 1].

    El campo se construye como superposición de n_peaks gaussianas
    bidimensionales con centros, amplitudes y dispersiones aleatorias,
    lo que produce una superficie continua con varios picos bien diferenciados
    sin depender de librerías externas a NumPy.

    Args:
        N:       Dimensión del mapa cuadrado (N×N celdas).
        n_peaks: Número de picos gaussianos superpuestos.
        seed:    Semilla aleatoria para reproducibilidad.

    Returns:
        Array NumPy de forma (N, N), valores en [0, 1].
    """
    rng = np.random.default_rng(seed)
    field = np.zeros((N, N), dtype=float)

    # Grids de coordenadas para operaciones vectorizadas.
    y_grid, x_grid = np.mgrid[0:N, 0:N]

    for _ in range(n_peaks):
        cx = rng.uniform(0.0, float(N - 1))
        cy = rng.uniform(0.0, float(N - 1))
        sigma = rng.uniform(N / 8.0, N / 3.0)
        amplitude = rng.uniform(0.5, 1.0)
        field += amplitude * np.exp(
            -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2.0 * sigma ** 2)
        )

    # Normalización lineal a [0, 1].
    field -= field.min()
    if field.max() > 0.0:
        field /= field.max()

    return field


# ────────────────────────────────────────────────────────────────────────── #
# Clase principal del escenario                                              #
# ────────────────────────────────────────────────────────────────────────── #

class MaxInformativePath(Problem):
    """
    Problema de planificación de caminos informativos sobre una cuadrícula N×N.

    El agente debe diseñar una trayectoria desde (0, 0) que maximice la
    información acumulada de las celdas visitadas por primera vez, sin
    exceder el presupuesto de distancia euclidiana disponible.

    El entorno es completamente observable y determinista: dada una acción,
    el estado sucesor queda unívocamente determinado.

    Attributes:
        info_map:   Array (N, N) con el valor informativo de cada celda ∈ [0,1].
        N:          Dimensión de la cuadrícula.
        max_budget: Presupuesto de distancia euclidiana total.
    """

    def __init__(self, info_map: np.ndarray, max_budget: float) -> None:
        """
        Inicializa el problema IPP.

        Args:
            info_map:   Mapa informativo de forma (N, N) con valores en [0, 1].
            max_budget: Presupuesto total de distancia euclidiana.

        Raises:
            AssertionError: Si info_map no es cuadrado.
        """
        assert info_map.ndim == 2 and info_map.shape[0] == info_map.shape[1], (
            "info_map debe ser cuadrado (N×N)."
        )
        self.info_map: np.ndarray = info_map.astype(float)
        self.N: int = info_map.shape[0]
        self.max_budget: float = float(max_budget)

        # Figura matplotlib persistente (se crea en el primer render).
        self._fig: plt.Figure | None = None
        self._ax: plt.Axes | None = None
        self._path: list[tuple[int, int]] = []

    def initial_state(self) -> dict:
        """
        Devuelve el estado inicial: agente en (0, 0) con presupuesto completo.

        La celda de inicio se incluye en `visited` desde el comienzo (actúa
        como depósito de partida). Su valor informativo no se computa como
        recompensa de transición.

        Returns:
            Diccionario con campos 'position', 'budget' y 'visited'.
        """
        start = (0, 0)
        return {
            "position": start,
            "budget":   self.max_budget,
            "visited":  frozenset({start}),
        }

    # ------------------------------------------------------------------ #
    # Implementación de la interfaz Problem                               #
    # ------------------------------------------------------------------ #

    def get_actions(self, state: dict) -> list[str]:
        """
        Devuelve las acciones feasibles desde el estado actual.

        Una acción es feasible si cumple dos condiciones:
          1. La celda destino está dentro de los límites del mapa.
          2. El presupuesto restante cubre el coste euclidiano del paso.

        Args:
            state: Estado actual.

        Returns:
            Lista de nombres de acción aplicables (vacía en estados terminales).
        """
        if self.is_terminal(state):
            return []

        x, y = state["position"]
        budget = state["budget"]
        actions = []

        for action, (dx, dy) in _ACTION_DELTAS.items():
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < self.N
                and 0 <= ny < self.N
                and budget >= _ACTION_COSTS[action]
            ):
                actions.append(action)

        return actions

    def transition(self, state: dict, action: str) -> tuple[dict, float]:
        """
        Aplica la acción al estado y devuelve el estado sucesor y la recompensa.

        El agente se mueve a la celda adyacente, consume el coste euclidiano
        del presupuesto y registra la nueva celda en `visited`.

        Args:
            state:  Estado actual.
            action: Acción a ejecutar.

        Returns:
            (next_state, reward) donde reward es la información recogida.
        """
        reward = self.reward(state, action)
        next_pos = self._next_position(state["position"], action)
        next_state = {
            "position": next_pos,
            "budget":   state["budget"] - _ACTION_COSTS[action],
            "visited":  state["visited"] | frozenset({next_pos}),
        }
        return next_state, reward

    def reward(self, state: dict, action: str) -> float:
        """
        Recompensa inmediata: valor informativo de la celda destino si es
        nueva; cero si ya fue visitada.

        Args:
            state:  Estado actual.
            action: Acción evaluada.

        Returns:
            info_map[y, x] ∈ [0, 1] si la celda es nueva; 0.0 en otro caso.
        """
        next_pos = self._next_position(state["position"], action)
        if next_pos not in state["visited"]:
            x, y = next_pos
            return float(self.info_map[y, x])
        return 0.0

    def is_terminal(self, state: dict) -> bool:
        """
        El estado es terminal cuando el presupuesto es inferior al coste
        mínimo de cualquier paso (1.0, paso ortogonal).

        Args:
            state: Estado a evaluar.

        Returns:
            True si budget < 1.0.
        """
        return state["budget"] < _MIN_COST

    def utility(self, state: dict) -> float:
        """
        Estimación optimista de la información adicional colectable.

        Ordena de mayor a menor los valores de todas las celdas no visitadas
        y suma los primeros k = ⌊budget / min_cost⌋. Esta cota es válida
        como upper-bound porque asume poder visitar k celdas arbitrarias al
        coste mínimo, ignorando la geometría del mapa.

        Se usa como evaluación de nodos hoja en MCTS y Sparse Sampling para
        proporcionar gradiente hacia zonas informativas incluso cuando los
        rollouts no agotan el horizonte.

        Args:
            state: Estado a evaluar.

        Returns:
            Suma de los k mayores valores no visitados (cota superior optimista).
        """
        budget = state["budget"]
        visited = state["visited"]
        max_steps = int(budget / _MIN_COST)

        unvisited_values = sorted(
            (
                float(self.info_map[y, x])
                for y in range(self.N)
                for x in range(self.N)
                if (x, y) not in visited
            ),
            reverse=True,
        )
        return sum(unvisited_values[:max_steps])

    def upper_bound(self, state: dict, action: str) -> float:
        """
        Cota superior Q*(s, a) para Branch and Bound.

        Recompensa inmediata de la acción más la utilidad optimista del
        estado sucesor resultante.

        Args:
            state:  Estado actual.
            action: Acción candidata.

        Returns:
            Cota superior de Q*(s, a).
        """
        immediate = self.reward(state, action)
        next_pos = self._next_position(state["position"], action)
        next_state = {
            "position": next_pos,
            "budget":   state["budget"] - _ACTION_COSTS[action],
            "visited":  state["visited"] | frozenset({next_pos}),
        }
        return immediate + self.utility(next_state)

    def observation(self, state: dict) -> dict:
        """
        En este entorno completamente observable, la observación es el
        estado completo del agente.

        Args:
            state: Estado actual.

        Returns:
            Copia superficial del estado.
        """
        return dict(state)

    def model_transition(self, state: dict, action: str) -> dict:
        """
        Modelo de transición determinista hacia adelante (sin recompensa).

        En este escenario no existe incertidumbre, por lo que el modelo
        coincide exactamente con la dinámica real del entorno.

        Args:
            state:  Estado actual.
            action: Acción a modelar.

        Returns:
            Estado predicho por el modelo.
        """
        next_state, _ = self.transition(state, action)
        return next_state

    def render(self, state: dict) -> None:
        """
        Visualización interactiva matplotlib del estado actual.

        En la primera llamada crea la figura; en las siguientes la actualiza
        sin abrir ventanas nuevas. Muestra:
          · Mapa informativo como fondo (gradiente YlOrRd).
          · Trayectoria acumulada como polilínea blanca semitransparente.
          · Celdas visitadas resaltadas con un overlay azul.
          · Posición actual del agente (marcador blanco con borde negro).
          · Estado del presupuesto y número de celdas visitadas en el título.

        Args:
            state: Estado a visualizar.
        """
        x, y = state["position"]

        # ── Actualizar trayectoria acumulada ──────────────────────────── #
        if not self._path or self._path[-1] != (x, y):
            self._path.append((x, y))

        # ── Inicializar figura en la primera llamada ──────────────────── #
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))

        ax: plt.Axes = self._ax  # type: ignore[assignment]
        ax.clear()

        # ── Fondo: mapa informativo ───────────────────────────────────── #
        ax.imshow(
            self.info_map,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            interpolation="nearest",
        )

        # ── Overlay azul sobre celdas visitadas ───────────────────────── #
        for vx, vy in state["visited"]:
            ax.add_patch(
                patches.Rectangle(
                    (vx - 0.5, vy - 0.5), 1.0, 1.0,
                    linewidth=0,
                    facecolor="steelblue",
                    alpha=0.35,
                )
            )

        # ── Trayectoria como polilínea ────────────────────────────────── #
        if len(self._path) > 1:
            xs, ys = zip(*self._path)
            ax.plot(xs, ys, color="white", linewidth=1.5, alpha=0.8, zorder=2)

        # ── Agente ───────────────────────────────────────────────────── #
        ax.plot(
            x, y,
            marker="o",
            color="white",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=3,
        )

        # ── Decoración ───────────────────────────────────────────────── #
        ax.set_title(
            f"Budget: {state['budget']:.2f}  |  Visitadas: {len(state['visited'])}",
            fontsize=11,
        )
        ax.set_xlim(-0.5, self.N - 0.5)
        ax.set_ylim(self.N - 0.5, -0.5)   # origen arriba-izquierda
        ax.set_xticks(range(self.N))
        ax.set_yticks(range(self.N))
        ax.tick_params(labelsize=7)

        self._fig.canvas.draw()
        plt.pause(1)

    # ------------------------------------------------------------------ #
    # Helpers privados                                                    #
    # ------------------------------------------------------------------ #

    def _next_position(
        self, position: tuple[int, int], action: str
    ) -> tuple[int, int]:
        """
        Calcula la celda destino de la acción sin verificar viabilidad.

        Args:
            position: Posición actual (x, y).
            action:   Acción a aplicar.

        Returns:
            Nueva posición (x + Δx, y + Δy).
        """
        dx, dy = _ACTION_DELTAS[action]
        return (position[0] + dx, position[1] + dy)
