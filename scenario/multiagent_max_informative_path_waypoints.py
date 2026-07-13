"""
Variante waypoint del IPP: las acciones son coordenadas destino (x, y).

A diferencia del escenario base —donde cada acción es un paso unitario en
una de las 8 direcciones—, aquí una acción especifica directamente el
punto de destino al que el agente quiere ir. El agente se desplaza por el
camino de mínimo coste euclidiano (greedy diagonal-primero con 8 dirs.) y
recoge la información de *todas* las celdas intermedias. El estado
resultante es la «foto» del sistema en el momento de llegar al destino.

Esta abstracción reduce el horizonte efectivo del árbol de búsqueda
(pocas acciones de gran alcance) pero ensancha enormemente el factor de
ramificación: en un mapa N×N hay hasta N²−1 destinos posibles por estado.
Para mantener el árbol manejable se puede proporcionar una *función
candidata* que filtre los destinos a considerar en cada nodo.

Acciones:
    tuplas (x, y) — coordenadas de la celda destino.

Estado:
    Idéntico al escenario base:
        position : (x, y)      — celda actual.
        budget   : float        — distancia euclidiana restante.
        visited  : frozenset    — celdas cuya información ya se recogió.

Coste de movimiento:
    _min_path_cost(start, end) = min(|Δx|,|Δy|)·√2 + ||Δx|−|Δy||·1.

Recompensa:
    Suma de info_map[y,x] de todas las celdas NUEVAS del camino,
    incluyendo celdas intermedias y el destino.

Terminal:
    budget < 1.0 (sin suficiente presupuesto para el paso mínimo).
"""

import math
from typing import Any, Callable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # allow imports from scenario/
from policies.algorithms.problem import Problem
from max_informative_path import _MIN_COST, _SQRT2, generate_smooth_map
from collections import defaultdict
# ────────────────────────────────────────────────────────────────────────── #
# Tipos                                                                      #
# ────────────────────────────────────────────────────────────────────────── #

# Una función candidata recibe el estado actual y devuelve una lista de
# posibles destinos (x, y). Permite restringir el espacio de acciones en
# cada nodo del árbol de búsqueda para evitar la explosión horizontal.
CandidateFn = Callable[[dict], list[tuple[int, int]]]


# ────────────────────────────────────────────────────────────────────────── #
# Utilidades de camino                                                       #
# ────────────────────────────────────────────────────────────────────────── #
def _min_path_cost(start: tuple[int, int], end: tuple[int, int]) -> float:
    """
    Coste mínimo del camino entre dos celdas con 8 direcciones.

    Con movimiento en 8 direcciones, el camino óptimo combina pasos
    diagonales (coste √2) para cubrir ambas componentes a la vez, seguidos
    de pasos rectos (coste 1) para la diferencia restante:

        coste = min(|Δx|, |Δy|) · √2 + ||Δx| − |Δy||

    Args:
        start: Posición de origen (x, y).
        end:   Posición de destino (x, y).

    Returns:
        Coste euclidiano mínimo del camino entre start y end.
    """
    dx = abs(end[0] - start[0])
    dy = abs(end[1] - start[1])
    return min(dx, dy) * _SQRT2 + abs(dx - dy)


def agent_priority_heuristic(state: dict) -> list[int]:
    """
    Heurística de prioridad de agentes: ordena por presupuesto restante.

    Los agentes con menos presupuesto restante se planifican primero,
    para evitar que se queden sin opciones mientras otros agentes
    consumen el espacio de acciones.

    Args:
        state: Estado actual con 'budget' y 'position'.

    Returns:
        Lista de índices de agentes ordenados por prioridad.
    """
    budgets = state["budget"]
    return sorted(range(len(budgets)), key=lambda i: budgets[i])


def _ma_min_path_cost(start: tuple[tuple[int, int], ...], end: tuple[tuple[int, int], ...]) -> tuple[float, ...]:
    """
    Coste mínimo del camino entre dos celdas con 8 direcciones.

    Con movimiento en 8 direcciones, el camino óptimo combina pasos
    diagonales (coste √2) para cubrir ambas componentes a la vez, seguidos
    de pasos rectos (coste 1) para la diferencia restante:

        coste = min(|Δx|, |Δy|) · √2 + ||Δx| − |Δy||

    Args:
        start: Posición de origen (x, y).
        end:   Posición de destino (x, y).

    Returns:
        Coste euclidiano mínimo del camino entre start y end.
    """
    costs = []
    for s, e in zip(start, end):
        if isinstance(s, list):
            s = tuple(s)
        if isinstance(e, list):
            e = tuple(e)
        if not (isinstance(s, tuple) and isinstance(e, tuple) and len(s) == 2 and len(e) == 2):
            raise ValueError("start y end deben ser tuplas de coordenadas (x, y).")
        dx = abs(e[0] - s[0])
        dy = abs(e[1] - s[1])
        cost = min(dx, dy) * _SQRT2 + abs(dx - dy)
        costs.append(cost)
    return costs


def _shortest_path(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    Calcula las celdas del camino mínimo de start (exclusive) a end
    (inclusive) usando la estrategia greedy diagonal-primero.

    En cada iteración se avanza en dirección al destino: si quedan
    desplazamientos en ambos ejes se toma un paso diagonal; si quedan
    en un solo eje, un paso recto. La secuencia resultante tiene coste
    igual a _min_path_cost(start, end) y longitud igual al número de
    pasos (no al coste).

    Args:
        start: Celda de origen (no incluida en el resultado).
        end:   Celda de destino (incluida en el resultado).

    Returns:
        Lista de celdas desde start+1 hasta end, en orden de recorrido.
    """
    cells: list[tuple[int, int]] = []
    x, y = start
    tx, ty = end

    while (x, y) != (tx, ty):
        dx = tx - x
        dy = ty - y
        sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        sy = 1 if dy > 0 else (-1 if dy < 0 else 0)
        x, y = x + sx, y + sy
        cells.append((x, y))

    return cells


# ────────────────────────────────────────────────────────────────────────── #
# Escenario                                                                  #
# ────────────────────────────────────────────────────────────────────────── #

class MultiagentMaxInformativePathWaypoints(Problem):
    """
    IPP con espacio de acciones de destinos (waypoints).

    Los agentes parten de la posición inicial y, en cada decisión, eligen una celda destino
    a la que desplazarse recorriendo el camino de menor coste. Durante el
    trayecto acumulan la información de todas las celdas nuevas que atraviesan.

    Para controlar el factor de ramificación del árbol MCTS se puede
    proporcionar una función candidata (``candidate_fn``) que restrinja
    el conjunto de destinos propuestos en cada estado. Si no se proporciona,
    ``get_actions`` enumera todos los destinos alcanzables dentro del
    presupuesto disponible.

    Attributes:
        info_map:      Mapa informativo (N, N) con valores en [0, 1].
        N:             Dimensión de la cuadrícula.
        max_budget:    Presupuesto total de distancia euclidiana.
        anneal_radius: Radio del sensor. Con 0 sólo se recoge la celda
                       atravesada; con r>0 se recogen todas las celdas
                       no visitadas a distancia euclidínea ≤ r.
        candidate_fn:  Función opcional state → list[(x,y)] que propone
                       destinos candidatos. Los destinos inalcanzables
                       dentro del presupuesto se filtran automáticamente.
    """

    def __init__(
        self,
        info_map: np.ndarray,
        max_budget: float | tuple[float, ...] | defaultdict[float],
        candidate_fn: CandidateFn | None = None,
        initial_position: tuple[tuple[int, int], ...] = ((0, 0),),
        gamma: float = 1.0,
        anneal_radius: int = 0,
        neg_reward: float = 0.0,
    ) -> None:
        """
        Inicializa el problema IPP con espacio de acciones waypoint.

        Args:
            info_map:      Mapa informativo cuadrado (N, N).
            max_budget:    Presupuesto total de distancia euclidiana.
            candidate_fn:  Función opcional que propone destinos candidatos.
            gamma:         Factor de descuento.
            anneal_radius: Radio del sensor en celdas (0 = solo la celda
                           atravesada, r > 0 = disco de radio r).

        Raises:
            AssertionError: Si info_map no es cuadrado.
        """
        assert info_map.ndim == 2 and info_map.shape[0] == info_map.shape[1], (
            "info_map debe ser cuadrado (N×N)."
        )
        self.info_map: np.ndarray = info_map.astype(float)
        self.N: int = info_map.shape[0]
        self.max_budget: float | tuple[float, ...] | defaultdict[float] = (
            [float(max_budget)] * len(initial_position)
            if isinstance(max_budget, (int, float))
            else max_budget 
        )
        self.candidate_fn: CandidateFn | None = candidate_fn
        self.gamma = gamma
        self.anneal_radius: int = max(0, anneal_radius)
        self.initial_position: tuple[tuple[int, int], ...] = initial_position
        self.neg_reward = neg_reward
        # Precalcular el vecindario de cada celda para el radio dado.
        # _anneal_map[(x,y)] es el frozenset de celdas dentro del disco.
        self._anneal_map: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
        for cx in range(self.N):
            for cy in range(self.N):
                if self.anneal_radius == 0:
                    self._anneal_map[(cx, cy)] = frozenset({(cx, cy)})
                else:
                    nbrs = {
                        (nx, ny)
                        for dx in range(-self.anneal_radius, self.anneal_radius + 1)
                        for dy in range(-self.anneal_radius, self.anneal_radius + 1)
                        if math.sqrt(dx * dx + dy * dy) <= self.anneal_radius
                        for nx, ny in [(cx + dx, cy + dy)]
                        if 0 <= nx < self.N and 0 <= ny < self.N
                    }
                    self._anneal_map[(cx, cy)] = frozenset(nbrs)

        # Estado interno de la figura matplotlib (render interactivo).
        self._fig: plt.Figure | None = None
        self._ax: plt.Axes | None = None
        self._waypoints: list[tuple[int, int]] = []

    def initial_state(self) -> dict:
        """
        Devuelve el estado inicial: agentes en initial position con presupuesto completo.

        La celda de inicio y su vecindario (según anneal_radius) se marcan
        como visitados desde el principio y no generan recompensa.

        Returns:
            Diccionario con campos 'position', 'budget' y 'visited'.
        """
        start = self.initial_position.copy()
        return {
            "position": start,
            "budget":   self.max_budget.copy(),
            "visited":  frozenset().union(*(self._anneal_map[(pos[0], pos[1])] for pos in start)),
            "last_position": start.copy(),
            "priority": agent_priority_heuristic({"budget": self.max_budget.copy()})
        }

    # ------------------------------------------------------------------ #
    # Implementación de la interfaz Problem                               #
    # ------------------------------------------------------------------ #

    def get_actions(self, state: dict) -> list[tuple[int, int]]:
        """
        Devuelve los destinos feasibles desde el estado actual.

        Si se ha configurado una ``candidate_fn``, se delega la propuesta
        de destinos en ella y se filtran los inalcanzables. Sin función
        candidata, se enumeran todas las celdas del mapa alcanzables con
        el presupuesto disponible (excepto la posición actual).

        Args:
            state: Estado actual.

        Returns:
            Lista de coordenadas (x, y) de destinos alcanzables.
        """
        if self.is_terminal(state):
            return []
        agent = state["priority"][0]  # agente con mayor prioridad
        pos = state["position"][agent]
        budget = state["budget"][agent]

        if self.candidate_fn is not None:
            return [
                dest for dest in self.candidate_fn(state)
                if dest != pos
                and _min_path_cost(pos, dest) <= budget
            ]

        # Sin función candidata: todos los destinos alcanzables.
        return [
            (x, y)
            for y in range(self.N)
            for x in range(self.N)
            if (x, y) != pos
            and _min_path_cost(pos, (x, y)) <= budget
        ]

    def transition(self, state: dict, action: tuple[int, int]) -> tuple[dict, float]:
        """
        Mueve el agente hasta el destino siguiendo el camino mínimo.

        Por cada celda nueva del trayecto recoge la información de todas
        las celdas no visitadas dentro de ``anneal_radius`` y las marca
        como visitadas.

        Args:
            state:  Estado actual.
            action: Celda destino (x, y).

        Returns:
            (next_state, total_reward).
        """
        agent = state["priority"][0]  # agente con mayor prioridad
        path = _shortest_path(state["position"][agent], action)
        visited = state["visited"]
        total_reward = 0.0
        last_position = state["position"].copy()

        for pos in path:
            if pos not in visited:
                for nbr in self._anneal_map[pos]:
                    if nbr not in visited:
                        nx, ny = nbr
                        total_reward += float(self.info_map[ny, nx]) - np.abs(self.neg_reward)
                        visited = visited | frozenset({nbr})

        cost = _min_path_cost(state["position"][agent], action)
        new_budget = list(state["budget"])
        new_budget[agent] -= cost
        if len(state["priority"]) > 1:
            priority = state["priority"][1:]
        else:
            priority = agent_priority_heuristic(state)
        new_position = list(state["position"])
        new_position[agent] = action

        next_state = {
            "position": new_position,
            "budget":   new_budget,
            "visited":  visited,
            "last_position": last_position,
            "priority": priority
        }
        return next_state, total_reward - np.abs(self.neg_reward)

    def reward(self, state: dict, action: tuple[int, int]) -> float:
        """
        Recompensa del trayecto hacia el destino, respetando anneal_radius.

        Args:
            state:  Estado actual.
            action: Celda destino (x, y).

        Returns:
            Suma de info_map de las celdas nuevas del trayecto y su vecindario.
        """
        agent = state["priority"][0]  # agente con mayor prioridad
        path = _shortest_path(state["position"][agent], action)
        visited = state["visited"]
        total = 0.0
        for pos in path:
            if pos not in visited:
                for nbr in self._anneal_map[pos]:
                    if nbr not in visited:
                        nx, ny = nbr
                        total += float(self.info_map[ny, nx]) 
                        visited = visited | frozenset({nbr})
        return total - np.abs(self.neg_reward)

    def is_terminal(self, state: dict) -> bool:
        """
        El estado es terminal cuando el presupuesto no cubre el paso mínimo.

        Args:
            state: Estado a evaluar.

        Returns:
            True si budget < 1.0.
        """
        return any(budget < _MIN_COST for budget in state["budget"])
    def is_action_node(self, state: dict) -> bool:
        """
        un nodo es de acción si no hay agentes que aún no hayan seleccionado acción
        """
        return len(state["priority"]) == len(state["budget"])
    def utility(self, state: dict) -> float:
        """
        Cota superior optimista de la información adicional colectable.

        Se calcula como la suma del valor de las celdas dentro de un radio igual al presupuesto restante
        """
        agent = state["priority"][0]  # agente con mayor prioridad
        x, y = state["position"][agent]
        budget = state["budget"][agent]
        max_distance = int(math.ceil(budget))
        total = 0.0
        for dy in range(-max_distance, max_distance + 1):
            for dx in range(-max_distance, max_distance + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.N and 0 <= ny < self.N:
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance <= budget and (nx, ny) not in state["visited"]:
                        total += float(self.info_map[ny, nx])
        return total
    
    def upper_bound(self, state: dict, action: tuple[int, int]) -> float:
        """
        Cota superior Q*(s, a) para Branch-and-Bound.

        Recompensa inmediata del trayecto más la utilidad optimista del
        estado sucesor.

        Args:
            state:  Estado actual.
            action: Destino candidato.

        Returns:
            Cota superior de Q*(s, a).
        """
        immediate = self.reward(state, action)
        agent = state["priority"][0]  # agente con mayor prioridad
        path = _shortest_path(state["position"][agent], action)
        cost = _min_path_cost(state["position"][agent], action)
        next_visited = state["visited"] | frozenset(path)
        new_budget = list(state["budget"])
        new_budget[agent] -= cost
        next_state = {
            "position": state["position"],
            "budget":   new_budget,
            "visited":  next_visited,
            "last_position" : state["position"]
        }
        return immediate + self.gamma * self.utility(next_state)

    def observation(self, state: dict) -> dict:
        """
        Entorno completamente observable: la observación es el estado completo.

        Args:
            state: Estado actual.

        Returns:
            Copia superficial del estado.
        """
        return dict(state)

    def model_transition(self, state: dict, action: tuple[int, int]) -> dict:
        """
        Modelo de transición determinista (sin recompensa).

        Args:
            state:  Estado actual.
            action: Destino.

        Returns:
            Estado predicho.
        """
        next_state, _ = self.transition(state, action)
        return next_state

    # ------------------------------------------------------------------ #
    # Funciones candidatas (factories)                                    #
    # ------------------------------------------------------------------ #

    def candidates_top_k(self, k: int = 5) -> "CandidateFn":
        """
        k destinos no visitados muestreados con probabilidad ∝ info/dist.

        Favorece celdas muy informativas y cercanas. Si hay k o menos
        candidatos los devuelve todos sin muestreo.
        """
        def candidate_fn(state: dict) -> list[tuple[int, int]]:
            agent = state["priority"][0]  # agente con mayor prioridad
            pos = state["position"][agent]
            visited = state["visited"]
            scored = []
            for x in range(self.N):
                for y in range(self.N):
                    if (x, y) in visited:
                        continue
                    dist = math.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
                    scored.append(((x, y), self.info_map[y, x] / (dist + 1e-5)))
            if len(scored) <= k:
                return [dest for dest, _ in scored]
            scores = np.array([s for _, s in scored])
            probs = scores / scores.sum()
            idx = np.random.choice(len(scored), size=k, replace=False, p=probs)
            return [scored[i][0] for i in idx]

        return candidate_fn

    def candidates_adaptive(
        self, min_resolution: int = 2, max_resolution: int = 10
    ) -> "CandidateFn":
        """
        8 vecinos con paso proporcional al valor de info en la posición actual.

        Mucha info → paso corto (exploración local).
        Poca info  → paso largo (salto hacia zonas más ricas).
        """
        info_min = float(self.info_map.min())
        info_range = float(self.info_map.max() - info_min) or 1.0

        def candidate_fn(state: dict) -> list[tuple[int, int]]:
            agent = state["priority"][0]  # agente con mayor prioridad
            x, y = state["position"][agent]
            ratio = (float(self.info_map[y, x]) - info_min) / info_range
            
            if ratio > 0.8:
                step = min_resolution
            else:
                step = max_resolution - round((max_resolution - min_resolution) * ratio)
            candidates = []
            for dx in (-step, 0, step):
                for dy in (-step, 0, step):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.N and 0 <= ny < self.N:
                        if (nx,ny) != (state["last_position"][agent]):
                            candidates.append((nx, ny))
            return candidates

        return candidate_fn

    def candidates_8grid(self, resolution: int = 1) -> "CandidateFn":
        """8 vecinos (diagonales incluidas) a paso fijo `resolution`."""
        def candidate_fn(state: dict) -> list[tuple[int, int]]:
            agent = state["priority"][0]  # agente con mayor prioridad
            x, y = state["position"][agent]
            candidates = []
            for dx in (-resolution, 0, resolution):
                for dy in (-resolution, 0, resolution):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.N and 0 <= ny < self.N:
                        candidates.append((nx, ny))
            return candidates

        return candidate_fn

    def candidates_subgrid(self, resolution: int = 2) -> "CandidateFn":
        """Todos los puntos del mapa a paso `resolution` (grid regular)."""
        def candidate_fn(state: dict) -> list[tuple[int, int]]:
            agent = state["priority"][0]  # agente con mayor prioridad
            pos = state["position"][agent]
            return [
                (x, y)
                for x in range(0, self.N, resolution)
                for y in range(0, self.N, resolution)
                if (x, y) != pos
            ]

        return candidate_fn
    

    # ------------------------------------------------------------------ #
    # Visualización                                                        #
    # ------------------------------------------------------------------ #

    def render(self, state: dict) -> None:
        """
        Visualización interactiva matplotlib del estado actual.

        En la primera llamada crea la figura; en las siguientes la actualiza
        sin abrir ventanas nuevas. Muestra:

          · Mapa informativo como fondo (gradiente YlOrRd).
          · Celdas visitadas resaltadas con overlay azul.
          · Camino completo reconstruido (celda a celda) desde los waypoints.
          · Waypoints intermedios como marcadores cuadrados numerados.
          · Posición actual del agente (estrella blanca con borde negro).
          · Presupuesto y número de celdas visitadas en el título.

        Args:
            state: Estado a visualizar.
        """
        agent = state["priority"][0]  # agente con mayor prioridad
        x, y = state["position"][agent]

        # ── Actualizar historial de waypoints ────────────────────────── #
        if not self._waypoints or self._waypoints[-1] != (x, y):
            self._waypoints.append((x, y))

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
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            interpolation="nearest",
        )

        # ── Overlay azul sobre celdas visitadas ───────────────────────── #
        agent = state["priority"][0]  # agente con mayor prioridad
        for vx, vy in state["visited"][agent]:
            ax.add_patch(
                patches.Rectangle(
                    (vx - 0.5, vy - 0.5), 1.0, 1.0,
                    linewidth=0,
                    facecolor="steelblue",
                    alpha=0.35,
                )
            )

        # ── Camino completo reconstruido desde waypoints ──────────────── #
        if len(self._waypoints) > 1:
            full: list[tuple[int, int]] = [self._waypoints[0]]
            for i in range(1, len(self._waypoints)):
                full.extend(_shortest_path(self._waypoints[i - 1], self._waypoints[i]))
            xs, ys = zip(*full)
            ax.plot(xs, ys, color="white", linewidth=1.5, alpha=0.8, zorder=2)

        # ── Waypoints intermedios como cuadrados numerados ────────────── #
        for i, (wx, wy) in enumerate(self._waypoints[:-1]):
            ax.plot(
                wx, wy,
                marker="s",
                color="lightblue",
                markersize=7,
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=3,
            )
            ax.text(wx + 0.15, wy - 0.15, str(i), color="white", fontsize=6, zorder=4)

        # ── Agente (posición actual) ──────────────────────────────────── #
        ax.plot(
            x, y,
            marker="*",
            color="white",
            markersize=13,
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=5,
        )

        # ── Decoración ───────────────────────────────────────────────── #
        ax.set_title(
            f"Budget: {state['budget'][agent]:.2f}  |  Visitadas: {len(state['visited'][agent])}",
            fontsize=11,
        )
        ax.set_xlim(-0.5, self.N - 0.5)
        ax.set_ylim(self.N - 0.5, -0.5)
        ax.set_xticks(range(self.N))
        ax.set_yticks(range(self.N))
        ax.tick_params(labelsize=7)

        self._fig.canvas.draw()
        plt.pause(0.05)
        
        return ax
