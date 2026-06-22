"""
Escenario de navegación en cuadrícula (Grid World).

Un robot equipado con una batería limitada debe encontrar el camino desde
su posición inicial hasta una meta en una cuadrícula bidimensional. El
entorno contiene paredes que bloquean el paso. El robot percibe su posición
y nivel de batería en todo momento (entorno completamente observable).

Estado:
    position : (x, y) — coordenadas del robot en la cuadrícula.
    battery  : int    — unidades de batería restantes.

Sistema de coordenadas:
    El origen (0, 0) se sitúa en la esquina superior-izquierda.
    El eje x crece hacia la derecha; el eje y crece hacia abajo.

Acciones:
    "move_up"    — desplaza el robot una celda hacia arriba    (y − 1).
    "move_down"  — desplaza el robot una celda hacia abajo     (y + 1).
    "move_left"  — desplaza el robot una celda hacia la izquierda (x − 1).
    "move_right" — desplaza el robot una celda hacia la derecha   (x + 1).

Dinámica:
    Cada movimiento consume una unidad de batería. Si el robot intenta
    moverse a una celda ocupada por una pared o fuera de los límites, su
    posición no cambia (pero la batería se consume igualmente).

Recompensas:
    +10  al alcanzar la celda objetivo.
    − 1  por cada paso ordinario.

Condición de terminal:
    - El robot alcanza la celda objetivo.
    - La batería llega a cero.
"""

from typing import Any

from onlineplanning.algorithms.problem import Problem


# Desplazamientos (Δx, Δy) para cada acción.
_ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "move_up":    ( 0, -1),
    "move_down":  ( 0, +1),
    "move_left":  (-1,  0),
    "move_right": (+1,  0),
}

# Conjunto ordenado de todas las acciones del dominio.
_ALL_ACTIONS: list[str] = list(_ACTION_DELTAS.keys())


class GridWorld(Problem):
    """
    Problema de navegación de un robot en una cuadrícula con obstáculos.

    El agente opera en una cuadrícula de width×height celdas. Su objetivo
    es alcanzar la celda goal consumiendo la menor cantidad de batería
    posible (equivalente a minimizar el número de pasos), evitando las
    paredes que bloquean el movimiento.

    El entorno es determinista: dada una acción, el estado sucesor está
    completamente determinado por el estado actual.

    Attributes:
        width:           Número de columnas de la cuadrícula.
        height:          Número de filas de la cuadrícula.
        goal:            Coordenadas (x, y) de la celda objetivo.
        initial_battery: Unidades de batería al inicio del episodio.
        walls:           Conjunto de coordenadas (x, y) bloqueadas.
    """

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        goal: tuple[int, int] = (4, 4),
        initial_battery: int = 20,
        walls: set[tuple[int, int]] | None = None,
    ) -> None:
        """
        Inicializa el entorno de cuadrícula.

        Args:
            width:           Número de columnas.
            height:          Número de filas.
            goal:            Celda objetivo (x, y).
            initial_battery: Batería inicial del robot.
            walls:           Conjunto de celdas bloqueadas. Si es None se
                             usa la configuración por defecto del escenario.
        """
        self.width = width
        self.height = height
        self.goal = goal
        self.initial_battery = initial_battery
        self.walls: set[tuple[int, int]] = walls if walls is not None else set()

    def initial_state(self) -> dict:
        """
        Devuelve el estado inicial canónico del episodio.

        El robot parte de la esquina superior-izquierda con la batería
        completamente cargada.

        Returns:
            Diccionario con los campos 'position' y 'battery'.
        """
        return {"position": (0, 0), "battery": self.initial_battery}

    # ------------------------------------------------------------------ #
    # Implementación de la interfaz Problem                               #
    # ------------------------------------------------------------------ #

    def get_actions(self, state: dict) -> list[str]:
        """
        Devuelve las acciones disponibles desde el estado dado.

        Las cuatro acciones direccionales están siempre disponibles en
        estados no terminales. Desde estados terminales no se puede actuar.

        Args:
            state: El estado actual del sistema.

        Returns:
            Lista de nombres de acción, o lista vacía en estados terminales.
        """
        if self.is_terminal(state):
            return []
        return list(_ALL_ACTIONS)

    def transition(self, state: dict, action: str) -> tuple[dict, float]:
        """
        Aplica la acción al estado y devuelve el estado sucesor y la recompensa.

        El movimiento consume una unidad de batería. Si la celda destino es
        una pared o está fuera de límites, el robot permanece en su celda
        actual (la batería se reduce igualmente).

        Args:
            state:  El estado actual con campos 'position' y 'battery'.
            action: Una de las cuatro acciones direccionales.

        Returns:
            Una tupla (next_state, reward).
        """
        next_position = self._next_position(state["position"], action)
        next_state = {
            "position": next_position,
            "battery":  state["battery"] - 1,
        }
        return next_state, self.reward(state, action)

    def reward(self, state: dict, action: str) -> float:
        """
        Devuelve la recompensa inmediata por tomar la acción desde el estado.

        Se otorga una recompensa de +10 si el movimiento conduce a la celda
        objetivo; −1 en cualquier otro caso.

        Args:
            state:  El estado actual.
            action: La acción ejecutada.

        Returns:
            +10.0 al alcanzar el objetivo; −1.0 en otro caso.
        """
        if self._next_position(state["position"], action) == self.goal:
            return +10.0
        return -1.0

    def is_terminal(self, state: dict) -> bool:
        """
        Determina si el estado es terminal.

        El episodio termina cuando el robot alcanza la meta o agota su batería.

        Args:
            state: El estado a evaluar.

        Returns:
            True si position == goal o battery <= 0.
        """
        return state["position"] == self.goal or state["battery"] <= 0

    def utility(self, state: dict) -> float:
        """
        Cota inferior U(s) ≤ V*(s) basada en la distancia de Manhattan.

        El robot necesita al menos manhattan_distance(s, goal) pasos para
        alcanzar el objetivo, cada uno con coste −1. En el peor caso nunca
        llega, por lo que la cota inferior más ajustada es:

            U(s) = −manhattan_distance(s, goal)

        Esta es una cota inferior válida: V*(s) ≥ −dist siempre que la
        recompensa en la meta (+10) sea positiva y las paredes no obliguen
        a dar más pasos del mínimo Manhattan.

        Args:
            state: El estado a evaluar.

        Returns:
            Negativo de la distancia Manhattan hasta la celda objetivo.
        """
        x, y = state["position"]
        gx, gy = self.goal
        return -float(abs(x - gx) + abs(y - gy))

    def upper_bound(self, state: dict, action: str) -> float:
        """
        Cota superior U(s, a) ≥ Q*(s, a) basada en el mejor caso optimista.

        Tras ejecutar la acción a desde s, el robot llega a s'. Asumiendo
        de forma optimista que desde s' puede alcanzar el objetivo en
        exactamente manhattan_distance(s', goal) pasos sin obstáculos, la
        cota superior del retorno acumulado con descuento es:

            U(s, a) = R(s, a)
                    + γ   · (−1)
                    + γ²  · (−1)
                    + …
                    + γ^d · (+10)

        donde d = manhattan_distance(s', goal). Esta suma geométrica
        representa el retorno del camino óptimo sin obstáculos.

        Si el robot está a 0 pasos del objetivo (ya está en la meta tras
        la acción), U(s, a) = R(s, a) directamente.

        Args:
            state:  El estado actual.
            action: La acción a evaluar.

        Returns:
            Cota superior escalar de Q*(s, a).
        """
        # Recompensa inmediata de la acción.
        immediate = self.reward(state, action)

        # Estado sucesor según la dinámica real.
        next_state, _ = self.transition(state, action)
        nx, ny = next_state["position"]
        gx, gy = self.goal
        dist = abs(nx - gx) + abs(ny - gy)

        if dist == 0:
            # El robot ya está en la meta: no hay retorno futuro.
            return immediate

        # Suma de los costes de pasos intermedios (geométrica) más la
        # recompensa de la meta, todos descontados desde el paso siguiente.
        # step_cost_total = Σ_{k=1}^{dist-1} γ^k · (−1)
        # goal_reward     = γ^dist · 10
        # (Usamos γ = 1.0 de forma conservadora para obtener una cota válida
        #  sin conocer γ en tiempo de diseño; la subclase puede refinarlo.)
        gamma = getattr(self, "_gamma_hint", 1.0)
        step_costs = sum(-(gamma ** k) for k in range(1, dist))
        goal_value = (gamma ** dist) * 10.0

        return immediate + step_costs + goal_value

    def observation(self, state: dict) -> dict:
        """
        Devuelve la observación del agente, que en este entorno completamente
        observable coincide con el estado completo.

        Args:
            state: El estado actual.

        Returns:
            Diccionario con 'position' y 'battery'.
        """
        return {"position": state["position"], "battery": state["battery"]}

    def model_transition(self, state: dict, action: str) -> dict:
        """
        Predice el siguiente estado según el modelo interno del agente.

        Este modelo optimista ignora las paredes: asume que cualquier
        movimiento es siempre posible dentro de los límites de la cuadrícula.
        Puede usarse como cota superior del alcance del robot.

        Args:
            state:  El estado actual.
            action: La acción a modelar.

        Returns:
            Estado predicho (sin comprobar colisiones con paredes).
        """
        dx, dy = _ACTION_DELTAS[action]
        x, y = state["position"]
        nx = max(0, min(self.width - 1,  x + dx))
        ny = max(0, min(self.height - 1, y + dy))
        return {"position": (nx, ny), "battery": state["battery"]}

    def render(self, state: dict) -> str:
        """
        Genera una representación ASCII de la cuadrícula con el estado actual.

        Leyenda:
            R  — posición actual del robot.
            G  — celda objetivo.
            #  — pared (celda bloqueada).
            .  — celda libre.

        Args:
            state: El estado a representar.

        Returns:
            Cadena multilínea con la cuadrícula y el nivel de batería.
        """
        border = "+" + "-" * (self.width * 2 + 1) + "+"
        rows = [border]

        for y in range(self.height):
            cells = []
            for x in range(self.width):
                pos = (x, y)
                if pos == state["position"]:
                    cells.append("R")
                elif pos == self.goal:
                    cells.append("G")
                elif pos in self.walls:
                    cells.append("#")
                else:
                    cells.append(".")
            rows.append("| " + " ".join(cells) + " |")

        rows.append(border)
        rows.append(f"  Battery: {state['battery']}")
        return "\n".join(rows)

    # ------------------------------------------------------------------ #
    # Helpers privados                                                    #
    # ------------------------------------------------------------------ #

    def _next_position(
        self, position: tuple[int, int], action: str
    ) -> tuple[int, int]:
        """
        Calcula la posición resultante de aplicar la acción, respetando los
        límites de la cuadrícula y las paredes.

        Si la celda destino está fuera de los límites o es una pared, el
        robot permanece en su posición actual.

        Args:
            position: La posición actual (x, y).
            action:   La acción a aplicar.

        Returns:
            La nueva posición (x', y').
        """
        dx, dy = _ACTION_DELTAS[action]
        nx = position[0] + dx
        ny = position[1] + dy

        # Comprobar límites de la cuadrícula.
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return position

        # Comprobar colisión con una pared.
        if (nx, ny) in self.walls:
            return position

        return (nx, ny)
