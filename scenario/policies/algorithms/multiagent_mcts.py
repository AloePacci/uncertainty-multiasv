"""
Monte Carlo Tree Search con UCT — Algoritmo 4.9 (MCTS / UCT).

MCTS construye de forma incremental un árbol de búsqueda utilizando el
modelo generativo del problema. A diferencia de Sparse Sampling —que trata
cada llamada de forma independiente—, MCTS acumula estadísticas entre
simulaciones y reutiliza la información de iteraciones anteriores para
guiar la exploración futura.

El algoritmo consiste en cuatro fases que se repiten por cada simulación:

  1. Selección:     Desde la raíz, se recorre el árbol eligiendo en cada
                    nodo la acción que maximiza el criterio UCB1 (Upper
                    Confidence Bound), que equilibra explotación de acciones
                    con alto valor empírico y exploración de acciones poco
                    visitadas.

  2. Expansión:     Cuando se alcanza un nodo hoja (estado nunca visitado),
                    se inicializa su entrada en el árbol con estadísticas a
                    cero y se realiza un rollout desde él.

  3. Simulación:    Se estima el valor del estado hoja mediante un rollout:
                    simulación con política aleatoria hasta el horizonte
                    máximo o un estado terminal.

  4. Retropropagación: El valor obtenido en la simulación se propaga hacia
                    atrás actualizando los contadores de visitas N(s, a) y
                    los valores medios Q(s, a) de cada par (estado, acción)
                    en el camino raíz–hoja.

Criterio UCB1 (bandit) aplicado a árboles (UCT):

    UCB1(s, a) = Q(s, a)  +  c · √(ln N(s) / N(s, a))

donde:
    Q(s, a) es la media incremental de retornos del par (s, a).
    N(s, a) es el número de veces que se ha ejecutado a desde s.
    N(s)    es el número total de visitas al estado s.
    c       es el parámetro de exploración (por defecto √2).

Estructura de árbol:
    Cada nodo del árbol es un objeto MCTSNode que almacena directamente
    las estadísticas N(s,·) y Q(s,·) del estado asociado, junto con
    referencias explícitas a sus hijos. Esto evita la conversión de estado
    a clave hashable en cada visita y la sobrecarga de los defaultdict de
    la implementación anterior basada en diccionarios globales.

Reutilización del árbol (reuse_tree=True):
    Cuando se activa, tras cada llamada a select_action el planificador
    avanza la raíz interna al hijo correspondiente a la acción elegida.
    La llamada siguiente arranca con todas las estadísticas acumuladas en
    ese subárbol, reduciendo el número de simulaciones necesarias para
    alcanzar buenas estimaciones.

Referencia:
    Kochenderfer, M. J. (2015). Decision Making Under Uncertainty:
    Theory and Application. MIT Press. Capítulo 4 (MCTS / UCT).

    Kocsis, L., & Szepesvári, C. (2006). Bandit Based Monte-Carlo Planning.
    ECML 2006. LNCS 4212, pp. 282–293.
"""

import math
import random
from typing import Any, Callable, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # allow imports from scenario/
from problem import Problem

# Tipo de una política de rollout: recibe un estado y devuelve una acción o None.
RolloutPolicy = Callable[[dict], Any]


# ──────────────────────────────────────────────────────────────────────────── #
# Nodo del árbol                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

class MCTSNode:
    """
    Nodo del árbol de búsqueda MCTS.

    Almacena las estadísticas UCB1 del estado asociado y referencias
    directas a los hijos ya expandidos. El uso de __slots__ reduce la
    huella de memoria y acelera el acceso a atributos en comparación con
    el __dict__ por defecto de Python.

    Attributes:
        state:        Diccionario de estado asociado a este nodo.
        parent:       Nodo padre (None para la raíz).
        action_taken: Acción que llevó desde el padre a este nodo.
        children:     Mapa acción → nodo hijo ya creado en el árbol.
        N:            Mapa acción → número de visitas N(s, a).
        Q:            Mapa acción → media incremental de retornos Q(s, a).
        N_total:      Número total de visitas al nodo, Σ_a N(s, a).
        expanded:     True si el nodo ya fue expandido (ha tenido al menos
                      un rollout y las siguientes visitas usarán UCB1).
        cached_actions: Conjunto de acciones fijado en el primer paso UCB1.
                      Congela el espacio de acciones del nodo para evitar
                      que funciones candidatas estocásticas generen un árbol
                      de anchura infinita (el árbol sólo crecería en anchura
                      nunca en profundidad si se re-muestreara cada vez).
    """

    __slots__ = (
        "state", "parent", "action_taken",
        "children", "N", "Q", "N_total", "expanded", "cached_actions",
    )

    def __init__(
        self,
        state: dict,
        parent: Optional["MCTSNode"] = None,
        action_taken: Any = None,
    ) -> None:
        self.state: dict = state
        self.parent: Optional["MCTSNode"] = parent
        self.action_taken: Any = action_taken
        self.children: dict[Any, "MCTSNode"] = {}
        self.N: dict[Any, int] = {}
        self.Q: dict[Any, float] = {}
        self.N_total: int = 0
        self.expanded: bool = False
        self.cached_actions: list | None = None



# ──────────────────────────────────────────────────────────────────────────── #
# Planificador MCTS                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

class MAMCTS:
    """
    Selección de acciones online mediante Monte Carlo Tree Search (UCT).

    El árbol se construye de forma incremental a lo largo de n_simulations
    simulaciones. Cada nodo del árbol es un MCTSNode que almacena las
    estadísticas de visita directamente, evitando la conversión de estado
    a clave hashable presente en la implementación basada en defaultdict.

    Con reuse_tree=True, el árbol persiste entre llamadas a select_action:
    tras devolver la acción elegida, la raíz interna avanza al nodo hijo
    correspondiente. La siguiente planificación parte de ese subárbol con
    estadísticas ya acumuladas.

    Attributes:
        problem:         La instancia del MDP con su modelo generativo.
        n_simulations:   Número de simulaciones (iteraciones) por decisión.
        depth:           Horizonte máximo del árbol y de los rollouts.
        gamma:           Factor de descuento γ ∈ (0, 1].
        exploration_c:   Constante de exploración c en UCB1 (por defecto √2).
        rollout_policy:  Política π₀ usada en la fase de rollout.
        reuse_tree:      Si True, reutiliza el subárbol de la acción elegida
                         en la siguiente llamada a select_action.
        control_horizon: Número de acciones a extraer por cada replanificación.
                         Con control_horizon=1 se replanifica en cada paso.
                         Con control_horizon>1 se planifica una vez y se
                         ejecutan las siguientes acciones del árbol sin
                         recalcular (open-loop) hasta agotar la cola.
    """

    def __init__(
        self,
        problem: Problem,
        n_simulations: int = 500,
        depth: int = 20,
        gamma: float = 1.0,
        exploration_c: float = math.sqrt(2),
        rollout_policy: Optional[RolloutPolicy] = None,
        reuse_tree: bool = False,
        control_horizon: int = 1,
    ) -> None:
        self.problem = problem
        self.n_simulations = n_simulations
        self.depth = depth
        self.gamma = gamma
        self.exploration_c = exploration_c
        self.rollout_policy: RolloutPolicy = (
            rollout_policy
            if rollout_policy is not None
            else lambda s: random.choice(self.problem.get_actions(s))
        )
        self.reuse_tree = reuse_tree
        self.control_horizon: int = max(1, control_horizon)
        self._root: Optional[MCTSNode] = None
        # Raíz de la última decisión (ANTES de avanzar al hijo elegido).
        # Permite renderizar el árbol completo desde el punto de decisión.
        self._decision_root: Optional[MCTSNode] = None
        # Cola de acciones pendientes cuando control_horizon > 1.
        # Cada entrada es (acción, nodo_siguiente) para poder avanzar _root
        # con reuse_tree aunque no se replantifique en ese paso.
        self._action_queue: list[tuple[Any, Optional[MCTSNode]]] = []

    # ------------------------------------------------------------------ #
    # Interfaz pública                                                    #
    # ------------------------------------------------------------------ #

    @property
    def root(self) -> Optional[MCTSNode]:
        """Raíz del árbol construido en la última llamada a select_action."""
        return self._root

    @property
    def decision_root(self) -> Optional[MCTSNode]:
        """Raíz del árbol en el punto de la última decisión (antes de avanzar).

        Con reuse_tree=True, self.root apunta al hijo elegido; esta propiedad
        conserva el nodo del estado desde el que se tomó la decisión, que
        contiene todas las estadísticas N(s,a) y Q(s,a) exploradas.
        """
        return self._decision_root

    def select_action(self, state: dict) -> tuple[Any, float]:
        """
        Devuelve la acción de mayor valor estimado y su valor Q empírico.

        Si control_horizon=1 (por defecto) replanifica en cada llamada.
        Si control_horizon>1, la primera llamada construye el árbol y llena
        una cola interna con las siguientes `control_horizon` acciones del
        camino greedy; las llamadas siguientes consumen la cola sin
        replanificar (open-loop) hasta agotarla, momento en que se vuelve
        a planificar.

        Args:
            state: El estado actual del sistema.

        Returns:
            Una tupla (a*, v*) donde a* es la acción recomendada y v* es
            su valor empírico Q(s, a*). El valor es 0.0 para acciones
            servidas desde la cola (no hay estimación Q del estado actual).
        """
        # ── Servir desde la cola si quedan acciones pendientes ─────────── #
        if self._action_queue:
            action, next_node = self._action_queue.pop(0)
            if self.reuse_tree and next_node is not None:
                next_node.parent = None
                self._root = next_node
            return (action, 0.0)

        # ── Determinar la raíz de esta planificación ──────────────────── #
        if (
            self.reuse_tree
            and self._root is not None
            and self._root.state == state
        ):
            root = self._root
        else:
            root = MCTSNode(state)

        # ── Simulaciones ──────────────────────────────────────────────── #
        for _ in range(self.n_simulations):
            self._simulate(root, self.depth)

        # ── Selección final: arg max_a Q(s, a) ────────────────────────── #
        # IMPORTANTE: usar cached_actions cuando estén disponibles.
        actions = root.cached_actions if root.cached_actions is not None \
            else self.problem.get_actions(state)
        if not actions:
            self._decision_root = root
            self._root = root
            return (None, 0.0)
        best_action = max(actions, key=lambda a: root.Q.get(a, float("-inf"))/ (root.N.get(a, 0) or 1))
        best_value = root.Q.get(best_action, 0.0)

        # ── Guardar raíz de decisión ANTES de avanzar ─────────────────── #
        self._decision_root = root

        # ── Rellenar cola con las acciones restantes del horizonte ─────── #
        if self.control_horizon > 1:
            node = root.children.get(best_action)
            for _ in range(self.control_horizon - 1):
                if node is None:
                    break
                acts = node.cached_actions if node.cached_actions is not None \
                    else self.problem.get_actions(node.state)
                if not acts:
                    break
                next_action = max(acts, key=lambda a: node.Q.get(a, float("-inf")/ (node.N.get(a, 0) or 1)))
                next_node = node.children.get(next_action)
                self._action_queue.append((next_action, next_node))
                node = next_node

        # ── Avanzar la raíz para la próxima replanificación ───────────── #
        if self.reuse_tree and best_action in root.children:
            self._root = root.children[best_action]
            self._root.parent = None
        else:
            self._root = root

        return (best_action, best_value)

    # ------------------------------------------------------------------ #
    # Fases internas de MCTS                                              #
    # ------------------------------------------------------------------ #

    def _simulate(self, node: MCTSNode, depth: int) -> float:
        """
        Ejecuta una simulación completa (selección → expansión → rollout →
        retropropagación) y devuelve el retorno estimado desde el nodo.

        Args:
            node:  Nodo raíz de esta simulación.
            depth: Profundidad restante.

        Returns:
            Retorno estimado con descuento desde el estado del nodo.
        """
        state = node.state

        # Caso base: horizonte agotado o estado terminal.
        if depth == 0 or self.problem.is_terminal(state):
            return self.problem.utility(state)

        # ── Expansión: primera visita al nodo ────────────────────────── #
        if not node.expanded:
            node.expanded = True
            return self._rollout(state, depth, self.rollout_policy)

        # ── Selección: UCB1 sobre las acciones disponibles ────────────── #
        action = self._ucb_action(node)
        if action is None:
            # Estado sin acciones (terminal de facto).
            return self.problem.utility(state)

        # ── Transición con el modelo generativo ──────────────────────── #
        next_state, reward = self.problem.transition(state, action)

        # Crear el nodo hijo si aún no existe en el árbol.
        if action not in node.children:
            node.children[action] = MCTSNode(
                next_state, parent=node, action_taken=action
            )
        child = node.children[action]

        # ── Recursión ─────────────────────────────────────────────────── #
        future = self._simulate(child, depth - 1)
        q = reward + self.gamma * future

        # ── Retropropagación: media incremental Q ← Q + (q − Q) / N ──── #
        n = node.N.get(action, 0) + 1
        node.N[action] = n
        node.N_total += 1
        #node.Q[action] = node.Q.get(action, 0.0) + (q - node.Q.get(action, 0.0)) / n
        node.Q[action] =  node.Q.get(action, 0.0) + q
        return q

    def _ucb_action(self, node: MCTSNode) -> Any:
        """
        Selecciona la acción que maximiza el criterio UCB1 en el nodo dado.

        Las acciones no visitadas tienen prioridad máxima (UCB1 = +∞) y se
        devuelven inmediatamente al encontrarse. Para acciones ya visitadas
        se calcula Q(s,a) + c·√(ln N(s) / N(s,a)).

        El conjunto de acciones se congela en la primera llamada
        (cached_actions). Esto es esencial cuando la función candidata es
        estocástica: sin caché, cada simulación encontraría acciones nuevas
        con N=0, el árbol crecería indefinidamente en anchura y nunca
        profundizaría más allá del primer nivel.

        Args:
            node: Nodo en el que seleccionar la acción.

        Returns:
            Acción con mayor UCB1, o None si no hay acciones disponibles.
        """
        # Fijar el espacio de acciones la primera vez (congelar).
        if node.cached_actions is None:
            node.cached_actions = self.problem.get_actions(node.state)

        log_ns = math.log(node.N_total) if node.N_total > 0 else 0.0
        best_action = None
        best_ucb = float("-inf")

        for action in node.cached_actions:
            n_sa = node.N.get(action, 0)
            if n_sa == 0:
                return action   # acción no explorada → UCB1 = +∞

            #ucb = node.Q[action] + self.exploration_c * math.sqrt(log_ns / n_sa)
            ucb = node.Q[action]/n_sa + self.exploration_c * math.sqrt(log_ns / n_sa)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        return best_action

    def _rollout(self, state: dict, depth: int, rollout_policy: RolloutPolicy) -> float:
        """
        Estima el valor de un nodo hoja mediante la política π₀ (Algoritmo 4.10).

        Ejecuta la política iterativamente hasta el horizonte o un estado
        terminal, acumulando el retorno con descuento. Al finalizar añade
        la utilidad heurística del estado final como cota de valor residual.

        Args:
            state:          Estado inicial del rollout.
            depth:          Profundidad máxima del rollout.
            rollout_policy: Política π₀: state → action (o None si sin acciones).

        Returns:
            Retorno acumulado con descuento más utilidad heurística final.
        """
        total_return = 0.0
        discount = 1.0
        current_state = state

        for _ in range(depth):
            if self.problem.is_terminal(current_state):
                break

            actions = self.problem.get_actions(current_state)
            if not actions:
                break

            action = rollout_policy(current_state)
            if action is None:
                break

            current_state, reward = self.problem.transition(current_state, action)
            total_return += discount * reward
            discount *= self.gamma

        total_return += discount * self.problem.utility(current_state)
        return total_return
