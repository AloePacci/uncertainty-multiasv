"""
Búsqueda hacia adelante (Forward Search) — Algoritmo 4.6.

La búsqueda hacia adelante es un método de selección de acciones online que
explora el árbol de lookahead desde el estado actual hasta un horizonte d.
En cada nodo interno se evalúan todas las acciones disponibles y se calcula
el valor esperado con descuento retropropagando las recompensas hacia la raíz.

La complejidad computacional en el caso peor es O(|A|^d · |S(s,a)|), es decir,
exponencial en la profundidad. Para problemas con factor de ramificación
pequeño u horizontes cortos resulta no obstante exacta y elegante.

Referencia:
    Kochenderfer, M. J. (2015). Decision Making Under Uncertainty:
    Theory and Application. MIT Press. Algoritmo 4.6.
"""

from typing import Any, Optional

from scenario.models.algorithms.problem import Problem


class ForwardSearch:
    """
    Selección de acciones online mediante búsqueda hacia adelante.

    El algoritmo expande recursivamente el árbol de estados desde s hasta
    profundidad d, acumulando las recompensas con descuento y retropropagando
    el valor estimado hasta la raíz. La acción que maximiza este valor es la
    acción óptima bajo el horizonte dado.

    Attributes:
        problem: La instancia del MDP que define la dinámica y las recompensas.
        depth:   La profundidad máxima de búsqueda d.
        gamma:   El factor de descuento γ ∈ (0, 1].
    """

    def __init__(self, problem: Problem, depth: int, gamma: float = 1.0) -> None:
        """
        Inicializa el planificador de búsqueda hacia adelante.

        Args:
            problem: Instancia del problema MDP a resolver.
            depth:   Horizonte de planificación (profundidad máxima del árbol).
            gamma:   Factor de descuento temporal. Valores cercanos a 1 tratan
                     recompensas futuras casi igual que las inmediatas; valores
                     menores privilegian recompensas a corto plazo.
        """
        self.problem = problem
        self.depth = depth
        self.gamma = gamma

    def select_action(
        self, state: dict, depth: Optional[int] = None
    ) -> tuple[Any, float]:
        """
        Devuelve la acción óptima y el valor estimado del estado.

        Implementa la función recursiva SelectAction(s, d) del Algoritmo 4.6.
        En cada llamada el algoritmo:
          1. Evalúa la condición de parada (horizonte alcanzado o estado terminal).
          2. Itera sobre A(s), las acciones disponibles desde s.
          3. Para cada acción, suma la recompensa inmediata R(s, a) más el
             retorno esperado con descuento sobre todos los sucesores S(s, a):

                 v(s, a) = R(s, a) + γ · Σ_{s'} T(s'|s,a) · V*(s', d-1)

          4. Retorna la acción a* que maximiza v(s, a).

        Args:
            state: El estado desde el cual seleccionar la acción.
            depth: Profundidad restante de búsqueda. Si es None se utiliza
                   self.depth (llamada de nivel superior).

        Returns:
            Una tupla (a*, v*) donde:
                a* es la acción de mayor valor estimado (None en nodos hoja).
                v* es el valor estimado asociado a dicha acción.
        """
        if depth is None:
            depth = self.depth

        # Caso base — horizonte alcanzado o estado terminal:
        # no se acumula más recompensa futura.
        if depth == 0 or self.problem.is_terminal(state):
            return (None, 0.0)

        best_action: Any = None
        best_value: float = float("-inf")

        for action in self.problem.get_actions(state):
            # R(s, a): recompensa esperada inmediata por tomar la acción.
            value: float = self.problem.reward(state, action)

            # Σ_{s' ∈ S(s,a)} T(s'|s,a) · V*(s', d-1):
            # suma sobre todos los estados sucesores con su probabilidad.
            for next_state, probability in self.problem.successors(state, action):
                _, future_value = self.select_action(next_state, depth - 1)
                value += self.gamma * probability * future_value

            # Actualización del mejor par (acción, valor) encontrado.
            if value > best_value:
                best_action = action
                best_value = value

        return (best_action, best_value)
