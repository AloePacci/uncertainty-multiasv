"""
Branch and Bound — Algoritmo 4.7.

Branch and Bound extiende la búsqueda hacia adelante (Forward Search,
Algoritmo 4.6) aprovechando cotas inferior y superior del valor para podar
ramas del árbol de búsqueda que no pueden mejorar la solución actual.

La diferencia clave respecto a Forward Search son dos:

  1. Nodos hoja evaluados con la cota inferior U(s) en lugar de 0, lo que
     permite propagar información heurística desde el fondo del árbol.

  2. Las acciones se ordenan de mayor a menor cota superior U(s, a) antes
     de expandirse. Si la cota superior de la siguiente acción es menor que
     el mejor valor ya encontrado (v*), se garantiza que ninguna acción
     restante puede mejorar v*, y el bucle se termina anticipadamente.

Complejidad:
    - Caso peor: idéntica a Forward Search — O(|A|^d · |S(s,a)|).
    - Caso promedio: significativamente mejor gracias a la poda; depende
      de la tightness de las cotas proporcionadas por el dominio.

Referencia:
    Kochenderfer, M. J. (2015). Decision Making Under Uncertainty:
    Theory and Application. MIT Press. Algoritmo 4.7.
"""

from typing import Any, Optional

from scenario.models.algorithms.problem import Problem


class BranchAndBound:
    """
    Selección de acciones online mediante Branch and Bound.

    Extiende la búsqueda hacia adelante con poda basada en cotas. Para que
    la poda sea efectiva el problema debe proporcionar:
      - problem.utility(s)         → cota inferior U(s) ≤ V*(s)
      - problem.upper_bound(s, a)  → cota superior U(s,a) ≥ Q*(s,a)

    Attributes:
        problem: La instancia del MDP que define la dinámica y las recompensas.
        depth:   La profundidad máxima de búsqueda d.
        gamma:   El factor de descuento γ ∈ (0, 1].
    """

    def __init__(self, problem: Problem, depth: int, gamma: float = 1.0) -> None:
        """
        Inicializa el planificador Branch and Bound.

        Args:
            problem: Instancia del problema MDP a resolver.
            depth:   Horizonte de planificación (profundidad máxima del árbol).
            gamma:   Factor de descuento temporal.
        """
        self.problem = problem
        self.depth = depth
        self.gamma = gamma
        # Propaga γ al problema para que upper_bound pueda usarlo si lo necesita.
        self.problem._gamma_hint = gamma

    def select_action(
        self, state: dict, depth: Optional[int] = None
    ) -> tuple[Any, float]:
        """
        Devuelve la acción óptima y una cota inferior de su valor.

        Implementa SelectAction(s, d) del Algoritmo 4.7. El procedimiento es:

          1. Caso base (d = 0 o terminal): devuelve U(s) como evaluación del
             nodo hoja mediante la cota inferior del valor óptimo.

          2. Ordena A(s) en orden descendente de U(s, a) (cota superior):
             esto maximiza la probabilidad de poda en iteraciones posteriores.

          3. Para cada acción a:
               a) Si U(s, a) < v*, la cota superior no puede superar el
                  mejor valor conocido → poda: se devuelve (a*, v*) de
                  inmediato sin evaluar las acciones restantes.
               b) En otro caso, se evalúa la acción calculando el retorno
                  esperado con descuento sobre todos sus sucesores.

          4. Actualiza (a*, v*) si la acción mejora el mejor valor actual.

        La corrección del algoritmo depende del orden descendente de U(s,a):
        si U(s, a_i) < v*, entonces U(s, a_j) < v* para todo j > i, por lo
        que todos los subárboles restantes pueden descartarse.

        Args:
            state: El estado desde el cual seleccionar la acción.
            depth: Profundidad restante. Si es None se usa self.depth.

        Returns:
            Una tupla (a*, v*) donde:
                a* es la acción con mayor valor estimado (None en nodos hoja).
                v* es la cota inferior del valor de dicha acción.
        """
        if depth is None:
            depth = self.depth

        # Caso base — nodo hoja: evaluar con la cota inferior U(s).
        # A diferencia de Forward Search (que retornaría 0), aquí se
        # propaga información heurística sobre el valor futuro esperado.
        if depth == 0 or self.problem.is_terminal(state):
            return (None, self.problem.utility(state))

        best_action: Any = None
        best_value: float = float("-inf")

        # Las acciones se ordenan de mayor a menor cota superior U(s, a).
        # Explorar primero las acciones más prometedoras maximiza la
        # probabilidad de que v* sea alto al llegar a acciones peores,
        # lo que activa la condición de poda más pronto.
        actions = sorted(
            self.problem.get_actions(state),
            key=lambda a: self.problem.upper_bound(state, a),
            reverse=True,
        )

        for action in actions:
            # Poda: si la cota superior de esta acción no puede superar v*,
            # tampoco lo harán las acciones restantes (están ordenadas).
            if self.problem.upper_bound(state, action) < best_value:
                return (best_action, best_value)

            # Evaluar la acción: recompensa inmediata más valor futuro.
            value: float = self.problem.reward(state, action)

            # Σ_{s' ∈ S(s,a)} T(s'|s,a) · V*(s', d-1):
            for next_state, probability in self.problem.successors(state, action):
                _, future_value = self.select_action(next_state, depth - 1)
                value += self.gamma * probability * future_value

            if value > best_value:
                best_action = action
                best_value = value

        return (best_action, best_value)
