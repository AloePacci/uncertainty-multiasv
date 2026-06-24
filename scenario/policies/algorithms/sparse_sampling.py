"""
Sparse Sampling — Algoritmo 4.8.

Sparse Sampling es un método de planificación basado en muestreo que evita
la explosión combinatoria de Forward Search reemplazando la enumeración
explícita de todos los estados sucesores por un número fijo n de muestras
del modelo generativo G(s, a) → (s', r).

La clave del algoritmo es que toda la información sobre la dinámica del
sistema (distribución de transiciones y recompensas) se accede únicamente
a través del modelo generativo, sin necesidad de representar explícitamente
las probabilidades T(s' | s, a) ni la función de recompensa R(s, a).

Complejidad:
    - Determinista:  O((n · |A|)^d)  en el número de llamadas al modelo.
    - La profundidad d y el número de muestras n controlan el compromiso
      entre coste computacional y calidad de la aproximación. Con n → ∞
      el algoritmo converge a Forward Search exacto.

Garantía de aproximación (Kearns et al., 2002):
    Para cualquier ε > 0 y δ > 0, existe un valor de n tal que con
    probabilidad ≥ 1 − δ la acción seleccionada es ε-óptima, con
    coste O((1/(ε(1−γ)))^(O(1/(1−γ)))) llamadas al modelo generativo
    (independiente del tamaño del espacio de estados).

Referencia:
    Kochenderfer, M. J. (2015). Decision Making Under Uncertainty:
    Theory and Application. MIT Press. Algoritmo 4.8.

    Kearns, M., Mansour, Y., & Ng, A. Y. (2002). A sparse sampling
    algorithm for near-optimal planning in large Markov decision processes.
    Machine Learning, 49(2–3), 193–208.
"""

from typing import Any, Optional

from scenario.models.algorithms.problem import Problem


class SparseSampling:
    """
    Selección de acciones online mediante Sparse Sampling.

    En cada estado el algoritmo muestrea n trayectorias de un paso por
    acción usando el modelo generativo del problema. El valor de cada acción
    se estima como la media muestral de los retornos:

        Q̂(s, a) ≈ (1/n) Σ_{i=1}^{n} [r_i + γ · V̂(s'_i, d−1)]

    donde (s'_i, r_i) ~ G(s, a) es la i-ésima muestra del modelo generativo.

    Attributes:
        problem:  La instancia del MDP con su modelo generativo.
        depth:    Horizonte de planificación d.
        n_samples: Número de muestras n por acción en cada nivel del árbol.
        gamma:    Factor de descuento γ ∈ (0, 1].
    """

    def __init__(
        self,
        problem: Problem,
        depth: int,
        n_samples: int = 10,
        gamma: float = 1.0,
    ) -> None:
        """
        Inicializa el planificador de Sparse Sampling.

        Args:
            problem:   Instancia del problema MDP a resolver.
            depth:     Horizonte de planificación (profundidad del árbol).
            n_samples: Número de muestras del modelo generativo por acción
                       y nivel del árbol. Valores mayores dan estimaciones
                       más precisas a mayor coste computacional.
            gamma:     Factor de descuento temporal.
        """
        self.problem = problem
        self.depth = depth
        self.n_samples = n_samples
        self.gamma = gamma

    def select_action(
        self, state: dict, depth: Optional[int] = None
    ) -> tuple[Any, float]:
        """
        Devuelve la acción con mayor valor estimado y su valor aproximado.

        Implementa SelectAction(s, d) del Algoritmo 4.8. Para cada acción
        a ∈ A(s) se realizan n consultas al modelo generativo G(s, a) y se
        promedia el retorno observado:

            Q̂(s, a) = (1/n) Σ_{i=1}^{n} [r_i + γ · V̂(s'_i, d−1)]

        La recursión se detiene cuando d = 0 o cuando el estado es terminal,
        devolviendo valor cero (no se usa utilidad heurística, a diferencia
        de Branch and Bound).

        Args:
            state: El estado desde el cual seleccionar la acción.
            depth: Profundidad restante. Si es None se usa self.depth.

        Returns:
            Una tupla (a*, Q̂(s, a*)) donde a* es la acción de mayor valor
            estimado por muestreo y Q̂(s, a*) es dicho valor esperado.
        """
        if depth is None:
            depth = self.depth

        # Caso base: horizonte agotado o estado terminal.
        # En lugar de retornar 0 (como en la versión del libro), se evalúa
        # el nodo hoja con la cota inferior U(s) del problema. Esto permite
        # que el algoritmo funcione correctamente incluso cuando la
        # profundidad de búsqueda es menor que el horizonte del episodio:
        # sin heurística, todos los caminos evaluarían igual (Σ(−1)) y el
        # algoritmo tomaría decisiones aleatorias. Es una extensión práctica
        # habitual del algoritmo base descrito en el libro.
        if depth == 0 or self.problem.is_terminal(state):
            return (None, self.problem.utility(state))

        best_action: Any = None
        best_value: float = float("-inf")

        for action in self.problem.get_actions(state):
            # Estimación Monte Carlo de Q(s, a) con n muestras del modelo
            # generativo G(s, a). El valor es la media del retorno muestral.
            value: float = 0.0

            for _ in range(self.n_samples):
                # Muestra (s', r) ~ G(s, a) usando el modelo generativo.
                next_state, reward = self.problem.transition(state, action)

                # Retorno futuro estimado desde el estado muestreado.
                _, future_value = self.select_action(next_state, depth - 1)

                # Acumula el retorno con descuento, promediado sobre n muestras.
                value += (reward + self.gamma * future_value) / self.n_samples

            if value > best_value:
                best_action = action
                best_value = value

        return (best_action, best_value)
