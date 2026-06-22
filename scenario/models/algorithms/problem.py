"""
Interfaz abstracta para problemas de planificación secuencial.

Un problema queda definido como un Proceso de Decisión de Markov (MDP) con
horizonte finito. La propiedad de Markov garantiza que la transición al
siguiente estado depende únicamente del estado actual y la acción tomada,
no de la historia pasada.

Referencia:
    Kochenderfer, M. J. (2015). Decision Making Under Uncertainty:
    Theory and Application. MIT Press. Capítulo 4.
"""

from abc import ABC, abstractmethod
from typing import Any


class Problem(ABC):
    """
    Clase base abstracta que define la interfaz común para todos los problemas
    de planificación online.

    Las subclases concretas implementan dominios específicos (p. ej. navegación
    en cuadrícula, brazos robóticos, juegos de mesa) instanciando cada uno de
    los métodos abstractos siguientes.

    La dinámica del problema se representa mediante:
        - Un espacio de estados  S
        - Un espacio de acciones A
        - Una función de transición  T(s' | s, a)
        - Una función de recompensa  R(s, a)
        - Un factor de descuento    γ ∈ (0, 1] 
    """

    # ------------------------------------------------------------------ #
    # Métodos abstractos — toda subclase debe implementarlos              #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_actions(self, state: dict) -> list:
        """
        Devuelve el conjunto de acciones disponibles desde el estado dado.

        Corresponde a A(s) en la notación del libro. En estados terminales
        este método debe devolver una lista vacía.

        Args:
            state: El estado actual del sistema.

        Returns:
            Lista de acciones aplicables.
        """

    @abstractmethod
    def transition(self, state: dict, action: Any) -> tuple[dict, float]:
        """
        Aplica una acción al estado y devuelve el estado sucesor y la recompensa.

        Actúa como modelo generativo G(s, a) → (s', r). Para problemas
        deterministas es una función pura; para problemas estocásticos debe
        muestrear del proceso subyacente.

        Args:
            state:  El estado actual.
            action: La acción a ejecutar.

        Returns:
            Una tupla (next_state, reward).
        """

    @abstractmethod
    def reward(self, state: dict, action: Any) -> float:
        """
        Devuelve la recompensa esperada inmediata R(s, a).

        En problemas estocásticos debe retornar la esperanza matemática
        E[r | s, a] en lugar de un único muestreo.

        Args:
            state:  El estado actual.
            action: La acción ejecutada.

        Returns:
            Recompensa escalar.
        """

    @abstractmethod
    def is_terminal(self, state: dict) -> bool:
        """
        Devuelve True si el estado es absorbente (no hay más transiciones).

        Los estados terminales representan el final del episodio, ya sea por
        haber alcanzado el objetivo o por haber agotado algún recurso.

        Args:
            state: El estado a evaluar.

        Returns:
            True si el estado es terminal, False en caso contrario.
        """

    @abstractmethod
    def utility(self, state: dict) -> float:
        """
        Devuelve la utilidad heurística del estado dado.

        Actúa como cota inferior U(s) ≤ V*(s) del valor verdadero. En
        Branch and Bound se usa para evaluar los nodos hoja en lugar del
        valor cero de Forward Search, lo que permite podar ramas con menor
        información.

        Args:
            state: El estado a evaluar.

        Returns:
            Cota inferior escalar del retorno óptimo desde state.
        """

    @abstractmethod
    def upper_bound(self, state: dict, action: Any) -> float:
        """
        Devuelve una cota superior U(s, a) ≥ Q*(s, a) del valor óptimo de
        la acción a en el estado s.

        Branch and Bound usa esta función para ordenar las acciones de mayor
        a menor cota superior y para podar ramas: si la cota superior de la
        siguiente acción es inferior al mejor valor encontrado hasta el
        momento, ninguna evaluación posterior puede mejorar ese valor, por
        lo que el resto de la búsqueda se descarta.

        La calidad de la poda es proporcional a la tightness de esta cota:
        cuanto más ajustada sea a Q*(s, a), más ramas se podaran.

        Args:
            state:  El estado actual.
            action: La acción candidata.

        Returns:
            Cota superior escalar de Q*(s, a).
        """

    @abstractmethod
    def observation(self, state: dict) -> Any:
        """
        Devuelve la observación percibida por el agente en el estado dado.

        En entornos completamente observables, la observación es simplemente
        el estado en sí. En entornos parcialmente observables la observación
        puede ser un subconjunto ruidoso del estado.

        Args:
            state: El estado actual.

        Returns:
            La observación del estado.
        """

    @abstractmethod
    def model_transition(self, state: dict, action: Any) -> dict:
        """
        Devuelve el estado predicho por el modelo interno del agente.

        A diferencia de transition(), este es un modelo hacia adelante
        determinista utilizado para planificación. Puede diferir de la
        dinámica real del entorno (modelo aproximado u optimista).

        Args:
            state:  El estado actual.
            action: La acción a ejecutar.

        Returns:
            El estado predicho por el modelo.
        """

    @abstractmethod
    def render(self, state: dict) -> str:
        """
        Devuelve una representación textual del estado para su visualización.

        Args:
            state: El estado a representar.

        Returns:
            Cadena de texto con la representación visual del estado.
        """

    # ------------------------------------------------------------------ #
    # Métodos concretos — pueden sobreescribirse en subclases             #
    # ------------------------------------------------------------------ #

    def successors(self, state: dict, action: Any) -> list[tuple[dict, float]]:
        """
        Devuelve la lista de estados sucesores posibles con sus probabilidades.

        Representa el conjunto S(s, a) junto con las probabilidades T(s' | s, a)
        necesarios para los algoritmos de búsqueda hacia adelante (Forward
        Search, Branch and Bound).

        La implementación por defecto delega en transition() y trata la
        dinámica como determinista (probabilidad = 1.0). Las subclases deben
        sobreescribir este método para entornos estocásticos donde múltiples
        sucesores son posibles desde un mismo par (estado, acción).

        Args:
            state:  El estado actual.
            action: La acción ejecutada.

        Returns:
            Lista de tuplas (next_state, probability) donde las probabilidades
            suman 1.
        """
        next_state, _ = self.transition(state, action)
        return [(next_state, 1.0)]
