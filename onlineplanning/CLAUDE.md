# Algoritmos de planificación en línea clásicos

Me estoy leyendo el libro Decision Making Under Uncertainty de Mykel Kochenderfer. Particularmente el capítulo online planning.

## 1. Descripción del problema

### 1.1. El estado

El estado es la representación de la situación actual del sistema. Puede ser un vector de características, una imagen, o cualquier otra representación que capture la información relevante para la toma de decisiones. 

En Python será un diccionario, y sus campos serán las características relevantes para la toma de decisiones.

### 1.2. La acción
La acción es la decisión que el agente toma en cada paso de tiempo. Será una acción discreta, pero no tine por qué ser un número. Puede ser una cadena de texto, un objeto, o cualquier otra representación que capture la acción que el agente quiere tomar.

### 1.3. La función de utilidad

La función de utilidad es una función que asigna un valor a cada estado. Este valor representa la "utilidad" o "valor" de estar en ese estado. El objetivo del agente es maximizar la utilidad a lo largo del tiempo. En Python, la función de utilidad será una función que tome un estado como entrada y devuelva un número como salida.

### 1.4. La función de transición

La función de transición es una función que describe cómo el estado del sistema cambia en respuesta a las acciones del agente. Es una función que toma un estado y una acción como entrada y devuelve un nuevo estado como salida. En Python, la función de transición será una función que tome un estado y una acción como entrada y devuelva un nuevo estado como salida.

### 1.5. El modelo de observación

El modelo de observación es una función que describe cómo el agente percibe el estado del sistema. Es una función que toma un estado como entrada y devuelve una observación como salida. En Python, el modelo de observación será una función que tome un estado como entrada y devuelva una observación como salida.

### 1.6. El modelo de transición

El modelo de transición es una función que intenta modelar cómo el estado del sistema cambia en respuesta a las acciones del agente. Es una función que toma un estado y una acción como entrada y devuelve un nuevo estado como salida. En Python, el modelo de transición será una función que tome un estado y una acción como entrada y devuelva un nuevo estado como salida. Es un modelo hacia adelante, es decir, intenta predecir el futuro a partir del presente.

## 2. Algoritmos de planificación en línea

A continuación, los algoritmos que se van a implementar. 

### 2.1. Forward Search

Forward search  is a simple online action-selection method that looks ahead from some initial state s0 to some horizon (or depth) d . The forward search function SelectAction(s , d ) returns the optimal action a∗ and its value v∗. The pseudocode uses A(s) to represent the set of actions available from state s , which may be a subset of the full action space A. The set of possible states that can follow immediately from s after executing action a is denoted S (s , a), which may be a small subset of the
full state spaces S .

Algorithm 4.6 Forward search
1: function SelectAction(s , d )
2: if d = 0
3: return (nil, 0)
4: (a∗, v∗) ← (nil, −∞)
5: for a ∈ A(s )
6: v ← R(s , a)
7: for s ′ ∈ S (s , a)
8: (a′, v′) ← SelectAction(s ′, d − 1)
9: v ← v + γ T (s ′ | s , a)v′
10: if v > v∗
11: (a∗, v∗) ← (a, v)
12: return (a∗, v∗)

### 2.2. Branch and Bound

Branch and bound search (Algorithm 4.7) is an extension to forward search that uses
knowledge of the upper and lower bounds of the value function to prune portions of
the search tree. This algorithm assumes that prior knowledge is available that allows us
to easily compute a lower bound on the value function U (s ) and an upper bound on
the state-action value function U (s , a). The pseudocode is identical to Algorithm 4.6,
except for the use of the lower bound in Line 3 and the pruning check in Line 6. The
call to SelectAction(s , d ) returns the action to execute and a lower bound on the
value function. The order in which we iterate over the actions in Line 5 is important. In order to
prune, the actions must be in descending order of upper bound. In other words, if
action ai is evaluated before aj , then U (s , ai ) ≥ U (s , aj ). The tighter we are able to
make the upper and lower bounds, the more we can prune the search space and decrease
computation time. The worst-case computational complexity, however, remains the
same as for forward search.

Algorithm 4.7 Branch-and-bound search

1: function SelectAction(s , d )
2: if d = 0
3: return (nil, U (s ))
4: (a∗, v∗) ← (nil, −∞)
5: for a ∈ A(s )
6: if U (s , a) < v∗
7: return (a∗, v∗)
8: v ← R(s , a)
9: for s ′ ∈ S (s , a)
10: (a′, v′) ← SelectAction(s ′, d − 1)
11: v ← v + γ T (s ′ | s , a)v′
12: if v > v∗
13: (a∗, v∗) ← (a, v)
14: return (a∗, v∗)

## 2.3. Sparse Sampling

Sampling methods can be used to avoid the worst-case exponential complexity of
forward and branch-and-bound search. Although these methods are not guaranteed
to produce the optimal action, they can be shown to produce approximately optimal
actions most of the time and can work well in practice. One of the simplest approaches
is referred to as sparse sampling (Algorithm 4.8). Sparse sampling uses a generative model G to produce samples of the next state s ′ and reward r . An advantage of using a generative model is that it is often easier to implement code for drawing random samples from a complex, multidimensional
distribution rather than explicitly representing probabilities. Line 8 of the algorithm
draws (s ′, r ) ∼ G (s , a). All of the information about the state transitions and rewards
is represented by G ; the state transition probabilities T (s ′ | s , a) and expected reward
function R(s , a) are not used directly.

Algorithm 4.8 Sparse sampling

1: function SelectAction(s , d )
2: if d = 0
3: return (nil, 0)
4: (a∗, v∗) ← (nil, −∞)
5: for a ∈ A(s )
6: v ← 0
7: for i ← 1 to n
8: (s ′, r ) ∼ G (s , a)
9: (a′, v′) ← SelectAction(s ′, d − 1)
10: v ← v + (r + γ v′)/n
11: if v > v∗
12: (a∗, v∗) ← (a, v)
13: return (a∗, v∗)

## 2.4. Monte Carlo Tree Search

This algorithm involves running many simulations from the current state while
updating an estimate of the state-action value function Q (s , a). There are three stages
in each simulation:

- Search: If the current state in the simulation is in the set T (initially empty), then
we enter the search stage. Otherwise we proceed to the expansion stage. During
the search stage, we update Q (s , a) for the states and actions visited and tried in
our search. We also keep track of the number of times we have taken an action
from a state N (s , a). During the search, we execute the action that maximizes

Q (s , a) + c √(ln N (s )/N (s , a))

where N (s ) = ∑a N (s , a) and c is a parameter that controls the amount of
exploration in the search (exploration will be covered in depth in the next chapter).
The second term is an exploration bonus that encourages selecting actions that have
not been tried as frequently. This bonus is infinite if N (s , a) = 0.

- Expansion: Once we have reached a state that is not in the set T , we iterate over
all of the actions available from that state and initialize N (s , a) and Q (s , a) with
N0(s , a) and Q0(s , a), respectively. The functions N0 and Q0 can be based on
prior expert knowledge of the problem; if none is available, then they can both be
initialized to 0. We then add the current state to the set T .

- Rollout: After the expansion stage, we simply select actions according to some
rollout (or default) policy π0 until the desired depth is reached (Algorithm 4.10).
Typically, rollout policies are stochastic, and so the action to execute is sampled
a ∼ π0(s ). The rollout policy does not have to be close to optimal, but it is a way
for an expert to bias the search into areas that are promising. The expected value
is returned and used in the search to update the value for Q (s , a).

Simulations are run until some stopping criterion is met, often simply a fixed number
of iterations. We then execute the action that maximizes Q (s , a). Once that action has
been executed, we can rerun the Monte Carlo tree search to select the next action. It is
common to carry over the values of N (s , a) and Q (s , a) computed in the previous step.

Algorithm 4.9 Monte Carlo tree search
1: function SelectAction(s , d )
2: loop
3:  Simulate(s , d , π0)
4: return arg maxa Q (s , a)

5: function Simulate(s , d , π0)
6:  if d = 0
7:      return 0
8:  if s 6 ∈ T
9:      for a ∈ A(s )
10:         (N (s , a), Q (s , a)) ← (N0(s , a), Q0(s , a))
11:         T = T ∪ {s }
12:     return Rollout(s , d , π0)
13: a ← arg maxa∈A(s ) [Q (s , a) + c sqrt(log N (s )/N (s ,a))]

14: (s ′, r ) ∼ G (s , a)
15: q ← r + γ Simulate(s ′, d − 1, π0)
16: N (s , a) ← N (s , a) + 1
17: Q (s , a) ← Q (s , a) + q−Q (s ,a) /N (s , a)
18: return q


Algorithm 4.10 Rollout evaluation
1: function Rollout(s , d , π0)
2:  if d = 0
3:      return 0
4:  a ∼ π0(s )
5: (s ′, r ) ∼ G (s , a)
6: return r + γ Rollout(s ′, d − 1, π0)

# 2. Common interface

El objetivo es tener una interfaz común para todos los algoritmos de planificación en línea. La interfaz debe ser lo suficientemente flexible como para permitir la implementación de diferentes algoritmos, pero también lo suficientemente simple como para ser fácil de usar.

Un problema queda definido como una clase que tiene los siguientes métodos:

- `get_actions(state)`: devuelve una lista de acciones disponibles desde el estado dado.
- `transition(state, action)`: devuelve el nuevo estado resultante de tomar la acción dada y la recompensa obtenida al tomar la acción dada desde el estado dado.
- `reward(state, action)`: devuelve la recompensa obtenida al tomar la acción dada desde el estado dado.
- `is_terminal(state)`: devuelve True si el estado dado es un estado terminal, False en caso contrario.
- `utility(state)`: devuelve la utilidad del estado dado.
- `observation(state)`: devuelve la observación del estado dado.
- `model_transition(state, action)`: devuelve el nuevo estado predicho por el modelo de transición al tomar la acción dada desde el estado dado.
- `render(state)`: devuelve una representación visual del estado dado.

La función de transición y la función de recompensa pueden ser estocásticas, es decir, pueden devolver diferentes resultados cada vez que se llaman con los mismos argumentos. 

La función de transición no depende de valores pasados. Es decir, el nuevo estado depende únicamente del estado actual y la acción tomada, no de la historia de estados y acciones anteriores. Esto se conoce como la propiedad de Markov. 


# 3. Ejemplo de escenario

Supongamos que tenemos un robot que se encuentra en una cuadrícula y quiere llegar a una meta. El estado del robot se puede representar como un diccionario con las siguientes claves:

- `position`: una tupla (x, y) que representa la posición del robot en la cuadrícula.
- `battery`: un número que representa la cantidad de batería restante del robot.

Las acciones disponibles para el robot son:
- `move_up`: mueve el robot hacia arriba.
- `move_down`: mueve el robot hacia abajo.
- `move_left`: mueve el robot hacia la izquierda.
- `move_right`: mueve el robot hacia la derecha.







