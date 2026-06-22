# Algoritmos de planificación online

Todos los algoritmos implementan la misma interfaz: reciben un problema
(`Problem`) y exponen un método `select_action(state) → (action, value)`.
Esto permite intercambiarlos sin modificar el código del entorno ni el
bucle de simulación.

---

## Interfaz `Problem`

Todo problema de planificación debe subclasificar `algorithms.problem.Problem`
e implementar los siguientes métodos:

| Método | Firma | Descripción |
|--------|-------|-------------|
| `get_actions` | `(state) → list` | Acciones aplicables en el estado. Lista vacía en terminales. |
| `transition` | `(state, action) → (next_state, reward)` | Modelo generativo G(s,a). |
| `reward` | `(state, action) → float` | Recompensa inmediata esperada R(s,a). |
| `is_terminal` | `(state) → bool` | True si el episodio ha terminado. |
| `utility` | `(state) → float` | Cota heurística inferior U(s) ≤ V\*(s). |
| `upper_bound` | `(state, action) → float` | Cota superior Q(s,a) ≥ Q\*(s,a). Solo requerida por Branch and Bound. |
| `observation` | `(state) → Any` | Observación percibida (igual al estado en entornos completamente observables). |
| `model_transition` | `(state, action) → next_state` | Transición sin recompensa (para wrappers internos). |

El estado se representa como un diccionario Python `dict`. No existe
ningún requisito sobre su contenido; cada dominio define sus propias claves.

### Ejemplo mínimo

```python
from algorithms.problem import Problem

class MiProblema(Problem):
    def get_actions(self, state):
        return ["izq", "der"] if not self.is_terminal(state) else []

    def transition(self, state, action):
        x = state["x"] + (-1 if action == "izq" else 1)
        reward = 1.0 if x == 5 else 0.0
        return {"x": x}, reward

    def reward(self, state, action):
        return self.transition(state, action)[1]

    def is_terminal(self, state):
        return state["x"] == 5

    def utility(self, state):
        return 0.0

    def upper_bound(self, state, action):
        return float("inf")

    def observation(self, state):
        return dict(state)

    def model_transition(self, state, action):
        return self.transition(state, action)[0]
```

---

## Forward Search

> `algorithms.forward_search.ForwardSearch`  
> Referencia: Kochenderfer (2015) — Algoritmo 4.6

Explora el árbol de lookahead desde el estado actual hasta profundidad `d`,
evaluando todas las acciones en cada nodo. Los valores se retropropagan con
descuento γ hasta la raíz.

La complejidad es **O(|A|^d)**, exponencial en el horizonte. Es exacta
para problemas deterministas con factores de ramificación o profundidades
pequeñas.

### Parámetros

| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `problem` | `Problem` | — | Instancia del problema. |
| `depth` | `int` | — | Horizonte de búsqueda d. |
| `gamma` | `float` | `1.0` | Factor de descuento. |

### Uso

```python
from algorithms.forward_search import ForwardSearch

planner = ForwardSearch(problem, depth=5, gamma=0.95)
action, value = planner.select_action(state)
```

---

## Branch and Bound

> `algorithms.branch_and_bound.BranchAndBound`  
> Referencia: Kochenderfer (2015) — Algoritmo 4.7

Extiende Forward Search con **poda** basada en cotas:

1. Las acciones se ordenan de mayor a menor cota superior `upper_bound(s,a)`
   antes de expandirse.
2. Si la cota superior de la siguiente acción es inferior al mejor valor ya
   encontrado, el resto del bucle se descarta.
3. Los nodos hoja se evalúan con `utility(s)` en lugar de 0, propagando
   información heurística desde el fondo del árbol.

La calidad de la poda depende de cuán ajustadas sean las cotas
a los valores reales Q\*(s,a).

### Parámetros

| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `problem` | `Problem` | — | Instancia del problema (debe implementar `utility` y `upper_bound`). |
| `depth` | `int` | — | Horizonte de búsqueda. |
| `gamma` | `float` | `1.0` | Factor de descuento. |

### Uso

```python
from algorithms.branch_and_bound import BranchAndBound

planner = BranchAndBound(problem, depth=8, gamma=0.95)
action, value = planner.select_action(state)
```

---

## Sparse Sampling

> `algorithms.sparse_sampling.SparseSampling`  
> Referencia: Kochenderfer (2015) — Algoritmo 4.8; Kearns et al. (2002)

Reemplaza la enumeración exhaustiva de Forward Search por **n muestras del
modelo generativo** por acción en cada nivel del árbol:

$$\hat{Q}(s, a) \approx \frac{1}{n} \sum_{i=1}^{n} \left[ r_i + \gamma \hat{V}(s'_i, d-1) \right]$$

La complejidad es **O((n·|A|)^d)** en llamadas al modelo generativo,
independiente del tamaño del espacio de estados. Con n → ∞ converge a
Forward Search exacto.

### Parámetros

| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `problem` | `Problem` | — | Instancia del problema. |
| `depth` | `int` | — | Horizonte de búsqueda. |
| `n_samples` | `int` | — | Muestras por acción por nivel. |
| `gamma` | `float` | `1.0` | Factor de descuento. |

### Uso

```python
from algorithms.sparse_sampling import SparseSampling

planner = SparseSampling(problem, depth=5, n_samples=20, gamma=0.95)
action, value = planner.select_action(state)
```

---

## MCTS / UCT

> `algorithms.mcts.MCTS`  
> Referencia: Kochenderfer (2015) — Algoritmo 4.9; Kocsis & Szepesvári (2006)

Monte Carlo Tree Search construye de forma **incremental** un árbol de
búsqueda a lo largo de múltiples simulaciones. A diferencia de Sparse
Sampling, acumula estadísticas entre simulaciones y las reutiliza para
guiar la exploración futura.

### Las cuatro fases

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐
│  Selección  │ →  │  Expansión  │ →  │  Rollout    │ →  │ Retropropagación    │
│  (UCB1)     │    │  (nodo hoja)│    │  (política) │    │  Q(s,a) incremental │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────────────┘
```

- **Selección**: desde la raíz se desciende el árbol eligiendo en cada nodo
  la acción que maximiza UCB1:

$$\text{UCB1}(s, a) = Q(s,a) + c \cdot \sqrt{\frac{\ln N(s)}{N(s,a)}}$$

- **Expansión**: el primer nodo no expandido dispara un rollout y se marca
  como expandido. Las visitas posteriores usarán UCB1.
- **Rollout**: se simula hasta el horizonte con la política π₀ (por defecto
  aleatoria; se puede pasar una política greedy).
- **Retropropagación**: el retorno estimado actualiza Q(s,a) con una media
  incremental: Q ← Q + (q − Q)/N.

### Estructura de árbol — `MCTSNode`

Cada nodo del árbol es una instancia de `MCTSNode` con los atributos
siguientes (declarados en `__slots__` para eficiencia):

| Atributo | Tipo | Descripción |
|----------|------|-------------|
| `state` | `dict` | Estado asociado al nodo. |
| `parent` | `MCTSNode \| None` | Referencia al padre. |
| `action_taken` | `Any` | Acción que produjo este nodo desde su padre. |
| `children` | `dict[action, MCTSNode]` | Hijos ya creados en el árbol. |
| `N` | `dict[action, int]` | Contador de visitas N(s,a) por acción. |
| `Q` | `dict[action, float]` | Valor medio estimado Q(s,a) por acción. |
| `N_total` | `int` | Número total de visitas al nodo Σ_a N(s,a). |
| `expanded` | `bool` | True tras el primer rollout (activa UCB1). |
| `cached_actions` | `list \| None` | Espacio de acciones congelado (ver abajo). |

### Particularidades de implementación

#### `cached_actions` — congelación del espacio de acciones

Cuando la función candidata (`candidate_fn`) es **estocástica**, cada
llamada a `get_actions` puede devolver un conjunto diferente de acciones.
Si se re-llamara en cada paso de selección UCB1, siempre habría acciones
nuevas con N=0 (UCB1 = +∞), el árbol crecería indefinidamente en
**anchura** y nunca profundizaría más allá del primer nivel.

La solución es congelar el espacio de acciones la primera vez que se
evalúa un nodo en `_ucb_action`:

```python
if node.cached_actions is None:
    node.cached_actions = self.problem.get_actions(node.state)
```

La selección final en `select_action` también usa `cached_actions` para
garantizar que evalúa las mismas acciones que se exploraron durante las
simulaciones.

#### `reuse_tree` — reutilización del subárbol

Con `reuse_tree=True`, tras devolver la acción elegida el planificador
avanza la raíz interna al nodo hijo correspondiente:

```
llamada t:    raíz = s₀   →   elige a*   →   raíz ← hijo(a*)
llamada t+1:  raíz = s₁  (subárbol con estadísticas ya acumuladas)
```

La propiedad `decision_root` conserva la raíz **antes** del avance para
permitir visualizar el árbol completo en el momento de cada decisión.

#### `control_horizon` — horizonte de control open-loop

Con `control_horizon > 1` el planificador planifica una vez y extrae las
siguientes `control_horizon` acciones del camino greedy del árbol, que
encola internamente. Las llamadas siguientes a `select_action` consumen la
cola sin replanificar:

```
planifica → [a₀, a₁, a₂]
step 0: devuelve a₀, avanza _root a n(a₀)   (con reuse_tree)
step 1: devuelve a₁, avanza _root a n(a₁)   (sin replanificar)
step 2: devuelve a₂, avanza _root a n(a₂)   (sin replanificar)
step 3: cola vacía → replanifica desde n(a₂)
```

Esto reduce el coste de planificación en un factor `control_horizon`
a cambio de ejecutar acciones open-loop entre replanificaciones.

### Parámetros de `MCTS`

| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `problem` | `Problem` | — | Instancia del problema. |
| `n_simulations` | `int` | `500` | Simulaciones por decisión. |
| `depth` | `int` | `20` | Horizonte del árbol y de los rollouts. |
| `gamma` | `float` | `1.0` | Factor de descuento. |
| `exploration_c` | `float` | `√2` | Constante de exploración c en UCB1. |
| `rollout_policy` | `Callable` | `None` | Política π₀ para rollouts. Si `None`, usa política aleatoria uniforme. |
| `reuse_tree` | `bool` | `False` | Reutilizar el subárbol entre decisiones. |
| `control_horizon` | `int` | `1` | Acciones open-loop entre replanificaciones. |

### Propiedades públicas

| Propiedad | Tipo | Descripción |
|-----------|------|-------------|
| `root` | `MCTSNode \| None` | Raíz actual (después de avanzar con `reuse_tree`). |
| `decision_root` | `MCTSNode \| None` | Raíz en el momento de la última decisión (antes de avanzar). Útil para visualizar el árbol completo. |

### Uso

```python
from algorithms.mcts import MCTS

def greedy_rollout(state):
    actions = problem.get_actions(state)
    return max(actions, key=lambda a: problem.reward(state, a)) if actions else None

planner = MCTS(
    problem=problem,
    n_simulations=500,
    depth=10,
    gamma=1.0,
    exploration_c=1.0,
    rollout_policy=greedy_rollout,
    reuse_tree=True,
    control_horizon=3,
)

state = problem.initial_state()
while not problem.is_terminal(state):
    action, q = planner.select_action(state)
    state, reward = problem.transition(state, action)
```

### Consideraciones sobre el factor de ramificación

El árbol necesita suficientes simulaciones para poder profundizar. Una
regla práctica es:

```
n_simulations >> branching_factor^depth_deseado
```

Con `branching_factor = 8` (8 vecinos) y `depth = 5`, se necesitan del
orden de 8⁵ = 32 768 simulaciones para cubrir el árbol completo. En la
práctica MCTS explora de forma no uniforme (UCB1 concentra el presupuesto
en las ramas prometedoras), por lo que valores menores son suficientes para
obtener buenas estimaciones en las acciones de mayor valor.
