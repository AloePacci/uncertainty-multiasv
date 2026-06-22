# Escenarios

Cada escenario implementa la interfaz `Problem` para un dominio específico.
Se describen aquí sus estados, acciones, recompensas y parámetros de construcción.

---

## GridWorld

> `scenarios.grid_world.GridWorld`

Navegación de un robot en una cuadrícula 2D con obstáculos y batería limitada.

### Estado

| Clave | Tipo | Descripción |
|-------|------|-------------|
| `position` | `(int, int)` | Coordenadas (x, y) del robot. Origen en la esquina superior-izquierda. |
| `battery` | `int` | Unidades de batería restantes. |

### Acciones

`"move_up"`, `"move_down"`, `"move_left"`, `"move_right"`. Cada movimiento
consume una unidad de batería. Si el destino es una pared o está fuera de
límites, la posición no cambia pero la batería sí se consume.

### Recompensas

| Evento | Recompensa |
|--------|-----------|
| Alcanzar la celda objetivo | `+10` |
| Cualquier paso ordinario | `−1` |

### Terminal

- El robot alcanza la celda objetivo.
- La batería llega a cero.

### Constructor

```python
GridWorld(
    grid: list[str],    # lista de cadenas; '#' = pared, 'S' = inicio, 'G' = meta, ' ' = libre
    max_battery: int,   # batería inicial
)
```

---

## MaxInformativePath

> `scenarios.max_informative_path.MaxInformativePathWaypoints`

Planificación de caminos informativos con **acciones unitarias** (8 direcciones).
El agente recorre la cuadrícula celda a celda maximizando la información
acumulada en celdas no visitadas.

### Estado

| Clave | Tipo | Descripción |
|-------|------|-------------|
| `position` | `(int, int)` | Celda actual (x, y). Origen superior-izquierdo. |
| `budget` | `float` | Distancia euclidiana restante. |
| `visited` | `frozenset` | Celdas cuya información ya fue recogida. |

### Acciones

8 direcciones: `N`, `S`, `E`, `W`, `NE`, `NW`, `SE`, `SW`.

| Tipo | Coste |
|------|-------|
| Ortogonales | 1.0 |
| Diagonales | √2 ≈ 1.414 |

### Recompensas

`info_map[y, x]` al llegar por primera vez a (x, y). Cero si ya fue visitada.

### Terminal

`budget < 1.0`.

### Generación del mapa

```python
from scenarios.max_informative_path import generate_smooth_map

info_map = generate_smooth_map(N=50, n_peaks=6, seed=7)
# Devuelve np.ndarray (N, N) con valores en [0, 1]
# generado como superposición de gaussianas 2D
```

---

## MaxInformativePathWaypoints

> `scenarios.max_informative_path_waypoints.MaxInformativePathWaypoints`

Variante del IPP donde una **acción es un destino** (x, y) en lugar de un
paso unitario. El agente se desplaza por el camino de mínimo coste
(diagonal-primero) y recoge la información de todas las celdas que atraviesa.

Esta abstracción reduce el horizonte del árbol (pocas acciones de gran
alcance) pero puede aumentar el factor de ramificación si no se usa una
función candidata.

### Estado

Igual que `MaxInformativePath`.

### Camino mínimo entre dos celdas

El coste del camino se calcula con la métrica Chebyshev extendida:

$$\text{coste}(\text{start}, \text{end}) = \min(|\Delta x|, |\Delta y|) \cdot \sqrt{2} + \bigl||\Delta x| - |\Delta y|\bigr|$$

### Sensor — `anneal_radius`

Al atravesar una celda, el agente recoge la información de todas las celdas
dentro del radio euclídeo `anneal_radius`:

- `anneal_radius = 0`: solo la celda atravesada (comportamiento por defecto).
- `anneal_radius = r`: disco de radio r alrededor de cada celda del camino.

Los vecindarios se precalculan en el constructor para evitar cómputo
repetido durante las simulaciones.

### Funciones candidatas

La clase expone cuatro métodos *factory* que devuelven una `CandidateFn`
(callable `state → list[(x,y)]`) y que se asignan a `problem.candidate_fn`:

| Método | Descripción | Determinista |
|--------|-------------|:---:|
| `candidates_8grid(resolution=1)` | 8 vecinos a paso fijo `resolution`. | ✓ |
| `candidates_subgrid(resolution=2)` | Todos los puntos del mapa a paso `resolution`. | ✓ |
| `candidates_adaptive(min_resolution=1, max_resolution=6)` | 8 vecinos con paso proporcional al valor de `info_map` en la posición actual. Mucha info → paso corto; poca info → paso largo. | ✓ |
| `candidates_top_k(k=5)` | k destinos no visitados muestreados con probabilidad ∝ info / dist. | ✗ |

> **Atención con funciones estocásticas**: al usar `candidates_top_k` con
> MCTS, el atributo `cached_actions` del nodo congela las acciones en la
> primera visita UCB1. Esto impide que el árbol crezca indefinidamente en
> anchura. Ver [algorithms.md — `cached_actions`](algorithms.md#cached_actions----congelación-del-espacio-de-acciones).

### Constructor

```python
MaxInformativePathWaypoints(
    info_map: np.ndarray,    # mapa (N, N) con valores en [0, 1]
    max_budget: float,       # presupuesto de distancia
    candidate_fn=None,       # función candidata (ver arriba)
    gamma: float = 1.0,      # factor de descuento
    anneal_radius: int = 0,  # radio del sensor
)
```

### Ejemplo completo

```python
from scenarios.max_informative_path import generate_smooth_map
from scenarios.max_informative_path_waypoints import MaxInformativePathWaypoints
from algorithms.mcts import MCTS

info_map = generate_smooth_map(N=50, n_peaks=6, seed=7)

problem = MaxInformativePathWaypoints(
    info_map=info_map,
    max_budget=150.0,
    gamma=0.95,
    anneal_radius=2,
)
problem.candidate_fn = problem.candidates_adaptive(min_resolution=1, max_resolution=6)

planner = MCTS(problem, n_simulations=500, depth=10, reuse_tree=True)

state = problem.initial_state()
while not problem.is_terminal(state):
    action, q = planner.select_action(state)
    if action is None:
        break
    state, reward = problem.transition(state, action)
```
