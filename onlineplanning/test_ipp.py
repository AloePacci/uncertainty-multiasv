"""
Test del algoritmo MCTS sobre el escenario MaxInformativePath.

Se genera un mapa gaussiano suave de 12×12 celdas con semilla fija. Se
comparan tres estrategias sobre el mismo mapa:

    1. MCTS + política greedy como π₀
    2. Greedy puro (sin planificación, acción codiciosa en cada paso)
    3. Aleatorio (baseline)

Los resultados se visualizan con matplotlib: tres paneles con el mapa de
calor y el camino recorrido por cada estrategia, más una tabla comparativa.

Parámetros del experimento:
    N            : 12 × 12 celdas
    max_budget   : 20.0 (unidades euclidianas)
    n_simulations: 300 simulaciones MCTS por decisión
    depth        : 18 (horizonte del árbol y de los rollouts)
    seed (mapa)  : 7
"""

import random
import time

import matplotlib.pyplot as plt
import numpy as np

from scenario.policies.algorithms.mcts import MCTS
from scenario.max_informative_path import MaxInformativePath, generate_smooth_map


# ────────────────────────────────────────────────────────────────────────── #
# Políticas de rollout                                                       #
# ────────────────────────────────────────────────────────────────────────── #

def make_greedy_policy(problem: MaxInformativePath):
    """
    Construye una política greedy informada para el problema dado.

    En cada estado elige la acción que maximiza la recompensa inmediata
    (valor de la celda destino). Si varias acciones empatan (e.g. todas
    las celdas vecinas ya fueron visitadas), la selección es aleatoria
    entre ellas para evitar ciclos.

    Args:
        problem: Instancia del problema IPP.

    Returns:
        Callable state → action que implementa la política greedy.
    """
    def policy(state: dict) -> str:
        actions = problem.get_actions(state)
        best_val = max(problem.reward(state, a) for a in actions)
        best_actions = [a for a in actions if problem.reward(state, a) == best_val]
        return random.choice(best_actions)

    return policy


# ────────────────────────────────────────────────────────────────────────── #
# Funciones de episodio                                                      #
# ────────────────────────────────────────────────────────────────────────── #

def run_mcts_episode(
    problem: MaxInformativePath,
    n_simulations: int = 300,
    depth: int = 18,
    verbose: bool = True,
) -> tuple[list[tuple[int, int]], float]:
    """
    Ejecuta un episodio completo con planificación MCTS.

    En cada paso de tiempo se corre MCTS desde el estado actual con la
    política greedy como π₀. Tras las simulaciones se ejecuta la acción
    con mayor valor Q empírico y se acumula la recompensa.

    Args:
        problem:       Instancia del problema IPP.
        n_simulations: Simulaciones MCTS por decisión.
        depth:         Horizonte del árbol y de los rollouts.
        verbose:       Si True, imprime el progreso paso a paso.

    Returns:
        (path, total_info) donde path es la lista de posiciones visitadas
        en orden y total_info es la información total acumulada.
    """
    planner = MCTS(
        problem=problem,
        n_simulations=n_simulations,
        depth=depth,
        gamma=0.95,
        rollout_policy=make_greedy_policy(problem),
    )

    state = problem.initial_state()
    path: list[tuple[int, int]] = [state["position"]]
    total_info: float = 0.0
    step = 0

    while not problem.is_terminal(state):
        action, q_value = planner.select_action(state)
        state, reward = problem.transition(state, action)
        total_info += reward
        path.append(state["position"])
        step += 1

        if verbose:
            print(
                f"  paso {step:>3}  acción={action:<2}  "
                f"r={reward:.4f}  Q={q_value:.4f}  "
                f"budget={state['budget']:.2f}"
            )
        problem.render(state)

    return path, total_info


def run_greedy_episode(
    problem: MaxInformativePath,
) -> tuple[list[tuple[int, int]], float]:
    """
    Ejecuta un episodio con política puramente greedy (sin planificación).

    Sirve como línea de referencia superior a la aleatoria: la política
    greedy es miope pero siempre prefiere celdas no visitadas de alto valor.

    Args:
        problem: Instancia del problema IPP.

    Returns:
        (path, total_info).
    """
    policy = make_greedy_policy(problem)
    state = problem.initial_state()
    path: list[tuple[int, int]] = [state["position"]]
    total_info: float = 0.0

    while not problem.is_terminal(state):
        action = policy(state)
        state, reward = problem.transition(state, action)
        total_info += reward
        path.append(state["position"])

    return path, total_info


def run_random_episode(
    problem: MaxInformativePath,
    seed: int | None = None,
) -> tuple[list[tuple[int, int]], float]:
    """
    Ejecuta un episodio con política aleatoria uniforme (baseline).

    Args:
        problem: Instancia del problema IPP.
        seed:    Semilla aleatoria para reproducibilidad.

    Returns:
        (path, total_info).
    """
    if seed is not None:
        random.seed(seed)

    state = problem.initial_state()
    path: list[tuple[int, int]] = [state["position"]]
    total_info: float = 0.0

    while not problem.is_terminal(state):
        action = random.choice(problem.get_actions(state))
        state, reward = problem.transition(state, action)
        total_info += reward
        path.append(state["position"])
        

    return path, total_info


# ────────────────────────────────────────────────────────────────────────── #
# Visualización                                                              #
# ────────────────────────────────────────────────────────────────────────── #

def plot_path(
    ax: plt.Axes,
    info_map: np.ndarray,
    path: list[tuple[int, int]],
    total_info: float,
    title: str,
    path_color: str = "white",
):
    """
    Dibuja el mapa informativo con el camino del agente superpuesto.

    El mapa se muestra como un heatmap con colormap 'YlOrRd'. El camino
    se representa como una polilinea con marcadores en cada celda visitada.
    Los puntos de inicio y fin tienen marcadores especiales.

    Args:
        ax:         Eje de matplotlib donde dibujar.
        info_map:   Mapa N×N de valores informativos en [0, 1].
        path:       Lista ordenada de posiciones (x, y) visitadas.
        total_info: Información total acumulada.
        title:      Título del subgráfico.
        path_color: Color de la línea del camino y los marcadores.

    Returns:
        Objeto AxesImage para la barra de color compartida.
    """
    N = info_map.shape[0]

    # Mapa de calor de fondo.
    # imshow indexa [fila, columna] con origin='upper' y extent alineado
    # con la convención (x=col, y=row) del problema.
    im = ax.imshow(
        info_map,
        origin="upper",
        extent=[-0.5, N - 0.5, N - 0.5, -0.5],
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
        aspect="equal",
    )

    # Línea del camino.
    if len(path) > 1:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, "-", color=path_color, linewidth=1.8, alpha=0.85, zorder=3)

    # Marcadores de celdas visitadas (excluye inicio y fin).
    if len(path) > 2:
        mid_xs = [p[0] for p in path[1:-1]]
        mid_ys = [p[1] for p in path[1:-1]]
        ax.scatter(
            mid_xs, mid_ys,
            c=path_color, s=18, zorder=4, alpha=0.7, linewidths=0,
        )

    # Punto de inicio (0, 0).
    ax.scatter(
        [path[0][0]], [path[0][1]],
        c="cyan", s=100, zorder=6, marker="*", edgecolors="black", linewidths=0.5,
        label="Inicio",
    )

    # Punto final.
    ax.scatter(
        [path[-1][0]], [path[-1][1]],
        c="lime", s=80, zorder=6, marker="X", edgecolors="black", linewidths=0.5,
        label="Final",
    )

    n_steps = len(path) - 1
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_xlabel("x (columna)")
    ax.set_ylabel("y (fila)")
    ax.set_title(
        f"{title}\nInfo: {total_info:.4f}  |  Pasos: {n_steps}",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="lower right")
    

    
    return im


# ────────────────────────────────────────────────────────────────────────── #
# Programa principal                                                         #
# ────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    # ------------------------------------------------------------------ #
    # Configuración del experimento                                       #
    # ------------------------------------------------------------------ #
    N = 30
    MAX_BUDGET = 400.0
    MAP_SEED = 7
    N_SIMULATIONS = 300
    DEPTH = 200

    SEP = "─" * 52

    print("=" * 52)
    print("  MaxInformativePath  ·  MCTS vs Greedy vs Aleatorio")
    print("=" * 52)
    print(f"  Mapa: {N}×{N}  |  Budget: {MAX_BUDGET}  |  Semilla: {MAP_SEED}")
    print(SEP)

    print("Generando mapa informativo...")
    info_map = generate_smooth_map(N=N, n_peaks=6, seed=MAP_SEED)
    problem = MaxInformativePath(info_map=info_map, max_budget=MAX_BUDGET)

    # ------------------------------------------------------------------ #
    # Episodio 1: MCTS + política greedy como π₀                         #
    # ------------------------------------------------------------------ #
    print(f"\n[1/3] MCTS  (simulaciones={N_SIMULATIONS}, profundidad={DEPTH})")
    print(SEP)
    t0 = time.perf_counter()
    path_mcts, info_mcts = run_mcts_episode(
        problem, n_simulations=N_SIMULATIONS, depth=DEPTH, verbose=True
    )
    t_mcts = time.perf_counter() - t0
    print(SEP)
    print(f"  → Info total: {info_mcts:.4f}  |  Pasos: {len(path_mcts)-1}  |  {t_mcts:.1f} s")

    # ------------------------------------------------------------------ #
    # Episodio 2: Greedy puro (sin planificación)                         #
    # ------------------------------------------------------------------ #
    print(f"\n[2/3] Greedy puro")
    print(SEP)
    path_greedy, info_greedy = run_greedy_episode(problem)
    print(f"  → Info total: {info_greedy:.4f}  |  Pasos: {len(path_greedy)-1}")

    # ------------------------------------------------------------------ #
    # Episodio 3: Aleatorio (baseline)                                   #
    # ------------------------------------------------------------------ #
    print(f"\n[3/3] Aleatorio (baseline, semilla=42)")
    print(SEP)
    path_random, info_random = run_random_episode(problem, seed=42)
    print(f"  → Info total: {info_random:.4f}  |  Pasos: {len(path_random)-1}")

    # ------------------------------------------------------------------ #
    # Tabla comparativa                                                   #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 52)
    print("  Comparativa")
    print("=" * 52)
    print(f"  {'Método':<22} {'Info':>8} {'Pasos':>6} {'Tiempo':>8}")
    print("  " + "─" * 46)
    print(f"  {'MCTS + greedy rollout':<22} {info_mcts:>8.4f} {len(path_mcts)-1:>6} {t_mcts:>7.1f}s")
    print(f"  {'Greedy puro':<22} {info_greedy:>8.4f} {len(path_greedy)-1:>6} {'—':>8}")
    print(f"  {'Aleatorio':<22} {info_random:>8.4f} {len(path_random)-1:>6} {'—':>8}")
    print("=" * 52)

    # Estado ASCII final del episodio MCTS.
    # Reconstruimos el budget restante a partir del camino recorrido.
    import math as _math
    _DIAGONALS = {"NE", "NW", "SE", "SW"}
    _DELTAS_INV = {
        ( 0, -1): "N",  ( 0, +1): "S",
        (+1,  0): "E",  (-1,  0): "W",
        (+1, -1): "NE", (-1, -1): "NW",
        (+1, +1): "SE", (-1, +1): "SW",
    }
    budget_used = sum(
        _math.sqrt(2) if _DELTAS_INV.get(
            (path_mcts[i+1][0] - path_mcts[i][0], path_mcts[i+1][1] - path_mcts[i][1]),
            "N",
        ) in _DIAGONALS else 1.0
        for i in range(len(path_mcts) - 1)
    )
    final_state_mcts = {
        "position": path_mcts[-1],
        "budget":   MAX_BUDGET - budget_used,
        "visited":  frozenset(path_mcts),
    }
    print("\nMapa final (MCTS):")
    print(problem.render(final_state_mcts))

    # ------------------------------------------------------------------ #
    # Visualización matplotlib                                            #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        f"MaxInformativePath  ·  N={N}  ·  Budget={MAX_BUDGET}",
        fontsize=13, fontweight="bold",
    )

    im = plot_path(axes[0], info_map, path_mcts,   info_mcts,   "MCTS + greedy rollout", path_color="white")
    plot_path(axes[1], info_map, path_greedy, info_greedy, "Greedy puro",           path_color="deepskyblue")
    plot_path(axes[2], info_map, path_random, info_random, "Aleatorio",             path_color="lightgray")

    # Barra de color compartida al margen derecho.
    fig.colorbar(
        im,
        ax=axes.tolist(),
        orientation="vertical",
        fraction=0.015,
        pad=0.02,
        label="Valor informativo",
    )

    plt.tight_layout()
    plt.savefig("test_ipp_result.png", dpi=150, bbox_inches="tight")
    print("\n  Figura guardada en test_ipp_result.png")
    plt.show()


if __name__ == "__main__":
    main()
