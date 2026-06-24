"""Comparativa MCTS vs Forward Search vs Greedy sobre MaxInformativePathWaypoints."""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from scenario.policies.algorithms.forward_search import ForwardSearch
from scenario.policies.algorithms.mcts import MCTS, MCTSNode
from scenario.max_informative_path import generate_smooth_map
from scenario.max_informative_path_waypoints import MaxInformativePathWaypoints, _shortest_path


# ────────────────────────────────────────────────────────────────────────── #
# Visualización del árbol MCTS                                               #
# ────────────────────────────────────────────────────────────────────────── #

def render_mcts_tree(
    ax: plt.Axes,
    root: MCTSNode,
    color: str = "deepskyblue",
    linewidth: float = 0.5,
    min_alpha: float = 0.05,
    max_alpha: float = 0.75,
) -> None:
    """
    Dibuja el árbol MCTS sobre un eje matplotlib.

    La opacidad de cada arista es proporcional a N(s,a) / N_max,
    resaltando los caminos más explorados. Requiere que cada estado del
    árbol tenga un campo ``position`` con coordenadas (x, y).
    """
    edges: list[tuple[tuple, tuple, int]] = []
    max_n = 1
    stack = [root]
    while stack:
        node = stack.pop()
        p_pos = node.state["position"]
        for action, child in node.children.items():
            n_sa = node.N.get(action, 0)
            if n_sa > 0:
                edges.append((p_pos, child.state["position"], n_sa))
                if n_sa > max_n:
                    max_n = n_sa
            stack.append(child)
    if not edges:
        return
    base_rgba = to_rgba(color)
    segments = [[(px, py), (cx, cy)] for (px, py), (cx, cy), _ in edges]
    colors = [
        (*base_rgba[:3], min_alpha + (max_alpha - min_alpha) * (n / max_n))
        for _, _, n in edges
    ]
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=linewidth, zorder=2))
# ────────────────────────────────────────────────────────────────────────── #
# Ejecución de episodios                                                     #
# ────────────────────────────────────────────────────────────────────────── #

def run_mcts_episode(
    problem: MaxInformativePathWaypoints,
    n_simulations: int,
    depth: int,
    verbose: bool = True,
) -> tuple[list[tuple[int, int]], float, MCTS]:
    """
    Ejecuta un episodio completo con MCTS.

    La política de rollout es greedy: siempre se mueve al candidato de
    mayor recompensa inmediata disponible.

    Args:
        problem:       Instancia del problema (con candidate_fn ya asignada).
        n_simulations: Simulaciones MCTS por decisión.
        depth:         Profundidad del árbol.
        verbose:       Si True, imprime el progreso paso a paso.

    Returns:
        (waypoints, total_info, planner) donde waypoints es la lista de
        destinos visitados en orden, total_info la información acumulada
        y planner el objeto MCTS con el árbol del último paso disponible
        en planner.root para visualización.
    """
    def greedy_rollout(state: dict) -> tuple[int, int] | None:
        actions = problem.get_actions(state)
        if not actions:
            return None
        return max(actions, key=lambda a: problem.reward(state, a))

    planner = MCTS(
        problem=problem,
        n_simulations=n_simulations,
        depth=depth,
        gamma=0.99,
        exploration_c=0.1,
        rollout_policy=greedy_rollout,
        reuse_tree=True,
        control_horizon=1,
    )

    state = problem.initial_state()
    waypoints: list[tuple[int, int]] = [state["position"]]
    total_info: float = 0.0
    step = 0

    while not problem.is_terminal(state):
        actions = problem.get_actions(state)
        if not actions:
            break

        action, q_value = planner.select_action(state)
        if action is None:
            break
        state, reward = problem.transition(state, action)
        total_info += reward
        waypoints.append(state["position"])
        step += 1

        if verbose:
            print(
                f"  paso {step:>3}  destino={str(action):<10}  "
                f"r={reward:.4f}  Q={q_value:.4f}  "
                f"budget={state['budget']:.2f}  visitadas={len(state['visited'])}"
            )

        # Render interactivo: mapa con el estado actual + árbol MCTS.
        ax = problem.render(state)
        if planner.decision_root is not None:
            render_mcts_tree(ax, planner.decision_root)
        ax.figure.canvas.draw()
        plt.pause(0.01)

    return waypoints, total_info, planner


def run_forward_search_episode(
    problem: MaxInformativePathWaypoints,
    depth: int,
    verbose: bool = True,
) -> tuple[list[tuple[int, int]], float]:
    """
    Ejecuta un episodio completo con Forward Search.

    ForwardSearch llama internamente a problem.successors(). Como el escenario
    waypoints es determinista, se inyecta un wrapper que devuelve
    [(next_state, 1.0)] para cada (state, action), adaptando la interfaz sin
    modificar el escenario.

    Args:
        problem: Instancia del problema (con candidate_fn ya asignada).
        depth:   Profundidad de lookahead por decisión.
        verbose: Si True, imprime el progreso paso a paso.

    Returns:
        (waypoints, total_info).
    """
    class _DeterministicWrapper:
        def __getattr__(self, name):
            return getattr(problem, name)

        def successors(self, state, action):
            next_state, _ = problem.transition(state, action)
            return [(next_state, 1.0)]

    wrapped = _DeterministicWrapper()
    planner = ForwardSearch(problem=wrapped, depth=depth, gamma=1.0)

    state = problem.initial_state()
    waypoints: list[tuple[int, int]] = [state["position"]]
    total_info: float = 0.0
    step = 0

    while not problem.is_terminal(state):
        actions = problem.get_actions(state)
        if not actions:
            break

        action, value = planner.select_action(state)
        if action is None:
            break
        state, reward = problem.transition(state, action)
        total_info += reward
        waypoints.append(state["position"])
        step += 1

        if verbose:
            print(
                f"  paso {step:>3}  destino={str(action):<10}  "
                f"r={reward:.4f}  V={value:.4f}  "
                f"budget={state['budget']:.2f}  visitadas={len(state['visited'])}"
            )

    return waypoints, total_info


def run_greedy_episode(
    problem: MaxInformativePathWaypoints,
    verbose: bool = True,
) -> tuple[list[tuple[int, int]], float]:
    """
    Ejecuta un episodio con política puramente greedy.

    En cada paso se elige el destino candidato que maximiza la recompensa
    inmediata del trayecto completo.

    Args:
        problem: Instancia del problema (con candidate_fn ya asignada).
        verbose: Si True, imprime el progreso paso a paso.

    Returns:
        (waypoints, total_info).
    """
    state = problem.initial_state()
    waypoints: list[tuple[int, int]] = [state["position"]]
    total_info: float = 0.0
    step = 0

    while not problem.is_terminal(state):
        actions = problem.get_actions(state)
        if not actions:
            break

        best = max(actions, key=lambda a: problem.reward(state, a) + problem.utility(state))
        state, reward = problem.transition(state, best)
        total_info += reward
        waypoints.append(state["position"])
        step += 1

        if verbose:
            print(
                f"  paso {step:>3}  destino={str(best):<10}  "
                f"r={reward:.4f}  budget={state['budget']:.2f}  "
                f"visitadas={len(state['visited'])}"
            )

    return waypoints, total_info


# ────────────────────────────────────────────────────────────────────────── #
# Visualización                                                              #
# ────────────────────────────────────────────────────────────────────────── #

def reconstruct_full_path(
    waypoints: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Reconstruye el camino celda-a-celda a partir de la lista de waypoints.

    Args:
        waypoints: Secuencia de destinos visitados (incluyendo el origen).

    Returns:
        Lista completa de celdas recorridas en orden.
    """
    if not waypoints:
        return []
    full = [waypoints[0]]
    for i in range(1, len(waypoints)):
        full.extend(_shortest_path(waypoints[i - 1], waypoints[i]))
    return full


def plot_result(
    ax: plt.Axes,
    info_map: np.ndarray,
    waypoints: list[tuple[int, int]],
    total_info: float,
    title: str,
    path_color: str = "white",
    planner: MCTS | None = None,
) -> None:
    """
    Dibuja el mapa con el camino del agente y los waypoints superpuestos.

    Si se proporciona un planner MCTS, superpone el árbol de búsqueda
    construido en el último paso: cada arista se dibuja con opacidad
    proporcional a N(s,a) / max_N.

    Args:
        ax:         Eje matplotlib.
        info_map:   Mapa N×N con valores en [0, 1].
        waypoints:  Lista de destinos visitados en orden.
        total_info: Información total acumulada (para el subtítulo).
        title:      Título del panel.
        path_color: Color de la polilínea del camino.
        planner:    Objeto MCTS opcional. Si se pasa, se dibuja el árbol
                    de búsqueda sobre el mapa con render_tree.
    """
    ax.imshow(
        info_map, cmap="viridis", vmin=0.0, vmax=1.0,
        origin="upper", interpolation="nearest",
    )

    # Árbol MCTS superpuesto (debajo del camino agente para no taparlo).
    if planner is not None and planner.decision_root is not None:
        render_mcts_tree(ax, planner.decision_root)

    # Camino completo celda a celda.
    full_path = reconstruct_full_path(waypoints)
    if len(full_path) > 1:
        xs, ys = zip(*full_path)
        ax.plot(xs, ys, color=path_color, linewidth=1.5, alpha=0.85, zorder=6)

    # Waypoints intermedios (cuadrados numerados).
    for i, (wx, wy) in enumerate(waypoints[:-1]):
        ax.plot(
            wx, wy, marker="s", color="lightblue", markersize=5,
            markeredgecolor="black", markeredgewidth=0.5, zorder=7,
        )
        ax.text(wx + 0.15, wy - 0.15, str(i), color="white", fontsize=5, zorder=8)

    # Posición final (estrella dorada).
    if waypoints:
        fx, fy = waypoints[-1]
        ax.plot(
            fx, fy, marker="*", color="gold", markersize=12,
            markeredgecolor="black", markeredgewidth=0.8, zorder=9,
        )

    ax.set_title(f"{title}\nInformación: {total_info:.4f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


# ────────────────────────────────────────────────────────────────────────── #
# Programa principal                                                         #
# ────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    # ── Parámetros del experimento ─────────────────────────────────
    N             = 50
    MAX_BUDGET    = 150.0
    MAP_SEED      = 0
    K_CANDIDATES  = 5    # destinos propuestos por la función candidata (top-k)
    RESOLUTION    = 2   # subgrid resolution: paso N/RESOLUTION → (N/res)² candidatos
                         # resolution=10 → 25 candidatos; resolution=5 → 100; resolution=2 → 625
    N_SIMULATIONS = 2000  # simulaciones MCTS por decisión                        
                         # regla práctica: >> branching_factor para que el árbol profundice
    DEPTH_MCTS    = 200   # profundidad MCTS — el episodio tiene ~7-15 saltos en total
    DEPTH_FS      = 5    # profundidad Forward Search (coste: branching_factor^DEPTH_FS)

    info_map = generate_smooth_map(N, n_peaks=6, seed=MAP_SEED)

    # ── MCTS con candidatos subgrid ────────────────────────────────────
    print("=" * 65)
    print("MCTS  —  candidatos subgrid (resolution=2)")
    print("=" * 65)

    problem_mcts = MaxInformativePathWaypoints(
        info_map=info_map,
        max_budget=MAX_BUDGET,
        gamma=0.99,
        anneal_radius=2
        
    )
    
    problem_mcts.candidate_fn = problem_mcts.candidates_adaptive()
    
    n_candidates = (N // RESOLUTION) ** 2
    print(f"  branching factor ≈ {n_candidates}  (N_SIMULATIONS={N_SIMULATIONS} → árbol profundidad ~{N_SIMULATIONS // n_candidates})")
    
    t0 = time.time()
    waypoints_mcts, info_mcts, planner_mcts = run_mcts_episode(
        problem_mcts, n_simulations=N_SIMULATIONS, depth=DEPTH_MCTS, verbose=True
    )
    t_mcts = time.time() - t0

    # ── Forward Search con candidatos top-k ───────────────────────────
    print("\n" + "=" * 65)
    print("Forward Search  —  candidatos subgrid  (resolution=2, d={})".format(DEPTH_FS))
    print("=" * 65)

    problem_fs = MaxInformativePathWaypoints(
        info_map=info_map,
        max_budget=MAX_BUDGET,
    )
    problem_fs.candidate_fn = problem_fs.candidates_8grid(resolution=2)

    t0 = time.time()
    waypoints_fs, info_fs = run_forward_search_episode(
        problem_fs, depth=DEPTH_FS, verbose=True
    )
    t_fs = time.time() - t0

    # ── Greedy con candidatos 8-vecinos ───────────────────────────────
    print("\n" + "=" * 65)
    print("Greedy  —  8-vecinos")
    print("=" * 65)

    problem_greedy = MaxInformativePathWaypoints(
        info_map=info_map,
        max_budget=MAX_BUDGET,
    )
    problem_greedy.candidate_fn = problem_greedy.candidates_8grid()

    t0 = time.time()
    waypoints_greedy, info_greedy = run_greedy_episode(
        problem_greedy, verbose=True
    )
    t_greedy = time.time() - t0

    # ── Tabla comparativa ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Método':<30} {'Waypoints':>10} {'Información':>13} {'Tiempo (s)':>12}")
    print("-" * 65)
    print(f"{'MCTS subgrid':<30} {len(waypoints_mcts):>10} {info_mcts:>13.4f} {t_mcts:>12.2f}")
    print(f"{'FS subgrid d={}'.format(DEPTH_FS):<30} {len(waypoints_fs):>10} {info_fs:>13.4f} {t_fs:>12.2f}")
    print(f"{'Greedy 8-vecinos':<30} {len(waypoints_greedy):>10} {info_greedy:>13.4f} {t_greedy:>12.2f}")
    print("=" * 65)

    # ── Visualización estática final ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_result(
        axes[0], info_map, waypoints_mcts, info_mcts,
        f"MCTS subgrid  |  {N_SIMULATIONS} sims  |  d={DEPTH_MCTS}",
        planner=planner_mcts,
    )
    plot_result(
        axes[1], info_map, waypoints_fs, info_fs,
        f"Forward Search  subgrid  d={DEPTH_FS}",
        path_color="lime",
    )
    plot_result(
        axes[2], info_map, waypoints_greedy, info_greedy,
        "Greedy  8-vecinos",
        path_color="cyan",
    )

    fig.suptitle(
        f"IPP Waypoints  —  N={N}  budget={MAX_BUDGET}  k={K_CANDIDATES}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig("test_ipp_waypoints_result.png", dpi=150)
    print("\nFigura guardada en test_ipp_waypoints_result.png")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
