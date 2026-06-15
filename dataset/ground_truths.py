"""
Ground truth simulators for uncertainty estimation benchmarks.

OilSpillGroundTruth: particle-based oil spill simulation on a water/land grid.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from typing import Generator


# ── Default parameters ──

DEFAULT_OIL_SPILL_PARAMS: dict = {
    # Particles
    "N_PARTICLES": 5_000,           # Initial number of pollution particles
    "MAX_FUEL": 100_000,            # Max total active particles (initial + emitted)
    "MAX_PARTICLES_PER_PIXEL": 50,  # Hard cap: particles coexisting per cell
    "SPILL_CENTER": None,           # (row, col); None → random water cell
    "SPILL_RADIUS": 3.0,            # Initial Gaussian spread (pixels)

    # Continuous emission from source_mask
    "EMISSION_RATE": 50,            # New particles added per step from source cells
    "EMISSION_SPREAD": 0.5,         # σ of Gaussian jitter around each source cell (pixels)

    # Time
    "N_STEPS": 200,                 # Number of simulation steps
    "DT": 0.5,                      # Δt per step

    # Velocity components and their weights (p_next = p + dt*(w_r*v_r + w_w*v_w + w_t*v_t))
    "W_RANDOM": 0.4,                # Weight of Brownian / diffusion term
    "W_WIND": 0.3,                  # Weight of wind advection
    "W_TIDE": 0.3,                  # Weight of tidal current

    "V_RANDOM_SCALE": 0.5,          # σ of random velocity (pixels / step)
    "WIND_SPEED": 0.8,              # Wind vector magnitude (pixels / step)
    "WIND_ANGLE": None,             # Wind direction in radians; None → drawn randomly
    "TIDE_SPEED": 0.6,              # Tidal current magnitude (pixels / step)

    # Output
    "GAUSSIAN_FILTER_SIZE": 2.0,    # σ for density smoothing

    # Reproducibility
    "SEED": None,                   # int or None
}


# ── Tidal field (Rosenbrock stream function) ────────────────

def _build_tidal_field(
    height: int,
    width: int,
    speed: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a divergence-free tidal vector field using the Rosenbrock function
    as a stream function ψ.

        ψ(x, y) = (1 - x)² + b·(y - x²)²

    A stream function gives automatically ∇·v = 0 (incompressible flow) via:

        v_col =  ∂ψ/∂y = 2b·(y - x²)
        v_row = -∂ψ/∂x = 2·(1-x) + 4b·x·(y - x²)

    The Rosenbrock banana-valley creates a curved channel with recirculation
    zones on either side—qualitatively similar to coastal tidal patterns.

    Parameters
    ----------
    height, width : int
        Grid dimensions (rows, cols).
    speed : float
        Magnitude to which each vector is normalised.

    Returns
    -------
    v_row, v_col : ndarray of shape (height, width), float32
        Row (↓) and column (→) velocity components.
    """
    # Map pixel grid to Rosenbrock domain: x ∈ [-2, 2], y ∈ [-1, 3]
    x_lin = np.linspace(-2.0, 2.0, width,  dtype=np.float64)
    y_lin = np.linspace(-1.0, 3.0, height, dtype=np.float64)
    X, Y = np.meshgrid(x_lin, y_lin)   # X ~ col direction, Y ~ row direction

    b = 5.0
    u = Y - X ** 2                      # y - x²

    v_col =  2.0 * b * u                         # ∂ψ/∂y
    v_row =  2.0 * (1.0 - X) + 4.0 * b * X * u  # -∂ψ/∂x

    mag = np.hypot(v_row, v_col)
    mag = np.where(mag < 1e-8, 1.0, mag)

    return (
        (v_row / mag * speed).astype(np.float32),
        (v_col / mag * speed).astype(np.float32),
    )


# ── Simulator ──────────

def OilSpillGroundTruth(
    water_mask: np.ndarray,
    params: dict | None = None,
    source_mask: np.ndarray | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Simulate an oil spill as a Lagrangian particle system on a water/land grid.

    Particle dynamics
    -----------------
        p[t+1] = p[t] + dt · (W_RANDOM·v_random + W_WIND·v_wind + W_TIDE·v_tide)

    - v_random  ~ N(0, V_RANDOM_SCALE²) i.i.d. per particle per step (diffusion).
    - v_wind    is a fixed 2-D vector drawn once at random per simulation run.
    - v_tide    is looked up from a precomputed, divergence-free vector field
                derived from the Rosenbrock function as a stream function.

    Particles that would move onto land are kept at their current position.
    When more than MAX_PARTICLES_PER_PIXEL particles occupy the same cell the
    excess is removed (randomly, via stable-sort grouping — fully vectorised).

    Parameters
    ----------
    water_mask : ndarray of shape (H, W)
        Binary grid: 1 = navigable water, 0 = land.
    params : dict, optional
        Override any key from DEFAULT_OIL_SPILL_PARAMS.
    source_mask : ndarray of shape (H, W), optional
        Binary grid: 1 = candidate zone for the spill origin, 0 = excluded.
        One cell is chosen at random from the valid (water ∩ source_mask)
        cells and used as the single emission point for the whole simulation.
        - Initial burst: N_PARTICLES drawn around that point (σ = SPILL_RADIUS).
        - Each step: EMISSION_RATE new particles drawn around that same point
          (σ = EMISSION_SPREAD), modelling a continuous leak.
        If None, the spill origin falls back to SPILL_CENTER / SPILL_RADIUS.

    Yields
    ------
    ndarray of shape (H, W), float32
        Gaussian-smoothed particle density at each simulation step.
        Values are proportional to local pollution concentration.
    """
    p = {**DEFAULT_OIL_SPILL_PARAMS, **(params or {})}
    rng = np.random.default_rng(p["SEED"])

    height, width = water_mask.shape
    water = water_mask.astype(bool)

    # ── Precomputed fields ────────────────

    tide_vr, tide_vc = _build_tidal_field(height, width, speed=float(p["TIDE_SPEED"]))

    # Wind: fixed magnitude, direction fixed per dataset or drawn randomly per sim
    # If WIND_ANGLE is provided (set once per dataset in DatasetGeneration),
    # all simulations in the dataset share the same wind direction.
    # If None, the angle is drawn from the per-simulation rng (original behaviour).
    if p.get("WIND_ANGLE") is not None:
        wind_angle = float(p["WIND_ANGLE"])
    else:
        wind_angle = rng.uniform(0.0, 2.0 * np.pi)
    wind_speed = float(p["WIND_SPEED"])
    wind_vr = np.float32(np.sin(wind_angle) * wind_speed)   # row component
    wind_vc = np.float32(np.cos(wind_angle) * wind_speed)   # col component

    # ── Emission origin ─

    emission_rate   = int(p["EMISSION_RATE"])
    emission_spread = float(p["EMISSION_SPREAD"])

    if source_mask is not None:
        # Pick ONE random cell from (source_mask ∩ water) as the spill origin
        candidates = np.argwhere(source_mask.astype(bool) & water)
        if candidates.size == 0:
            raise ValueError("source_mask has no cells that overlap with water_mask.")
        origin = candidates[rng.integers(len(candidates))].astype(np.float32)
    else:
        origin = None   # resolved below alongside SPILL_CENTER

    # ── Initialise particles ──────────────

    n_init = int(p["N_PARTICLES"])

    if origin is not None:
        # Burst from the chosen origin with SPILL_RADIUS spread
        pos = rng.normal(loc=origin, scale=float(p["SPILL_RADIUS"]), size=(n_init, 2)).astype(np.float32)
    else:
        if p["SPILL_CENTER"] is None:
            water_cells = np.argwhere(water)
            origin = water_cells[rng.integers(len(water_cells))].astype(np.float32)
        else:
            origin = np.array(p["SPILL_CENTER"], dtype=np.float32)
        pos = rng.normal(loc=origin, scale=float(p["SPILL_RADIUS"]), size=(n_init, 2)).astype(np.float32)

    pos[:, 0] = np.clip(pos[:, 0], 0.0, height - 1.0)
    pos[:, 1] = np.clip(pos[:, 1], 0.0, width  - 1.0)

    # Remove particles that land on non-water cells at initialisation
    ri = np.round(pos[:, 0]).astype(np.int32)
    ci = np.round(pos[:, 1]).astype(np.int32)
    pos = pos[water[ri, ci]]

    # ── Simulation constants (avoid repeated dict lookups) ──

    dt        = np.float32(p["DT"])
    w_r       = np.float32(p["W_RANDOM"])
    w_w       = np.float32(p["W_WIND"])
    w_t       = np.float32(p["W_TIDE"])
    v_scale   = np.float32(p["V_RANDOM_SCALE"])
    max_pp    = int(p["MAX_PARTICLES_PER_PIXEL"])
    sigma     = float(p["GAUSSIAN_FILTER_SIZE"])
    row_max   = np.float32(height - 1)
    col_max   = np.float32(width  - 1)

    # Time loop

    for _ in range(int(p["N_STEPS"])):

        n_act = len(pos)

        # Continuous emission from the single origin point

        if emission_rate > 0 and n_act < p["MAX_FUEL"]:
            new_pts = rng.normal(loc=origin, scale=emission_spread, size=(emission_rate, 2)).astype(np.float32)
            new_pts[:, 0] = np.clip(new_pts[:, 0], 0.0, row_max)
            new_pts[:, 1] = np.clip(new_pts[:, 1], 0.0, col_max)
            # Discard any that fell on land
            nr = np.round(new_pts[:, 0]).astype(np.int32)
            nc = np.round(new_pts[:, 1]).astype(np.int32)
            new_pts = new_pts[water[nr, nc]]
            pos = np.concatenate([pos, new_pts], axis=0) if len(pos) > 0 else new_pts
            n_act = len(pos)

        if n_act == 0:
            yield np.zeros((height, width), dtype=np.float32)
            continue

        # Snapshot positions after emission, before any movement.
        # Blocked particles (land or overcrowding) revert here.
        old_pos = pos.copy()

        # ── Velocity components ───────────

        # (1) Brownian diffusion — i.i.d. Gaussian, shape (n_act, 2)
        v_rand = rng.standard_normal((n_act, 2)).astype(np.float32) * v_scale

        # (2) Wind — broadcast scalar to all particles
        #     (row component = sin, col component = cos)

        # (3) Tidal current — bilinear interpolation from precomputed field
        coords = [pos[:, 0], pos[:, 1]]
        vt_r = map_coordinates(tide_vr, coords, order=1, mode="nearest").astype(np.float32)
        vt_c = map_coordinates(tide_vc, coords, order=1, mode="nearest").astype(np.float32)

        # ── Position update ───────────────

        delta_r = dt * (w_r * v_rand[:, 0] + w_w * wind_vr + w_t * vt_r)
        delta_c = dt * (w_r * v_rand[:, 1] + w_w * wind_vc + w_t * vt_c)

        new_pos = pos + np.stack([delta_r, delta_c], axis=1)
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0.0, row_max)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0.0, col_max)

        # Land boundary: particles that would enter land revert to old_pos
        nri = new_pos[:, 0].astype(np.int32)
        nci = new_pos[:, 1].astype(np.int32)
        on_water = water[nri, nci]
        pos = np.where(on_water[:, np.newaxis], new_pos, old_pos)

        # ── MAX_PARTICLES_PER_PIXEL enforcement (vectorised) 
        #
        # Particles that would overfill a cell revert to old_pos instead of
        # being deleted. Within each cell, the first max_pp (by stable-sort
        # order) keep their new position; the rest return to old_pos.

        ri_int = pos[:, 0].astype(np.int32)
        ci_int = pos[:, 1].astype(np.int32)
        pix = ri_int * width + ci_int           # linear pixel index, shape (n_act,)

        order      = np.argsort(pix, kind="stable")
        pix_sorted = pix[order]

        # Mark the start of each new pixel group
        is_start = np.empty(n_act, dtype=bool)
        is_start[0] = True
        is_start[1:] = pix_sorted[1:] != pix_sorted[:-1]

        # Group-start absolute indices; propagate forward with cummax
        group_start = np.where(is_start, np.arange(n_act, dtype=np.int32), 0)
        np.maximum.accumulate(group_start, out=group_start)

        rank_in_group = np.arange(n_act, dtype=np.int32) - group_start

        fits = np.empty(n_act, dtype=bool)
        fits[order] = rank_in_group < max_pp
        pos = np.where(fits[:, np.newaxis], pos, old_pos)

        # ── Density map ─

        ri_out = pos[:, 0].astype(np.int32)
        ci_out = pos[:, 1].astype(np.int32)

        density = np.zeros((height, width), dtype=np.float32)
        np.add.at(density, (ri_out, ci_out), 1.0)

        # Normalise to [0, 1] by dividing by the hard per-pixel cap.
        # MAX_PARTICLES_PER_PIXEL is the theoretical maximum count any pixel
        # can hold before the Gaussian filter is applied, so dividing by it
        # maps the smoothed density to the (0, 1] range.
        unnormalized = gaussian_filter(density, sigma=sigma).astype(np.float32) * water_mask
        # Min Max Normalization
        normalized = (unnormalized - unnormalized.min()) / (unnormalized.max() - unnormalized.min() + 1e-5)

        yield normalized


# ── Quick visual demo ───

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation


    # Simple map:
    H, W = 100, 100
    water_mask = np.ones((H, W), dtype=np.uint8)
    # Set to 0 the squared borders
    water_mask[:8, :] = 0
    water_mask[:, :8] = 0
    water_mask[-8:, :] = 0
    water_mask[:, -8:] = 0

    # Creamos un source_mask que sea una línea diagonal
    source_mask = np.zeros_like(water_mask)
    source_mask[np.arange(10, 90), np.arange(10, 90)] = 1


    sim_params = {
    # Particles
    "N_PARTICLES": 1_000,           # Initial number of pollution particles
    "MAX_FUEL": 15_000,              # Max total particles (initial + emitted)
    "MAX_PARTICLES_PER_PIXEL": 50,  # Hard cap: particles coexisting per cell
    "SPILL_CENTER": None,           # (row, col); None → random water cell
    "SPILL_RADIUS": 3.0,            # Initial Gaussian spread (pixels)

    # Continuous emission from source_mask
    "EMISSION_RATE": 50,            # New particles added per step from source cells
    "EMISSION_SPREAD": 0.5,         # σ of Gaussian jitter around each source cell (pixels)

    # Time
    "N_STEPS": 200,                 # Number of simulation steps
    "DT": 0.5,                      # Δt per step

    # Velocity components and their weights (p_next = p + dt*(w_r*v_r + w_w*v_w + w_t*v_t))
    "W_RANDOM": 2.0,                # Weight of Brownian / diffusion term
    "W_WIND": 0.5,                  # Weight of wind advection
    "W_TIDE": 0.5,                  # Weight of tidal current

    "V_RANDOM_SCALE": 1,          # σ of random velocity (pixels / step)
    "WIND_SPEED": 1,              # Wind vector magnitude (pixels / step)
    "TIDE_SPEED": 2,              # Tidal current magnitude (pixels / step)

    # Output
    "GAUSSIAN_FILTER_SIZE": 2.0,    # σ for density smoothing

    # Reproducibility
    "SEED": 2,                   # int or None
}

    frames = list(
        OilSpillGroundTruth(water_mask, sim_params, source_mask=source_mask)
    )

    # Show tidal field
    vr, vc = _build_tidal_field(H, W)
    step = 6
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_title("Tidal vector field (Rosenbrock stream function)")
    axes[0].imshow(water_mask, cmap="Blues", vmin=0, vmax=1.5)

    axes[0].imshow(source_mask, cmap="Greens", vmin=0, vmax=1, alpha=source_mask)
    axes[0].quiver(
        np.arange(0, W, step),
        np.arange(0, H, step),
        vc[::step, ::step],
        -vr[::step, ::step],   # flip row axis for display
        color="white", scale=30, width=0.004,
    )

    im = axes[1].imshow(frames[0], cmap="hot", vmin=0, interpolation="bicubic")
    axes[1].imshow(water_mask == 0, cmap="Greens", alpha=0.4)
    axes[1].set_title("Oil spill density")

    def _update(i):
        im.set_data(frames[i])
        axes[1].set_title(f"Oil spill — step {i}")
        return (im,)

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), interval=60, blit=True)
    plt.tight_layout()
    plt.show()
