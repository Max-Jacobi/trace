"""
tests/test_integration_blast.py

Exploratory end-to-end integration test: 2D asymmetric blast wave.

Setup
-----
A Sedov-Taylor-like blast wave expands radially from an off-centre point
(X_C, Y_C) = (0.5, 0.3).  The shock front is spherical (in the effective
elliptic metric) so analytic tracer trajectories are available inside the
shock.  The density field is angularly asymmetric to give an interesting
M(<r) profile.

Velocity field (exact inside shock, zero outside)
    v_x(x, y, t) = ALPHA / (1 + t) * (x - X_C) * f(xi)
    v_y(x, y, t) = ALPHA / (1 + t) * (y - Y_C) * f(xi)
where f(xi) = 0.5 * (1 - tanh((xi - 1) / SIGMA)) is a smooth shock
indicator and xi = r / R_s(t) with R_s(t) = (1 + t)^ALPHA.

Exact tracer trajectory inside shock (xi0 < 1):
    x_exact(t) = X_C + (x0 - X_C) * ((1 + t) / (1 + T_START))^ALPHA
    y_exact(t) = Y_C + (y0 - Y_C) * ((1 + t) / (1 + T_START))^ALPHA

Density field
    rho(xi, theta) = ambient * angular(theta) * (1 - f(xi))
                   + rho_inner * f(xi)
                   + rho_peak * angular(theta) * exp(-((xi-1)/sig_rho)^2)
where angular(theta) = 1 + 0.5*sin(2*theta + 0.8).
This makes the mass distribution genuinely asymmetric even though the
velocity field is radially symmetric.

Schemes compared
----------------
  1. RegularInterpolator3D  (linear spatial)  + ForwardEuler  (1st order)
  2. PchipInterpolator3D    (cubic spatial)   + ImplicitTrapezoid (2nd order)
  3. PchipInterpolator3D    (cubic spatial)   + RK4 (4th order, monotone PCHIP time)

Plots
-----
All plots are saved to tests/plots/:
  blast_wave_final_positions.png  - scatter of tracer positions at T_END
                                    on a density background; one panel per
                                    scheme.
  blast_wave_M_R.png              - cumulative enclosed mass M(<r) at 4
                                    timesteps; all schemes on each panel.
  blast_wave_errors.png           - RMS position error vs time for tracers
                                    well inside the shock (xi0 < 0.6),
                                    where the analytic trajectory is exact.

No failure conditions: the test passes as long as no exception is raised.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from multiprocessing.shared_memory import SharedMemory

from src.integrators.base import IntegratorBase
from src.integrators.impl_trapezoid import ImplicitTrapezoid
from src.integrators.rk4 import RK4
from src.interpolators.regular import RegularInterpolator3D
from src.interpolators.pchip import PchipInterpolator3D


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

X_C   = 0.5    # explosion centre x
Y_C   = 0.3    # explosion centre y (off-centre for visual asymmetry)
ALPHA = 0.5    # R_shock(t) = (1 + t)^ALPHA  (2-D Sedov exponent)
SIGMA = 0.06   # shock-front width (tanh half-width in xi units)

RHO_SIGMA = 0.08   # density-spike half-width in xi
RHO_PEAK  = 4.0    # density at shock shell (compressed material)
RHO_INNER = 0.05   # density inside cavity (swept-out)
RHO_AMB   = 1.0    # ambient density outside shock


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------

T_START = 1.0
T_END   = 4.0
N_STEPS = 20
DT = (T_END - T_START) / N_STEPS

# TIMES has N_STEPS + 3 entries:
#   index 0            = T_START - DT         (left padding for RK4)
#   index 1            = T_START
#   index N_STEPS + 1  = T_END
#   index N_STEPS + 2  = T_END + DT           (right padding for RK4)
#
# Integration step i  (0-indexed, i in [0, N_STEPS-1]) advances the tracers
# from TIMES[i+1] to TIMES[i+2].
TIMES = np.linspace(T_START - DT, T_END + DT, N_STEPS + 3)


# ---------------------------------------------------------------------------
# Spatial grid (3-D; z is a thin dummy axis so PCHIP stencils stay valid)
# ---------------------------------------------------------------------------

NX, NY, NZ = 60, 60, 7
X_GRID = np.linspace(-4.0, 6.0, NX)
Y_GRID = np.linspace(-5.0, 5.0, NY)
Z_GRID = np.linspace(0.0, 1.0, NZ)

# All tracers are fixed at this z value.
# With Z_GRID = linspace(0,1,7), dz = 1/6.
# iz = floor((0.5 - 0) / (1/6)) = 3, iz0 = iz-1 = 2.
# Valid: iz0 >= 0 and iz0+4 = 6 <= 7.  OK.
Z_C = 0.5


# ---------------------------------------------------------------------------
# Analytic blast wave
# ---------------------------------------------------------------------------

def _xi(x, y, t):
    """
    Dimensionless shock-normalised radius: xi = 1 at shock, xi < 1 inside.

    R_shock(t) = (1 + t)^ALPHA.
    """
    R = (1.0 + t) ** ALPHA
    return np.sqrt((x - X_C) ** 2 + (y - Y_C) ** 2) / R


def _f_inside(xi):
    """Smooth indicator: ~1 deep inside shock, ~0 outside."""
    return 0.5 * (1.0 - np.tanh((xi - 1.0) / SIGMA))


def _angular(x, y):
    """
    Angular density variation: range [0.5, 1.5].
    Creates an asymmetric mass distribution (denser in upper-right).
    """
    theta = np.arctan2(y - Y_C, x - X_C)
    return 1.0 + 0.5 * np.sin(2.0 * theta + 0.8)


def analytic_vx(x, y, t):
    """x-velocity of the blast wave."""
    return (ALPHA / (1.0 + t)) * (x - X_C) * _f_inside(_xi(x, y, t))


def analytic_vy(x, y, t):
    """y-velocity of the blast wave."""
    return (ALPHA / (1.0 + t)) * (y - Y_C) * _f_inside(_xi(x, y, t))


def analytic_rho(x, y, t):
    """
    Density field.

    Structure:
    - Compressed shell at xi ~ 1 (spike amplitude modulated by angular factor)
    - Swept-out cavity inside (xi < 1)
    - Ambient medium outside (xi > 1), also modulated by angular factor
    """
    xi = _xi(x, y, t)
    ang = _angular(x, y)
    f_in = _f_inside(xi)
    spike = RHO_PEAK * ang * np.exp(-((xi - 1.0) / RHO_SIGMA) ** 2)
    ambient = RHO_AMB * ang * (1.0 - f_in)
    cavity  = RHO_INNER * f_in
    return ambient + cavity + spike


def exact_position(x0, y0, t):
    """
    Analytic tracer position at time t for tracers starting inside the shock.

    Valid when xi(x0, y0, T_START) << 1, i.e., f_inside ~ 1 throughout.
    The velocity field is then v = ALPHA/(1+t) * dr_vec, giving:
        r(t) = r0 * ((1+t)/(1+T_START))^ALPHA
    which means xi = r(t)/R_s(t) = r0/R_s(T_START) = const (xi is conserved).
    """
    ratio = ((1.0 + t) / (1.0 + T_START)) ** ALPHA
    x_ex = X_C + (x0 - X_C) * ratio
    y_ex = Y_C + (y0 - Y_C) * ratio
    return x_ex, y_ex


# ---------------------------------------------------------------------------
# Shared-memory snapshot helper
# ---------------------------------------------------------------------------

class _SnapInterp:
    """
    Velocity interpolator for one time snapshot backed by shared memory.

    The callable interface accepts position coordinates of shape (3, n) and
    returns velocity of shape (3, n) = [vx, vy, vz], compatible with the
    integrators in src/integrators.  vz is identically zero (2-D field).
    """

    def __init__(self, t: float, interp_class):
        xx, yy = np.meshgrid(X_GRID, Y_GRID, indexing="ij")  # (NX, NY)
        vx2d = analytic_vx(xx, yy, t)
        vy2d = analytic_vy(xx, yy, t)

        shape = (NX, NY, NZ)
        fields = {
            "vx": np.broadcast_to(vx2d[:, :, np.newaxis], shape).copy(),
            "vy": np.broadcast_to(vy2d[:, :, np.newaxis], shape).copy(),
            "vz": np.zeros(shape),
        }

        self._shm: dict[str, SharedMemory] = {}
        shm_names: dict[str, str] = {}
        for key, arr in fields.items():
            buf = np.asarray(arr, dtype=np.float64)
            shm = SharedMemory(create=True, size=max(buf.nbytes, 1))
            np.ndarray(shape, dtype=np.float64, buffer=shm.buf)[:] = buf
            self._shm[key] = shm
            shm_names[key] = shm.name

        self._interp = interp_class(X_GRID, Y_GRID, Z_GRID,
                                    shm=shm_names, shape=shape)
        self._interp.load()
        self._closed = False

    def __call__(self, xn: np.ndarray) -> np.ndarray:
        """Evaluate velocity at positions xn of shape (3, n); returns (3, n)."""
        return self._interp(xn)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._interp.unload()
        for shm in self._shm.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Forward Euler integrator (1st-order, single snapshot)
# ---------------------------------------------------------------------------

class _ForwardEuler(IntegratorBase):
    """Forward Euler: x_{n+1} = x_n + dt * v(x_n, t_n).  Baseline only."""

    n_snapshots = 1

    def __call__(self, xn, dt, interps, snap_times=None):
        return xn + dt * interps[0](xn)


# ---------------------------------------------------------------------------
# Integration driver
# ---------------------------------------------------------------------------

def _run_scheme(interp_class, integrator, x0: np.ndarray, y0: np.ndarray):
    """
    Integrate tracers through the blast wave with a given spatial interpolator
    class and time integrator.

    Parameters
    ----------
    interp_class : type
        RegularInterpolator3D or PchipInterpolator3D.
    integrator : IntegratorBase instance
        _ForwardEuler, ImplicitTrapezoid, or RK4.
    x0, y0 : ndarray, shape (n_tr,)
        Initial tracer positions at T_START.

    Returns
    -------
    traj : ndarray, shape (N_STEPS + 1, 3, n_tr)
        Tracer positions at each of the N_STEPS + 1 stored time levels
        (T_START to T_END inclusive).

    Notes
    -----
    TIMES has N_STEPS + 3 entries.  Integration step i advances from
    TIMES[i+1] to TIMES[i+2].

    Snapshot indices consumed per step:
      n_snapshots == 1  (Euler):    [i+1]
      n_snapshots == 2  (Trap.):    [i+1, i+2]
      n_snapshots == 4  (RK4):      [i, i+1, i+2, i+3]  (cubic time interp)
    """
    n_tr = len(x0)
    xn = np.array([x0, y0, np.full(n_tr, Z_C)], dtype=float)  # (3, n_tr)
    traj = [xn.copy()]

    n_snap = integrator.n_snapshots
    snap_cache: dict[int, _SnapInterp] = {}

    def _get(k: int) -> _SnapInterp:
        if k not in snap_cache:
            snap_cache[k] = _SnapInterp(TIMES[k], interp_class)
        return snap_cache[k]

    def _evict(k_min: int):
        stale = [k for k in snap_cache if k < k_min]
        for k in stale:
            snap_cache[k].close()
            del snap_cache[k]

    for step_i in range(N_STEPS):
        if n_snap == 4:
            needed = [step_i, step_i + 1, step_i + 2, step_i + 3]
        elif n_snap == 2:
            needed = [step_i + 1, step_i + 2]
        else:  # n_snap == 1
            needed = [step_i + 1]

        _evict(min(needed))
        interps = tuple(_get(k) for k in needed)
        dt = float(TIMES[step_i + 2] - TIMES[step_i + 1])
        st = TIMES[needed] if n_snap == 4 else None

        xn = integrator(xn, dt, interps, snap_times=st)
        traj.append(xn.copy())

    for snap in snap_cache.values():
        snap.close()

    return np.array(traj)  # (N_STEPS+1, 3, n_tr)


# ---------------------------------------------------------------------------
# Tracer setup
# ---------------------------------------------------------------------------

def _init_tracers():
    """
    Place 225 tracers on a 15x15 Cartesian grid covering both the interior
    and exterior of the initial shock.

    Initial shock radius at T_START = 1.0:
        R_s(T_START) = (1 + 1)^0.5 = sqrt(2) ~ 1.41

    Tracer x range [-2.5, 3.5], y range [-3.0, 3.0].

    Returns
    -------
    x0, y0 : ndarray, shape (225,)
    masses  : ndarray, shape (225,)
        m_i = rho(x_i, y_i, T_START) * dA  (Lagrangian mass, conserved).
    """
    xi_vals = np.linspace(-2.5, 3.5, 15)
    yi_vals = np.linspace(-3.0, 3.0, 15)
    xx0, yy0 = np.meshgrid(xi_vals, yi_vals, indexing="ij")
    x0 = xx0.ravel()
    y0 = yy0.ravel()

    dA = (xi_vals[1] - xi_vals[0]) * (yi_vals[1] - yi_vals[0])
    masses = analytic_rho(x0, y0, T_START) * dA
    return x0, y0, masses


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _density_bg(t, ax):
    """Draw density field at time t as a colour mesh on ax."""
    xx, yy = np.meshgrid(X_GRID, Y_GRID, indexing="ij")
    rho = analytic_rho(xx, yy, t)
    ax.pcolormesh(X_GRID, Y_GRID, rho.T, cmap="inferno",
                  shading="auto", vmin=0.0, vmax=RHO_PEAK * 0.6)


def _shock_circle(t, ax, **kw):
    """Overlay the analytic shock circle at time t."""
    theta = np.linspace(0, 2.0 * np.pi, 300)
    R = (1.0 + t) ** ALPHA
    ax.plot(X_C + R * np.cos(theta), Y_C + R * np.sin(theta), **kw)


def _M_R(traj_step, masses, n_bins=80, r_max=5.0):
    """
    Compute cumulative enclosed mass M(<r) from tracer positions at one step.

    Parameters
    ----------
    traj_step : ndarray, shape (3, n_tr)
    masses    : ndarray, shape (n_tr,)
    n_bins    : int
    r_max     : float

    Returns
    -------
    r_centres : ndarray, shape (n_bins,)
    M_cumul   : ndarray, shape (n_bins,)
    """
    xpos, ypos = traj_step[0], traj_step[1]
    r = np.sqrt((xpos - X_C) ** 2 + (ypos - Y_C) ** 2)
    bins = np.linspace(0.0, r_max, n_bins + 1)
    M_bin, _ = np.histogram(r, bins=bins, weights=masses)
    return 0.5 * (bins[:-1] + bins[1:]), np.cumsum(M_bin)


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------

SCHEMES = [
    ("Linear + Euler",       RegularInterpolator3D, _ForwardEuler()),
    ("PCHIP + Impl.Trap.",   PchipInterpolator3D,   ImplicitTrapezoid()),
    ("PCHIP + RK4",          PchipInterpolator3D,   RK4(monotone=True)),
]

COLORS     = ["tab:blue", "tab:orange", "tab:green"]
LINE_STYLES = ["--", "-.", "-"]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestBlastWaveIntegration:
    """
    Exploratory end-to-end integration: 2D asymmetric blast wave.

    Runs three schemes on a prescribed blast-wave velocity field sampled on a
    60x60x7 grid with 20 time steps (T = 1 -> 4).  Saves three comparison
    plots to tests/plots/.

    No failure conditions: the test always passes if no exception is raised.
    """

    def test_blast_wave_comparison(self):
        """
        Run all three schemes, record tracer trajectories, and save plots.
        Expected runtime: ~30-60 s (PCHIP builds x-interpolator cache lazily).
        """
        os.makedirs(PLOT_DIR, exist_ok=True)

        x0, y0, masses = _init_tracers()
        n_tr = len(x0)

        # ---- run all three schemes ----------------------------------------
        results: dict[str, np.ndarray] = {}
        for name, interp_cls, integr in SCHEMES:
            print(f"\n  [{name}] integrating {n_tr} tracers x {N_STEPS} steps ...",
                  flush=True)
            traj = _run_scheme(interp_cls, integr, x0, y0)
            results[name] = traj
            print(f"  [{name}] done.  traj shape: {traj.shape}", flush=True)

        # Actual time at each stored level: TIMES[1..N_STEPS+1]
        t_levels = TIMES[1: N_STEPS + 2]  # length N_STEPS+1

        # ================================================================
        # Plot 1: final tracer positions on density background
        # ================================================================
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        t_end = t_levels[-1]  # = T_END

        for ax, (name, _, _), color in zip(axes, SCHEMES, COLORS):
            _density_bg(t_end, ax)
            traj = results[name]
            ax.scatter(traj[-1, 0], traj[-1, 1], s=8, c=color,
                       label="tracers", zorder=3)
            _shock_circle(t_end, ax, color="white", lw=1.5, ls="--",
                          label=f"shock (R={( 1+t_end)**ALPHA:.2f})")
            ax.set_xlim(-3.0, 5.0)
            ax.set_ylim(-4.5, 4.5)
            ax.set_title(name)
            ax.set_xlabel("x")
            ax.legend(fontsize=7, loc="upper right")

        axes[0].set_ylabel("y")
        fig.suptitle(
            f"Tracer final positions at t = {t_end:.2f}"
            f"  |  Initial shock radius R_s(T_START) = {(1+T_START)**ALPHA:.2f}"
            f",  Final R_s(T_END) = {(1+T_END)**ALPHA:.2f}",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        out1 = os.path.join(PLOT_DIR, "blast_wave_final_positions.png")
        fig.savefig(out1, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: {out1}")

        # ================================================================
        # Plot 2: M(<r) at 4 time levels
        # ================================================================
        step_ids = [0, N_STEPS // 3, 2 * N_STEPS // 3, N_STEPS]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)

        for ax, si in zip(axes.ravel(), step_ids):
            t_lev = t_levels[si]
            for (name, _, _), color, ls in zip(SCHEMES, COLORS, LINE_STYLES):
                r_c, M_c = _M_R(results[name][si], masses)
                ax.plot(r_c, M_c, color=color, ls=ls, lw=1.8, label=name)

            R_s = (1.0 + t_lev) ** ALPHA
            ax.axvline(R_s, color="black", ls=":", lw=1.2,
                       label=f"shock R = {R_s:.2f}")
            ax.set_title(f"t = {t_lev:.2f}")
            ax.set_xlabel("r  (from explosion centre)")
            ax.set_ylabel("M(<r)")
            ax.legend(fontsize=7)

        fig.suptitle(
            "Enclosed mass M(<r) at 4 timesteps\n"
            "Vertical dashed line: analytic shock radius",
            fontsize=11,
        )
        fig.tight_layout()
        out2 = os.path.join(PLOT_DIR, "blast_wave_M_R.png")
        fig.savefig(out2, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out2}")

        # ================================================================
        # Plot 3: RMS position error vs time (tracers well inside shock)
        # ================================================================
        # The analytic solution x(t) = X_C + (x0-X_C)*((1+t)/(1+T_START))^ALPHA
        # is exact when xi0 << 1 (f_inside ~ 1 throughout the integration).
        xi0 = _xi(x0, y0, T_START)
        inside = xi0 < 0.6   # well inside; f_inside > 0.9999 for xi < 0.6
        x_in, y_in = x0[inside], y0[inside]
        n_in = inside.sum()

        rms: dict[str, list[float]] = {name: [] for name, _, _ in SCHEMES}

        for si, t in enumerate(t_levels):
            x_ex, y_ex = exact_position(x_in, y_in, t)
            for name, _, _ in SCHEMES:
                xn = results[name][si, 0, inside]
                yn = results[name][si, 1, inside]
                err = np.sqrt(np.nanmean((xn - x_ex) ** 2 + (yn - y_ex) ** 2))
                rms[name].append(float(err))

        fig, ax = plt.subplots(figsize=(9, 5))
        for (name, _, _), color, ls in zip(SCHEMES, COLORS, LINE_STYLES):
            ax.semilogy(t_levels, rms[name], color=color, ls=ls, lw=2.2,
                        label=name)

        ax.set_xlabel("t")
        ax.set_ylabel("RMS position error")
        ax.set_title(
            f"RMS position error vs time  ({n_in} tracers with xi0 < 0.6,"
            " exact analytic solution valid)"
        )
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out3 = os.path.join(PLOT_DIR, "blast_wave_errors.png")
        fig.savefig(out3, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out3}")

        # ================================================================
        # Summary table printed to stdout
        # ================================================================
        print("\n  RMS position error at T_END:")
        for name, _, _ in SCHEMES:
            print(f"    {name:<25s}  {rms[name][-1]:.3e}")
        print()
