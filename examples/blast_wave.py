"""
examples/blast_wave.py

Exploratory end-to-end example: 2D asymmetric blast wave.

This is not a pytest test -- it is a runnable script that exercises the
tracer-integration pipeline end-to-end and saves comparison plots.  Run it
directly:

    PYTHONPATH=. python examples/blast_wave.py

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

Reference ("ground truth") tracer trajectory, valid for every tracer
---------------------------------------------------------------------
Unlike the analytic xi0 << 1 approximation used previously (only valid
deep inside the shock), the reference here is a direct high-order
numerical integration (scipy.integrate.solve_ivp, DOP853) of the exact
analytic velocity field -- no spatial interpolation needed, since
analytic_vx/vy are closed-form.  See reference_trajectory() for the
(important) caveat that tracer paths are *not* all self-similar: xi(t)
evolves according to dxi/dt = xi*ALPHA/(1+t)*(f(xi)-1), which depends on
xi itself, so tracers crossing the shock transition (xi0 ~ 1) follow
genuinely different curves than those deep inside or far outside.

Density field
    rho(xi, theta) = ambient * angular(theta) * (1 - f(xi))
                   + rho_inner * f(xi)
                   + rho_peak * angular(theta) * exp(-((xi-1)/sig_rho)^2)
where angular(theta) = 1 + 0.5*sin(2*theta + 0.8).
This makes the mass distribution genuinely asymmetric even though the
velocity field is radially symmetric.

Comparison (a): integrators, interpolator held fixed
------------------------------------------------------
This script isolates the time-integrator's contribution to the error by
holding the spatial interpolator fixed at PCHIP (cubic) for all three:
  1. PCHIP + ForwardEuler        (1st order)
  2. PCHIP + ExplicitTrapezoid   (2nd order)
  3. PCHIP + RK4 (4th order, monotone PCHIP time)

See examples/blast_wave_interpolators.py for the complementary comparison
(b): time integrator held fixed at RK4, interpolator varied
(Linear vs PCHIP).

Parallelisation
---------------
Each step's tracer batch is split into bunches and dispatched to a
persistent worker pool (see examples/_parallel.py), mirroring the
production pattern in src/tracers.py: tracers are sorted by their (y, z)
grid bin first so that tracers needing the same PCHIP y/z stencil end up in
the same bunch, maximising cache hits in PchipInterpolator3D's
per-(y,z)-column cache.

Plots
-----
All plots are saved to examples/plots/ as ``{prefix}_*`` (prefix
"blast_wave" for this script, "blast_wave_interpolators" for comparison b):
  {prefix}_final_positions.png  - scatter of tracer positions at T_END
                                   on a density background; one panel per
                                   scheme.
  {prefix}_trajectories.png     - (x(t), y(t)) paths of a few representative
                                   tracers, one colour per scheme.
  {prefix}_M_R.png              - cumulative enclosed mass M(<r) at 4
                                   timesteps; all schemes on each panel.
  {prefix}_errors.png           - RMS position error vs time, all tracers,
                                   vs the solve_ivp reference trajectory.
  {prefix}_animation.mp4        - animation, tracers coloured by log10
                                   position error vs the solve_ivp reference.

This script has no pass/fail assertions; it always "succeeds" unless an
exception is raised.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing.shared_memory import SharedMemory

from src.integrators.base import IntegratorBase
from src.integrators.expl_trapezoid import ExplicitTrapezoid
from src.integrators.rk4 import RK4
from src.interpolators.regular import RegularInterpolator3D
from src.interpolators.pchip import PchipInterpolator3D

import matplotlib.animation as _mpl_animation
from matplotlib.colors import LogNorm

from examples._parallel import N_CPU, N_BUNCHES, worker_pool


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

X_C   = 0.5    # explosion centre x
Y_C   = 0.3    # explosion centre y (off-centre for visual asymmetry)
ALPHA = 0.8    # R_shock(t) = (1 + t)^ALPHA.  0.5 is the literal 2-D Sedov
               # exponent, but it gives a narrow shock-crossing window
               # (initial xi0 in only ~[1.05, 1.50] get overtaken by the
               # shock within [T_START, T_END] -- see _init_tracers).  0.8
               # widens that to ~[1.05, 2.00] and increases velocities
               # (v ~ ALPHA), making crossings more spread out in time and
               # more dynamically pronounced -- this example is illustrative
               # rather than a strict physical Sedov solution anyway.
SIGMA = 0.06   # shock-front width (tanh half-width in xi units)

RHO_SIGMA = 0.08   # density-spike half-width in xi
RHO_PEAK  = 4.0    # density at shock shell (compressed material)
RHO_INNER = 0.05   # density inside cavity (swept-out)
RHO_AMB   = 1.0    # ambient density outside shock


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------
#
# DT matches examples/disk_wind.py's reference simulation timestep, per the
# "use the realistic dt for all examples" choice.

T_START = 1.0
T_END   = 4.0
DT = 0.1
N_STEPS = round((T_END - T_START) / DT)

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
#
# Cartesian, per the "keep blast wave Cartesian" choice.  NX, NY match
# disk_wind.py's N_R (256) -- "grid dimensions match" -- which gives
# dx = dy = 10/255 ~ 0.039.  That's ~7x coarser than disk_wind's innermost
# radial spacing (~0.0056, geometric spacing concentrates resolution at
# small r); matching it exactly would need ~1786 points/axis (~22M grid
# points total) which is too expensive for an example. 256x256x7 (~460k
# points) is the practical compromise: same order of magnitude, cheap.

NX, NY, NZ = 256, 256, 7
X_GRID = np.linspace(-4.0, 6.0, NX)
Y_GRID = np.linspace(-5.0, 5.0, NY)
Z_GRID = np.linspace(0.0, 1.0, NZ)
GRID_SHAPE = (NX, NY, NZ)

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


def reference_trajectory(x0: np.ndarray, y0: np.ndarray, t_eval: np.ndarray):
    """
    High-accuracy numerical ground truth, valid for every tracer (not just
    xi0 << 1 the way the old analytic approximation was).

    The blast-wave velocity field is known in closed form, so no spatial
    interpolation is needed for the reference: integrate
    dx/dt = analytic_vx(x, y, t), dy/dt = analytic_vy(x, y, t) directly with
    an adaptive high-order method (DOP853).  Different tracers don't
    interact (each one's RHS only depends on its own position), so all of
    them are solved together as one batched ODE system (2*n_tr equations)
    in a single solve_ivp call rather than one call per tracer.

    Note that, despite the field being radially symmetric, tracer paths are
    *not* all the same shape: xi(t) = r(t)/R_shock(t) evolves according to
    dxi/dt = xi * ALPHA/(1+t) * (f(xi) - 1), which depends on xi itself, so
    tracers starting at different xi0 (especially near xi0 ~ 1, where they
    cross the shock transition) follow genuinely different curves -- only
    deep inside (f~1) or far outside (f~0) is the motion close to
    self-similar.

    Parameters
    ----------
    x0, y0 : ndarray, shape (n_tr,)
    t_eval : ndarray, shape (n_frames,)
        Times to evaluate at; t_eval[0] must equal T_START.

    Returns
    -------
    x_t, y_t : ndarray, shape (n_frames, n_tr)
    """
    n_tr = len(x0)
    state0 = np.concatenate([x0, y0])

    def rhs(t, state):
        x, y = state[:n_tr], state[n_tr:]
        return np.concatenate([analytic_vx(x, y, t), analytic_vy(x, y, t)])

    sol = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), state0,
        method="DOP853", t_eval=t_eval, rtol=1e-12, atol=1e-14,
    )
    return sol.y[:n_tr].T, sol.y[n_tr:].T


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

        shape = GRID_SHAPE
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

        self.shm_names = shm_names
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
# Parallel tracer integration
# ---------------------------------------------------------------------------
#
# Each step's tracer batch is split into bunches and dispatched to a
# persistent worker pool, mirroring the production pattern in
# src/tracers.py (_ensure_interps / _integrate_positions / sort_tracers):
# tracers are sorted by their (y, z) grid bin first so that tracers needing
# the same PCHIP y/z stencil end up in the same bunch, maximising cache hits
# in PchipInterpolator3D's per-(y,z)-column cache.  Workers attach to the
# existing shared-memory snapshot by name (created once in the main
# process) and cache the wrapped interpolator across calls so repeated
# bunches in the same step don't reattach/reload.

_worker_cache: dict = {}


def _get_worker_interp(interp_class, shm_names: dict):
    """Return a cached (or freshly attached) interpolator for shm_names."""
    key = (interp_class, tuple(sorted(shm_names.items())))
    interp = _worker_cache.get(key)
    if interp is None:
        interp = interp_class(X_GRID, Y_GRID, Z_GRID,
                              shm=shm_names, shape=GRID_SHAPE)
        # track=False: this worker only attaches to memory owned (created
        # and unlinked) by the main process's _SnapInterp.
        interp.load(track=False)
        _worker_cache[key] = interp
        if len(_worker_cache) > 8:
            stale_key = next(iter(_worker_cache))
            _worker_cache.pop(stale_key).unload()
    return interp


def _integrate_bunch(args):
    """Worker task: advance one bunch of tracers by one step."""
    xn_bunch, dt, integrator, interp_class, shm_names_list, snap_times = args
    interps = tuple(_get_worker_interp(interp_class, names) for names in shm_names_list)
    return integrator(xn_bunch, dt, interps, snap_times=snap_times)


def _sort_order(xn: np.ndarray) -> np.ndarray:
    """Order tracer indices by (y, z) grid bin for PCHIP cache locality."""
    _, y, z = xn
    y_bins = np.digitize(y, Y_GRID)
    z_bins = np.digitize(z, Z_GRID)
    return np.lexsort((z_bins, y_bins))


# ---------------------------------------------------------------------------
# Integration driver
# ---------------------------------------------------------------------------

def _run_scheme(interp_class, integrator, x0: np.ndarray, y0: np.ndarray, pool):
    """
    Integrate tracers through the blast wave with a given spatial
    interpolator class and time integrator, parallelising each step's
    tracer batch across ``pool``'s worker processes.

    Parameters
    ----------
    interp_class : type
        RegularInterpolator3D or PchipInterpolator3D.
    integrator : IntegratorBase instance
        _ForwardEuler, ImplicitTrapezoid, or RK4.
    x0, y0 : ndarray, shape (n_tr,)
        Initial tracer positions at T_START.
    pool : multiprocessing.Pool
        Persistent worker pool used to parallelise tracer bunches.

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

    n_bunches = min(N_BUNCHES, n_tr)

    for step_i in range(N_STEPS):
        if n_snap == 4:
            needed = [step_i, step_i + 1, step_i + 2, step_i + 3]
        elif n_snap == 2:
            needed = [step_i + 1, step_i + 2]
        else:  # n_snap == 1
            needed = [step_i + 1]

        _evict(min(needed))
        snaps = [_get(k) for k in needed]
        shm_names_list = [s.shm_names for s in snaps]
        dt = float(TIMES[step_i + 2] - TIMES[step_i + 1])
        st = TIMES[needed] if n_snap == 4 else None

        order = _sort_order(xn)
        xn_sorted = xn[:, order]
        pos_bunches = [b for b in np.array_split(xn_sorted, n_bunches, axis=1)
                       if b.shape[1] > 0]

        tasks = [
            (bunch, dt, integrator, interp_class, shm_names_list, st)
            for bunch in pos_bunches
        ]
        results = pool.map(_integrate_bunch, tasks)
        new_xn_sorted = np.concatenate(results, axis=1)

        xn = np.empty_like(new_xn_sorted)
        xn[:, order] = new_xn_sorted
        traj.append(xn.copy())

    for snap in snap_cache.values():
        snap.close()

    return np.array(traj)  # (N_STEPS+1, 3, n_tr)


# ---------------------------------------------------------------------------
# Tracer setup
# ---------------------------------------------------------------------------

def _init_tracers(xi0_vals=(1.1, 1.3, 1.5, 1.8), n_th=4):
    """
    Place tracers on a small polar grid (xi0_vals radii x n_th angles)
    centred on the explosion point, chosen so every tracer actually gets
    overtaken by the expanding shock during [T_START, T_END] -- that
    crossing is the most dynamically interesting part of this scenario.

    Only tracers starting *outside* the shock (xi0 > 1) can cross it: a
    tracer that starts inside (xi0 < 1) is already comoving with the
    swept-up shell and stays frozen at constant xi forever (self-similar
    inside-shock motion), so it never "crosses" anything.  A tracer
    starting just outside sits in the near-zero-velocity ambient medium
    while the shock radius R_s(t) grows underneath it, until R_s(t)
    catches up and sweeps the tracer up -- xi drops through 1.  The
    default xi0_vals were picked (see exploration in git history /
    conversation, not reproduced here) to cross at well-spread times
    across the window: t ~ 1.33, 1.88, 2.44, 3.32 respectively, for
    ALPHA = 0.8.  (Removed entirely: the old xi0 in [0.2, 1.6] linspace,
    which wasted half the tracer budget on never-crossing deep-inside
    tracers.)

    Initial shock radius at T_START = 1.0: R_s(T_START) = 2^ALPHA.

    Returns
    -------
    x0, y0 : ndarray, shape (len(xi0_vals) * n_th,)
    masses  : ndarray, shape (len(xi0_vals) * n_th,)
        m_i = rho(x_i, y_i, T_START) * dA_i (Lagrangian mass, conserved).
    """
    R_s0 = (1.0 + T_START) ** ALPHA
    r_vals = np.asarray(xi0_vals) * R_s0
    th_vals = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    rr0, tt0 = np.meshgrid(r_vals, th_vals, indexing="ij")
    r0 = rr0.ravel()
    th0 = tt0.ravel()

    x0 = X_C + r0 * np.cos(th0)
    y0 = Y_C + r0 * np.sin(th0)

    # r_vals isn't uniformly spaced (xi0_vals were chosen for crossing
    # times, not even spacing), so use a per-radius cell width (central
    # differences) instead of a single scalar dr.
    dr_per_radius = np.gradient(r_vals)
    dr = np.repeat(dr_per_radius, n_th)  # broadcast to match r0's ravel order
    dth = th_vals[1] - th_vals[0]
    dA = r0 * dr * dth  # polar-cell area element at each tracer's radius

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


def _log_ylim_skip_first(series_list, skip=1, pad_decades=0.3):
    """
    Compute a log-scale (ymin, ymax) from a list of 1D sequences, ignoring
    the first ``skip`` points of each.  The first timestep's error is ~0
    (machine precision, since the integrated and analytic positions are
    identical at t=T_START), which would otherwise drag matplotlib's
    semilogy autoscale down to an unreadably low floor and compress the
    rest of the curve.
    """
    vals = np.concatenate([np.asarray(s)[skip:] for s in series_list])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return None
    pad = 10 ** pad_decades
    return vals.min() / pad, vals.max() * pad


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


def _sample_by_value(metric: np.ndarray, n_sample: int) -> np.ndarray:
    """
    Return indices of ``n_sample`` tracers spread evenly across the *range*
    of ``metric`` (e.g. xi0) rather than evenly by rank -- avoids
    over-representing whichever value is most numerous when the underlying
    tracer grid isn't uniform in ``metric`` (e.g. a Cartesian tracer grid
    has many more tracers at large xi0 than small xi0).
    """
    order = np.argsort(metric)
    sorted_metric = metric[order]
    targets = np.linspace(sorted_metric[0], sorted_metric[-1], n_sample)
    sample_pos = np.unique(np.searchsorted(sorted_metric, targets).clip(0, len(order) - 1))
    return order[sample_pos]


def plot_trajectories(scheme_names, trajs, x0, y0, output_path, n_sample=5):
    """
    Compare r(t) = distance from the explosion centre for a handful of
    representative tracers across schemes, in a single panel: one line per
    (tracer, scheme), coloured by scheme.  Motion in this scenario is
    purely radial, so plotting (x(t), y(t)) directly makes every tracer's
    path a short straight segment out of the centre -- visually
    indistinguishable from each other.  r(t) instead clearly shows each
    tracer's radial expansion (or near-stagnation, for xi0 well outside the
    shock) and exactly where the schemes start to diverge.  The shock
    radius R_s(t) is overlaid for reference.  The sampled tracers are
    spread evenly across initial shock-normalised radius xi0 (inside the
    shock to well outside).
    """
    xi0 = _xi(x0, y0, T_START)
    sample_idx = _sample_by_value(xi0, n_sample)
    t_levels = TIMES[1: N_STEPS + 2]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.get_cmap("tab10").colors

    for s, (name, traj) in enumerate(zip(scheme_names, trajs)):
        color = colors[s % len(colors)]
        for k, i in enumerate(sample_idx):
            r_t = np.sqrt((traj[:, 0, i] - X_C) ** 2 + (traj[:, 1, i] - Y_C) ** 2)
            ax.plot(t_levels, r_t, color=color, lw=1.5, alpha=0.85,
                    label=name if k == 0 else None)

    R_shock_t = (1.0 + t_levels) ** ALPHA
    ax.plot(t_levels, R_shock_t, color="black", lw=1.2, ls="--",
            label="Shock radius R_s(t)")

    ax.set_xlabel("t")
    ax.set_ylabel("r  (distance from explosion centre)")
    ax.set_title(
        f"r(t) for {len(sample_idx)} representative tracers "
        f"(xi0 = {xi0[sample_idx].min():.2f} - {xi0[sample_idx].max():.2f}) across schemes",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Animation helper
# ---------------------------------------------------------------------------

def _animate_tracers(
    scheme_names: list,
    trajs: list,
    t_levels: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    output_path: str,
    fps: int = 4,
    **_kwargs,          # absorb legacy 'inside' kwarg if passed
):
    """
    Save an animation comparing one or more integration schemes.

    Each panel shows the blast-wave domain at the current time:
    - Density field (Greys colourmap) as background; updated every frame.
    - All tracers coloured by log10 position error vs the solve_ivp
      reference trajectory (valid for every tracer; hot: white = large
      error, dark = small error); shared colourbar.
    - Analytic shock circle (white dashed).
    """
    n_panels = len(scheme_names)
    n_frames = len(t_levels)
    n_tr = trajs[0].shape[-1]

    x_ref, y_ref = reference_trajectory(x0, y0, t_levels)  # (n_frames, n_tr)

    errors = []
    for p in range(n_panels):
        traj = trajs[p]  # (n_frames, 3, n_tr)
        err = np.sqrt((traj[:, 0] - x_ref) ** 2 + (traj[:, 1] - y_ref) ** 2)
        errors.append(np.maximum(err, 1e-6))  # floor to avoid log(0) issues

    flat = np.concatenate([e for pe in errors for e in pe[1:]])
    flat = flat[np.isfinite(flat) & (flat > 0)]
    vmin_err = float(np.nanpercentile(flat, 2))
    vmax_err = float(np.nanpercentile(flat, 99))
    norm = LogNorm(vmin=max(vmin_err, 1e-6), vmax=vmax_err)
    err_cmap = "seismic"

    xx_bg, yy_bg = np.meshgrid(X_GRID, Y_GRID, indexing="ij")
    rho_frames = [analytic_rho(xx_bg, yy_bg, t) for t in t_levels]
    rho_vmax = max(r.max() for r in rho_frames)

    theta_c = np.linspace(0, 2 * np.pi, 300)
    shock_xy = []
    for t in t_levels:
        R = (1.0 + t) ** ALPHA
        shock_xy.append((X_C + R * np.cos(theta_c), Y_C + R * np.sin(theta_c)))

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(6 * n_panels, 6),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    meshes, sc_list, circle_list = [], [], []

    for p, (ax, name) in enumerate(zip(axes, scheme_names)):
        traj = trajs[p]

        mesh = ax.pcolormesh(
            X_GRID, Y_GRID, rho_frames[0].T,
            cmap="Greys", shading="auto",
            vmin=0.0, vmax=rho_vmax,
        )
        meshes.append(mesh)

        sc = ax.scatter(
            traj[0, 0], traj[0, 1],
            s=20, c=errors[p][0],
            cmap=err_cmap, norm=norm,
            linewidths=0, zorder=4,
        )
        sc_list.append(sc)

        line, = ax.plot(shock_xy[0][0], shock_xy[0][1],
                        color="white", lw=1.5, ls="--", zorder=5)
        circle_list.append(line)

        ax.set_xlim(-3.0, 5.0)
        ax.set_ylim(-4.5, 4.5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x")

    axes[0].set_ylabel("y")

    sm = plt.cm.ScalarMappable(cmap=err_cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes,
                 label="Position error vs solve_ivp reference (log scale)",
                 fraction=0.018, pad=0.02)

    title_obj = fig.suptitle(
        f"t = {t_levels[0]:.3f}  |  frame 1 / {n_frames}",
        fontsize=11,
    )

    def _update(frame):
        t = t_levels[frame]

        for p in range(n_panels):
            traj = trajs[p]
            meshes[p].set_array(rho_frames[frame].T.ravel())
            sc_list[p].set_offsets(
                np.column_stack([traj[frame, 0], traj[frame, 1]])
            )
            sc_list[p].set_array(errors[p][frame])
            circle_list[p].set_data(shock_xy[frame][0], shock_xy[frame][1])

        title_obj.set_text(
            f"t = {t:.3f}  |  frame {frame + 1} / {n_frames}"
        )
        return meshes + sc_list + circle_list

    ani = _mpl_animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000 // fps, blit=False,
    )
    writer = _mpl_animation.FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(output_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"  Saved animation: {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------
#
# Comparison (a): spatial interpolator held fixed at PCHIP, vary only the
# time integrator.  This isolates the integrator's contribution to the
# error, decoupled from the choice of spatial interpolation order -- see
# examples/blast_wave_interpolators.py for the complementary comparison
# (b): time integrator held fixed at RK4, interpolator varied.

SCHEMES = [
    ("PCHIP + Euler",      PchipInterpolator3D, _ForwardEuler()),
    ("PCHIP + Expl.Trap.", PchipInterpolator3D, ExplicitTrapezoid()),
    ("PCHIP + RK4",        PchipInterpolator3D, RK4(monotone=True)),
]


def _scheme_style(n: int):
    """Return ``n`` (color, linestyle) pairs, cycling if there are many schemes."""
    colors = plt.get_cmap("tab10").colors
    linestyles = ["-", "--", "-.", ":"]
    return ([colors[i % len(colors)] for i in range(n)],
            [linestyles[i % len(linestyles)] for i in range(n)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(schemes, output_prefix):
    """
    Run every (interpolator, integrator) scheme in ``schemes`` on the
    blast-wave field and save the comparison plots/animation to
    ``examples/plots/{output_prefix}_*``.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    x0, y0, masses = _init_tracers()
    n_tr = len(x0)
    n_panels = len(schemes)
    colors, line_styles = _scheme_style(n_panels)

    print(f"  Using a worker pool of {N_CPU} processes "
          f"({N_BUNCHES} tracer bunches per step).", flush=True)

    results: dict[str, np.ndarray] = {}
    with worker_pool(N_CPU) as pool:
        for name, interp_cls, integr in schemes:
            print(f"\n  [{name}] integrating {n_tr} tracers x {N_STEPS} steps ...",
                  flush=True)
            traj = _run_scheme(interp_cls, integr, x0, y0, pool)
            results[name] = traj
            print(f"  [{name}] done.  traj shape: {traj.shape}", flush=True)

    # Actual time at each stored level: TIMES[1..N_STEPS+1]
    t_levels = TIMES[1: N_STEPS + 2]  # length N_STEPS+1

    # ====================================================================
    # Plot 1: final tracer positions on density background
    # ====================================================================
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]
    t_end = t_levels[-1]  # = T_END

    for ax, (name, _, _), color in zip(axes, schemes, colors):
        _density_bg(t_end, ax)
        traj = results[name]
        ax.scatter(traj[-1, 0], traj[-1, 1], s=8, c=[color],
                   label="tracers", zorder=3)
        _shock_circle(t_end, ax, color="white", lw=1.5, ls="--",
                      label=f"shock (R={(1+t_end)**ALPHA:.2f})")
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
    out1 = os.path.join(PLOT_DIR, f"{output_prefix}_final_positions.png")
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out1}")

    # ====================================================================
    # Plot 1b: tracer trajectories in the xy-plane
    # ====================================================================
    plot_trajectories(
        scheme_names=[name for name, _, _ in schemes],
        trajs=[results[name] for name, _, _ in schemes],
        x0=x0, y0=y0,
        output_path=os.path.join(PLOT_DIR, f"{output_prefix}_trajectories.png"),
    )

    # ====================================================================
    # Plot 2: M(<r) at 4 time levels
    # ====================================================================
    step_ids = [0, N_STEPS // 3, 2 * N_STEPS // 3, N_STEPS]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)

    for ax, si in zip(axes.ravel(), step_ids):
        t_lev = t_levels[si]
        for (name, _, _), color, ls in zip(schemes, colors, line_styles):
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
    out2 = os.path.join(PLOT_DIR, f"{output_prefix}_M_R.png")
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out2}")

    # ====================================================================
    # Plot 3: RMS position error vs time, all tracers, vs the solve_ivp
    # reference trajectory (valid everywhere, unlike the old xi0<<1-only
    # analytic approximation).
    # ====================================================================
    x_ref, y_ref = reference_trajectory(x0, y0, t_levels)  # (n_frames, n_tr)

    rms: dict[str, list[float]] = {name: [] for name, _, _ in schemes}
    for si in range(len(t_levels)):
        x_ex, y_ex = x_ref[si], y_ref[si]
        for name, _, _ in schemes:
            xn = results[name][si, 0]
            yn = results[name][si, 1]
            err = np.sqrt(np.nanmean((xn - x_ex) ** 2 + (yn - y_ex) ** 2))
            rms[name].append(float(err))

    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, _, _), color, ls in zip(schemes, colors, line_styles):
        ax.semilogy(t_levels, rms[name], color=color, ls=ls, lw=2.2,
                    label=name)

    ylim = _log_ylim_skip_first([rms[name] for name, _, _ in schemes])
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("t")
    ax.set_ylabel("RMS position error")
    ax.set_title(
        f"RMS position error vs time  ({n_tr} tracers, vs solve_ivp "
        "reference, valid everywhere)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out3 = os.path.join(PLOT_DIR, f"{output_prefix}_errors.png")
    fig.savefig(out3, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out3}")

    # ====================================================================
    # Summary table printed to stdout
    # ====================================================================
    print("\n  RMS position error at T_END:")
    for name, _, _ in schemes:
        print(f"    {name:<25s}  {rms[name][-1]:.3e}")
    print()

    # ====================================================================
    # Animation: tracers coloured by error vs the solve_ivp reference
    # ====================================================================
    print("  Building blast-wave animation ...", flush=True)
    anim_path = os.path.join(PLOT_DIR, f"{output_prefix}_animation.mp4")
    _animate_tracers(
        scheme_names=[name for name, _, _ in schemes],
        trajs=[results[name] for name, _, _ in schemes],
        t_levels=t_levels,
        x0=x0,
        y0=y0,
        output_path=anim_path,
    )


if __name__ == "__main__":
    run_comparison(SCHEMES, "blast_wave")
