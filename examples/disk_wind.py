"""
examples/disk_wind.py

Exploratory end-to-end example: differentially-rotating disk-wind outflow,
mimicking ejecta launched from a post-merger (BNS) accretion disk.

This is not a pytest test -- it is a runnable script that exercises the
tracer-integration pipeline end-to-end and saves comparison plots.  Run it
directly:

    PYTHONPATH=. python examples/disk_wind.py

Setup
-----
Matter is launched radially outward from a reference radius R0 while still
carrying the disk's specific angular momentum.  The radial motion is true
homologous (ballistic) outflow: every fluid element moves at its own fixed
radial velocity once launched, v0 = r0 / T0, so

    r(t) = r0 * (1 + LAMBDA * (t - T_START))           with LAMBDA = 1 / T0

reproduces a self-similar v_r(r, t) = r / (t - T_START + T0) profile.  T0 is
fixed by requiring the radial velocity at the launch radius R0 at T_START to
equal V_R0 (a representative value taken from real post-merger ejecta).

Physical parameters are chosen to be representative of an actual GRMHD
post-merger simulation (geometric units, c = 1, lengths/times in ms):

    R0     = 0.75 ms   launch radius
    R_OUT  = 5.0  ms    outer radial extent of the velocity-field grid
    V_R0   = 0.05       radial velocity at the launch radius (-> T0 = R0/V_R0 = 15 ms)
    OMEGA0 = 0.4 / ms   angular velocity at the launch radius
    DT     = 0.1 ms     reference simulation timestep

Tangential speed at launch is Omega0 * R0 = 0.3, so the velocity magnitude
across the ejecta ranges over ~0.05-0.3 c, matching real ejecta.

The Eulerian velocity field (spherical coordinates, time-dependent through
the homologous radial term) is, with r the full spherical radius:

    v_r(r, t)  = r / (t - T_START + T0)             (homologous radial outflow)
    Omega(r)   = OMEGA0 * (R0 / r) ** Q              (angular velocity, about z)

with Q = 2, i.e. angular momentum L = Omega * r^2 is conserved -- the ejecta
are torque-free once unbound, rather than staying on Keplerian circular
orbits (Q = 3/2).  In Cartesian components (rotation strictly about z):

    vx = (v_r/r) * x - Omega(r) * y
    vy = (v_r/r) * y + Omega(r) * x
    vz = (v_r/r) * z

Tracers are seeded in the equatorial plane (z0 = 0); since Omega depends
only on r and the radial term is purely along r_hat, dz/dt = (v_r/r) * z is
satisfied by z(t) = 0 for all t, so tracers stay exactly in the plane and
the closed-form trajectory below is exact.

Exact tracer trajectory (valid everywhere; Q != 1 so the integral below is
elementary)
    r(t)     = r0 * (1 + LAMBDA * (t - T_START))
    theta(t) = theta0 + OMEGA0 * (R0 / r0) ** Q / (LAMBDA * (Q - 1))
               * (1 - (1 + LAMBDA * (t - T_START)) ** (1 - Q))

Spatial grid
------------
The velocity field is sampled on a spherical (r, theta, phi) grid matching
the resolution of a real simulation snapshot:

    r     : 256 points, geometric spacing, r in [R0, R_OUT]
    theta : 128 points, uniform spacing, theta in [0, pi]
    phi   : 256 points, uniform spacing, phi in [0, 2*pi)
            (+ 3 ghost points on each side for periodic wraparound, needed
            because tracers can sit arbitrarily close to the phi = 0 / 2*pi
            seam and the cubic stencil needs 4 in-bounds nodes)

Cartesian (x, y, z) tracer positions are converted to (r, theta, phi) via
``CartesianToSpherical`` before each interpolator query.

Density field (visualisation only; not required to satisfy continuity)
    Sigma(r, theta) = angular(theta) * RHO_PEAK / (1 + (r / R0) ** 2)
with the same angular(theta) asymmetry trick as before.

Comparison (c): integrators, interpolator held fixed
------------------------------------------------------
This script isolates the time-integrator's contribution to the error by
holding the spatial interpolator fixed at PCHIP (cubic) for all three:
  1. PCHIP + ForwardEuler        (1st order)
  2. PCHIP + ExplicitTrapezoid   (2nd order)
  3. PCHIP + RK4 (4th order, monotone PCHIP time)

See examples/disk_wind_interpolators.py for the complementary comparison
(d): time integrator held fixed at RK4, interpolator varied
(Linear vs PCHIP).

Plots
-----
All plots are saved to examples/plots/:
  disk_wind_final_positions.png  - scatter of tracer positions at T_END on a
                                    density background; one panel per scheme.
  disk_wind_trajectories.png     - (x(t), y(t)) paths of every tracer, one
                                    panel per scheme, coloured by launch radius.
  disk_wind_M_R.png              - cumulative enclosed mass M(<r) at 4
                                    timesteps; all schemes on each panel.
  disk_wind_errors.png           - RMS position error vs time (exact solution
                                    is valid for all tracers here).
  disk_wind_animation.mp4        - 3-panel animation, tracers coloured by
                                    log10 position error vs the analytic
                                    solution.

This script has no pass/fail assertions; it always "succeeds" unless an
exception is raised.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing.shared_memory import SharedMemory

from examples._parallel import N_CPU, worker_pool

from src.integrators.base import IntegratorBase
from src.integrators.expl_trapezoid import ExplicitTrapezoid
from src.integrators.rk4 import RK4
from src.interpolators.regular import RegularInterpolator3D
from src.interpolators.pchip import PchipInterpolator3D
from src.interpolators.coordinate_transformations import CartesianToSpherical

import matplotlib.animation as _mpl_animation
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")


# ---------------------------------------------------------------------------
# Physical parameters (representative of a real post-merger GRMHD run)
# ---------------------------------------------------------------------------

R0     = 0.75   # launch radius [ms]
R_OUT  = 5.0    # outer radial extent of the velocity-field grid [ms]
V_R0   = 0.05   # radial velocity at the launch radius, at T_START [c]
OMEGA0 = 0.4    # angular velocity at the launch radius [1/ms]
Q      = 2.0    # angular-momentum-conserving: Omega(r) = OMEGA0*(R0/r)^Q

T0     = R0 / V_R0    # homologous time elapsed since launch, at t=T_START [ms]
LAMBDA = 1.0 / T0      # 1/T0: v_r(r,t) = r / (t - T_START + T0)

RHO_PEAK = 4.0   # density normalisation at the launch radius
RHO_SOFT = R0    # softening scale for the density profile


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------

T_START = 0.0
DT      = 0.1     # reference simulation timestep [ms]
T_END   = T0       # one homologous time unit of integration
N_STEPS = round((T_END - T_START) / DT)

# TIMES has N_STEPS + 3 entries:
#   index 0            = T_START - DT         (left padding for RK4)
#   index 1            = T_START
#   index N_STEPS + 1  = T_END
#   index N_STEPS + 2  = T_END + DT           (right padding for RK4)
TIMES = np.linspace(T_START - DT, T_END + DT, N_STEPS + 3)


# ---------------------------------------------------------------------------
# Spatial grid: spherical (r, theta, phi), matching a real simulation's
# resolution.  phi gets ghost points on each side for periodic wraparound.
# ---------------------------------------------------------------------------

N_R, N_TH, N_PH = 256, 128, 256
N_GHOSTS = PchipInterpolator3D.n_ghosts  # 3

_R_MARGIN = 1.0 - 1e-6  # tiny safety margin: see note below
R_GRID = np.geomspace(R0 * _R_MARGIN, R_OUT, N_R)
TH_GRID = np.linspace(0.0, np.pi, N_TH)

# Note on _R_MARGIN: tracers are seeded at r0 = R0 exactly, but
# r = sqrt((r0*cos(th0))**2 + (r0*sin(th0))**2) is not always exactly R0
# in floating point (cos^2 + sin^2 != 1 to the last bit for most th0). With
# the grid's lower edge at exactly R0 and extrapolate=False, that rounding
# error alone can push a query just below the domain and return NaN at the
# very first step. The margin keeps the grid's true edge a hair below the
# nominal R0 so this can't happen, without materially changing "r in
# [R0, R_OUT]".

_dphi = 2.0 * np.pi / N_PH
_phi_phys = np.linspace(0.0, 2.0 * np.pi, N_PH, endpoint=False)
_phi_ghost = np.arange(1, N_GHOSTS + 1) * _dphi
PH_GRID = np.concatenate([
    _phi_phys[0] - _phi_ghost[::-1],
    _phi_phys,
    _phi_phys[-1] + _phi_ghost,
])
N_PH_TOTAL = N_PH + 2 * N_GHOSTS

GRID_SHAPE = (N_R, N_TH, N_PH_TOTAL)

# Equatorial plane index, used only for sanity/debugging (theta is uniform).
TH_EQ_INDEX = N_TH // 2


# ---------------------------------------------------------------------------
# Analytic disk wind
# ---------------------------------------------------------------------------

def _xi(x, y, t):
    """Radius normalised by the launch radius R0 (xi = 1 at the launch ring)."""
    return np.sqrt(x ** 2 + y ** 2) / R0


def _omega(r):
    """Angular velocity at radius r (angular-momentum-conserving, Q = 2)."""
    return OMEGA0 * (R0 / np.maximum(r, 1e-12)) ** Q


def _v_r_over_r(t):
    """Homologous radial term v_r(r,t) / r = 1 / (t - T_START + T0)."""
    return 1.0 / (t - T_START + T0)


def _angular(x, y):
    """
    Angular density variation: range [0.5, 1.5].
    Creates an asymmetric ejecta mass distribution.
    """
    theta = np.arctan2(y, x)
    return 1.0 + 0.5 * np.sin(2.0 * theta + 0.8)


def analytic_vx(x, y, z, t):
    """x-velocity of the disk wind (homologous radial term, time-dependent)."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return _v_r_over_r(t) * x - _omega(r) * y


def analytic_vy(x, y, z, t):
    """y-velocity of the disk wind (homologous radial term, time-dependent)."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return _v_r_over_r(t) * y + _omega(r) * x


def analytic_vz(x, y, z, t):
    """z-velocity of the disk wind (rotation is strictly about z)."""
    return _v_r_over_r(t) * z


def analytic_rho(x, y, t):
    """
    Density field: angularly-modulated, falling off away from the launch
    radius.  Visualisation only (equatorial plane) -- not constrained to
    satisfy continuity under the velocity field above.
    """
    r = np.sqrt(x ** 2 + y ** 2)
    ang = _angular(x, y)
    return ang * RHO_PEAK / (1.0 + (r / RHO_SOFT) ** 2)


def exact_position(x0, y0, t):
    """
    Analytic tracer position at time t, valid for every tracer (no
    near-centre restriction).  Tracers seeded with z0 = 0 stay exactly in
    the equatorial plane (dz/dt = (v_r/r)*z is satisfied by z = 0).

    r(t) = r0 * (1 + LAMBDA * (t - T_START))                  (ballistic)
    theta(t) = theta0 + OMEGA0/(LAMBDA*(Q-1)) * (R0/r0)^Q
               * (1 - (1 + LAMBDA*(t - T_START)) ** (1 - Q))
    """
    dx0 = x0
    dy0 = y0
    r0 = np.sqrt(dx0 ** 2 + dy0 ** 2)
    theta0 = np.arctan2(dy0, dx0)

    dt_ = t - T_START
    growth = 1.0 + LAMBDA * dt_
    r_t = r0 * growth
    theta_t = theta0 + (OMEGA0 / (LAMBDA * (Q - 1.0))) * (R0 / r0) ** Q * (
        1.0 - growth ** (1.0 - Q)
    )

    x_t = r_t * np.cos(theta_t)
    y_t = r_t * np.sin(theta_t)
    return x_t, y_t


# ---------------------------------------------------------------------------
# Shared-memory snapshot helper
# ---------------------------------------------------------------------------

_GEOM_CACHE = None        # (xx, yy, zz, omega*xx, omega*yy)
_GEOM_CACHE_KEY = None    # (R0, OMEGA0, Q) the cache above was built with


def _geometry_cache():
    """
    Return the time-independent geometry needed to build a snapshot:
    Cartesian grid coordinates (xx, yy, zz) and the rotational term
    (omega*xx, omega*yy).  None of this depends on t -- only the scalar
    v_r_over_r(t) does -- so it would be wasteful to recompute it (two
    trig-heavy arrays plus Omega(r)) on every single snapshot construction,
    which happens once per integration step.  Cached at module level and
    built on first use (lazily, since R_GRID etc. must already exist).

    Keyed by (R0, OMEGA0, Q) rather than cached unconditionally: scripts
    like trapezoid_comparison.py override the OMEGA0 module attribute
    (see its docstring) to stiffen the rotation for their own process, and
    an earlier version of this cache kept serving geometry built from
    whichever OMEGA0 was active the *first* time a snapshot was built,
    silently ignoring later overrides -- a real bug, caught because
    re-running a parameter sweep over OMEGA0 gave suspiciously identical
    results at every value.
    """
    global _GEOM_CACHE, _GEOM_CACHE_KEY
    key = (R0, OMEGA0, Q)
    if _GEOM_CACHE is None or _GEOM_CACHE_KEY != key:
        rr, tt, pp = np.meshgrid(R_GRID, TH_GRID, PH_GRID, indexing="ij")
        sin_t = np.sin(tt)
        xx = rr * sin_t * np.cos(pp)
        yy = rr * sin_t * np.sin(pp)
        zz = rr * np.cos(tt)
        omega = _omega(rr)
        _GEOM_CACHE = (xx, yy, zz, omega * xx, omega * yy)
        _GEOM_CACHE_KEY = key
    return _GEOM_CACHE


class _SnapInterp:
    """
    Velocity interpolator for one time snapshot backed by shared memory.

    The callable interface accepts Cartesian position coordinates of shape
    (3, n) and returns velocity of shape (3, n) = [vx, vy, vz].  Internally
    the field is sampled on the spherical (r, theta, phi) grid and wrapped
    with ``CartesianToSpherical`` so Cartesian tracer positions can be
    queried directly.
    """

    def __init__(self, t: float, interp_class):
        xx, yy, zz, omega_xx, omega_yy = _geometry_cache()
        v_r_over_r = _v_r_over_r(t)  # the only actually time-dependent piece

        fields = {
            "vx": v_r_over_r * xx - omega_yy,
            "vy": v_r_over_r * yy + omega_xx,
            "vz": v_r_over_r * zz,
        }

        self._shm: dict[str, SharedMemory] = {}
        shm_names: dict[str, str] = {}
        for key, arr in fields.items():
            buf = np.asarray(arr, dtype=np.float64)
            shm = SharedMemory(create=True, size=max(buf.nbytes, 1))
            np.ndarray(GRID_SHAPE, dtype=np.float64, buffer=shm.buf)[:] = buf
            self._shm[key] = shm
            shm_names[key] = shm.name

        self.shm_names = shm_names
        self._interp = CartesianToSpherical(
            interp_class, R_GRID, TH_GRID, PH_GRID,
            shm=shm_names, shape=GRID_SHAPE,
        )
        self._interp.load()
        self._closed = False

    def __call__(self, xn: np.ndarray) -> np.ndarray:
        """Evaluate velocity at Cartesian positions xn of shape (3, n)."""
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
# tracers are sorted by their (theta, phi) grid bin first so that tracers
# needing the same PCHIP y/z stencil end up in the same bunch, maximising
# cache hits in PchipInterpolator3D's per-(theta,phi)-column cache.
# Workers attach to the existing shared-memory snapshot by name (it is
# created once in the main process) and cache the wrapped interpolator
# across calls so repeated bunches in the same step don't reattach/reload.

N_BUNCHES = 3 * N_CPU

_worker_cache: dict = {}


def _get_worker_interp(interp_class, shm_names: dict):
    """Return a cached (or freshly attached) interpolator for shm_names."""
    key = (interp_class, tuple(sorted(shm_names.items())))
    interp = _worker_cache.get(key)
    if interp is None:
        interp = CartesianToSpherical(
            interp_class, R_GRID, TH_GRID, PH_GRID,
            shm=shm_names, shape=GRID_SHAPE,
        )
        # track=False: this worker only attaches to memory owned (created
        # and unlinked) by the main process's _SnapInterp.
        interp.load(track=False)
        _worker_cache[key] = interp
        # Bound the per-worker cache; evict the oldest entry once full.
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
    """
    Order tracer indices by (theta, phi) grid bin for PCHIP cache locality
    (same idea as CartesianToSpherical.sort_tracers, applied to a raw
    position array instead of Tracer objects).
    """
    x, y, z = xn
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    safe_r = np.where(r > 0, r, 1.0)
    theta = np.arccos(np.clip(z / safe_r, -1.0, 1.0))
    phi = (np.arctan2(y, x) + 2.0 * np.pi) % (2.0 * np.pi)
    theta_bins = np.digitize(theta, TH_GRID)
    phi_bins = np.digitize(phi, PH_GRID)
    return np.lexsort((phi_bins, theta_bins))


# ---------------------------------------------------------------------------
# Integration driver
# ---------------------------------------------------------------------------

def _run_scheme(interp_class, integrator, x0: np.ndarray, y0: np.ndarray, pool):
    """
    Integrate tracers through the disk wind with a given spatial interpolator
    class and time integrator, parallelising each step's tracer batch across
    ``pool``'s worker processes.

    Parameters
    ----------
    interp_class : type
        RegularInterpolator3D or PchipInterpolator3D.
    integrator : IntegratorBase instance
        _ForwardEuler, ExplicitTrapezoid, ImplicitTrapezoid, or RK4.
    x0, y0 : ndarray, shape (n_tr,)
        Initial tracer positions at T_START.
    pool : multiprocessing.Pool
        Persistent worker pool used to parallelise tracer bunches.

    Returns
    -------
    traj : ndarray, shape (N_STEPS + 1, 3, n_tr)
        Tracer positions at each of the N_STEPS + 1 stored time levels
        (T_START to T_END inclusive).
    """
    n_tr = len(x0)
    xn = np.array([x0, y0, np.zeros(n_tr)], dtype=float)  # (3, n_tr)
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

def _init_tracers(n_r=3, n_th=6):
    """
    Place tracers on a small ``n_r`` (radius) x ``n_th`` (angle) polar grid
    covering the disk-wind launch region, from the launch radius R0 out to
    2*R0.  The setup is rotationally symmetric, so a few radii x a few
    angles is enough to see the schemes' qualitative differences cheaply.

    Returns
    -------
    x0, y0 : ndarray, shape (n_r * n_th,)
    masses  : ndarray, shape (n_r * n_th,)
        m_i = rho(x_i, y_i, T_START) * dA_i (Lagrangian mass, conserved),
        with dA_i the polar-cell area for an annular/angular grid.
    """
    r_vals = np.linspace(R0, 2.0 * R0, n_r)
    th_vals = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    rr0, tt0 = np.meshgrid(r_vals, th_vals, indexing="ij")
    r0 = rr0.ravel()
    th0 = tt0.ravel()

    x0 = r0 * np.cos(th0)
    y0 = r0 * np.sin(th0)

    dr = r_vals[1] - r_vals[0]
    dth = th_vals[1] - th_vals[0]
    dA = r0 * dr * dth  # polar-cell area element at each tracer's radius

    masses = analytic_rho(x0, y0, T_START) * dA
    return x0, y0, masses


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

VIEW = 4.0  # half-width of the xy view used for plots/animation [ms]
_PLOT_N = 300
_PLOT_X = np.linspace(-VIEW, VIEW, _PLOT_N)
_PLOT_Y = np.linspace(-VIEW, VIEW, _PLOT_N)


def _density_bg(t, ax):
    """Draw density field at time t as a colour mesh on ax (equatorial plane)."""
    xx, yy = np.meshgrid(_PLOT_X, _PLOT_Y, indexing="ij")
    rho = analytic_rho(xx, yy, t)
    ax.pcolormesh(_PLOT_X, _PLOT_Y, rho.T, cmap="inferno",
                  shading="auto", vmin=0.0, vmax=RHO_PEAK * 0.6)


def _launch_circle(t, ax, **kw):
    """Overlay the launch-radius circle r = R0 (stiffness reference scale)."""
    theta = np.linspace(0, 2.0 * np.pi, 300)
    ax.plot(R0 * np.cos(theta), R0 * np.sin(theta), **kw)


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


def _M_R(traj_step, masses, n_bins=80, r_max=None):
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
    if r_max is None:
        r_max = VIEW
    xpos, ypos = traj_step[0], traj_step[1]
    r = np.sqrt(xpos ** 2 + ypos ** 2)
    bins = np.linspace(0.0, r_max, n_bins + 1)
    M_bin, _ = np.histogram(r, bins=bins, weights=masses)
    return 0.5 * (bins[:-1] + bins[1:]), np.cumsum(M_bin)


def _sample_by_value(metric: np.ndarray, n_sample: int) -> np.ndarray:
    """
    Return indices of ``n_sample`` tracers spread evenly across the *range*
    of ``metric`` (e.g. launch radius, xi0) rather than evenly by rank --
    avoids over-representing whichever value is most numerous when the
    underlying tracer grid isn't uniform in ``metric``.
    """
    order = np.argsort(metric)
    sorted_metric = metric[order]
    targets = np.linspace(sorted_metric[0], sorted_metric[-1], n_sample)
    sample_pos = np.unique(np.searchsorted(sorted_metric, targets).clip(0, len(order) - 1))
    return order[sample_pos]


def plot_trajectories(scheme_names, trajs, x0, y0, output_path, n_sample=5):
    """
    Compare a handful of representative tracers' (x(t), y(t)) paths across
    schemes in a single panel: one line per (tracer, scheme), coloured by
    scheme so the schemes' divergence for the same tracer is directly
    visible.  The sampled tracers are spread evenly across launch radius
    r0 (the physically interesting axis here) rather than chosen randomly.
    """
    r0 = np.sqrt(x0 ** 2 + y0 ** 2)
    sample_idx = _sample_by_value(r0, n_sample)

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.get_cmap("tab10").colors

    for s, (name, traj) in enumerate(zip(scheme_names, trajs)):
        color = colors[s % len(colors)]
        for k, i in enumerate(sample_idx):
            ax.plot(traj[:, 0, i], traj[:, 1, i],
                    color=color, lw=1.5, alpha=0.85,
                    label=name if k == 0 else None)

    for i in sample_idx:
        ax.scatter(x0[i], y0[i], color="black", s=25, zorder=5, marker="o")

    _launch_circle(0.0, ax, color="grey", lw=1.2, ls="--",
                   label=f"Launch radius R0={R0:.2f}")
    ax.set_xlim(-VIEW, VIEW)
    ax.set_ylim(-VIEW, VIEW)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Trajectories of {len(sample_idx)} representative tracers "
        f"(r0 = {r0[sample_idx].min():.2f} - {r0[sample_idx].max():.2f}) across schemes",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper right")
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

    Each panel shows the disk-wind domain at the current time:
    - Density field (Greys colourmap) as background; updated every frame.
    - All tracers coloured by log10 position error vs the analytic solution
      (valid for every tracer in this scenario; hot: white = large error,
      dark = small error); shared colourbar.
    - Launch-radius circle (white dashed).
    """
    n_panels = len(scheme_names)
    n_frames = len(t_levels)
    n_tr = trajs[0].shape[-1]

    # Error vs the analytic solution at each frame (not vs the "best" panel).
    errors = []
    for p in range(n_panels):
        traj = trajs[p]  # (n_frames, 3, n_tr)
        err = np.empty((n_frames, n_tr))
        for fi, t in enumerate(t_levels):
            x_ex, y_ex = exact_position(x0, y0, t)
            err[fi] = np.sqrt((traj[fi, 0] - x_ex) ** 2 + (traj[fi, 1] - y_ex) ** 2)
        errors.append(np.maximum(err, 1e-6))

    flat = np.concatenate([e for pe in errors for e in pe[1:]])
    flat = flat[np.isfinite(flat) & (flat > 0)]
    vmin_err = float(np.nanpercentile(flat, 2))
    vmax_err = float(np.nanpercentile(flat, 99))
    norm = LogNorm(vmin=max(vmin_err, 1e-6), vmax=vmax_err)
    err_cmap = "seismic"

    xx_bg, yy_bg = np.meshgrid(_PLOT_X, _PLOT_Y, indexing="ij")
    rho_frames = [analytic_rho(xx_bg, yy_bg, t) for t in t_levels]
    rho_vmax = max(r.max() for r in rho_frames)

    theta_c = np.linspace(0, 2 * np.pi, 300)
    launch_xy = (R0 * np.cos(theta_c), R0 * np.sin(theta_c))

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(6 * n_panels, 6),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    meshes, sc_list, circle_list = [], [], []

    for p, (ax, name) in enumerate(zip(axes, scheme_names)):
        traj = trajs[p]

        mesh = ax.pcolormesh(
            _PLOT_X, _PLOT_Y, rho_frames[0].T,
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

        line, = ax.plot(*launch_xy, color="white", lw=1.5, ls="--", zorder=5)
        circle_list.append(line)

        ax.set_xlim(-VIEW, VIEW)
        ax.set_ylim(-VIEW, VIEW)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x")

    axes[0].set_ylabel("y")

    sm = plt.cm.ScalarMappable(cmap=err_cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes,
                 label="Position error vs analytic solution (log scale)",
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
            # Launch circle is static in this scenario; nothing to update.

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
# Comparison (c): spatial interpolator held fixed at PCHIP, vary only the
# time integrator.  This isolates the integrator's contribution to the
# error, decoupled from the choice of spatial interpolation order -- see
# examples/disk_wind_interpolators.py for the complementary comparison
# (integrator held fixed at RK4, vary the interpolator).

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
    disk-wind field and save the comparison plots/animation to
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

    t_levels = TIMES[1: N_STEPS + 2]  # length N_STEPS+1

    # ====================================================================
    # Plot 1: final tracer positions on density background
    # ====================================================================
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]
    t_end = t_levels[-1]

    for ax, (name, _, _), color in zip(axes, schemes, colors):
        _density_bg(t_end, ax)
        traj = results[name]
        ax.scatter(traj[-1, 0], traj[-1, 1], s=8, c=[color],
                   label="tracers", zorder=3)
        _launch_circle(t_end, ax, color="white", lw=1.5, ls="--",
                      label=f"launch radius R0={R0:.2f}")
        ax.set_xlim(-VIEW, VIEW)
        ax.set_ylim(-VIEW, VIEW)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.legend(fontsize=7, loc="upper right")

    axes[0].set_ylabel("y")
    fig.suptitle(
        f"Tracer final positions at t = {t_end:.2f}  |  "
        f"R0 = {R0:.2f}, Omega0 = {OMEGA0:.2f}, T0 = {T0:.2f}",
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

        ax.axvline(R0, color="black", ls=":", lw=1.2,
                   label=f"launch R0 = {R0:.2f}")
        ax.set_title(f"t = {t_lev:.2f}")
        ax.set_xlabel("r  (from disk centre)")
        ax.set_ylabel("M(<r)")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Enclosed mass M(<r) at 4 timesteps\n"
        "Vertical dotted line: launch radius R0",
        fontsize=11,
    )
    fig.tight_layout()
    out2 = os.path.join(PLOT_DIR, f"{output_prefix}_M_R.png")
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out2}")

    # ====================================================================
    # Plot 3: RMS position error vs time (exact solution valid for all)
    # ====================================================================
    rms: dict[str, list[float]] = {name: [] for name, _, _ in schemes}

    for si, t in enumerate(t_levels):
        x_ex, y_ex = exact_position(x0, y0, t)
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
        f"RMS position error vs time  ({n_tr} tracers, "
        "exact analytic solution valid everywhere)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out3 = os.path.join(PLOT_DIR, f"{output_prefix}_errors.png")
    fig.savefig(out3, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out3}")

    print("\n  RMS position error at T_END:")
    for name, _, _ in schemes:
        print(f"    {name:<25s}  {rms[name][-1]:.3e}")
    print()

    print("  Building disk-wind animation ...", flush=True)
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
    run_comparison(SCHEMES, "disk_wind")
