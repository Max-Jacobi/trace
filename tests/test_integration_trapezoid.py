"""
tests/test_integration_trapezoid.py

Exploratory comparison: ExplicitTrapezoid vs ImplicitTrapezoid.

Both schemes use PchipInterpolator3D for spatial interpolation and are
driven through the same 2-D asymmetric blast wave defined in
test_integration_blast.py.

Both methods are formally 2nd-order accurate in time.  The key
differences are:

  ExplicitTrapezoid (Heun / predictor-corrector):
    x* = x_n + dt * v(x_n, t_n)              (predictor: explicit Euler)
    x_{n+1} = x_n + dt/2 * (v(x_n, t_n) + v(x*, t_{n+1}))  (corrector)

  ImplicitTrapezoid (Crank-Nicolson, Picard iteration):
    x_{n+1} = x_n + dt/2 * (v(x_n, t_n) + v(x_{n+1}, t_{n+1}))
    (iterated to convergence; the corrector position is refined
    rather than fixed by a single predictor step)

For smooth flows both achieve similar O(dt^2) accuracy.  The implicit
method has a smaller error constant and is unconditionally stable (A-
stable), while the explicit method requires the predictor to land in a
region of compatible velocity.  The difference becomes visible:

  - At large DT, where the predictor step is large enough to overshoot
    across the shock transition region, sampling a very different
    velocity than the converged implicit solution.
  - For tracers whose initial xi is close to 1 (near the shock front),
    where the velocity changes most steeply.

Three analyses and plots
------------------------
  1. trapezoid_convergence.png
     Log-log convergence: RMS position error vs DT for inside-shock
     tracers (xi0 < 0.6, analytic solution valid).  Both methods lie on
     O(dt^2) lines; the implicit method sits lower.

  2. trapezoid_disagreement.png
     Position disagreement |ExplTrap - ImplTrap| at T_END as a function
     of each tracer's initial xi0.  Run at a large step size (DT = 0.5)
     to make the near-shock discrepancy visible.

  3. trapezoid_final_positions.png
     Side-by-side scatter of final tracer positions at large DT overlaid
     on the density background.  White cross markers show the analytic
     positions for inside-shock tracers.

No failure conditions: the test passes if no exception is raised.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.integrators.expl_trapezoid import ExplicitTrapezoid
from src.integrators.impl_trapezoid import ImplicitTrapezoid
from src.interpolators.pchip import PchipInterpolator3D

# Physics, grid constants and helpers from the blast-wave test module.
# _SnapInterp uses X_GRID/Y_GRID/Z_GRID/NX/NY/NZ from that module.
from tests.test_integration_blast import (
    analytic_rho, exact_position, _xi,
    X_GRID, Y_GRID, Z_GRID, NX, NY, NZ, Z_C,
    T_START, T_END, ALPHA, X_C, Y_C,
    PLOT_DIR,
    _density_bg, _shock_circle, _init_tracers, _SnapInterp,
)


# ---------------------------------------------------------------------------
# Integration driver (explicit times, no module-level state)
# ---------------------------------------------------------------------------

def _run(integrator, x0: np.ndarray, y0: np.ndarray, times: np.ndarray):
    """
    Integrate tracers through the blast wave using PchipInterpolator3D.

    Parameters
    ----------
    integrator : IntegratorBase instance
        ExplicitTrapezoid or ImplicitTrapezoid (n_snapshots == 2).
    x0, y0 : ndarray, shape (n_tr,)
        Initial tracer positions at times[1] = T_START.
    times : ndarray, shape (n_steps + 3,)
        Padded snapshot times.
        times[0]            = T_START - DT    (left buffer, unused by 2-snap methods)
        times[1..n_steps+1] = T_START..T_END  (actual integration window)
        times[n_steps+2]    = T_END + DT      (right buffer, unused)

    Returns
    -------
    traj : ndarray, shape (n_steps + 1, 3, n_tr)
        Positions at each stored time level times[1..n_steps+1].
    """
    n_steps = len(times) - 3
    n_tr = len(x0)
    xn = np.array([x0, y0, np.full(n_tr, Z_C)], dtype=float)  # (3, n_tr)
    traj = [xn.copy()]

    snap_cache: dict[int, _SnapInterp] = {}

    def _get(k: int) -> _SnapInterp:
        if k not in snap_cache:
            snap_cache[k] = _SnapInterp(times[k], PchipInterpolator3D)
        return snap_cache[k]

    def _evict(k_min: int):
        for k in [k for k in list(snap_cache) if k < k_min]:
            snap_cache[k].close()
            del snap_cache[k]

    for step_i in range(n_steps):
        # 2-snapshot methods use snap at start and end of step.
        needed = [step_i + 1, step_i + 2]
        _evict(needed[0])
        interps = (_get(needed[0]), _get(needed[1]))
        dt = float(times[step_i + 2] - times[step_i + 1])
        xn = integrator(xn, dt, interps)
        traj.append(xn.copy())

    for snap in snap_cache.values():
        snap.close()
    return np.array(traj)  # (n_steps+1, 3, n_tr)


def _make_times(dt: float):
    """
    Build a padded times array for a given nominal step size.

    Returns (times, n_steps, dt_actual) where dt_actual may differ
    slightly from dt so that T_END is hit exactly.
    """
    n_steps = max(1, round((T_END - T_START) / dt))
    dt_actual = (T_END - T_START) / n_steps
    times = np.linspace(T_START - dt_actual, T_END + dt_actual, n_steps + 3)
    return times, n_steps, dt_actual


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestTrapezoidComparison:
    """
    Exploratory comparison: ExplicitTrapezoid vs ImplicitTrapezoid.
    Both use PchipInterpolator3D; identical spatial treatment.
    No failure conditions.
    """

    def test_trapezoid_comparison(self):
        """
        Run three analyses and save three plots to tests/plots/.
        Expected runtime: ~2-5 min (multiple DT values x 2 schemes).
        """
        os.makedirs(PLOT_DIR, exist_ok=True)

        x0, y0, masses = _init_tracers()
        xi0 = _xi(x0, y0, T_START)

        # Masks for different radial zones
        inside   = xi0 < 0.6          # well inside: analytic solution exact
        near_shk = (xi0 >= 0.7) & (xi0 <= 1.3)  # near shock transition

        x_in, y_in = x0[inside], y0[inside]

        expl = ExplicitTrapezoid()
        impl = ImplicitTrapezoid()

        # ================================================================
        # Analysis 1: Convergence study (inside-shock tracers, xi0 < 0.6)
        # ================================================================
        dt_vals = [0.1, 0.15, 0.2, 0.3, 0.5, 0.75]
        errs_expl, errs_impl, dt_actual_vals = [], [], []

        print("\n  Convergence study (xi0 < 0.6):", flush=True)
        print(f"    {'DT':>6}  {'n_steps':>7}  {'ExplTrap':>10}  {'ImplTrap':>10}")

        for dt_nom in dt_vals:
            times, n_steps, dt_act = _make_times(dt_nom)
            dt_actual_vals.append(dt_act)

            traj_e = _run(expl, x0, y0, times)
            traj_i = _run(impl, x0, y0, times)

            x_ex, y_ex = exact_position(x_in, y_in, T_END)

            def _rms(traj):
                xn = traj[-1, 0, inside]
                yn = traj[-1, 1, inside]
                return float(np.sqrt(np.nanmean((xn - x_ex) ** 2 + (yn - y_ex) ** 2)))

            ee = _rms(traj_e)
            ei = _rms(traj_i)
            errs_expl.append(ee)
            errs_impl.append(ei)
            print(f"    {dt_act:6.3f}  {n_steps:7d}  {ee:10.3e}  {ei:10.3e}", flush=True)

        dt_arr = np.array(dt_actual_vals)
        # O(dt^2) reference anchored to the ExplTrap value at the smallest DT
        ref = errs_expl[0] * (dt_arr / dt_arr[0]) ** 2

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(dt_arr, errs_expl, "o-",  color="tab:blue",   lw=2, label="ExplicitTrapezoid")
        ax.loglog(dt_arr, errs_impl, "s--", color="tab:orange",  lw=2, label="ImplicitTrapezoid")
        ax.loglog(dt_arr, ref,       "k:",              lw=1.5, label="O(dt^2) reference")
        ax.set_xlabel("DT")
        ax.set_ylabel("RMS position error at T_END")
        ax.set_title(
            "Convergence: ExplicitTrapezoid vs ImplicitTrapezoid\n"
            f"(tracers with xi0 < 0.6, PCHIP spatial, T = {T_START} -> {T_END})"
        )
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out1 = os.path.join(PLOT_DIR, "trapezoid_convergence.png")
        fig.savefig(out1, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: {out1}")

        # ================================================================
        # Analysis 2: Disagreement vs xi0 at large DT = 0.5
        # ================================================================
        DT_LARGE = 0.5
        times_l, n_l, dt_l = _make_times(DT_LARGE)

        print(f"\n  Large DT = {DT_LARGE} run (n_steps = {n_l}) ...", flush=True)
        traj_expl_l = _run(expl, x0, y0, times_l)
        traj_impl_l = _run(impl, x0, y0, times_l)

        # Absolute position difference between the two methods at T_END
        diff_x = traj_expl_l[-1, 0] - traj_impl_l[-1, 0]
        diff_y = traj_expl_l[-1, 1] - traj_impl_l[-1, 1]
        r_diff = np.sqrt(diff_x ** 2 + diff_y ** 2)

        # Also compute error of each method vs analytic for inside-shock tracers
        x_ex_end, y_ex_end = exact_position(x_in, y_in, T_END)
        err_expl_inside = np.sqrt(
            (traj_expl_l[-1, 0, inside] - x_ex_end) ** 2 +
            (traj_expl_l[-1, 1, inside] - y_ex_end) ** 2
        )
        err_impl_inside = np.sqrt(
            (traj_impl_l[-1, 0, inside] - x_ex_end) ** 2 +
            (traj_impl_l[-1, 1, inside] - y_ex_end) ** 2
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: scatter of disagreement vs xi0
        ax = axes[0]
        sc = ax.scatter(xi0, r_diff, c=xi0, cmap="plasma", s=15,
                        vmin=0.0, vmax=min(2.5, xi0.max()))
        plt.colorbar(sc, ax=ax, label="xi0")
        ax.axvline(1.0, color="red", ls="--", lw=1.5,
                   label="Shock front (xi0 = 1)")
        ax.set_xlabel("xi0 = r0 / R_shock(T_START)")
        ax.set_ylabel("|ExplTrap - ImplTrap| at T_END")
        ax.set_title(f"Position disagreement vs xi0  (DT = {DT_LARGE})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: individual errors vs analytic for inside-shock tracers
        ax2 = axes[1]
        xi0_in = xi0[inside]
        ax2.scatter(xi0_in, err_expl_inside, color="tab:blue",   s=25,
                    label="ExplTrap vs analytic")
        ax2.scatter(xi0_in, err_impl_inside, color="tab:orange", s=25,
                    marker="s", label="ImplTrap vs analytic")
        ax2.set_xlabel("xi0 (inside-shock tracers)")
        ax2.set_ylabel("Position error vs analytic at T_END")
        ax2.set_title(f"Inside-shock errors  (DT = {DT_LARGE})")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            f"ExplicitTrapezoid vs ImplicitTrapezoid  |  DT = {DT_LARGE}  "
            f"(n_steps = {n_l})\n"
            "Implicit iteration removes the predictor-overshoot error near the shock",
            fontsize=10,
        )
        fig.tight_layout()
        out2 = os.path.join(PLOT_DIR, "trapezoid_disagreement.png")
        fig.savefig(out2, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out2}")

        # ================================================================
        # Analysis 3: Side-by-side final positions at large DT
        # ================================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        t_end = T_END

        for ax, (traj, label, color) in zip(axes, [
            (traj_expl_l, f"ExplicitTrapezoid  (DT={DT_LARGE})", "tab:blue"),
            (traj_impl_l, f"ImplicitTrapezoid  (DT={DT_LARGE})", "tab:orange"),
        ]):
            _density_bg(t_end, ax)
            ax.scatter(traj[-1, 0], traj[-1, 1],
                       s=8, c=color, zorder=3, label="Tracer positions")
            # Analytic positions for inside-shock tracers
            ax.scatter(x_ex_end, y_ex_end,
                       s=30, c="white", marker="+", zorder=5, linewidths=0.8,
                       label="Analytic (xi0 < 0.6)")
            _shock_circle(t_end, ax, color="white", lw=1.5, ls="--",
                          label=f"Shock R={( 1+t_end)**ALPHA:.2f}")
            ax.set_xlim(-3.0, 5.0)
            ax.set_ylim(-4.5, 4.5)
            ax.set_title(label)
            ax.set_xlabel("x")
            ax.legend(fontsize=7, loc="upper right")

        axes[0].set_ylabel("y")
        fig.suptitle(
            f"Final tracer positions at T_END = {T_END}  |  DT = {DT_LARGE}\n"
            "White + markers = analytic positions for tracers with xi0 < 0.6",
            fontsize=10,
        )
        fig.tight_layout()
        out3 = os.path.join(PLOT_DIR, "trapezoid_final_positions.png")
        fig.savefig(out3, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out3}")

        # ================================================================
        # Summary
        # ================================================================
        print(f"\n  Results at DT = {DT_LARGE} (n_steps = {n_l}):")
        print(f"    ExplTrap RMS error (inside, xi0<0.6): {err_expl_inside.mean():.3e}")
        print(f"    ImplTrap RMS error (inside, xi0<0.6): {err_impl_inside.mean():.3e}")
        print(f"    Disagreement between methods:")
        print(f"      All tracers:        mean {r_diff.mean():.3e},  max {r_diff.max():.3e}")
        near_diff = r_diff[near_shk]
        if near_diff.size > 0:
            print(f"      Near-shock (0.7<xi0<1.3): mean {near_diff.mean():.3e},  "
                  f"max {near_diff.max():.3e}")
        print()
