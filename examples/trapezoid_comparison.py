"""
examples/trapezoid_comparison.py

Exploratory comparison: ExplicitTrapezoid vs ImplicitTrapezoid.

This is not a pytest test -- it is a runnable script.  Run it directly:

    PYTHONPATH=. python examples/trapezoid_comparison.py

Both schemes use PchipInterpolator3D for spatial interpolation and are
driven through the same differentially-rotating disk-wind outflow defined
in examples/disk_wind.py (post-BNS-merger ejecta analogue: matter launched
radially outward from a reference radius R0 while conserving the angular
momentum it carried in the disk).

Both methods are formally 2nd-order accurate in time.  The key difference
is unconditional (A-) stability, not local truncation error.  Locally,
rotation at fixed r behaves like a harmonic oscillator with purely
imaginary eigenvalue +-i*Omega(r):

  ExplicitTrapezoid (Heun / predictor-corrector) has amplification factor
      |g| = sqrt(1 + (Omega*dt)^4 / 4) > 1   for any nonzero dt,
  so it spirals outward every step -- slowly when Omega*dt is small, fast
  once Omega*dt approaches order unity.

  ImplicitTrapezoid (Crank-Nicolson, Picard iteration) has |g| = 1 exactly
  for any dt, so converged tracers stay on their exact spiral trajectory
  indefinitely.

Because Omega(r) = OMEGA0 * (R0/r)^Q falls off sharply with radius, this
divergence is concentrated near the launch radius R0 (large Omega, stiff)
and vanishes for tracers that have already streamed far outward and
decoupled from the rotation (small Omega).

Locally stiffened rotation (this script only)
------------------------------------------------
With disk_wind.py's default OMEGA0 = 0.4, even the largest DT tested here
only reaches Omega(R0)*DT ~ 0.1 -- nowhere near the O(1) threshold where
ExplicitTrapezoid's instability actually shows up, so the comparison would
mostly just confirm both methods are O(dt^2) without ever showing the
divergence that's the point of this script.  OMEGA0 is therefore
overridden to OMEGA0_LOCAL (see below) for this script's process only, by
mutating the disk_wind module's attribute directly (not just a locally
imported name) so that disk_wind.py's own functions (_omega, analytic_vx
etc., which look up OMEGA0 in disk_wind's namespace) pick up the new
value.  Since each example script runs as its own process, this has no
effect on examples/disk_wind.py's own comparisons (c)/(d).

Parallelisation
---------------
Each step's tracer batch is split into bunches and dispatched to a
persistent worker pool (see examples/_parallel.py), reusing
examples.disk_wind's worker-side interpolator cache and (theta, phi)
cache-locality sort directly.

Four analyses and plots
------------------------
  1. trapezoid_convergence.png
     Log-log convergence: RMS position error vs DT (the analytic solution
     is valid for every tracer in this scenario).  Both methods lie on
     O(dt^2) lines at small DT; the explicit curve bends upward once
     Omega(R0)*dt approaches order unity, showing the onset of the
     instability.  The implicit method sits lower and stays on the
     O(dt^2) line throughout.

  2. trapezoid_disagreement.png
     Position disagreement |ExplTrap - ImplTrap| at T_END as a function
     of each tracer's launch-normalised radius xi0 = r0 / R0.  Run at a
     step size large enough that Omega(R0)*dt = O(1) to make the
     near-launch (stiff) discrepancy visible.

  3. trapezoid_final_positions.png
     Side-by-side scatter of final tracer positions at large DT overlaid
     on the density background.  White cross markers show the analytic
     positions.

  4. trapezoid_trajectories.png
     (x(t), y(t)) paths of every tracer at the large DT, one panel per
     scheme -- explicit trapezoid's outward spiral near the launch radius
     is directly visible here.

This script has no pass/fail assertions; it always "succeeds" unless an
exception is raised.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.integrators.expl_trapezoid import ExplicitTrapezoid
from src.integrators.impl_trapezoid import ImplicitTrapezoid
from src.interpolators.pchip import PchipInterpolator3D

# Physics, grid constants and parallel-worker helpers from the disk-wind
# example module.
import examples.disk_wind as _dw
from examples.disk_wind import (
    analytic_rho, exact_position, _xi,
    T_START, T_END, LAMBDA, R0,
    PLOT_DIR,
    _density_bg, _launch_circle, _init_tracers, _SnapInterp,
    _animate_tracers, plot_trajectories,
    _sort_order, _integrate_bunch,
)
from examples._parallel import N_CPU, N_BUNCHES, worker_pool

# See "Locally stiffened rotation" in the module docstring above: this
# overrides disk_wind.py's module attribute (not just a local name) so
# disk_wind's own functions pick up the new value too.  OMEGA0 here always
# refers to the (possibly overridden) value actually in effect.
OMEGA0_LOCAL = 3.0
_dw.OMEGA0 = OMEGA0_LOCAL
OMEGA0 = OMEGA0_LOCAL


# ---------------------------------------------------------------------------
# Integration driver
# ---------------------------------------------------------------------------
#
# The disk-wind velocity field carries a homologous radial term that makes
# it genuinely time-dependent, so a fresh snapshot must be built at every
# new time level.  Only one new snapshot per step is needed though -- the
# trailing endpoint of step n is the leading endpoint of step n+1.  Each
# step's tracer batch is parallelised across `pool` exactly like
# examples.disk_wind._run_scheme.

def _run(integrator, x0: np.ndarray, y0: np.ndarray, n_steps: int,
         dt: float, pool, t_start: float = T_START):
    """
    Integrate tracers through the (time-dependent) disk wind.

    Parameters
    ----------
    integrator : IntegratorBase instance
        ExplicitTrapezoid or ImplicitTrapezoid (n_snapshots == 2).
    x0, y0 : ndarray, shape (n_tr,)
        Initial tracer positions at t_start.
    n_steps : int
        Number of steps of size dt from t_start to t_start + n_steps * dt.
    dt : float
        Step size.
    pool : multiprocessing.Pool
        Persistent worker pool used to parallelise tracer bunches.
    t_start : float
        Start time.

    Returns
    -------
    traj : ndarray, shape (n_steps + 1, 3, n_tr)
    """
    n_tr = len(x0)
    xn = np.array([x0, y0, np.zeros(n_tr)], dtype=float)  # (3, n_tr)
    traj = [xn.copy()]
    n_bunches = min(N_BUNCHES, n_tr)

    t = t_start
    snap_a = _SnapInterp(t, PchipInterpolator3D)
    for _ in range(n_steps):
        t_next = t + dt
        snap_b = _SnapInterp(t_next, PchipInterpolator3D)
        shm_names_list = [snap_a.shm_names, snap_b.shm_names]

        order = _sort_order(xn)
        xn_sorted = xn[:, order]
        pos_bunches = [b for b in np.array_split(xn_sorted, n_bunches, axis=1)
                       if b.shape[1] > 0]
        tasks = [
            (bunch, dt, integrator, PchipInterpolator3D, shm_names_list, None)
            for bunch in pos_bunches
        ]
        results = pool.map(_integrate_bunch, tasks)
        new_xn_sorted = np.concatenate(results, axis=1)

        xn = np.empty_like(new_xn_sorted)
        xn[:, order] = new_xn_sorted

        traj.append(xn.copy())
        snap_a.close()
        snap_a = snap_b
        t = t_next
    snap_a.close()

    return np.array(traj)  # (n_steps+1, 3, n_tr)


def _n_steps_for(dt: float):
    """
    Returns (n_steps, dt_actual) where dt_actual may differ slightly from
    the requested dt so that T_END is hit exactly.
    """
    n_steps = max(1, round((T_END - T_START) / dt))
    dt_actual = (T_END - T_START) / n_steps
    return n_steps, dt_actual


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    x0, y0, masses = _init_tracers()
    xi0 = _xi(x0, y0, T_START)  # = r0 / R0, ranges over [1, 2]

    # Masks for different stiffness zones (Omega(r) falls off as r^-Q)
    near_launch = xi0 <= 1.2   # large Omega: stiff, where explicit suffers
    decoupled = xi0 >= 1.8     # small Omega: tracers have outrun the spin-up

    expl = ExplicitTrapezoid()
    impl = ImplicitTrapezoid()

    print(f"  Using a worker pool of {N_CPU} processes "
          f"({N_BUNCHES} tracer bunches per step).", flush=True)

    with worker_pool(N_CPU) as pool:
        # ================================================================
        # Analysis 1: Convergence study (RMS error over all tracers; the
        # analytic solution is valid everywhere in this scenario).
        # ================================================================
        # Fewer points than a "real" convergence study would use, chosen to
        # keep cost down (each new DT needs a full fresh run, and every step
        # needs a freshly-built velocity snapshot since the field is
        # time-dependent) while still spanning from clearly-stable
        # (Omega(R0)*dt ~ 0.1, smallest) to clearly-unstable (~0.6, largest).
        dt_vals = [0.04, 0.08, 0.12, 0.16, 0.2]
        errs_expl, errs_impl, dt_actual_vals = [], [], []

        print("\n  Convergence study (all tracers):", flush=True)
        print(f"    {'DT':>6}  {'n_steps':>7}  {'ExplTrap':>10}  {'ImplTrap':>10}", flush=True)

        for dt_nom in dt_vals:
            n_steps, dt_act = _n_steps_for(dt_nom)
            dt_actual_vals.append(dt_act)

            traj_e = _run(expl, x0, y0, n_steps, dt_act, pool)
            traj_i = _run(impl, x0, y0, n_steps, dt_act, pool)

            x_ex, y_ex = exact_position(x0, y0, T_END)

            def _rms(traj):
                xn = traj[-1, 0]
                yn = traj[-1, 1]
                return float(np.sqrt(np.nanmean((xn - x_ex) ** 2 + (yn - y_ex) ** 2)))

            ee = _rms(traj_e)
            ei = _rms(traj_i)
            errs_expl.append(ee)
            errs_impl.append(ei)
            print(f"    {dt_act:6.3f}  {n_steps:7d}  {ee:10.3e}  {ei:10.3e}", flush=True)

        dt_arr = np.array(dt_actual_vals)
        # O(dt^2) reference anchored to the ImplTrap value at the smallest DT
        # (the implicit curve stays on the true asymptotic slope throughout).
        ref = errs_impl[0] * (dt_arr / dt_arr[0]) ** 2

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(dt_arr, errs_expl, "o-",  color="tab:blue",   lw=2, label="ExplicitTrapezoid")
        ax.loglog(dt_arr, errs_impl, "s--", color="tab:orange",  lw=2, label="ImplicitTrapezoid")
        ax.loglog(dt_arr, ref,       "k:",              lw=1.5, label="O(dt^2) reference")
        ax.set_xlabel("DT")
        ax.set_ylabel("RMS position error at T_END")
        ax.set_title(
            "Convergence: ExplicitTrapezoid vs ImplicitTrapezoid\n"
            f"(disk-wind outflow, PCHIP spatial, T = {T_START} -> {T_END}, "
            f"Omega(R0) = {OMEGA0:.2f})"
        )
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out1 = os.path.join(PLOT_DIR, "trapezoid_convergence.png")
        fig.savefig(out1, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: {out1}")

        # ================================================================
        # Analysis 2: Disagreement vs xi0 at large DT (Omega(R0)*DT ~ O(1))
        # ================================================================
        DT_LARGE = 0.2
        n_l, dt_l = _n_steps_for(DT_LARGE)

        print(f"\n  Large DT = {DT_LARGE} run (n_steps = {n_l}, "
              f"Omega(R0)*DT = {OMEGA0 * dt_l:.2f}) ...", flush=True)
        traj_expl_l = _run(expl, x0, y0, n_l, dt_l, pool)
        traj_impl_l = _run(impl, x0, y0, n_l, dt_l, pool)

        # Absolute position difference between the two methods at T_END
        diff_x = traj_expl_l[-1, 0] - traj_impl_l[-1, 0]
        diff_y = traj_expl_l[-1, 1] - traj_impl_l[-1, 1]
        r_diff = np.sqrt(diff_x ** 2 + diff_y ** 2)

        # Also compute error of each method vs the analytic solution (valid
        # for all tracers here).
        x_ex_end, y_ex_end = exact_position(x0, y0, T_END)
        err_expl = np.sqrt(
            (traj_expl_l[-1, 0] - x_ex_end) ** 2 +
            (traj_expl_l[-1, 1] - y_ex_end) ** 2
        )
        err_impl = np.sqrt(
            (traj_impl_l[-1, 0] - x_ex_end) ** 2 +
            (traj_impl_l[-1, 1] - y_ex_end) ** 2
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: scatter of disagreement vs xi0
        ax = axes[0]
        sc = ax.scatter(xi0, r_diff, c=xi0, cmap="plasma", s=15)
        plt.colorbar(sc, ax=ax, label="xi0")
        ax.axvline(1.0, color="red", ls="--", lw=1.5,
                   label="Launch radius (xi0 = 1)")
        ax.set_xlabel("xi0 = r0 / R0")
        ax.set_ylabel("|ExplTrap - ImplTrap| at T_END")
        ax.set_title(f"Position disagreement vs xi0  (DT = {DT_LARGE})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: individual errors vs analytic, all tracers
        ax2 = axes[1]
        ax2.scatter(xi0, err_expl, color="tab:blue",   s=25,
                    label="ExplTrap vs analytic")
        ax2.scatter(xi0, err_impl, color="tab:orange", s=25,
                    marker="s", label="ImplTrap vs analytic")
        ax2.set_yscale("log")
        ax2.set_xlabel("xi0 = r0 / R0")
        ax2.set_ylabel("Position error vs analytic at T_END")
        ax2.set_title(f"Error vs analytic, all tracers  (DT = {DT_LARGE})")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            f"ExplicitTrapezoid vs ImplicitTrapezoid  |  DT = {DT_LARGE}  "
            f"(n_steps = {n_l}, Omega(R0)*DT = {OMEGA0 * dt_l:.2f})\n"
            "Implicit iteration removes the rotational instability near the launch radius",
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
        view = 4.0

        for ax, (traj, label, color) in zip(axes, [
            (traj_expl_l, f"ExplicitTrapezoid  (DT={DT_LARGE})", "tab:blue"),
            (traj_impl_l, f"ImplicitTrapezoid  (DT={DT_LARGE})", "tab:orange"),
        ]):
            _density_bg(t_end, ax)
            ax.scatter(traj[-1, 0], traj[-1, 1],
                       s=8, c=color, zorder=3, label="Tracer positions")
            ax.scatter(x_ex_end, y_ex_end,
                       s=30, c="white", marker="+", zorder=5, linewidths=0.8,
                       label="Analytic position")
            _launch_circle(t_end, ax, color="white", lw=1.5, ls="--",
                           label=f"Launch radius R0={R0:.2f}")
            ax.set_xlim(-view, view)
            ax.set_ylim(-view, view)
            ax.set_title(label)
            ax.set_xlabel("x")
            ax.legend(fontsize=7, loc="upper right")

        axes[0].set_ylabel("y")
        fig.suptitle(
            f"Final tracer positions at T_END = {T_END}  |  DT = {DT_LARGE}\n"
            "White + markers = analytic positions (valid for all tracers)",
            fontsize=10,
        )
        fig.tight_layout()
        out3 = os.path.join(PLOT_DIR, "trapezoid_final_positions.png")
        fig.savefig(out3, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out3}")

        # ================================================================
        # Analysis 4: tracer trajectories at large DT (xy-plane)
        # ================================================================
        plot_trajectories(
            scheme_names=["ExplicitTrapezoid", "ImplicitTrapezoid"],
            trajs=[traj_expl_l, traj_impl_l],
            x0=x0, y0=y0,
            output_path=os.path.join(PLOT_DIR, "trapezoid_trajectories.png"),
        )

        # ================================================================
        # Summary
        # ================================================================
        print(f"\n  Results at DT = {DT_LARGE} (n_steps = {n_l}):")
        print(f"    ExplTrap RMS error (all tracers): {err_expl.mean():.3e}")
        print(f"    ImplTrap RMS error (all tracers): {err_impl.mean():.3e}")
        print(f"    Disagreement between methods:")
        print(f"      All tracers:    mean {r_diff.mean():.3e},  max {r_diff.max():.3e}")
        near_diff = r_diff[near_launch]
        far_diff = r_diff[decoupled]
        if near_diff.size > 0:
            print(f"      Near launch (xi0<=1.2): mean {near_diff.mean():.3e},  "
                  f"max {near_diff.max():.3e}")
        if far_diff.size > 0:
            print(f"      Decoupled  (xi0>=1.8):  mean {far_diff.mean():.3e},  "
                  f"max {far_diff.max():.3e}")
        print()

        # ================================================================
        # Animation: 2-panel ExplTrap vs ImplTrap at DT=0.1
        # ================================================================
        DT_ANIM = 0.1
        n_anim, dt_anim = _n_steps_for(DT_ANIM)
        t_levels_anim = T_START + dt_anim * np.arange(n_anim + 1)

        traj_expl_a = _run(expl, x0, y0, n_anim, dt_anim, pool)
        traj_impl_a = _run(impl, x0, y0, n_anim, dt_anim, pool)

        print("  Building trapezoid animation ...", flush=True)
        anim_path = os.path.join(PLOT_DIR, "trapezoid_animation.mp4")
        _animate_tracers(
            scheme_names=["ExplicitTrapezoid", "ImplicitTrapezoid"],
            trajs=[traj_expl_a, traj_impl_a],
            t_levels=t_levels_anim,
            x0=x0,
            y0=y0,
            output_path=anim_path,
        )


if __name__ == "__main__":
    main()
