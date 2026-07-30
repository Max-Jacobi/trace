#!/usr/bin/env python3
"""
Mass-weighted histogram panel of per-tracer summary quantities -- the
standard "what does this ejecta look like" overview: peak temperature,
composition/entropy/expansion-timescale at the nucleosynthesis-relevant
freeze-out temperature, where the ejecta ends up (angle, radius), and how
fast it's ultimately moving.

Every histogram is weighted by each tracer's represented mass, so the
y-axis is always "ejecta mass per bin" (M_sun), not "tracer count per bin"
-- a handful of high-mass tracers matter more than a swarm of low-mass
ones.

Example
-------
    python analysis/histograms.py \\
        --tracer-dirs data/tracers_out data/tracers_out_surf \\
        --t-ref-gk 5.0 --output histograms.png
"""

import argparse
import pathlib as pl

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis._common import (
    add_tracer_loading_args, load_trajectories, add_spherical_fields,
    tracer_masses, get_units, asymptotic_velocity,
)

ALL_PANELS = (
    'Tmax', 'Ye_ref', 's_ref', 'tau_ref', 'theta_final', 'phi_final', 'r_final', 'v_final',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mass-weighted histogram panel of per-tracer summary quantities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_tracer_loading_args(parser)
    parser.add_argument('--t-ref-gk', type=float, default=5.0,
                         help="Reference ('NSE dropout') temperature in GK, at which Ye/entropy/expansion "
                              "timescale are sampled -- typical literature values are 5-6 GK.")
    parser.add_argument('--panels', nargs='+', default=list(ALL_PANELS), choices=list(ALL_PANELS),
                         help="Which panels to include.")
    parser.add_argument('--r-ref-km', type=float, nargs='+', default=[],
                         help="Reference radii in km. For each, an extra figure is written with Ye, s, T, "
                              "|v| (coordinate + asymptotic), theta and phi sampled at the time each tracer "
                              "first crosses outward through that radius.")
    parser.add_argument('--bins', type=int, default=40, help="Number of histogram bins per panel.")
    parser.add_argument('--output', default='histograms.png', help="Output image path.")
    parser.add_argument('--dpi', type=int, default=300, help="Output image DPI.")
    return parser.parse_args()


def _reference_values(traj, t_ref_gk: float, Tfac: float) -> tuple:
    """
    Ye, entropy, and expansion timescale (code time units) at the time the
    tracer's temperature first drops through t_ref_gk, walking forward from
    its hottest recorded point. Returns (nan, nan, nan) if it never reaches
    t_ref_gk in that range.
    """
    T = traj.data['T'] * Tfac
    i_hot = np.argmax(T)
    T_after = T[i_hot:]
    # size guard: np.gradient needs >=2 points; chained comparison is False for NaN T
    if T_after.size < 2 or not (T_after.min() <= t_ref_gk <= T_after.max()):
        return np.nan, np.nan, np.nan

    time_after = traj.data['time'][i_hot:]
    ye_after = traj.data['r_0'][i_hot:]
    s_after = traj.data['s'][i_hot:]
    rho_after = traj.data['rho'][i_hot:]
    drhodt = np.gradient(rho_after, time_after)
    tau_after = rho_after / (np.abs(drhodt) + 1e-300)

    # np.interp needs increasing x; T_after is decreasing from the hottest point onward.
    T_rev = T_after[::-1]
    ye_ref = np.interp(t_ref_gk, T_rev, ye_after[::-1])
    s_ref = np.interp(t_ref_gk, T_rev, s_after[::-1])
    tau_ref = np.interp(t_ref_gk, T_rev, tau_after[::-1])
    return float(ye_ref), float(s_ref), float(tau_ref)


def _crossing_time(traj, r_ref: float) -> float:
    """
    Time (code units) at which the tracer first crosses outward through
    radius `r_ref` (code units), linearly interpolated between the two
    samples bracketing the crossing. If the tracer already starts outside
    `r_ref`, its first recorded time is returned; NaN if it never reaches
    `r_ref`.
    """
    r, t = traj.data['r'], traj.data['time']
    outside = r >= r_ref
    if not outside.any():
        return np.nan
    i = int(np.argmax(outside))
    if i == 0:
        return float(t[0])
    f = (r_ref - r[i - 1]) / (r[i] - r[i - 1])
    return float(t[i - 1] + f * (t[i] - t[i - 1]))


def _values_at_radius(trajs: list, r_ref_code: float, keys: list) -> dict:
    """Each key of traj.data interpolated to the tracer's outward crossing of r_ref_code; NaN if it never crosses."""
    out = {k: np.full(len(trajs), np.nan) for k in keys}
    for j, tr in enumerate(trajs):
        t_c = _crossing_time(tr, r_ref_code)
        if not np.isfinite(t_c):
            continue
        for k in keys:
            out[k][j] = np.interp(t_c, tr.data['time'], tr.data[k])
    return out


def _velocity_panel(ax, v_coord, v_inf_geo, v_inf_bernoulli, masses, total_mass, bins, coord_label) -> None:
    """Overlaid mass-weighted histograms of coordinate |v| and the (optional) asymptotic velocities."""
    for v, label in ((v_coord, coord_label),
                     (v_inf_geo, "asymptotic (geodesic)"),
                     (v_inf_bernoulli, "asymptotic (Bernoulli)")):
        if v is None:
            continue
        finite = np.isfinite(v)
        if finite.any():
            mass_frac = 100 * masses[finite].sum() / total_mass
            ax.hist(v[finite], bins=bins, weights=masses[finite],
                    histtype="step", label=f"{label}, {mass_frac:.1f}% of mass")
    ax.set_xlabel(r"$|v|$ (c)")
    ax.legend(fontsize=7)


def _radius_figure(trajs, masses, total_mass, r_ref_km, units, bins):
    """Histogram figure of Ye, s, T, |v|, theta, phi sampled where tracers cross r_ref_km."""
    keys = ['r_0', 's', 'T', 'theta', 'phi', 'V_u_x', 'V_u_y', 'V_u_z']
    keys += [k for k in ('u_t', 'hu_t') if k in trajs[0].data]
    vals = _values_at_radius(trajs, r_ref_km / units.length_km, keys)

    n_missed = np.sum(~np.isfinite(vals['T']))
    if n_missed:
        print(f"r_ref={r_ref_km:g} km: {n_missed} of {len(trajs)} tracers never reach that radius "
              f"({100 * masses[~np.isfinite(vals['T'])].sum() / total_mass:.2f}% of mass; excluded).")

    v_coord = np.sqrt(vals['V_u_x']**2 + vals['V_u_y']**2 + vals['V_u_z']**2)
    v_inf_geo = asymptotic_velocity(-vals['u_t']) if 'u_t' in vals else None
    v_inf_bernoulli = asymptotic_velocity(-vals['hu_t']) if 'hu_t' in vals else None

    panels = [
        (vals['r_0'], rf"$Y_e$ at {r_ref_km:g} km", False),
        (vals['s'], rf"$s$ at {r_ref_km:g} km ($k_{{\rm B}}$)", False),
        (vals['T'] * units.temperature_gk, rf"$T$ at {r_ref_km:g} km (GK)", False),
        (vals['theta'], rf"$\theta$ at {r_ref_km:g} km (deg)", False),
        (vals['phi'] % 360, rf"$\phi$ at {r_ref_km:g} km (deg)", False),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.6), squeeze=False)
    ax_flat = ax.flatten()
    for a, (values, label, log) in zip(ax_flat, panels):
        _mass_weighted_hist(a, values, masses, bins, label, log)
    _velocity_panel(ax_flat[len(panels)], v_coord, v_inf_geo, v_inf_bernoulli,
                    masses, total_mass, bins, f"at {r_ref_km:g} km (coordinate)")
    for a in ax_flat:
        a.set_ylabel(r"$\Delta m$ ($M_\odot$)")
    fig.tight_layout()
    return fig


def _mass_weighted_hist(ax, values: np.ndarray, masses: np.ndarray, bins: int, label: str, log: bool = False) -> None:
    finite = np.isfinite(values)
    if not finite.any():
        ax.set_title("(no finite values)")
        ax.set_xlabel(label)
        return
    v, m = values[finite], masses[finite]
    bin_edges = np.geomspace(v[v > 0].min(), v.max(), bins) if log else np.linspace(v.min(), v.max(), bins)
    ax.hist(v, bins=bin_edges, weights=m, histtype="step")
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(label)
    if finite.sum() < len(values):
        ax.set_title(f"{len(values) - finite.sum()} tracer(s) excluded (no finite value)", fontsize=8)


def main() -> None:
    args = parse_args()
    units = get_units()

    trajs = load_trajectories(args.tracer_dirs, args.glob_pattern, args.n_cpu, tuple(args.status))
    masses = tracer_masses(trajs)
    total_mass = masses.sum()
    print(f"{len(trajs)} tracers, total mass {total_mass:.6g} Msun.")

    for tr in trajs:
        add_spherical_fields(tr)

    panel_values = {}

    if 'Tmax' in args.panels:
        panel_values['Tmax'] = np.array([np.max(tr.data['T']) for tr in trajs]) * units.temperature_gk

    if {'Ye_ref', 's_ref', 'tau_ref'} & set(args.panels):
        ref = [_reference_values(tr, args.t_ref_gk, units.temperature_gk) for tr in trajs]
        ye_ref, s_ref, tau_ref = (np.array(x) for x in zip(*ref))
        n_missed = np.isnan(ye_ref).sum()
        if n_missed:
            print(f"{n_missed} of {len(trajs)} tracers never reached T_ref={args.t_ref_gk} GK "
                  f"(excluded from the *_ref panels).")
        if 'Ye_ref' in args.panels:
            panel_values['Ye_ref'] = ye_ref
        if 's_ref' in args.panels:
            panel_values['s_ref'] = s_ref
        if 'tau_ref' in args.panels:
            panel_values['tau_ref'] = tau_ref * units.time_ms

    if 'theta_final' in args.panels:
        panel_values['theta_final'] = np.array([tr.data['theta'][-1] for tr in trajs])
    if 'phi_final' in args.panels:
        panel_values['phi_final'] = np.array([tr.data['phi'][-1] % 360 for tr in trajs])
    if 'r_final' in args.panels:
        panel_values['r_final'] = np.array([tr.data['r'][-1] for tr in trajs]) * units.length_km

    v_final = None
    v_inf_geo = None
    v_inf_bernoulli = None
    if 'v_final' in args.panels:
        v_final = np.array([
            np.sqrt(tr.data['V_u_x'][-1]**2 + tr.data['V_u_y'][-1]**2 + tr.data['V_u_z'][-1]**2)
            for tr in trajs
        ])
        if 'u_t' in trajs[0].data:
            u_t_final = np.array([tr.data['u_t'][-1] for tr in trajs])
            v_inf_geo = asymptotic_velocity(-u_t_final)
            n_bound = np.sum(~np.isfinite(v_inf_geo))
            print(f"Geodesic criterion (-u_t>1): {n_bound} of {len(trajs)} tracers bound "
                  f"({100 * masses[~np.isfinite(v_inf_geo)].sum() / total_mass:.2f}% of mass) at final time.")
        else:
            print("'u_t' not found in tracer data; skipping geodesic-criterion asymptotic velocity.")

        if 'hu_t' in trajs[0].data:
            hu_t_final = np.array([tr.data['hu_t'][-1] for tr in trajs])
            v_inf_bernoulli = asymptotic_velocity(-hu_t_final)
            n_bound_bernoulli = np.sum(~np.isfinite(v_inf_bernoulli))
            print(f"Bernoulli criterion (-h*u_t>1): {n_bound_bernoulli} of {len(trajs)} tracers bound "
                  f"({100 * masses[~np.isfinite(v_inf_bernoulli)].sum() / total_mass:.2f}% of mass) at final time.")
        else:
            print("'hu_t' not found in tracer data; skipping Bernoulli-criterion asymptotic velocity.")

    panels = [p for p in args.panels if p != 'v_final' or v_final is not None]
    n_panels = len(panels)
    n_cols = min(4, n_panels) if n_panels else 1
    n_rows = -(-n_panels // n_cols) if n_panels else 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows), squeeze=False)
    ax_flat = ax.flatten()

    labels = {
        'Tmax': (r"$T_{\rm max}$ (GK)", False),
        'Ye_ref': (rf"$Y_e$ at {args.t_ref_gk:g} GK", False),
        's_ref': (rf"$s$ at {args.t_ref_gk:g} GK ($k_{{\rm B}}$)", False),
        'tau_ref': (rf"$\tau_{{\rm exp}}$ at {args.t_ref_gk:g} GK (ms)", True),
        'theta_final': (r"$\theta_{\rm final}$ (deg)", False),
        'phi_final': (r"$\phi_{\rm final}$ (deg)", False),
        'r_final': (r"$r_{\rm final}$ (km)", False),
    }

    for a, panel in zip(ax_flat, panels):
        if panel == 'v_final':
            _velocity_panel(a, v_final, v_inf_geo, v_inf_bernoulli,
                            masses, total_mass, args.bins, "final (coordinate)")
            continue
        label, log = labels[panel]
        _mass_weighted_hist(a, panel_values[panel], masses, args.bins, label, log)

    for a in ax_flat[:n_panels]:
        a.set_ylabel(r"$\Delta m$ ($M_\odot$)")
    for a in ax_flat[n_panels:]:
        a.set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output}")

    out = pl.Path(args.output)
    for r_ref_km in args.r_ref_km:
        r_fig = _radius_figure(trajs, masses, total_mass, r_ref_km, units, args.bins)
        r_out = out.with_name(f"{out.stem}_r{r_ref_km:g}km{out.suffix}")
        r_fig.savefig(r_out, dpi=args.dpi, bbox_inches="tight")
        plt.close(r_fig)
        print(f"Wrote {r_out}")


if __name__ == "__main__":
    main()
