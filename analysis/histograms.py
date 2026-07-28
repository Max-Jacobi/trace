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
    if T_after.min() > t_ref_gk or T_after.max() < t_ref_gk:
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
            finite_v = np.isfinite(v_final)
            mass_frac = 100 * masses[finite_v].sum() / total_mass
            a.hist(v_final[finite_v], bins=args.bins, weights=masses[finite_v],
                   histtype="step", label=f"final (coordinate), {mass_frac:.1f}% of mass")
            for v_inf, style_label in ((v_inf_geo, "asymptotic (geodesic)"),
                                        (v_inf_bernoulli, "asymptotic (Bernoulli)")):
                if v_inf is None:
                    continue
                finite_vinf = np.isfinite(v_inf)
                if finite_vinf.any():
                    mass_frac = 100 * masses[finite_vinf].sum() / total_mass
                    a.hist(v_inf[finite_vinf], bins=args.bins, weights=masses[finite_vinf],
                           histtype="step", label=f"{style_label}, {mass_frac:.1f}% of mass")
            a.set_xlabel(r"$|v|$ (c)")
            a.legend(fontsize=7)
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


if __name__ == "__main__":
    main()
