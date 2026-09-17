#!/usr/bin/env python3
"""
Plot a random sample of tracer histories vs time.

Loads tracer output (from run_pipeline.py), picks a random subsample, and
plots one line per tracer for each requested field, colour-coded by the
tracer's average value of --color-by. Useful as a first sanity look at a
new run: are trajectories smooth, do fields evolve as expected, are there
obvious outliers?

By default only tracers that come inside 300 km at some point are shown
(--select 'r@min<300km', dropping material that never was near the
remnant, e.g. atmosphere seeded far out) and each history
starts at its last drop below 10 GK (--after T@10GK; tracers that never
get that hot are kept whole, tracers still above 10 GK at the end are
dropped). Pass --select or --after with no value to switch either off.
The spec grammar is the one histograms.py uses:
FIELD[@final|@5GK|@500km|@t10ms], Tmax.

Example
-------
    python analysis/plot_trajectories.py \\
        --tracer-dirs data/tracers_out data/tracers_out_surf \\
        --n-sample 200 --select 'Tmax>5GK' 'mass>0' --after T@10GK --before r@1000km
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from analysis._common import (
    add_tracer_loading_args, add_filter_args, apply_filters, load_trajectories, add_spherical_fields,
    get_units, tracer_masses, derived_fields,
)

FIELD_REGISTRY = {   # label, function of (derived-field dict, units), log y-axis
    'r':      (r"r (km)",                  lambda d, u: d['r'] * u.length_km,      False),
    'theta':  (r"theta (deg)",             lambda d, u: d['theta'],                False),
    'phi':    (r"phi (deg)",                lambda d, u: d['phi'],                  False),
    'rho_r3': (r"$\rho r^3$ ($M_\odot$)",  lambda d, u: d['rho'] * d['r']**3,      True),
    'Ye':     (r"$Y_e$",                    lambda d, u: d['Ye'],                   False),
    'T':      (r"T (GK)",                   lambda d, u: d['T'] * u.temperature_gk, False),
    'rho':    (r"$\rho$ (g/cm$^3$)",       lambda d, u: d['rho'] * u.density_cgs,  True),
    'v':      (r"$|v|$ (c)",                lambda d, u: d['v'],                    False),
    'v_r':    (r"$v_r$ (c)",                lambda d, u: d['v_r'],                  False),
    'u_t':    (r"$u_t$",                    lambda d, u: d['u_t'],                  False),
    'tau':    (r"$\tau$ (ms)",              lambda d, u: d['tau'] * u.time_ms,      True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a random sample of tracer histories vs time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_tracer_loading_args(parser)
    g = add_filter_args(parser)
    g.set_defaults(select=['r@min<300km'], after='T@10GK')
    parser.add_argument('--n-sample', type=int, default=200,
                         help="Number of tracers to randomly sample and plot.")
    parser.add_argument('--seed', type=int, default=None,
                         help="RNG seed for the random sample (default: nondeterministic).")
    parser.add_argument('--fields', nargs='+', default=['r', 'T', 'rho', 'Ye', 'v_r', 'u_t'],
                         choices=list(FIELD_REGISTRY), help="Fields to plot, one panel each.")
    parser.add_argument('--color-by', default='Ye', choices=list(FIELD_REGISTRY),
                         help="Colour each tracer's lines by its average value of this field.")
    parser.add_argument('--cmap', default='jet_r', help="Matplotlib colormap for --color-by.")
    parser.add_argument('--color-min', type=float, default=None, help="Colour scale lower bound (default: data min).")
    parser.add_argument('--color-max', type=float, default=None, help="Colour scale upper bound (default: data max).")
    parser.add_argument('--lw', type=float, default=0.3, help="Line width.")
    parser.add_argument('--alpha', type=float, default=0.5, help="Line alpha.")
    parser.add_argument('--output', default='trajectories.png', help="Output image path.")
    parser.add_argument('--dpi', type=int, default=300, help="Output image DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    units = get_units()

    trajs = load_trajectories(args.tracer_dirs, args.glob_pattern, args.n_cpu, tuple(args.status))
    for tr in trajs:
        add_spherical_fields(tr)
    data, _ = apply_filters([derived_fields(tr) for tr in trajs], tracer_masses(trajs), args, units)

    rng = np.random.default_rng(args.seed)
    n_sample = min(args.n_sample, len(data))
    sample = [data[i] for i in rng.choice(len(data), size=n_sample, replace=False)]

    color_label, color_fn, _ = FIELD_REGISTRY[args.color_by]
    color_vals = np.array([np.average(color_fn(d, units)) for d in sample])
    vmin = args.color_min if args.color_min is not None else float(color_vals.min())
    vmax = args.color_max if args.color_max is not None else float(color_vals.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(args.cmap)

    n_fields = len(args.fields)
    n_cols = min(3, n_fields)
    n_rows = -(-n_fields // n_cols)  # ceil division
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.3 * n_rows), squeeze=False,
                           layout="constrained")
    ax_flat = ax.flatten()

    for d, cval in zip(sample, color_vals):
        t = d['time'] * units.time_ms
        color = cmap(norm(cval))
        kw = dict(lw=args.lw, alpha=args.alpha, c=color)
        for a, field in zip(ax_flat, args.fields):
            _, fn, log = FIELD_REGISTRY[field]
            a.plot(t, fn(d, units), **kw)

    for a, field in zip(ax_flat, args.fields):
        label, _, log = FIELD_REGISTRY[field]
        a.set_xlabel("time (ms)")
        a.set_ylabel(label)
        if log:
            a.set_yscale("log")
        if field == 'v_r':
            a.axhline(0, color='k', lw=0.5, ls='--')
    for a in ax_flat[n_fields:]:
        a.set_visible(False)

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label=color_label, ax=ax_flat[:n_fields])
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output} ({n_sample} of {len(trajs)} tracers)")


if __name__ == "__main__":
    main()
