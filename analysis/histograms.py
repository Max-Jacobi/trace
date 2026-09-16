#!/usr/bin/env python3
"""
Mass-weighted histograms of per-tracer quantities.

Each --panels entry is one histogram:

    FIELD            value at the tracer's last recorded point
    FIELD@5GK        value where the tracer *last* crosses T = 5 GK
    FIELD@500km      value where the tracer *last* crosses r = 500 km
    FIELD@t10ms      value where the tracer last crosses time = 10 ms
    Tmax             peak temperature (GK)
    A,B,C            comma-joined specs overlaid on one axis (Ye@8GK,Ye@5GK,Ye@3GK)

The variable being crossed is inferred from the unit (GK -> T, km -> r,
ms -> time) and can be given explicitly (Ye@T5GK). FIELD is any key of the
tracer data plus the derived fields Ye, r, theta, phi, v (coordinate |v|),
tau (rho/|drho/dt|), vinf_geo (-u_t) and vinf_bern (-h u_t / h_inf, with
h_inf the global minimum enthalpy of the EOS table). With --eos (a
PyCompOSE HDF5 table) any EOS quantity evaluated along the tracer's
(rho, T, Ye) history is a FIELD too: entr, enth, pres, eps, cs2, Y[...],
... (see src/eos.py), and vinf_bern is computed from it. Without --eos,
vinf_bern falls back to the recorded hu_t, which the pipeline normalises
by the Ye-dependent minimum enthalpy instead (a different criterion).

Every histogram is weighted by each tracer's represented mass, so the
y-axis is ejecta mass per bin (M_sun), not tracer count.

Example
-------
    python analysis/histograms.py --tracer-dirs data/tracers_out \\
        --panels Tmax,T@400km Ye@8GK,Ye@5GK,Ye@3GK tau@5GK theta vinf_geo --output histograms.png
    python analysis/histograms.py --tracer-dirs data/tracers_out --eos SFHo.h5 \\
        --panels Ye@5GK entr@5GK tau@5GK vinf_bern
"""

import argparse
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis._common import (
    add_tracer_loading_args, load_trajectories, add_spherical_fields,
    tracer_masses, get_units, asymptotic_velocity,
)
from src.eos import PyCompOSEEOS

DEFAULT_PANELS = ['Tmax', 'Ye@5GK', 'tau@5GK', 'theta', 'phi', 'r', 'v']
SPEC_RE = re.compile(r'^([^@]+?)(?:@([A-Za-z]*?)([\d.eE+-]+)(GK|km|ms))?$')
UNIT_VAR = {'GK': 'T', 'km': 'r', 'ms': 'time'}
LOG_FIELDS = {'rho', 'tau', 'r', 'time', 'pres'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mass-weighted histograms of per-tracer quantities.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    add_tracer_loading_args(parser)
    parser.add_argument('--panels', nargs='+', default=DEFAULT_PANELS,
                        help="Panel specs, see below (default: %(default)s).")
    parser.add_argument('--eos', help="PyCompOSE HDF5 EOS table; enables EOS-derived FIELDs (entr, enth, ...).")
    parser.add_argument('--bins', type=int, default=40, help="Histogram bins per panel.")
    parser.add_argument('--output', default='histograms.png', help="Output image path.")
    parser.add_argument('--dpi', type=int, default=300, help="Output image DPI.")
    return parser.parse_args()


def parse_spec(spec: str) -> tuple:
    """'Ye@T5GK' -> ('Ye', 'T', 5.0, 'GK'); 'theta' -> ('theta', None, None, None)."""
    m = SPEC_RE.match(spec)
    if not m:
        raise SystemExit(f"Bad panel spec {spec!r}: expected FIELD, FIELD@5GK, FIELD@500km or FIELD@t10ms.")
    field, var, level, unit = m.groups()
    if unit:
        var = var or UNIT_VAR[unit]
        level = float(level)
    return field, var, level, unit


def panel_specs(panel: str) -> list:
    """'Tmax,T@400km' -> [parse_spec('Tmax'), parse_spec('T@400km')]: comma-joined specs share one axis."""
    return [parse_spec(sp) for sp in panel.split(',')]


def field_units(units) -> dict:
    """Multiply code-unit data by these to get the plotted unit; (factor, unit label)."""
    return {
        'T': (units.temperature_gk, 'GK'), 'r': (units.length_km, 'km'),
        'time': (units.time_ms, 'ms'), 'tau': (units.time_ms, 'ms'),
        'rho': (units.density_cgs, 'g/cm^3'), 'entr': (1.0, 'k_B/baryon'), 'pres': (1.0, 'MeV/fm^3'),
    }


def derived_fields(traj, fields, eos=None) -> dict:
    """Per-tracer time series: raw data, the derived aliases and the requested EOS fields, all in code units."""
    d = dict(traj.data)
    d['Ye'] = d['r_0']
    if eos is not None:
        if 'u_t' in d:   # recorded hu_t is h/h_ref(Ye) u_t; we want the global h_inf
            d['hu_t'] = eos('enth', d['rho'], d['T'], d['Ye']) * d['u_t'] / eos.h_inf
        for f in fields:
            if f not in d and f in eos.keys():
                d[f] = eos(f, d['rho'], d['T'], d['Ye'])
    d['v'] = np.sqrt(d['V_u_x']**2 + d['V_u_y']**2 + d['V_u_z']**2)
    if d['time'].size >= 2:
        d['tau'] = d['rho'] / (np.abs(np.gradient(d['rho'], d['time'])) + 1e-300)
    if 'u_t' in d:
        d['vinf_geo'] = asymptotic_velocity(-d['u_t'])
    if 'hu_t' in d:
        d['vinf_bern'] = asymptotic_velocity(-d['hu_t'])
    return d


def last_crossing_time(x: np.ndarray, t: np.ndarray, level: float) -> float:
    """Time of the last sign change of x - level, linearly interpolated; NaN if none."""
    s = np.sign(x - level)
    idx = np.flatnonzero((s[:-1] != s[1:]) & (s[:-1] != 0))
    if idx.size == 0:
        return np.nan
    i = idx[-1]
    f = (level - x[i]) / (x[i + 1] - x[i])
    return float(t[i] + f * (t[i + 1] - t[i]))


def sample(d: dict, field: str, var, level_code) -> float:
    if field == 'Tmax':
        return float(np.max(d['T']))
    if field not in d:
        return np.nan
    if var is None:
        return float(d[field][-1])
    t_c = last_crossing_time(d[var], d['time'], level_code)
    return np.nan if np.isnan(t_c) else float(np.interp(t_c, d['time'], d[field]))


def mass_weighted_hist(ax, curves, masses, bins, xlabel, log) -> None:
    """Overlay one step histogram per (values, label) in `curves` on shared bin edges."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Delta m$ ($M_\odot$)")
    allv = np.concatenate([v[np.isfinite(v) & ((v > 0) if log else True)] for v, _ in curves])
    if allv.size == 0:
        ax.set_title("(no finite values)")
        return
    edges = (np.geomspace if log else np.linspace)(allv.min(), allv.max(), bins)
    if log:
        ax.set_xscale("log")
    excluded = []
    for values, label in curves:
        ok = np.isfinite(values) & ((values > 0) if log else True)
        ax.hist(values[ok], bins=edges, weights=masses[ok], histtype="step", label=label)
        if not ok.all():
            excluded.append(f"{label}: {(~ok).sum()} excl. ({100 * masses[~ok].sum() / masses.sum():.1f}% mass)")
    if len(curves) > 1:
        ax.legend(fontsize=8)
    if excluded:
        ax.set_title("\n".join(excluded), fontsize=7)


def main() -> None:
    args = parse_args()
    units = get_units()
    funits = field_units(units)
    panels = args.panels
    specs = [sp for p in panels for sp in panel_specs(p)]

    trajs = load_trajectories(args.tracer_dirs, args.glob_pattern, args.n_cpu, tuple(args.status))
    masses = tracer_masses(trajs)
    print(f"{len(trajs)} tracers, total mass {masses.sum():.6g} Msun.")
    for tr in trajs:
        add_spherical_fields(tr)
    eos = PyCompOSEEOS(args.eos) if args.eos else None
    if eos is None and 'vinf_bern' in [f for f, *_ in specs]:
        print("Note: without --eos, vinf_bern uses the recorded hu_t (h / h_ref(Ye) u_t), "
              "not the global-h_inf Bernoulli criterion.")
    data = [derived_fields(tr, [f for f, *_ in specs], eos) for tr in trajs]

    for field, *_ in specs:
        if field != 'Tmax' and not any(field in d for d in data):
            raise SystemExit(f"Unknown field {field!r}. Available: {sorted(data[0])}"
                             + ("" if eos else " (EOS fields need --eos)."))

    n = len(panels)
    n_cols = min(4, n)
    n_rows = -(-n // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows), squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        curves = []
        for spec, (field, var, level, unit) in zip(panel.split(','), panel_specs(panel)):
            level_code = None if unit is None else level / funits[var][0]
            fac, _ = funits.get('T' if field == 'Tmax' else field, (1.0, ''))
            values = np.array([sample(d, field, var, level_code) for d in data]) * fac
            if field == 'phi':
                values %= 360
            curves.append((values, spec))
            n_missing = np.sum(~np.isfinite(values))
            if n_missing:
                print(f"{spec}: {n_missing} of {len(trajs)} tracers have no value (excluded).")
        field = panel_specs(panel)[0][0]
        ulabel = funits.get('T' if field == 'Tmax' else field, (1.0, ''))[1]
        xlabel = (', '.join(panel.split(',')) if len(curves) > 1 else curves[0][1]) + (f" ({ulabel})" if ulabel else '')
        mass_weighted_hist(ax, curves, masses, args.bins, xlabel, field in LOG_FIELDS)
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
