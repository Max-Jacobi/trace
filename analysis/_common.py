"""
Shared helpers for the scripts in ``analysis/``: loading tracer output in
parallel, physical-unit conversion, and spherical/kinematic fields derived
from the raw Cartesian tracer data.
"""

import argparse
import os
import re
import pathlib as pl
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.trajectory import Trajectory

try:
    import tabulatedEOS.unit_system as us
    _HAVE_TABULATED_EOS = True
except ImportError:
    _HAVE_TABULATED_EOS = False

ALL_STATUSES = ("done", "active", "failed", "not started")
DEFAULT_STATUSES = ("done", "active")


@dataclass(frozen=True)
class Units:
    """Multiply raw (geometric-units) tracer data by these to get physical units."""
    length_km: float
    time_ms: float
    temperature_gk: float
    density_cgs: float


def get_units() -> Units:
    """
    Physical-unit conversion factors for GeometricSolar (G=c=M_sun=1) data,
    via the external ``tabulatedEOS`` package (a sibling project, not
    distributed with this repo -- see analysis/README.md).
    """
    if not _HAVE_TABULATED_EOS:
        raise ImportError(
            "tabulatedEOS is required for unit conversion (length -> km, time -> ms, "
            "temperature -> GK, density -> g/cm^3) but could not be imported. It's an "
            "external sibling package, not distributed with this repo -- put it on "
            "PYTHONPATH. See analysis/README.md."
        )
    return Units(
        length_km=us.GeometricSolar.LengthConversion(us.CGS) * 1e-5,
        time_ms=us.GeometricSolar.TimeConversion(us.CGS) * 1000,
        temperature_gk=us.GeometricSolar.TemperatureConversion(us.CGS) / 1e9,
        density_cgs=us.GeometricSolar.MassDensityConversion(us.CGS),
    )


def add_tracer_loading_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add the --tracer-dirs/--glob-pattern/--n-cpu/--status argument group, shared by every analysis script."""
    g = parser.add_argument_group("tracer loading")
    g.add_argument('--tracer-dirs', required=True, nargs='+',
                    help="One or more directories of tracer_*.dat files (run_pipeline.py's --output-dir). "
                         "Pass both the volume- and surface-seeded output directories together to analyze "
                         "the full combined population.")
    g.add_argument('--glob-pattern', default='tracer_*.dat',
                    help="Glob pattern used to find tracer files within each --tracer-dirs entry.")
    g.add_argument('--n-cpu', type=int, default=os.cpu_count() or 1,
                    help="Parallel workers for loading tracer files.")
    g.add_argument('--status', nargs='+', default=list(DEFAULT_STATUSES), choices=list(ALL_STATUSES),
                    help="Only keep tracers with one of these statuses (see each .dat file's header comment).")
    return g


def _try_load_one(file_path: str):
    """Load one tracer file, returning None (instead of raising) on a parse failure."""
    try:
        return Trajectory.from_ascii(file_path)
    except Exception as e:
        return (file_path, str(e))


def load_trajectories(
    tracer_dirs: list[str],
    glob_pattern: str = 'tracer_*.dat',
    n_cpu: int = 1,
    statuses: Optional[tuple] = DEFAULT_STATUSES,
    verbose: bool = True,
) -> list[Trajectory]:
    """
    Load every tracer file under `tracer_dirs` matching `glob_pattern`, in
    parallel. Files that fail to parse (e.g. truncated by an interrupted
    copy/download -- a cut-off last line is the usual symptom) are skipped
    with a printed warning rather than aborting the whole load.
    """
    files = []
    for d in tracer_dirs:
        files += sorted(pl.Path(d).glob(glob_pattern))
    file_paths = [str(f) for f in files]
    if not file_paths:
        raise FileNotFoundError(f"No files matching {glob_pattern!r} found in {tracer_dirs}.")

    with Pool(n_cpu) as pool:
        results = list(tqdm(
            pool.imap_unordered(_try_load_one, file_paths),
            total=len(file_paths), ncols=0, unit="tracers",
            desc="Loading tracers", disable=not verbose,
        ))

    trajs = [r for r in results if isinstance(r, Trajectory)]
    failures = [r for r in results if not isinstance(r, Trajectory)]
    if failures:
        print(f"WARNING: {len(failures)} of {len(file_paths)} tracer file(s) failed to parse "
              f"and were skipped (likely truncated by an interrupted copy):")
        for file_path, err in failures[:10]:
            print(f"  {file_path}: {err}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    if statuses is not None:
        trajs = [t for t in trajs if t.props.get('status') in statuses]

    if not trajs:
        raise ValueError(
            f"No tracers left after filtering {len(file_paths)} loaded files to status in {statuses}."
        )

    return trajs


def add_spherical_fields(traj: Trajectory) -> None:
    """Add 'r' (code units), 'theta' and 'phi' (degrees, phi unwrapped) to traj.data, in place."""
    x, y, z = traj.data['x'], traj.data['y'], traj.data['z']
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    phi[phi < 0] += 2 * np.pi
    phi = np.unwrap(phi) * 180 / np.pi
    theta = np.arccos(np.clip(z / r, -1, 1)) * 180 / np.pi
    traj.data['r'] = r
    traj.data['phi'] = phi
    traj.data['theta'] = theta


def tracer_masses(trajs: list[Trajectory]) -> np.ndarray:
    """
    Each tracer's represented mass, **sign included**.

    A surface-seeded tracer's mass is the flux integral over its cell, which is
    negative wherever material crosses the sphere inward. That sign is
    meaningful: it subtracts from the net crossing mass exactly as the surface
    element it stands for does. The mass-weighted `-mc` seeding makes this
    routine, since it samples inflowing cells in proportion to their flux like
    any other. Taking the magnitude here would silently count inflow as ejecta.

    `mass` is whatever the run's file handler decided a mass is -- plain
    rest mass, or the conserved `D`-based one under `--adm-mass`. Output
    written before the two were merged carries a separate `mass_D`; prefer it
    where it exists, so those files keep reading correctly.
    """
    return np.array([t.props.get('mass_D', t.props['mass']) for t in trajs])


def asymptotic_velocity(specific_energy: np.ndarray) -> np.ndarray:
    """
    Asymptotic (t -> infinity) velocity implied by a conserved specific
    energy at infinity: if ``W_inf`` (>= 1) is the tracer's Lorentz factor
    once it reaches flat spacetime with no residual internal energy, then
    ``v_inf = sqrt(1 - 1/W_inf**2)``.

    Two conventions for ``specific_energy`` (i.e. ``W_inf``) are in use
    here, differing in whether internal/thermal energy is assumed to fully
    convert to kinetic energy by the time the tracer reaches infinity:

    - Geodesic criterion: ``specific_energy = -u_t``. Treats the tracer as
      an exact geodesic (gravity only); appropriate once pressure forces
      are negligible.
    - Bernoulli criterion: ``specific_energy = -h*u_t`` (``h`` = specific
      enthalpy). Also lets thermal/internal energy unbind or accelerate a
      tracer that the pure geodesic criterion would call bound -- usually
      gives a larger unbound fraction and higher asymptotic velocities.

    Returns NaN wherever ``specific_energy <= 1`` (bound under whichever
    criterion was passed in: the tracer's energy doesn't exceed its rest
    mass, so it has no well-defined escape velocity).
    """
    w_inf = np.asarray(specific_energy)
    v_inf = np.full_like(w_inf, np.nan, dtype=np.float64)
    unbound = w_inf > 1
    v_inf[unbound] = np.sqrt(1 - 1 / w_inf[unbound]**2)
    return v_inf


# ---------------------------------------------------------------------------
# Panel / filter spec grammar, shared by histograms.py and plot_trajectories.py
#
#   FIELD          value at the tracer's last recorded point (also FIELD@final)
#   FIELD@5GK      value where the tracer *last* crosses T = 5 GK
#   FIELD@500km    value where the tracer last crosses r = 500 km
#   FIELD@t10ms    value where the tracer last crosses time = 10 ms
#   Tmax           peak temperature
#
# The crossed variable follows from the unit (GK -> T, km -> r, ms -> time)
# unless given explicitly (Ye@T5GK).
# ---------------------------------------------------------------------------
AT_RE = re.compile(r'^([A-Za-z]*?)([\d.eE+-]+)(GK|km|ms)$')
UNIT_VAR = {'GK': 'T', 'km': 'r', 'ms': 'time'}
PRED_RE = re.compile(r'^(.+?)(<=|>=|==|!=|<|>)([\d.eE+-]+)([A-Za-z/0-9^]*)$')
_OPS = {'<': np.less, '<=': np.less_equal, '>': np.greater, '>=': np.greater_equal,
        '==': np.equal, '!=': np.not_equal}


def parse_spec(spec: str) -> tuple:
    """'Ye@T5GK' -> ('Ye', 'T', 5.0, 'GK'); 'theta' / 'r@final' -> ('theta', None, None, None)."""
    field, _, at = spec.partition('@')
    if not at or at == 'final':
        return field, None, None, None
    m = AT_RE.match(at)
    if not m:
        raise SystemExit(f"Bad spec {spec!r}: expected FIELD, FIELD@final, FIELD@5GK, FIELD@500km or FIELD@t10ms.")
    var, level, unit = m.groups()
    return field, var or UNIT_VAR[unit], float(level), unit


def unit_factors(units: Units) -> dict:
    """Multiply code-unit data by these to get the unit named by the label."""
    return {'GK': units.temperature_gk, 'km': units.length_km, 'ms': units.time_ms, 'g/cm3': units.density_cgs}


def field_units(units: Units) -> dict:
    """Plotted unit per field: (factor to multiply code-unit data by, unit label)."""
    return {
        'T': (units.temperature_gk, 'GK'), 'Tmax': (units.temperature_gk, 'GK'),
        'r': (units.length_km, 'km'), 'time': (units.time_ms, 'ms'), 'tau': (units.time_ms, 'ms'),
        'rho': (units.density_cgs, 'g/cm3'), 'entr': (1.0, 'k_B/baryon'), 'pres': (1.0, 'MeV/fm^3'),
    }


def derived_fields(traj: Trajectory, fields=(), eos=None) -> dict:
    """
    Per-tracer time series in code units: the raw data, numeric header props
    broadcast along time (mass, dV, ...), the derived aliases Ye, v, v_r, tau,
    vinf_geo, vinf_bern, and the requested `fields` looked up in `eos`
    (a src.eos.PyCompOSEEOS) along (rho, T, Ye) when they are not recorded.

    With an EOS, hu_t is always recomputed as h u_t / h_inf with the table's
    global minimum enthalpy; the pipeline's recorded hu_t uses the
    Ye-dependent minimum instead, a different Bernoulli criterion.
    """
    d = dict(traj.data)
    for k, v in traj.props.items():
        if isinstance(v, (int, float)) and k not in d:
            d[k] = np.full_like(d['time'], v, dtype=float)
    d['Ye'] = d['r_0']
    d['v'] = np.sqrt(d['V_u_x']**2 + d['V_u_y']**2 + d['V_u_z']**2)
    if 'r' in d:
        d['v_r'] = (d['x'] * d['V_u_x'] + d['y'] * d['V_u_y'] + d['z'] * d['V_u_z']) / d['r']
    if d['time'].size >= 2:
        d['tau'] = d['rho'] / (np.abs(np.gradient(d['rho'], d['time'])) + 1e-300)
    if eos is not None:
        if 'u_t' in d:
            d['hu_t'] = eos('enth', d['rho'], d['T'], d['Ye']) * d['u_t'] / eos.h_inf
        for f in fields:
            if f not in d and f in eos.keys():
                d[f] = eos(f, d['rho'], d['T'], d['Ye'])
    if 'u_t' in d:
        d['vinf_geo'] = asymptotic_velocity(-d['u_t'])
    if 'hu_t' in d:
        d['vinf_bern'] = asymptotic_velocity(-d['hu_t'])
    return d


def last_crossing_time(x: np.ndarray, t: np.ndarray, level: float) -> float:
    """Time of the last sign change of x - level, linearly interpolated; NaN if none."""
    s = np.sign(x - level)
    idx = np.flatnonzero((s[:-1] != s[1:]) & (x[:-1] != x[1:]))   # a point exactly at the level counts
    if idx.size == 0:
        return np.nan
    i = idx[-1]
    f = (level - x[i]) / (x[i + 1] - x[i])
    return float(t[i] + f * (t[i + 1] - t[i]))


def sample(d: dict, spec: tuple, ufac: dict) -> float:
    """Evaluate a parsed spec on one tracer's derived fields; code units, NaN if unavailable."""
    field, var, level, unit = spec
    if field == 'Tmax':
        return float(np.max(d['T']))
    if field not in d:
        return np.nan
    if var is None:
        return float(d[field][-1])
    t_c = last_crossing_time(d[var], d['time'], level / ufac[unit])
    return np.nan if np.isnan(t_c) else float(np.interp(t_c, d['time'], d[field]))


def select_tracers(data: list, predicates: list, units: Units) -> np.ndarray:
    """
    Boolean mask of tracers satisfying every predicate like 'Tmax>5GK',
    'Ye@5GK<0.4' or 'r@final>300km'. The threshold is in the named unit,
    or in the field's plotted unit when none is given. NaN never passes.
    """
    ufac, funits = unit_factors(units), field_units(units)
    keep = np.ones(len(data), dtype=bool)
    for pred in predicates:
        m = PRED_RE.match(pred)
        if not m:
            raise SystemExit(f"Bad --select {pred!r}: expected SPEC<op>VALUE[unit], e.g. Tmax>5GK.")
        spec, op, value, unit = m.groups()
        spec = parse_spec(spec)
        if spec[0] != 'Tmax' and not any(spec[0] in d for d in data):
            raise SystemExit(f"--select {pred!r}: unknown field {spec[0]!r}. Available: {sorted(data[0])}")
        if unit and unit not in ufac:
            raise SystemExit(f"--select {pred!r}: unknown unit {unit!r}, use one of {list(ufac)}.")
        fac = ufac[unit] if unit else funits.get(spec[0], (1.0, ''))[0]
        vals = np.array([sample(d, spec, ufac) for d in data])
        keep &= _OPS[op](vals, float(value) / fac)
    return keep


AFTER_SIDE = {'T': -1, 'r': +1, 'time': +1}   # which side of the level a tracer is on "after" it: cooled, moved out, later


def trim(d: dict, after: str | None, before: str | None, units: Units) -> dict | None:
    """
    Cut a tracer's derived fields to the window after the last crossing of
    `after` ('T@10GK') and/or before the last crossing of `before`
    ('r@500km'). "After T@10GK" means the cooled side (T < 10 GK), "after
    r@500km" the outer side; "before" is the opposite side.

    A tracer that never crosses is kept whole when it is entirely on the
    wanted side (never got hot at all), and dropped (None) when its end
    point (--after) or start point (--before) is on the wrong side.
    The cut point itself is inserted with interpolated values.
    """
    ufac = unit_factors(units)
    t0, t1 = -np.inf, np.inf
    pin = {}   # index -> (var, level): the crossed variable is set to the level exactly, no rounding
    for bound, is_after in ((after, True), (before, False)):
        if bound is None:
            continue
        var, _, level, unit = parse_spec(bound)
        if var not in AFTER_SIDE:
            raise SystemExit(f"--after/--before {bound!r}: can only cut at {list(AFTER_SIDE)}.")
        level_code = level / ufac[unit]
        side = AFTER_SIDE[var] * (1 if is_after else -1)
        if np.sign(d[var][-1 if is_after else 0] - level_code) != side:
            return None
        t_c = last_crossing_time(d[var], d['time'], level_code)
        if np.isnan(t_c):
            continue
        if is_after:
            t0 = t_c
            pin[0] = (var, level_code)
        else:
            t1 = t_c
            pin[-1] = (var, level_code)
    t = d['time']
    keep = (t >= t0) & (t <= t1)
    t_new = np.concatenate([[t0] if np.isfinite(t0) else [], t[keep], [t1] if np.isfinite(t1) else []])
    out = {k: np.interp(t_new, t, v) if isinstance(v, np.ndarray) and v.shape == t.shape else v
           for k, v in d.items()}
    for i, (var, level_code) in pin.items():
        out[var][i] = level_code
    return out


def add_filter_args(parser: argparse.ArgumentParser, trim: bool = True) -> argparse._ArgumentGroup:
    """Add --select (and, with trim=True, --after/--before), the tracer filtering shared by the analysis scripts."""
    g = parser.add_argument_group("tracer filtering")
    g.add_argument('--select', nargs='*', default=[],
                    help="Keep only tracers satisfying every predicate SPEC<op>VALUE[unit], e.g. "
                         "'Tmax>5GK' 'Ye@5GK<0.4' 'r@final>300km' 'mass>0'. A missing value never passes.")
    if trim:
        g.add_argument('--after', nargs='?', default=None, const=None, metavar='VAR@LEVEL',
                        help="Keep only the part of each history after its last crossing of VAR@LEVEL, e.g. "
                             "T@10GK (after the last drop below 10 GK). Tracers never reaching the level are "
                             "kept whole; tracers ending on the wrong side (still hot) are dropped. "
                             "A bare --after switches a script default off.")
        g.add_argument('--before', nargs='?', default=None, const=None, metavar='VAR@LEVEL',
                        help="Keep only the part of each history before its last crossing of VAR@LEVEL, "
                             "e.g. r@500km. Same never-crossing rules, judged at the start point.")
    return g


def apply_filters(data: list, masses: np.ndarray, args: argparse.Namespace, units: Units) -> tuple:
    """Apply --select, then --after/--before; returns (data, masses) of the surviving tracers."""
    n0 = len(data)
    keep = select_tracers(data, args.select, units)
    data, masses = [d for d, k in zip(data, keep) if k], masses[keep]
    if args.select:
        print(f"--select {' '.join(args.select)}: kept {len(data)} of {n0} tracers.")
    after, before = getattr(args, 'after', None), getattr(args, 'before', None)
    if after or before:
        trimmed = [trim(d, after, before, units) for d in data]
        ok = [d is not None for d in trimmed]
        data, masses = [d for d in trimmed if d is not None], masses[ok]
        print(f"--after {after} --before {before}: kept {sum(ok)} of {len(ok)} tracers.")
    if not data:
        raise SystemExit("No tracers left after filtering.")
    return data, masses
