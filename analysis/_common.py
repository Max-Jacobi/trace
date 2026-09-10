"""
Shared helpers for the scripts in ``analysis/``: loading tracer output in
parallel, physical-unit conversion, and spherical/kinematic fields derived
from the raw Cartesian tracer data.
"""

import argparse
import os
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

    Prefers `mass_D` (the conserved rest mass) when the seeding recorded one.
    """
    return np.array([t.props.get('mass_D', t.props['mass']) for t in trajs])


def value_at_reference_temperature(traj: Trajectory, key: str, t_ref_gk: float, Tfac: float) -> float:
    """
    Interpolate `traj.data[key]` to the time at which the tracer's temperature
    first crosses down through `t_ref_gk` (GK), walking forward in time from
    the trajectory's hottest point.

    Assumes temperature is not perfectly monotonic in general but is, on
    average, decreasing as the tracer's stored history progresses forward in
    time (the usual case for expanding ejecta); returns NaN if the tracer
    never reaches `t_ref_gk` (e.g. it was already cooler than that at the
    earliest recorded time).
    """
    T = traj.data['T'] * Tfac
    i_hot = np.argmax(T)
    T_after = T[i_hot:]
    v_after = traj.data[key][i_hot:]
    if T_after.min() > t_ref_gk or T_after.max() < t_ref_gk:
        return np.nan
    # np.interp needs increasing x; T_after is decreasing (from the hottest point onward).
    return float(np.interp(t_ref_gk, T_after[::-1], v_after[::-1]))


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
