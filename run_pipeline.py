#!/usr/bin/env python3
"""
General-purpose command-line launcher for the tracer-integration pipeline.

Reads simulation snapshots in any of the formats listed in FORMATS below,
seeds tracers either throughout a spherical volume or on a spherical
surface, integrates them through the velocity field between two times, and
writes one ASCII file per tracer.

Examples
--------
Seed tracers through a spherical shell and integrate backward in time,
from transformed GR-Athena++ surface files (see transform_files.py):

    python run_pipeline.py --data-dir data/transformed --output-dir data/out \\
        --start-t 11600 --end-t 0 \\
        volume --r-min 300 --r-max 1000 --n-r 30 --n-th 15 --n-ph 30

The same, from AthenaK spherical-grid vtk dumps:

    python run_pipeline.py --format athenak \\
        --data-dir data/vtk --output-dir data/out \\
        --start-t 6120 --end-t 6080 \\
        volume --r-min 300 --r-max 1000 --n-r 30 --n-th 15 --n-ph 30

Seed tracers on a spherical surface, sampling every 5th snapshot:

    python run_pipeline.py --data-dir data/transformed --output-dir data/out \\
        --start-t 11600 --end-t 0 \\
        surface --r-surf 300 --n-th 15 --n-ph 30 --every-n-files 5
"""

import argparse
import os
import sys

import numpy as np

from src.interpolators import (
    PchipInterpolator3D, RegularInterpolator3D, MeshblockPchipInterpolator,
)
from src.integrators import ExplicitTrapezoid, ImplicitTrapezoid, RK4
from src.athenak import AthenaKFileHandler
from src.athdf_spherical import SphericalAthdfFileHandler
from src.reduced_surface import ReducedSurfaceFileHandler
from src.seeds import spherical_by_volume, spherical_surface_by_area

DEFAULT_VEL_KEYS = ('V_u_x', 'V_u_y', 'V_u_z')

# Per-format defaults for the options whose sensible value depends on the
# data source.  Anything given explicitly on the command line wins.
FORMATS = {
    'reduced_surface': {
        'handler': ReducedSurfaceFileHandler,
        'file_pattern': '*.hdf5',
        # GR-Athena++ surface output is geometrically spaced in radius.
        'rad_transform': 'log',
        'keys': (
            'V_u_x', 'V_u_y', 'V_u_z',
            'T', 'hu_t',
            'u_t', 'rho', 'r_0',
            'F_nue', 'F_anue', 'F_nux',
            'eps_nue', 'eps_anue', 'eps_nux',
        ),
    },
    'athenak': {
        'handler': AthenaKFileHandler,
        'file_pattern': '*.vtk',
        # AthenaK's spherical grid is geometrically (log) spaced in radius,
        # from rmin to rmax -- see docs/formats/athenak.md. Older dumps used
        # a linear grid; pass --rad-transform none for those explicitly.
        'rad_transform': 'log',
        # No entropy and no hu_t in AthenaK's dumps. AthenaK evolves a 4th
        # (anux) neutrino species GR-Athena++ doesn't have; the default
        # --heavy-neutrinos=sum folds it into "nux", so these are the same
        # 3-species keys as the reduced_surface format -- pass
        # --heavy-neutrinos=separate and add F_anux/eps_anux to --keys to
        # keep all 4 species distinct instead. See src/athenak.py's
        # FIELD_MAP/_build_key_specs for how these map onto AthenaK's own
        # scalar names.
        'keys': (
            'V_u_x', 'V_u_y', 'V_u_z',
            'T', 'u_t', 'rho', 'r_0',
            'F_nue', 'F_anue', 'F_nux',
            'eps_nue', 'eps_anue', 'eps_nux',
        ),
    },
    'athdf_spherical': {
        'handler': SphericalAthdfFileHandler,
        'file_pattern': '*.athdf',
        # Interpolation happens on each meshblock's true node coordinates,
        # so no radial transform applies (the handler rejects one).
        'rad_transform': 'none',
        # The meshblock memory layout needs the meshblock interpolator; the
        # separable-grid interpolators cannot be constructed against it.
        'interpolator': 'meshblock',
        # Which output series (out1, out2, ...) carries each variable is
        # autodetected per file from its VariableNames metadata; files with
        # none of the requested keys (e.g. cons-only dumps) are skipped.
        # See src/athdf_spherical.py's FIELD_MAP for the canonical -> raw name map.
        'keys': (
            'V_u_x', 'V_u_y', 'V_u_z',
            'T', 'u_t', 'rho', 'r_0',
        ),
    },
}

INTEGRATOR_N_SNAPSHOTS = {'expl_trap': 2, 'impl_trap': 2, 'rk4': 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the tracer-integration pipeline over transformed surface snapshots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    io_group = parser.add_argument_group("input/output")
    io_group.add_argument('--format', choices=sorted(FORMATS), default='reduced_surface',
                           help="Snapshot data format. 'reduced_surface': transformed "
                                "GR-Athena++ surface hdf5 (see transform_files.py). "
                                "'athenak': AthenaK spherical-grid vtk. 'athdf_spherical': Athena++ "
                                "spherical meshblock hdf5 (with ghost zones). Sets the defaults "
                                "for --keys, --file-pattern, --rad-transform and --interpolator.")
    io_group.add_argument('--data-dir', required=True,
                           help="Directory of snapshot files.")
    io_group.add_argument('--output-dir', default='tracer_output',
                           help="Directory to write per-tracer ASCII files to.")
    io_group.add_argument('--file-pattern', default=None,
                           help="Glob pattern used to find snapshot files in --data-dir "
                                "(default: per --format).")
    io_group.add_argument('--start-t', required=True, type=float,
                           help="Time to seed and start integrating tracers from.")
    io_group.add_argument('--end-t', required=True, type=float,
                           help="Time to integrate tracers to (can be < start-t for backward integration).")

    field_group = parser.add_argument_group("fields")
    field_group.add_argument('--keys', nargs='+', default=None,
                              help="Field keys to load and carry along each tracer "
                                   "(default: per --format).")
    field_group.add_argument('--vel-keys', nargs='+', default=list(DEFAULT_VEL_KEYS),
                              help="Field keys (subset of --keys) used as the velocity vector.")
    field_group.add_argument('--density-key', default='rho',
                              help="Field key used as mass density for seed-mass integration.")
    field_group.add_argument('--heavy-neutrinos', choices=['sum', 'drop', 'separate'], default='sum',
                              help="--format=athenak only: how to handle AthenaK's 4th (anux) "
                                   "neutrino species, which GR-Athena++ doesn't have. 'sum' folds "
                                   "nux+anux into one GRA-style nux (number-flux-weighted average "
                                   "energy); 'drop' omits nux/anux entirely; 'separate' keeps all "
                                   "4 species distinct. Ignored for --format=reduced_surface.")
    field_group.add_argument('--bh-mass', type=float, default=1.0,
                              help="--format=athdf_spherical only: black hole mass in code units, used "
                                   "in the Schwarzschild lapse/metric factors of the velocity "
                                   "transform (not stored in the athdf metadata).")
    field_group.add_argument('--positive-keys', nargs='+', default=[],
                              help="Field keys (subset of --keys) that cannot physically be "
                                   "negative. Samples below zero are floored at 0 as each "
                                   "snapshot is loaded, before any interpolation or seed-mass "
                                   "integration. Use this when the writer's own output "
                                   "interpolation is non-monotone and undershoots at shock "
                                   "fronts. Off by default, since it modifies the loaded data.")

    perf_group = parser.add_argument_group("performance")
    perf_group.add_argument('--n-cpu', type=int,
                             default=int(os.environ.get('SLURM_NTASKS_PER_NODE', os.cpu_count() or 1)),
                             help="Number of worker processes. Defaults to $SLURM_NTASKS_PER_NODE or os.cpu_count().")
    mem_group = perf_group.add_mutually_exclusive_group()
    mem_group.add_argument('--files-per-step', type=int, default=None,
                            help="Number of snapshot files to keep loaded at once (default: 10).")
    mem_group.add_argument('--max-tot-memory-gb', type=float, default=None,
                            help="Alternative to --files-per-step. Caps the total shared memory used by loaded snapshots, counting every key.")
    perf_group.add_argument('--verbose', action='store_true',
                             help="Print per-file loading progress.")

    interp_group = parser.add_argument_group("interpolator")
    interp_group.add_argument('--interpolator', choices=['pchip', 'regular', 'meshblock'],
                               default=None,
                               help="Spatial interpolation scheme (default: per --format; "
                                    "'meshblock' for --format=athdf_spherical, 'pchip' otherwise). "
                                    "The athdf_spherical format's meshblock memory layout only works "
                                    "with 'meshblock'.")
    interp_group.add_argument('--interp-method',
                               choices=['linear', 'nearest', 'slinear', 'cubic', 'quintic', 'pchip'],
                               default='linear',
                               help="Method passed to RegularGridInterpolator (--interpolator=regular only).")
    interp_group.add_argument('--cache-size-gb', type=float, default=None,
                               help="Max cache size per PCHIP x-interpolator (--interpolator=pchip only). "
                                    "Default: auto-computed from free memory, --n-cpu, and the integrator's "
                                    "snapshot count.")
    interp_group.add_argument('--rad-transform', choices=['log', 'asinh', 'none'], default=None,
                               help="Coordinate transform applied to the radial axis before interpolation "
                                    "(default: per --format). 'asinh' requires --rad-scale.")
    interp_group.add_argument('--rad-scale', type=float, default=None,
                               help="Lin-log transition radius for --rad-transform=asinh (code units).")

    integ_group = parser.add_argument_group("integrator")
    integ_group.add_argument('--integrator', choices=['expl_trap', 'impl_trap', 'rk4'], default='impl_trap',
                              help="Time integration scheme.")
    integ_group.add_argument('--tol', type=float, default=1e-8,
                              help="Picard convergence tolerance (--integrator=impl_trap only).")
    integ_group.add_argument('--max-iter', type=int, default=20,
                              help="Max Picard iterations per step (--integrator=impl_trap only).")
    integ_group.add_argument('--relax', type=float, default=1.0,
                              help="Picard relaxation factor in (0, 1] (--integrator=impl_trap only).")
    integ_group.add_argument('--rk4-lagrange', action='store_true',
                              help="Use cubic Lagrange time interpolation instead of the default monotone "
                                   "PCHIP blend (--integrator=rk4 only).")

    parser.add_argument('--seed-rng', type=int, default=None,
                         help="Seed numpy's RNG (used for in-cell random jitter) for reproducible runs.")

    subparsers = parser.add_subparsers(dest='seed_mode', required=True,
                                        help="How to seed initial tracer positions.")

    volume = subparsers.add_parser('volume', help="Seed tracers throughout a spherical volume.")
    volume.add_argument('--r-min', required=True, type=float, help="Minimum radius.")
    volume.add_argument('--r-max', required=True, type=float, help="Maximum radius.")
    volume.add_argument('--n-r', required=True, type=int, help="Number of radial bins.")
    volume.add_argument('--n-th', required=True, type=int, help="Number of theta bins.")
    volume.add_argument('--n-ph', required=True, type=int, help="Number of phi bins.")
    volume.add_argument('--phi-min-deg', type=float, default=0.0, help="Minimum azimuthal angle (degrees).")
    volume.add_argument('--phi-max-deg', type=float, default=360.0, help="Maximum azimuthal angle (degrees).")
    volume.add_argument('--theta-min-deg', type=float, default=0.0, help="Minimum polar angle (degrees).")
    volume.add_argument('--theta-max-deg', type=float, default=180.0, help="Maximum polar angle (degrees).")
    volume.add_argument('--n-quad', type=int, default=2,
                         help="Gauss-Legendre points per dimension for cell-mass integration.")
    volume.add_argument('--no-random-shift', action='store_false', dest='random_shift', default=True,
                         help="Place tracers at cell centres instead of jittering within each cell.")

    surface = subparsers.add_parser('surface', help="Seed tracers on a spherical surface.")
    surface.add_argument('--r-surf', required=True, type=float, help="Radius of the seeding surface.")
    surface.add_argument('--n-th', required=True, type=int, help="Number of theta bins.")
    surface.add_argument('--n-ph', required=True, type=int, help="Number of phi bins.")
    surface.add_argument('--phi-min-deg', type=float, default=0.0, help="Minimum azimuthal angle (degrees).")
    surface.add_argument('--phi-max-deg', type=float, default=360.0, help="Maximum azimuthal angle (degrees).")
    surface.add_argument('--theta-min-deg', type=float, default=0.0, help="Minimum polar angle (degrees).")
    surface.add_argument('--theta-max-deg', type=float, default=180.0, help="Maximum polar angle (degrees).")
    surface.add_argument('--n-quad', type=int, default=3,
                          help="Gauss-Legendre points per dimension for mass-flux integration.")
    surface.add_argument('--no-random-shift', action='store_false', dest='random_shift', default=True,
                          help="Use bin-centre angles/times instead of jittering within each cell/window.")
    surface.add_argument('--every-n-files', type=int, default=1,
                          help="Build one time slot every Nth snapshot between --start-t and --end-t.")

    args = parser.parse_args()

    fmt = FORMATS[args.format]
    if args.keys is None:
        args.keys = list(fmt['keys'])
    if args.file_pattern is None:
        args.file_pattern = fmt['file_pattern']
    if args.rad_transform is None:
        args.rad_transform = fmt['rad_transform']
    if args.interpolator is None:
        args.interpolator = fmt.get('interpolator', 'pchip')

    if args.rad_transform == 'asinh' and args.rad_scale is None:
        parser.error("--rad-transform=asinh requires --rad-scale")
    missing = [k for k in list(args.vel_keys) + [args.density_key] if k not in args.keys]
    if missing:
        parser.error(f"--vel-keys/--density-key entries not present in --keys: {missing}")
    missing = [k for k in args.positive_keys if k not in args.keys]
    if missing:
        parser.error(f"--positive-keys entries not present in --keys: {missing}")
    return args


def auto_cache_size_gb(n_cpu: int, integrator_name: str) -> float:
    """Estimate a safe per-interpolator PCHIP cache size from free memory."""
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                free_mem_gb = int(line.split()[1]) / 1024 ** 2  # kB -> GB
                break
        else:
            raise RuntimeError("Could not find MemAvailable in /proc/meminfo.")

    n_snapshots = INTEGRATOR_N_SNAPSHOTS[integrator_name]
    # Each worker holds n_snapshots velocity interpolators + 1 data interpolator concurrently.
    n_interps = n_cpu * (n_snapshots + 1)
    cache_size_gb = 0.8 * free_mem_gb / n_interps
    print(f"Auto cache size: {cache_size_gb:.3f} GB/interpolator "
          f"({n_interps} concurrent interpolators, {free_mem_gb:.1f} GB free).")
    return cache_size_gb


def build_integrator(args: argparse.Namespace):
    if args.integrator == 'expl_trap':
        return ExplicitTrapezoid()
    if args.integrator == 'impl_trap':
        return ImplicitTrapezoid(tol=args.tol, max_iter=args.max_iter, relax=args.relax)
    if args.integrator == 'rk4':
        return RK4(monotone=not args.rk4_lagrange)
    raise ValueError(f"Unknown integrator {args.integrator!r}")


def build_interpolator_cls_and_kwargs(args: argparse.Namespace):
    if args.interpolator == 'pchip':
        cache_size_gb = args.cache_size_gb
        if cache_size_gb is None:
            cache_size_gb = auto_cache_size_gb(args.n_cpu, args.integrator)
        return PchipInterpolator3D, {"max_cache_size_GB": cache_size_gb}
    if args.interpolator == 'regular':
        return RegularInterpolator3D, {"method": args.interp_method}
    if args.interpolator == 'meshblock':
        return MeshblockPchipInterpolator, {}
    raise ValueError(f"Unknown interpolator {args.interpolator!r}")


def build_file_handler(args: argparse.Namespace, interpolator_cls, interpolator_kwargs):
    files_per_step = args.files_per_step
    max_tot_memory = None
    if args.max_tot_memory_gb is not None:
        max_tot_memory = int(args.max_tot_memory_gb * 1024 ** 3)
        files_per_step = None
    elif files_per_step is None:
        files_per_step = 10

    if args.rad_transform == 'asinh':
        rad_transform = ('asinh', args.rad_scale)
    elif args.rad_transform == 'log':
        rad_transform = 'log'
    else:
        rad_transform = None

    common_kwargs = dict(
        interpolator=interpolator_cls,
        directory=args.data_dir,
        keys=list(args.keys),
        n_cpu=args.n_cpu,
        files_per_step=files_per_step,
        max_tot_memory=max_tot_memory,
        verbose=args.verbose,
        rad_transform=rad_transform,
        file_pattern=args.file_pattern,
        interpolator_kwargs=interpolator_kwargs,
        positive_keys=list(args.positive_keys),
    )

    if args.format == 'athenak':
        return AthenaKFileHandler(heavy_neutrinos=args.heavy_neutrinos, **common_kwargs)
    if args.format == 'athdf_spherical':
        return SphericalAthdfFileHandler(bh_mass=args.bh_mass, **common_kwargs)
    return FORMATS[args.format]['handler'](**common_kwargs)


def build_surface_t_start(args: argparse.Namespace, file_handler) -> np.ndarray:
    file_times = np.asarray(file_handler.times)
    if args.end_t > args.start_t:
        sel = file_times[(file_times >= args.start_t) & (file_times <= args.end_t)]
    else:
        sel = file_times[(file_times <= args.start_t) & (file_times >= args.end_t)][::-1]
    t_start = sel[::args.every_n_files]
    if len(t_start) < 2:
        raise ValueError(
            f"Only {len(t_start)} snapshot time(s) found between --start-t and --end-t with "
            f"--every-n-files={args.every_n_files}; need at least 2 to build a time slot."
        )
    return t_start


def seed_tracers(args: argparse.Namespace, integrator, file_handler):
    common_kwargs = dict(
        vel_keys=list(args.vel_keys),
        integrator=integrator,
        file_handler=file_handler,
    )
    phi_min, phi_max = np.radians(args.phi_min_deg), np.radians(args.phi_max_deg)
    theta_min, theta_max = np.radians(args.theta_min_deg), np.radians(args.theta_max_deg)

    if args.seed_mode == 'volume':
        return spherical_by_volume(
            r_min=args.r_min, r_max=args.r_max,
            n_r=args.n_r, n_th=args.n_th, n_ph=args.n_ph,
            start_t=args.start_t,
            phi_min=phi_min, phi_max=phi_max,
            theta_min=theta_min, theta_max=theta_max,
            random_shift_in_cell=args.random_shift,
            n_quad=args.n_quad,
            density_key=args.density_key,
            **common_kwargs,
        )

    t_start = build_surface_t_start(args, file_handler)
    return spherical_surface_by_area(
        r_surf=args.r_surf,
        t_start=t_start,
        n_th=args.n_th, n_ph=args.n_ph,
        phi_min=phi_min, phi_max=phi_max,
        theta_min=theta_min, theta_max=theta_max,
        random_shift_in_cell=args.random_shift,
        n_quad=args.n_quad,
        density_key=args.density_key,
        **common_kwargs,
    )


def write_output(tracers, output_dir: str, density_key: str = 'rho') -> None:
    os.makedirs(output_dir, exist_ok=True)
    filebase = f"{output_dir}/tracer_"

    for tr in tracers.tracers:
        if 'dV' in tr.props:
            i_tmax = np.argmax(tr.times)
            tr.props['mass'] = tr.props['dV'] * tr.data[density_key][i_tmax]

        # Strip any dotted group prefix ("group.field" -> "field") from data keys.
        short_keys = {key: key.split(".")[-1] for key in tr.data.keys()}
        for key, short in short_keys.items():
            if short == key:
                continue
            tr.data[short] = tr.data[key]
            del tr.data[key]

        tr.output_to_ascii(coords=['x', 'y', 'z'], filebase=filebase)


def main() -> None:
    import multiprocessing as mp
    mp.set_start_method("spawn")

    args = parse_args()

    if args.seed_rng is not None:
        np.random.seed(args.seed_rng)

    integrator = build_integrator(args)
    interpolator_cls, interpolator_kwargs = build_interpolator_cls_and_kwargs(args)
    file_handler = build_file_handler(args, interpolator_cls, interpolator_kwargs)

    # Snap --start-t to the nearest available snapshot time.  Tracers only
    # activate when their seed time matches a snapshot time (np.isclose in
    # src/tracers.py), and a --start-t rounded below the snapshot's exact
    # (float32) Time would also exclude that snapshot from the integration
    # range -- both silently yield "0 tracers active" for the whole run.
    snapped = float(file_handler.times[np.argmin(np.abs(file_handler.times - args.start_t))])
    if snapped != args.start_t:
        print(f"Snapping --start-t {args.start_t} to the nearest snapshot time {snapped}.")
        args.start_t = snapped

    tracers = seed_tracers(args, integrator, file_handler)
    tracers.integrate(args.start_t, args.end_t)

    write_output(tracers, args.output_dir, args.density_key)


if __name__ == "__main__":
    main()
