#!/usr/bin/env python3
"""
Check tracer-derived cumulative ejected mass, at some radius --r-check,
against a raw-simulation ground truth. This is the actual validation this
script performs -- pass --raw-sim-dir (see below).

volume and surface seeded tracers represent two genuinely distinct,
non-overlapping parts of the ejecta (see README.md's "volume and surface
seeding are complementary" section): volume seeding catches whatever is
still inside the domain at the shared reference time, surface seeding
catches whatever already crossed --r-check earlier. They are not expected
to match each other -- there's nothing to compare between them on their
own. What IS meaningful is their sum: the combined tracer-derived total
should equal an independent, direct measurement of the same quantity from
the raw simulation. Without --raw-sim-dir, this script has no ground
truth to check against and just plots the combined/surface-only
breakdown for inspection, not a pass/fail validation.

Both populations' tracers are advected backward from a late seed time
toward the (small-r, hot) merger epoch, so every tracer's radius
*decreases* going backward in time -- meaning every tracer, from either
population, sweeps downward through any given --r-check as long as
--r-check is smaller than its own seed radius (volume's --r-max, or
surface's --r-surf). Plots, vs time: the total tracer mass currently at
radius >= --r-check (i.e. that has not yet, going backward, fallen below
it), and its time derivative -- this is the tracer-derived cumulative
mass/mass-rate that crossed --r-check by a given time, in exactly the
same sense a raw simulation's own surface-flux diagnostic at that radius
measures it.

--r-check need NOT equal (and, for the volume population to show any
interesting time dependence at all, generally should NOT equal) either
seeding run's --r-max/--r-surf: those set where the two populations were
seeded; --r-check is just an observation radius you're curious about
(e.g. matching wherever a raw-simulation surface diagnostic exists),
typically well inside --r-min.

Example
-------
    python analysis/mass_conservation.py \\
        --volume-dirs data/tracers_out --surface-dirs data/tracers_out_surf \\
        --r-check 350 --raw-sim-dir data/ --output mass_conservation.png
"""

import argparse
import os

import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis._common import add_tracer_loading_args, load_trajectories, tracer_masses, get_units, DEFAULT_STATUSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check tracer-derived cumulative ejected mass against a raw-simulation ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--volume-dirs', required=True, nargs='+',
                         help="Output directory/directories from run_pipeline.py's volume seeding mode.")
    parser.add_argument('--surface-dirs', required=True, nargs='+',
                         help="Output directory/directories from run_pipeline.py's surface seeding mode.")
    parser.add_argument('--glob-pattern', default='tracer_*.dat',
                         help="Glob pattern used to find tracer files within each directory.")
    parser.add_argument('--n-cpu', type=int, default=os.cpu_count() or 1,
                         help="Parallel workers for loading tracer files.")
    parser.add_argument('--status', nargs='+', default=list(DEFAULT_STATUSES),
                         choices=['done', 'active', 'failed', 'not started'],
                         help="Only keep tracers with one of these statuses.")
    parser.add_argument('--r-check', required=True, type=float,
                         help="Radius (code units) to measure cumulative crossing mass/flux at. Should "
                              "generally be smaller than both seeding runs' --r-max/--r-surf (e.g. close to "
                              "--r-min, or wherever a raw-simulation diagnostic is available) -- it is NOT "
                              "the volume/surface seeding boundary itself. See the module docstring.")
    parser.add_argument('--min-innermost-r', type=float, default=None,
                         help="Optional sanity filter: drop tracers whose innermost (earliest-time) radius "
                              "never gets below this value (code units) -- e.g. to exclude tracers that "
                              "didn't integrate back far enough to reach a reference epoch. Disabled by default.")
    parser.add_argument('--densitize-mass', type=float, default=None,
                        help="ADM mass (code units) used to convert the rho-based "
                             "tracer masses to the conserved baryon mass "
                             "D = rho*W*sqrt(gamma), via W = -u_t/alpha and an "
                             "analytic isotropic-Schwarzschild metric. Omit to "
                             "leave the masses as the seeding produced them.")
    parser.add_argument('--mdot-dt', type=float, default=None,
                        help="Bin the tracer-derived mdot onto this time step "
                             "(code units) instead of the snapshot cadence. The "
                             "tracer rate is a sum over discrete crossings and "
                             "carries shot noise the Eulerian ground truth does "
                             "not; widening the differencing interval averages it "
                             "down. Leaves the cumulative mass curves untouched.")
    parser.add_argument('--t-min', type=float, default=None, help="Plot x-axis lower bound (default: full range).")
    parser.add_argument('--t-max', type=float, default=None, help="Plot x-axis upper bound (default: full range).")
    parser.add_argument('--output', default='mass_conservation.png', help="Output image path.")
    parser.add_argument('--dpi', type=int, default=300, help="Output image DPI.")

    raw_group = parser.add_argument_group(
        "raw-simulation ground truth (needs the external 'surface' package; this is the actual validation "
        "-- without it the script just plots the tracer-derived breakdown with nothing to check it against)")
    raw_group.add_argument('--raw-sim-dir', default=None,
                            help="Path to the raw (untransformed) simulation output (the directory "
                                 "transform_files.py reads its input from -- often called e.g. 'data/', "
                                 "distinct from the 'transformed/' directory run_pipeline.py reads). Omit to "
                                 "skip the ground-truth comparison entirely.")
    raw_group.add_argument('--batchtools', action='store_true',
                            help="Assume --raw-sim-dir has a batchtools subdirectory structure (output-0000...).")
    raw_group.add_argument('--isurf', type=int, default=1, help="Index of the raw surface diagnostic to use.")
    raw_group.add_argument('--irad', type=int, default=None,
                            help="Radius index of the raw surface diagnostic. Default: auto-inferred from "
                                 "--r-check by reading a sample raw file's radius grid and picking the "
                                 "closest match -- pass this explicitly only if that guess is wrong (e.g. "
                                 "unusual file naming) or you want a specific index regardless of --r-check.")
    raw_group.add_argument('--raw-n-cpu', type=int, default=os.cpu_count() or 1,
                            help="Parallel workers for reading the raw simulation surface diagnostic.")
    raw_group.add_argument('--verbose', action='store_true', help="Print raw-diagnostic loading progress.")

    return parser.parse_args()


def infer_irad(paths: list[str], isurf: int, r_check: float) -> tuple[int, float]:
    """
    Pick the raw-diagnostic radius index whose actual grid radius is
    closest to r_check, by reading the radius grid straight out of one
    matching raw surface file (same '*.surfaceN.*.hdf5' naming and
    per-radius 'coordinates/<key>/R' layout transform_files.py's input
    uses). Returns (irad, the actual radius at that index).
    """
    import glob
    from h5py import File

    sample_file = None
    for path in paths:
        matches = sorted(glob.glob(f"{path}/*.surface{isurf}.*.hdf5"))
        if matches:
            sample_file = matches[0]
            break
    if sample_file is None:
        raise FileNotFoundError(
            f"Could not find a *.surface{isurf}.*.hdf5 file under {paths} to infer --irad from --r-check. "
            f"Pass --irad explicitly instead."
        )

    with File(sample_file, "r") as f:
        rad_keys = sorted(f["coordinates"].keys(), key=lambda k: int(k))
        r_grid = np.array([float(f[f"coordinates/{key}/R"][0]) for key in rad_keys])

    irad = int(np.argmin(np.abs(r_grid - r_check)))
    return irad, float(r_grid[irad])


def crossing_mass_vs_time(trajs, masses: np.ndarray, times: np.ndarray, r_check: float) -> np.ndarray:
    """
    Total mass of `trajs` currently at radius >= r_check, at each time in
    `times` (a shared axis across both populations, which don't
    necessarily share individual seed times -- surface tracers are seeded
    across a range of times, not all at the global max).

    Beyond each tracer's own last-recorded time (either its own seed time,
    for a tracer whose seed radius is itself >= r_check, e.g. a surface
    tracer with r_check < r_surf; or wherever it left the interpolation
    domain) its radius is held at that last known value, since -- going
    forward from there -- it can only have kept moving further outward.
    It contributes 0 before its own first recorded time (not yet seeded).
    """
    tr_r = np.zeros((len(times), len(trajs)))
    for i, tr in enumerate(trajs):
        t_msk = np.isin(times, tr.data["time"])
        r = np.sqrt(tr.data["x"]**2 + tr.data["y"]**2 + tr.data["z"]**2)
        tr_r[t_msk, i] = r
        if t_msk.any():
            last_i = len(times) - 1 - np.argmax(t_msk[::-1])
            tr_r[last_i:, i] = tr_r[last_i, i]
    return np.array([np.sum(masses[r >= r_check]) for r in tr_r])



def densitize(trajs, masses, m_adm: float) -> np.ndarray:
    """
    Convert rho-based tracer masses to the conserved baryon mass.

    The seeding integrates `--density-key` (rho, the rest-mass density) over a
    coordinate cell, while the conserved rest mass is the integral of the
    densitized D = rho*W*sqrt(gamma). The missing factor is recoverable from
    the tracer's own u_t plus an analytic metric: W = -u_t/alpha holds to
    machine precision wherever the shift is negligible, and for isotropic
    Schwarzschild alpha = (1 - M/2r)/psi and sqrt(gamma) = psi^6 with
    psi = 1 + M/2r, so

        D/rho = W*sqrt(gamma) = (-u_t) * psi**7 / (1 - M/2r).

    Evaluated at the seed point, where the tracer's cell volume was measured
    and so where its conserved parcel mass is defined.
    """
    out = np.asarray(masses, dtype=float).copy()
    for i, tr in enumerate(trajs):
        if 'adm_mass' in tr.props or 'mass_D' in tr.props:
            # The mass is already densitized: 'adm_mass' says the run weighted
            # on D throughout, and 'mass_D' is the older output where
            # tracer_masses has returned the conserved value. Either way,
            # applying the factor again would double it.
            continue
        d = tr.data
        k = int(np.argmax(d["time"]))
        r = float(np.sqrt(d["x"][k] ** 2 + d["y"][k] ** 2 + d["z"][k] ** 2))
        psi = 1.0 + m_adm / (2.0 * r)
        out[i] *= float(-d["u_t"][k]) * psi ** 7 / (1.0 - m_adm / (2.0 * r))
    return out


def main() -> None:
    args = parse_args()
    units = get_units()

    vol_trajs = load_trajectories(args.volume_dirs, args.glob_pattern, args.n_cpu, tuple(args.status))
    srf_trajs = load_trajectories(args.surface_dirs, args.glob_pattern, args.n_cpu, tuple(args.status))
    print(f"Loaded {len(vol_trajs)} volume-seeded and {len(srf_trajs)} surface-seeded tracers.")

    if args.min_innermost_r is not None:
        def reaches_inner_r(tr):
            r = np.sqrt(tr.data["x"]**2 + tr.data["y"]**2 + tr.data["z"]**2)
            return r.min() < args.min_innermost_r
        vol_trajs = [t for t in vol_trajs if reaches_inner_r(t)]
        srf_trajs = [t for t in srf_trajs if reaches_inner_r(t)]
        print(f"{len(vol_trajs)} volume-seeded and {len(srf_trajs)} surface-seeded tracers "
              f"left after the --min-innermost-r filter.")

    vol_masses = tracer_masses(vol_trajs)
    srf_masses = tracer_masses(srf_trajs)

    if args.densitize_mass is not None:
        vol_masses = densitize(vol_trajs, vol_masses, args.densitize_mass)
        srf_masses = densitize(srf_trajs, srf_masses, args.densitize_mass)
        print(f"masses densitized to D = rho*W*sqrt(gamma) using M_ADM="
              f"{args.densitize_mass:g}; total {vol_masses.sum() + srf_masses.sum():.4e} Msun")

    times = np.unique(np.concatenate([tr.data["time"] for tr in (*vol_trajs, *srf_trajs)]))

    mtot_vol = crossing_mass_vs_time(vol_trajs, vol_masses, times, args.r_check)
    mtot_srf = crossing_mass_vs_time(srf_trajs, srf_masses, times, args.r_check)
    mtot_combined = mtot_vol + mtot_srf

    def rate(mtot):
        """
        d(mtot)/dt, optionally on a coarser grid than the snapshot cadence.

        The tracer-derived rate is a sum over discrete parcels crossing the
        sphere, so it carries shot noise that the Eulerian ground truth does
        not. Differencing the cumulative curve over a wider interval averages
        that down (as sqrt of the widening) without touching the cumulative
        curve itself, which is what the mass comparison actually uses.
        """
        if args.mdot_dt is None:
            t = times
            m = mtot
        else:
            n = max(2, int(np.ceil((times[-1] - times[0]) / args.mdot_dt)) + 1)
            t = np.linspace(times[0], times[-1], n)
            m = np.interp(t, times, mtot)
        d = np.zeros_like(m)
        d[1:] = np.diff(m) / np.diff(t)
        return t, d

    t_rate, mdot_combined = rate(mtot_combined)
    _, mdot_srf = rate(mtot_srf)
    if args.mdot_dt is not None:
        print(f"mdot binned onto dt={args.mdot_dt:g} ({len(t_rate)} points, "
              f"native cadence had {len(times)})")

    t_ms = times * units.time_ms
    t_rate_ms = t_rate * units.time_ms

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(t_rate_ms, mdot_combined / units.time_ms, label='combined (tracer-derived)')
    ax[0].plot(t_rate_ms, mdot_srf / units.time_ms, label='surface contribution only')
    ax[1].plot(t_ms, mtot_combined, label='combined (tracer-derived)')
    ax[1].plot(t_ms, mtot_srf, label='surface contribution only')

    if args.raw_sim_dir is not None:
        from surface import Surfaces
        from surface import surface_func as sf

        if args.batchtools:
            paths = [f"{args.raw_sim_dir}/{d}" for d in os.listdir(args.raw_sim_dir)
                     if os.path.isdir(f"{args.raw_sim_dir}/{d}") and d.startswith("output-")]
        else:
            paths = [args.raw_sim_dir]

        irad = args.irad
        if irad is None:
            irad, actual_r = infer_irad(paths, args.isurf, args.r_check)
            print(f"Auto-selected --irad={irad} (grid radius {actual_r:.3f}) closest to --r-check={args.r_check:.3f}.")

        s = Surfaces(paths, args.isurf, irad, n_cpu=args.raw_n_cpu, verbose=args.verbose)
        if not np.isclose(s.r, args.r_check, rtol=1e-3):
            print(f"WARNING: raw diagnostic surface radius ({s.r:.3f}) does not match "
                  f"--r-check ({args.r_check:.3f}); the overlay below is not measuring the same boundary.")

        # Surface dumps that carry the derived `tracer.*` group hand us rho and
        # V^i ready-made. The plain GR-Athena++ surface output (the one written
        # without that group) carries the evolved variables instead, and there
        # the same flux is D*V^i, which is what sf.mass_flux builds -- and for a
        # densitized flux like D the flat normal of integrate_flux_classical is
        # exact rather than approximate.
        tracer_flux = ("tracer.hydro.aux.V_u_x", "tracer.hydro.aux.V_u_y",
                       "tracer.hydro.aux.V_u_z")
        tracer_weight = "tracer.hydro.prim.rho"
        if all(k in s.fields for k in (*tracer_flux, tracer_weight)):
            mdot_sf = sf.integrate_flux_classical(flux=tracer_flux, weight=tracer_weight)
        else:
            missing = [k for k in sf.mass_flux.keys if k not in s.fields]
            if missing:
                raise KeyError(
                    f"surface{args.isurf} dumps carry neither the 'tracer.*' group nor "
                    f"the variables sf.mass_flux needs (missing {missing}). Point "
                    f"--isurf at a surface that dumps one of the two."
                )
            mdot_sf = sf.integrate_flux_classical(flux=sf.mass_flux)
        print(f"raw ground truth from {mdot_sf.name}")
        raw_data = s.process_h5_parallel((mdot_sf,), ordered=True)
        raw_mdot = np.array([d[0] for d in raw_data])
        # s.dts is already a centred trapezoid weight -- dts[i] = (dt[i]+dt[i-1])/2,
        # halved at the ends -- so sum(mdot*dts) is the correct total. But
        # cumsum of it overshoots every intermediate point by mdot[i]*dt[i]/2,
        # half a step, because it credits the whole of dump i's weight to a
        # curve plotted at t[i]. That makes the raw cumulative lead the
        # tracer-derived one during the steep rise. Integrating properly puts
        # the mass where it belongs and leaves the total unchanged.
        raw_mtot = cumulative_trapezoid(raw_mdot, s.times, initial=0.0)

        raw_t_ms = s.times * units.time_ms
        ax[0].plot(raw_t_ms, raw_mdot / units.time_ms, label='raw simulation (ground truth)', c='k')
        ax[1].plot(raw_t_ms, raw_mtot, label='raw simulation (ground truth)', c='k')

        # The point of the script, stated as a number rather than left to the
        # eye. Compare at the last time both curves cover, since the tracer and
        # raw diagnostics are written on different cadences and end points.
        t_end = min(times[-1], s.times[-1])
        m_tracer = float(np.interp(t_end, times, mtot_combined))
        m_raw = float(np.interp(t_end, s.times, raw_mtot))
        print(f"cumulative mass through R={args.r_check:g} at t={t_end:g} "
              f"({t_end * units.time_ms:.2f} ms):")
        print(f"  tracer-derived   {m_tracer:.4e} Msun")
        print(f"  raw ground truth {m_raw:.4e} Msun")
        print(f"  tracer/raw = {m_tracer / m_raw:.4f}  "
              f"({100 * (m_tracer / m_raw - 1):+.2f}%)")

    for a in ax:
        a.set_xlabel("time (ms)")
        if args.t_min is not None or args.t_max is not None:
            a.set_xlim(args.t_min, args.t_max)
    ax[0].set_ylabel(r"$\dot{M}$ (M$_\odot$/ms)")
    ax[1].set_ylabel(r"$M$ (M$_\odot$)")
    ax[1].legend()
    fig.suptitle(f"R = {args.r_check:.2f} M")

    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
