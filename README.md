# trace

`trace` advects massless Lagrangian tracer particles through a sequence of
velocity-field snapshots from a numerical-relativity simulation, recording
each tracer's interpolated thermodynamic history (density, temperature,
electron fraction, neutrino fluxes, ...) along the way. The resulting
per-tracer time series are the standard input for nucleosynthesis
post-processing (e.g. reaction-network codes) of merger ejecta.

Everything except the reader is agnostic about which code produced the
snapshots. Three formats ship with `trace`, and adding another means
implementing one class -- see
[Supporting a new data format](#supporting-a-new-data-format).

This document is a practical guide for *running* the tool: pointing it at
your data, invoking the pipeline, understanding every command-line option,
reading the output back, and running at scale on a cluster. For the
internals (module layout, class responsibilities) see the section
["How it fits together"](#how-it-fits-together) below; for the test suite
and exploratory examples see `tests/README.md` and `examples/README.md`;
and for post-processing real tracer output (plots, animations, mass-budget
checks) see [Analysis scripts](#analysis-scripts) and `analysis/README.md`.

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Step 1: point it at your data](#step-1-point-it-at-your-data)
- [Step 2: run the pipeline](#step-2-run-the-pipeline)
  - [Input/output options](#inputoutput-options)
  - [Field options](#field-options)
  - [Performance options](#performance-options)
  - [Interpolator options](#interpolator-options)
  - [Integrator options](#integrator-options)
  - [Seeding: `volume` mode](#seeding-volume-mode)
  - [Seeding: `surface` mode](#seeding-surface-mode)
  - [Seeding: `volume-mc` and `surface-mc` modes](#seeding-volume-mc-and-surface-mc-modes)
- [What counts as mass: `--adm-mass`](#what-counts-as-mass---adm-mass)
- [Step 3: read the output](#step-3-read-the-output)
- [Running on a cluster (SLURM)](#running-on-a-cluster-slurm)
- [Choosing options: a tuning guide](#choosing-options-a-tuning-guide)
  - [Matching cell shapes and masses between `volume` and `surface`](#matching-cell-shapes-and-masses-between-volume-and-surface)
- [Units](#units)
- [Analysis scripts](#analysis-scripts)
- [How it fits together](#how-it-fits-together)
- [Supporting a new data format](#supporting-a-new-data-format)
- [Tests and examples](#tests-and-examples)
- [Troubleshooting](#troubleshooting)

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `h5py`, `tqdm` (core pipeline)
- `matplotlib` (only needed for the plotting/analysis scripts, not for
  `run_pipeline.py` itself; install with the `plots` extra below)

Install with:

```bash
pip install -e .            # core pipeline only
pip install -e ".[plots]"   # + matplotlib, for analysis/ and examples/
```

This is an editable install (`pyproject.toml`), so it picks up local edits
immediately and puts `src`, `analysis`, and `examples` all on your import
path permanently -- no `PYTHONPATH` juggling needed afterward, from any
directory. If you'd rather not install anything, everything also runs
directly from the repo root without it: just make sure the repo root is
your working directory, or is on `PYTHONPATH`, when invoking any script
(`run_pipeline.py`, anything under `analysis/`, `examples/`, etc.).

## Quick start

```bash
# GR-Athena++ surface output: reduce the raw files once, then run.
# Both formats below keep their data on one global spherical grid, so
# --weight-grid native weights the sampling on the data as written, with no
# interpolation anywhere. It is what `auto` picks for them anyway.
python transform_files.py /path/to/raw/*.surface1.*.hdf5 --output_dir data/transformed
python run_pipeline.py --format reduced_surface \
    --data-dir data/transformed --output-dir data/tracers_out \
    --start-t 11600 --end-t 0 --adm-mass 2.7 \
    volume-mc --r-min 300 --r-max 1000 --n-tracers 20000 --weight-grid native

# AthenaK spherical-grid vtk: no preparation step, point it at the dumps.
python run_pipeline.py --format athenak \
    --data-dir /path/to/vtk --output-dir data/tracers_out \
    --start-t 6120 --end-t 6080 --adm-mass 2.7 \
    volume-mc --r-min 300 --r-max 1000 --n-tracers 20000 --weight-grid native

# Athena++ meshblock athdf (written with ghost zones): no preparation step.
# An octree has no single global grid, but its meshblocks tile the domain, so
# native works here too -- one entry per block.
python run_pipeline.py --format athdf_spherical \
    --data-dir /path/to/athdf --output-dir data/tracers_out \
    --start-t 40.9446 --end-t 0 --bh-mass 1.0 --adm-mass 2.7 \
    volume-mc --r-min 305 --r-max 1000 --n-tracers 20000 --weight-grid native

# Each tracer's full history is now one ASCII file:
ls data/tracers_out/           # tracer_000000.dat, tracer_000001.dat, ...
```

## Step 1: point it at your data

Pick the format with `--format`. It determines how snapshots are read, and
sets the defaults for `--file-pattern`, `--rad-transform`, `--interpolator`
and `--keys` -- which differ between formats because the grids and field
names do.

| `--format` | Data | Preparation | Details |
|---|---|---|---|
| `reduced_surface` (default) | GR-Athena++ surface output, HDF5 | Yes -- run `transform_files.py` once per simulation | [docs/formats/gr_athena.md](docs/formats/gr_athena.md) |
| `athenak` | AthenaK spherical-grid output, binary VTK | None, read directly | [docs/formats/athenak.md](docs/formats/athenak.md) |
| `athdf_spherical` | Athena++ meshblock output (spherical, with ghost zones), HDF5 | None, read directly | [docs/formats/athdf_spherical.md](docs/formats/athdf_spherical.md) |

Read your format's page before the first run: it lists the field names,
the grid conventions, and the format-specific caveats. Everything from
[Step 2](#step-2-run-the-pipeline) onward is the same either way.

For a format that isn't listed, see
[Supporting a new data format](#supporting-a-new-data-format).

## Step 2: run the pipeline

```
python run_pipeline.py [shared options...] {volume,surface,volume-mc,surface-mc} [mode-specific options...]
```

Run `python run_pipeline.py --help`, `python run_pipeline.py volume --help`,
or `python run_pipeline.py surface --help` at any time for the authoritative,
up-to-date option list -- what follows explains what each one *means* and
when to change it from its default.

### Input/output options

| Flag | Default | Meaning |
|---|---|---|
| `--format` | `reduced_surface` | Snapshot data format; see [Step 1](#step-1-point-it-at-your-data). Also supplies the defaults for `--file-pattern`, `--rad-transform`, `--interpolator` and `--keys`. |
| `--data-dir` (required) | -- | Directory of snapshot files. |
| `--output-dir` | `tracer_output` | Directory to write `tracer_NNNNNN.dat` files to (created if missing). |
| `--file-pattern` | per `--format` | Glob used to find snapshot files inside `--data-dir`. |
| `--start-t` (required) | -- | Time to seed tracers at and start integrating from. |
| `--end-t` (required) | -- | Time to integrate to. May be **less than** `--start-t` -- the pipeline detects the direction and integrates backward, loading snapshots in reverse. This is the common case for ejecta tracers: seed late (after the ejecta is fully unbound) and integrate backward to trace each tracer's thermodynamic history through the epoch that set its final composition. |

Both times must be given in the same units as the time recorded in your
snapshot files (see [Units](#units)); `--start-t`/`--end-t` don't need to
exactly match a snapshot time -- the nearest available snapshot times are
used as the integration bounds.

### Field options

| Flag | Default | Meaning |
|---|---|---|
| `--keys` | per `--format` | Every field loaded from the snapshots and recorded along each tracer's history. Must all exist in your snapshot files. |
| `--vel-keys` | `V_u_x V_u_y V_u_z` | The (3 of `--keys`) fields used as the Cartesian velocity vector that actually moves the tracers. |
| `--density-key` | `rho` | Field used as mass density when computing each tracer's represented mass (see the seeding sections below). |
| `--adm-mass` | off | ADM mass in code units. Makes every tracer's `mass` the conserved rest mass, weighting on `D = rho*W*sqrt(gamma)` instead of `rho` -- see [What counts as mass](#what-counts-as-mass---adm-mass). |
| `--ut-key` | `u_t` | Field key for `u_t`, from which `--adm-mass` reconstructs the Lorentz factor. Must be in `--keys` when `--adm-mass` is used. |

If your data carries different or extra fields (e.g. you skipped the M1
reduction, or ran without neutrino transport), override
`--keys`/`--vel-keys`/`--density-key` accordingly -- there is nothing
hard-coded elsewhere that depends on the *names*, only on `--vel-keys`
being 3 of `--keys` and `--density-key` being 1 of `--keys`.

Requesting a key that no snapshot file provides raises a `KeyError` at
start-up listing the names that *were* found, rather than silently writing
an all-NaN column.

### Performance options

| Flag | Default | Meaning |
|---|---|---|
| `--n-cpu` | `$SLURM_NTASKS_PER_NODE` if set, else `os.cpu_count()` | Worker processes for parallel tracer integration and file loading. |
| `--files-per-step` | `10` | Number of consecutive snapshots kept resident in shared memory at once. |
| `--max-tot-memory-gb` | -- | Alternative to `--files-per-step`. Picks the snapshot count automatically so total shared-memory use stays under this budget. The implied count is the budget divided by `per-snapshot field size x number of --keys`, so it shrinks as you add keys. Mutually exclusive with `--files-per-step`. |
| `--verbose` | off | Print per-file loading progress bars. |

`--files-per-step` (or its `--max-tot-memory-gb`-derived equivalent) must be
at least as large as the integrator's snapshot stencil (2 for the
trapezoid schemes, 4 for RK4) or the pipeline can't complete even a single
integration step per loaded chunk and will raise `ValueError`. Larger
values mean fewer, bigger snapshot-loading passes (less loading overhead,
more memory); see [Choosing options](#choosing-options-a-tuning-guide).

### Interpolator options

The interpolator reconstructs each field as a continuous function of
position from the snapshot's grid samples, so it can be evaluated at each
tracer's (off-grid) location.

| Flag | Default | Meaning |
|---|---|---|
| `--interpolator` | `pchip` | `pchip` (monotone cubic, `PchipInterpolator3D`) or `regular` (`RegularInterpolator3D`, wraps `scipy.interpolate.RegularGridInterpolator`). |
| `--interp-method` | `linear` | Method passed through to `RegularGridInterpolator` when `--interpolator=regular`: `linear`, `nearest`, `slinear`, `cubic`, `quintic`, or `pchip`. Ignored for `--interpolator=pchip`. |
| `--cache-size-gb` | auto | Max memory (GB) each PCHIP interpolator instance's internal cache of per-column 1-D interpolators may use before evicting the oldest entries. `--interpolator=pchip` only. If omitted, it's computed from `/proc/meminfo`'s free memory, `--n-cpu`, and the integrator's snapshot count (leaving an 80% safety margin) -- the computed value and its reasoning are printed at start-up. |
| `--rad-transform` | per `--format` | Coordinate transform applied to the radial axis before interpolation: `log` (`log10`, for a geometrically-spaced radial grid, as GR-Athena++ surface output uses), `asinh` (`arcsinh(r / scale)`, linear near the origin and logarithmic far out; requires `--rad-scale`), or `none` for a grid whose radial spacing is already linear, as AthenaK's is. |
| `--rad-scale` | -- | Lin-log transition radius (code units) for `--rad-transform=asinh`: the approximate radius where the `arcsinh` scaling switches from linear to logarithmic behaviour. Required with `asinh`, ignored otherwise. |

`pchip` (the default) guarantees the reconstructed field never overshoots
between grid samples -- important near steep gradients (shock fronts,
the ejecta's outer edge) where linear/cubic interpolation can produce
non-physical over/undershoots. `regular` with `--interp-method linear` is
faster and simpler, and a reasonable choice for smooth, well-resolved
fields or quick exploratory runs.

### Integrator options

| Flag | Default | Meaning |
|---|---|---|
| `--integrator` | `impl_trap` | `expl_trap` (explicit trapezoid), `impl_trap` (implicit/Crank-Nicolson trapezoid), or `rk4` (classical 4th-order Runge-Kutta with cubic time interpolation). |
| `--tol` | `1e-8` | Picard fixed-point convergence tolerance. `--integrator=impl_trap` only. |
| `--max-iter` | `20` | Max Picard iterations per step before giving up and using the latest iterate. `--integrator=impl_trap` only. |
| `--relax` | `1.0` | Picard relaxation factor in `(0, 1]`; values `< 1` damp oscillating iterations at the cost of slower convergence. `--integrator=impl_trap` only. |
| `--rk4-lagrange` | off (monotone PCHIP) | `--integrator=rk4` only: use cubic Lagrange time-interpolation for the mid-step velocity instead of the default monotone (Fritsch-Carlson PCHIP) blend. Lagrange is 4th-order accurate in smooth flow but can overshoot across a shock in time (e.g. a fast-moving ejecta front passing a grid point between snapshots); the default PCHIP blend trades a little accuracy for guaranteed no new extrema. |

`rk4` needs 4 consecutive snapshots per integration step (vs. 2 for the
trapezoid schemes) and is the most accurate option, especially with
widely-spaced or non-uniformly-spaced snapshots. `impl_trap` (the default)
is unconditionally stable and a solid default when snapshot spacing is
coarse relative to the flow's timescales; `expl_trap` is cheaper per step
but can become unstable for large timesteps near fast-changing flow (e.g.
close to the launch radius of a disk wind).

### Seeding: `volume` mode

Seeds tracers throughout a spherical shell, with one tracer per
`(r, theta, phi)` cell on a grid that's geometrically spaced in `r` and
equal-solid-angle in `(theta, phi)`. Geometric radial spacing makes `dr`
grow with `r`, so cell volume (`dV ~ r^2 dr`) grows as `r^3` outward, not
constant -- by design, since a typical homologous outflow's density falls
off as `rho ~ r^-3`, this `r^3` volume growth roughly cancels the density
falloff, giving tracers of roughly equal *mass* rather than equal volume.
Each tracer's represented mass is computed once at `--start-t` by
Gauss-Legendre-integrating `--density-key` over its actual cell (not
assumed from the grid alone), so this holds only approximately, and only
insofar as your flow actually resembles that `rho ~ r^-3` profile.

```bash
python run_pipeline.py --data-dir ... --output-dir ... --start-t ... --end-t ... \
    volume --r-min 300 --r-max 1000 --n-r 30 --n-th 15 --n-ph 30
```

| Flag | Default | Meaning |
|---|---|---|
| `--r-min`, `--r-max` (required) | -- | Radial range of the seeded shell. |
| `--n-r`, `--n-th`, `--n-ph` (required) | -- | Number of radial / polar / azimuthal bins -- total tracer count is `n_r * n_th * n_ph`. |
| `--phi-min-deg`, `--phi-max-deg` | `0`, `360` | Azimuthal range, in degrees. |
| `--theta-min-deg`, `--theta-max-deg` | `0`, `180` | Polar-angle range, in degrees (`0` = north pole, `180` = south pole). Restrict to seed one hemisphere, an equatorial band, etc. |
| `--n-quad` | `2` | Gauss-Legendre points per dimension (`n_quad**3` total per cell) used to integrate `--density-key` for each tracer's mass. `1` = single-point (cell-centre) approximation. |
| `--no-random-shift` | random shift **on** | By default each tracer is placed at a random point within its cell (not the cell centre) so that many tracers together sample the cell more representatively. Pass this to place every tracer exactly at its cell centre instead (useful for reproducible/deterministic test runs). |

### Seeding: `surface` mode

Seeds tracers on a fixed-radius spherical surface, injecting them at a
sequence of times ("slots") built automatically from the available
snapshot times between `--start-t` and `--end-t`. Each tracer's mass is
the mass flux `∫ ρ v_r dA dt` through its angular cell over its time slot,
integrated directly from the field data (not approximated from the
tracer's own trajectory) -- appropriate for surface-flux-based ejecta
estimates.

```bash
python run_pipeline.py --data-dir ... --output-dir ... --start-t ... --end-t ... \
    surface --r-surf 300 --n-th 15 --n-ph 30 --every-n-files 5
```

| Flag | Default | Meaning |
|---|---|---|
| `--r-surf` (required) | -- | Radius of the seeding surface. |
| `--n-th`, `--n-ph` (required) | -- | Number of polar / azimuthal bins -- one tracer per `(time slot, theta bin, phi bin)`. |
| `--phi-min-deg`, `--phi-max-deg`, `--theta-min-deg`, `--theta-max-deg` | `0`/`360`, `0`/`180` | Same meaning as in `volume` mode. |
| `--n-quad` | `3` | Gauss-Legendre points per dimension (`n_quad**2` total) for the surface mass-flux integral. |
| `--no-random-shift` | random shift **on** | By default each tracer's angular position is jittered within its cell and its injection time is drawn randomly from the snapshot times inside its slot (for smoother angular/temporal coverage across many tracers). Pass this to use bin-centre angles and the slot's first snapshot time instead. |
| `--every-n-files` | `1` | Build one time slot every Nth available snapshot between `--start-t` and `--end-t` (rather than one per snapshot). Increase this to seed fewer, more widely time-separated batches of tracers. |

Needs at least 2 snapshot times between `--start-t` and `--end-t` at the
chosen `--every-n-files` stride to form even one slot; it raises a clear
error if that's not satisfiable (narrow your time range, or lower
`--every-n-files`).

### Seeding: `volume-mc` and `surface-mc` modes

Same two geometries, but instead of one tracer per grid cell these draw a
fixed number of tracers at random, weighted by the mass. `volume-mc`
samples the shell with the density as weight, `surface-mc` samples the
`(theta, phi, t)` surface-time space with the mass flux magnitude
`|D v_r|` as weight. Every tracer then carries the *same* mass
`M_tot / --n-tracers`, and the tracer number density follows the mass
density. That under-resolves low-density regions on purpose: it is the
optimal way to spend a fixed tracer budget on tracking the global mass
flow.

In `surface-mc` an inflowing surface element (`v_r < 0`) is sampled with
the same probability as an outflowing one of equal magnitude, but its
tracers carry a **negative** mass, so they subtract from the net ejecta
just as their surface elements do. `M_tot` is the mass crossing the
sphere either way, and summing the signed tracer masses is an unbiased
estimate of the net crossing mass (this is standard importance sampling:
drawing at `p ∝ |w|` and weighting by `w/p` leaves `sign(w) M_tot/N`).

With `--adm-mass` the weight and the total use the conserved density
`D = sqrt(gamma) W rho` rather than `rho`, so the `mass` written per tracer
is the conserved rest mass. That choice is the file handler's, not the
seeder's -- see [What counts as mass](#what-counts-as-mass---adm-mass).

```bash
# --weight-grid native is what `auto` already picks for GR-Athena++ and
# AthenaK data; it is spelled out here to make the runs self-documenting.
python run_pipeline.py --data-dir ... --output-dir ... --start-t ... --end-t ... \
    --adm-mass 2.7 volume-mc --r-min 300 --r-max 1000 --n-tracers 5000 \
    --weight-grid native

python run_pipeline.py --data-dir ... --output-dir ... --start-t ... --end-t ... \
    --adm-mass 2.7 surface-mc --r-surf 300 --n-tracers 5000 --every-n-files 5 \
    --weight-grid native
```

| Flag | Default | Meaning |
|---|---|---|
| `--r-min`, `--r-max` (`volume-mc`, required) | -- | Radial range of the sampled shell. |
| `--r-surf` (`surface-mc`, required) | -- | Radius of the sampled surface. |
| `--n-tracers` (required) | -- | Number of tracers to draw. |
| `--phi-min-deg`, `--phi-max-deg`, `--theta-min-deg`, `--theta-max-deg` | `0`/`360`, `0`/`180` | Same meaning as in `volume` mode. |
| `--every-n-files` (`surface-mc`) | `1` | Sample the flux every Nth snapshot between `--start-t` and `--end-t`. Unlike in `surface` mode this does not change how many tracers you get, only how finely the flux is resolved in time. See [below](#--every-n-files-in-surface-mc). |
| `--weight-grid` | `auto` | Grid the sampling weights are built on: `native` uses the data's own grid with no interpolation at all, `helper` builds an interpolated one sized by `--cells-per-tracer`, `auto` takes native where the format provides it. See [`--weight-grid`](#where-the-sampling-weights-come-from---weight-grid). |
| `--cells-per-tracer` | `8` | Resolution of the *helper* weight grid (`--n-tracers` times this many cells, split over the axes so cells are roughly isotropic). Raise it to resolve a structured density/flux field better, at the price of more interpolations. Ignored under `--weight-grid native`, where the grid is whatever the data is. |
| `--weight-filter` | none | Zero the sampling weight where a field fails a cut, e.g. `--weight-filter 'T<1' 'r_0>0.1'`. Nothing is then seeded there, and the tracers carry the mass that passes the cut rather than the region's total. See [`--weight-filter`](#filtering-what-gets-seeded---weight-filter). |

The weight grid is only the sampling PDF and the mass normalisation: the
tracer positions themselves are drawn continuously (uniformly in volume,
resp. in solid angle) inside the drawn cell. Injection times are exact
snapshot times, since a tracer only activates on one.

#### `--every-n-files` in `surface-mc`

In `surface` mode this flag decides how many tracers you get, one per cell
per time slot. In `surface-mc` you always get `--n-tracers` of them, so it
does something else: it sets the time axis of the sampling grid. Four
consequences, in the order they usually matter.

- **How finely the flux is resolved in time.** Each sampled snapshot carries
  a trapezoidal `dt` taken from its neighbours, and a cell's weight is
  `flux * dt`. A stride of 5 makes one snapshot's flux stand for five
  snapshots' worth of ejecta. If `mdot` varies faster than that, the total is
  biased, not just noisier.
- **Which injection times exist.** A tracer activates only on a snapshot
  time, and it is injected at the exact time of the cell it was drawn from.
  With a stride of 5 the whole population sits on every 5th snapshot.
- **The angular resolution of the *helper* grid.** The cell budget
  `--cells-per-tracer` times `--n-tracers` is split over the sampled times,
  so sampling more times leaves fewer angular cells per time. Under
  `--weight-grid native` this does not apply, the angular grid being the
  data's own.
- **Cost, but less than it looks.** The flux loop still walks every snapshot
  in the range, chunk by chunk, and skips the ones not on the time axis. What
  a larger stride saves is the per-snapshot weight evaluation, not the file
  reading: on a GR-Athena++ surface run, about 1.4 s per snapshot on the
  helper grid against 4 ms on the native one.

So under `--weight-grid native` it is close to a pure accuracy knob and `1`
is a fine default. Under `helper` it is the main lever for making a long time
range affordable.

## Step 3: read the output

Every tracer becomes one file, `<output-dir>/tracer_NNNNNN.dat`
(`NNNNNN` is a 6-digit, zero-padded ID), a plain-text table:

```
# dV=35974213.06...; mass=3.90441377...e-05; status=active
#                    time                         x                         y                         z                     V_u_x ...
   1.4760000000000000e+04    9.7855884742544347e+01    2.2651893978032632e+02 ...
   1.4780000000000000e+04    9.7222327313242658e+01    2.2642132834699208e+02 ...
   ...
```

- **Line 1** (`#`-prefixed): tracer metadata as `key=value` pairs. Always
  includes `status` (`"active"` if the tracer was still inside the domain
  when integration stopped, `"done"` if it left the domain / hit a
  boundary and stopped early, `"failed"` if something went wrong writing
  its data, `"not started"` if it was never reached). `volume`-seeded
  tracers also carry `dV` (cell volume) and `mass`; `surface`-seeded
  tracers carry `mass` (flux-integrated). There is exactly one mass per
  tracer whatever the seeding mode, and `--adm-mass` changes what it is
  rather than adding a second -- see
  [What counts as mass](#what-counts-as-mass---adm-mass).
- **Line 2** (`#`-prefixed): column legend -- always `time x y z` followed
  by every field in `--keys`, in order.
- **Remaining lines**: one row per integration step, sorted by time,
  `%25.16e`-formatted.

To load these back into Python for analysis/plotting, use
`src/trajectory.py`'s `Trajectory` class:

```python
from src.trajectory import Trajectory

tr = Trajectory.from_ascii("data/tracers_out/tracer_000000.dat")
tr.props            # {'dV': ..., 'mass': ..., 'status': 'active', 'filename': ...}
tr.data["rho"]       # ndarray, density history
tr.data["time"]      # ndarray, matching time values
```

`analysis/` (see [Analysis scripts](#analysis-scripts) below) has worked,
argparse-driven examples of loading many `Trajectory` objects this way and
building plots/animations/mass budgets from them.

### What counts as mass: `--adm-mass`

Every tracer carries exactly one `mass`. What that mass *is* -- a rest-mass
integral or a conserved one -- is decided by the file handler, through the
`MassDensity` it builds (see [`src/mass.py`](src/mass.py)), and no seeder or
output writer knows which case it is in.

By default it is `--density-key`, which for every format shipped here is `rho`,
the rest-mass density. The conserved rest mass is not that integral but the
integral of the densitized `D = rho*W*sqrt(gamma)`, so a plain `rho` mass is
short by `W*sqrt(gamma)`, of order a few percent for merger ejecta and rising
with both velocity and depth in the potential. It is not a constant factor and
cannot be divided out afterwards.

Passing `--adm-mass M` (code units) switches the handler to `D`. Both factors
are reconstructed from data the snapshots already carry:

```
W            = -u_t / alpha          (exact where the shift is negligible)
sqrt(gamma)  = psi**6                (isotropic Schwarzschild, psi = 1 + M/2r)
alpha        = (1 - M/2r) / psi
-> D/rho     = (-u_t) * psi**7 / (1 - M/2r)
```

`u_t` (or whatever `--ut-key` names) must be in `--keys`; it is by default for
`reduced_surface`. The factor is applied wherever the mass is built: at every
cell of the weight grid in the `-mc` modes, at every quadrature point in
`volume` and `surface`, and at the tracer's own seed radius when the mass is
finished off at output time.

Checked against a GR-Athena++ BNS merger at `r = 400 M`, where the surface
dumps carry both `hydro.cons.D` and `hydro.aux.W`: the reconstructed
`sqrt(gamma)` was accurate to 0.008%, the reconstructed `W` matched the dumped
one to five decimals, and a 0.1 `M_sun` error in `--adm-mass` moves the result
by under 0.1%. Summing the densitized tracer masses over the tracers crossing
that sphere reproduced the simulation's own `D*V` flux integral to better than
1%, against 3.8% low for the plain `rho` mass.

The approximation degrades close to the remnant, where the shift stops being
negligible and the metric stops being Schwarzschild. If your snapshots carry
`D` itself, prefer pointing `--density-key` at it and ignoring all of this.

A format that needs something else entirely -- a Newtonian run, different
coordinates, a dumped metric to use instead of the analytic one -- overrides
`FileHandler.build_mass_density` and returns its own `MassDensity`. Nothing
else changes: the seeders call the same hook.

Trajectory files written before this consolidation carry a separate `mass_D`
header entry alongside a rho-based `mass`. `analysis/` still prefers `mass_D`
where it finds one, so old output keeps working.

### Where the sampling weights come from: `--weight-grid`

The `-mc` modes need the density (or the mass flux) over the seeding region to
build their sampling weights. Two ways to get it:

- `helper` lays a grid of its own, sized from `--cells-per-tracer`, and
  interpolates the fields onto it.
- `native` uses the data's own grid directly, with no interpolation anywhere.
- `auto` (the default) takes `native` where the format provides it and falls
  back to `helper` otherwise. `native` errors rather than falling back, so a
  run cannot silently change what it sampled.

Either way the region is exactly the one you gave. The helper grid is laid out
on your limits, and the native cells are cut at them: a cell `--r-max` or an
angular limit passes through keeps only the part inside, with its mass scaled
by that fraction of its measure, which is exact for the piecewise-constant
value a native cell holds. `surface-mc` puts its sphere at exactly `--r-surf`
and nothing is snapped, so **a `volume-mc` run to `--r-max R` and a
`surface-mc` run at `--r-surf R` share one boundary**: every parcel is in one
population or the other, never both and never neither.

On the native grid the sphere's flux comes from the cell it passes through,
measured at that cell's own sample radius `r_j` as `r_j**2 rho v_r` and carried
to `R` unchanged. That is a nearest-neighbour value in radius, but of
`r**2 rho v_r`, which a steady radial outflow conserves. Carrying `rho v_r`
instead and multiplying by `R**2` would be off by `(R/r_j)**2` -- up to one
radial cell's `q`, the same for every cell of a global grid, so it would not
average out. What remains is only the flow's departure from steady state
across half a cell.

Every format shipped here implements it: `reduced_surface` and `athenak` hold
one global grid, and `athdf_spherical` pools the cells of every meshblock. A
format that cannot enumerate its cells falls back to `helper` under `auto`.

`native` is both more accurate and **faster**, which is the opposite of what
the cell counts suggest. The helper route's cost is dominated by building an
interpolator per sampled snapshot, not by evaluating it, and the native route
has none to build. Measured on a GR-Athena++ BNS merger:

| mode | grid | cells | wall |
|---|---|---|---|
| `volume-mc` | helper | 13x25x50 | 61 s |
| `volume-mc` | native | 213x128x256 | **54 s** |
| `surface-mc` | helper | 34x68x7 | 169 s |
| `surface-mc` | native | 128x256x7 | **88 s** |

with the sampled totals agreeing to 0.4% either way. `--cells-per-tracer` is
ignored under `native`, since the grid is whatever the data is.

Note that refining the weight grid reduces bias in the weights, not the
sampling variance -- that is set by `--n-tracers` and stays `sqrt(N)` however
fine the grid. What `native` buys is a more accurate total and faithful sharp
features, not a smoother `mdot`.

To give a new format a native grid, implement `native_cell_weights` on its
`FileHandler` (see [docs/writing_a_reader.md](docs/writing_a_reader.md)). It
hands back the coordinate bounds of its cells one by one, plus the mass in
each, so it is optional, it needs no single grid, and the only real
requirement is that the cells **tile the region without overlapping**. Its
companion `native_cell_values` returns a raw field on those same cells, which
is what keeps `--weight-filter` free on the native grid. What a
mass is comes from the same class, via `build_mass_density` -- see
[What counts as mass](#what-counts-as-mass---adm-mass).

### Filtering what gets seeded: `--weight-filter`

Sometimes only part of the mass is worth tracing. Material still too hot to
have frozen out, a low-`ye` component, a wind picked out by its entropy: the
weight can be cut on the fields themselves, so the sampler never places a
tracer there.

```bash
# only material cooler than 1 MeV, and only above Ye = 0.1
python run_pipeline.py --data-dir ... --output-dir ... --start-t ... --end-t ... \
    volume-mc --r-min 300 --r-max 1000 --n-tracers 5000 \
    --weight-filter 'T<1' 'r_0>0.1'
```

Each cut is `field`, a comparison (`<`, `<=`, `>`, `>=`, `==`, `!=`) and a
number. Several cuts on the same field all have to hold. Every field named
must be in `--keys`, which is checked before a file is opened.

A cut multiplies that cell's sampling weight by 0 or 1, so cells are kept or
dropped whole, and the edge of the seeded region follows the cell boundaries
of the weight grid rather than the cut value exactly. On `--weight-grid native`
those are the data's own cells, which is as sharp as the data gets. In
`surface-mc` the cut is re-read at every sampled snapshot, so a region that
cools below the threshold starts being seeded when it does.

`M_tot` in the summary line, and so the mass each tracer carries, is then the
mass that **passes** the cut. The line says how much was filtered out:

```
Sampled 5000 tracers of mass 1.7241e-06 from a 213x128x256 native weight grid
(total mass 8.6203e-03 of 8.6203e-03 on the grid, weighted by rho (rest-mass
density, not densitized), filtering out 43.1% of it on T).
```

Reading the cut fields is free on the native grid, where the values are taken
straight off the cells. On the helper grid it costs one more interpolator build
per sampled snapshot, roughly +70% on the seeding step.

Anything the command line cannot spell -- a smooth taper instead of a hard
cut, or a condition on two fields at once -- is a `dict[str, Callable]` passed
straight to the seeder in Python:

```python
from src.seeds import spherical_by_volume_mc

tracers = spherical_by_volume_mc(
    r_min=300, r_max=1000, n_tracers=5000, start_t=t0,
    file_handler=fh, integrator=integrator, vel_keys=vel_keys,
    # any factor per cell, not just 0 or 1
    weight_filters={'T': lambda T: np.clip(2.0 - T, 0.0, 1.0)},
)
```

Each callable is handed its field's values on the cells, shape `(n_cells,)`,
and returns one factor per cell. The factors of all named fields multiply
together.

## Running on a cluster (SLURM)

`batch.sub` is a template SLURM job:

```bash
#!/bin/bash
#SBATCH -A PHY23001
#SBATCH --partition small
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=1
#SBATCH -N 1 --ntasks-per-node=56
#SBATCH -J tracers
#SBATCH -o tracers.out
#SBATCH -e tracers.err
#SBATCH --mem=0
#SBATCH --exclusive

python run_pipeline.py \
    --data-dir data/transformed \
    --output-dir data/tracers_out \
    --start-t 11600 --end-t 0 \
    --n-cpu "$SLURM_NTASKS_PER_NODE" \
    volume --r-min 300 --r-max 1000 --n-r 30 --n-th 15 --n-ph 30
```

Edit the account/partition/walltime `#SBATCH` lines for your allocation,
edit the `run_pipeline.py` arguments for your run (see
[Step 2](#step-2-run-the-pipeline) above for what each one means), then
submit with:

```bash
sbatch batch.sub
```

`--n-cpu "$SLURM_NTASKS_PER_NODE"` passes SLURM's per-node task count
straight through, so the job automatically uses however many tasks you
requested via `--ntasks-per-node` -- keep those two in sync. `--mem=0
--exclusive` requests the whole node's memory, which matters because
snapshot data is staged in `/dev/shm` (shared memory), not process heap
(see [Troubleshooting](#troubleshooting) if you hit shared-memory limits).

## Choosing options: a tuning guide

- **Which integrator?** Start with the default, `impl_trap` -- it's
  unconditionally stable, so it won't silently blow up if your snapshot
  cadence is coarse relative to the flow. Reach for `rk4` when you need
  higher time-accuracy (e.g. validating against an analytic solution, or
  snapshots that are widely/non-uniformly spaced) and can afford loading
  4 snapshots per step instead of 2. `expl_trap` is mainly useful as a
  cheap baseline for comparison (see `examples/trapezoid_comparison.py`).
- **Which interpolator?** `pchip` (default) for anything with sharp
  features (shocks, ejecta edges); `regular` with `--interp-method linear`
  for speed on smooth fields, or when you need a specific
  `RegularGridInterpolator` method for comparison.
- **`volume` and `surface` seeding are complementary, not alternatives.**
  The usual goal is to capture *all* the matter that ends up ejected by
  integrating backward from a late time. At that late `--start-t`, the
  ejecta splits into two populations: matter still inside the simulation
  domain (out to some outer radius `R`), and matter that has already left
  it (crossed `R` at some earlier time and kept going). `volume` seeded
  out to `--r-max R` catches the first population, once, at `--start-t`.
  `surface` seeded at `--r-surf R`, with a time range covering everything
  *before* `--start-t`, catches the second population at the moment each
  parcel crossed `R` -- which is exactly the information `volume` seeding
  can't see, since that matter is no longer inside `[--r-min, --r-max]` by
  `--start-t` to be caught by it. Run both against the same `R`, then
  integrate each backward from its own seed time; together they cover the
  ejecta without double-counting (a given parcel is either still inside at
  `--start-t`, or already crossed -- never both). The two share `R` exactly,
  not to within a cell: neither snaps, and native cells are cut at the limit
  (see [`--weight-grid`](#where-the-sampling-weights-come-from---weight-grid)).
  See
  [Matching cell shapes and masses between `volume` and `surface`](#matching-cell-shapes-and-masses-between-volume-and-surface)
  for how to size the two grids so they combine into one consistent
  tracer population.
- **`--n-cpu` / `--files-per-step`**: more snapshots resident at once
  means fewer, larger I/O passes but more `/dev/shm` usage (which scales
  with `--files-per-step x number of --keys x per-snapshot field size`,
  independent of `--n-cpu`); more worker processes speeds up the
  per-snapshot interpolation/integration work but each worker holds its
  own interpolator cache (`--cache-size-gb`), so total memory scales with
  `--n-cpu` too. If you hit the `MemoryError`/`ValueError` described in
  [Troubleshooting](#troubleshooting), turn one of these down before
  reaching for a bigger node.
- **`--cache-size-gb`**: the auto-computed default is deliberately
  conservative (80% of free memory divided across all concurrent
  interpolators). Raising it reduces cache evictions (and the
  scipy-`PchipInterpolator`-object reconstruction cost that comes with
  them) at the cost of using more memory per worker -- worth doing
  explicitly if you have headroom and are seeding a very large number of
  tracers spread across many grid columns.

### Matching cell shapes and masses between `volume` and `surface`

Since `volume` and `surface` are meant to be run together against the
same transition radius `R` (see above), it's worth sizing both grids so
the combined tracer population is reasonably uniform: roughly cubical
cells (no long, thin slivers that make one direction of the flow
under-resolved relative to the others), and roughly similar mass per
tracer whether it came from the `volume` or the `surface` half of the run
(so downstream analysis isn't implicitly weighting one population more
than the other).

**Cubical `volume` cells.** `--n-r`'s bins are geometrically spaced, so
the fractional radial step is constant:

```
Δ = ln(r_max / r_min) / n_r
```

and a cell's physical radial extent is `dr = r * Δ`. `--n-th`'s bins are
equally spaced in `cos(theta)`, so the angular step at a given `theta` is
`dtheta = (cos(theta_min) - cos(theta_max)) / (n_th * sin(theta))` -- its
physical (arc-length) extent is `r * dtheta`. `--n-ph`'s bins are equally
spaced in `phi`, giving physical extent `r * sin(theta) * dphi` with
`dphi = (phi_max - phi_min) / n_ph`. For a cell at a reference colatitude
`theta_ref` (pick the middle of your `--theta-min-deg`/`--theta-max-deg`
range, or 90 degrees/the equator for a disk-like geometry) to come out
roughly cubical, all three physical extents should match `r * Δ`, which
gives:

```
n_th ~ (cos(theta_min) - cos(theta_max)) / (Δ * sin(theta_ref))
n_ph ~ sin(theta_ref) * (phi_max - phi_min) / Δ
```

(Note this can't be made exact at every latitude simultaneously with a
single global `--n-th`/`--n-ph` -- the equal-`cos(theta)` spacing that
keeps solid angle per cell constant necessarily makes cells anisotropic
near the poles, since `dtheta` grows while `sin(theta)` shrinks there.
The formulas above match cell shape at your chosen `theta_ref`, which is
the right thing to optimize for if your ejecta of interest is
concentrated away from the poles, as it usually is for disk-wind/BNS
merger geometries.)

*Worked example:* full sphere (`theta_min=0`, `theta_max=180`, so
`cos(theta_min) - cos(theta_max) = 2`; `phi_min=0`, `phi_max=360`, so
`phi_max - phi_min = 2*pi`), `theta_ref = 90` degrees (`sin = 1`),
`--r-min 300 --r-max 1000 --n-r 30` gives `Δ = ln(1000/300)/30 ≈ 0.0401`.
Then `n_th ~ 2 / 0.0401 ≈ 50` and `n_ph ~ 2*pi / 0.0401 ≈ 157`.

**Matching `surface` tracer mass to `volume` tracer mass.** A `volume`
cell at the transition radius `R` with side length `ell = R * Δ` (from
above) represents mass `dm_volume ~ rho * ell**3`. A `surface` tracer's
angular cell, built with the *same* `Δ` so its footprint matches the
`volume` cell's angular footprint (i.e. `--n-th`/`--n-ph` computed the
same way, evaluated at `r = R`), has area `dA ~ ell**2`; over one time
slot of duration `dt_slot` it represents mass
`dm_surface ~ rho * v_r * ell**2 * dt_slot`, where `v_r` is the typical
radial velocity of material crossing `R`. Setting `dm_surface ~ dm_volume`
gives:

```
dt_slot ~ ell / v_r
```

i.e. pick the surface time-slot duration so that material crossing `R`
at its typical radial velocity covers about one cell-width `ell` per
slot -- the same "cubical" intuition as the spatial grid, applied to
time. Since slots are built from `--every-n-files` (an integer stride
over your snapshot cadence `dt_file`), solve for the stride:

```
--every-n-files ~ round(dt_slot / dt_file) = round(ell / (v_r * dt_file))
```

(clamped to at least `1`). *Continuing the worked example:* `ell = R*Δ`
with `R = 1000` (matching `--r-max` above) and `Δ ≈ 0.0401` gives
`ell ≈ 40.1`; for ejecta crossing `R` at a typical `v_r ≈ 0.1` (both in
geometric units, see [Units](#units)) and a snapshot cadence of
`dt_file = 20`, `dt_slot ≈ 40.1/0.1 = 401`, so
`--every-n-files ≈ round(401/20) ≈ 20`.

Treat all of the above as a starting point, not a hard rule: measure your
own simulation's characteristic `v_r` near `R` and its actual snapshot
cadence, and adjust from there.

## Units

`run_pipeline.py` doesn't do any unit conversion -- every length
(`--r-min`, `--r-max`, `--r-surf`) and time (`--start-t`, `--end-t`) is in
whatever units the radius and time recorded in your snapshot files use,
which for both GR-Athena++ and AthenaK output is geometric units with
`G = c = M_sun = 1`. `analysis/` (below) shows how to convert output back
to physical units (km, ms, GK, CGS density) for plotting, via
`tabulatedEOS.unit_system.GeometricSolar`.

## Analysis scripts

`analysis/` has four argparse-driven post-processing scripts for real
tracer output (as opposed to `examples/`'s synthetic-velocity-field
validation scripts): plotting a random sample of tracer histories vs
time, rendering a 3-D animation, checking the `volume`/`surface`
mass-budget consistency discussed above, and a mass-weighted histogram
panel of standard summary quantities (peak temperature, `Ye` and
expansion timescale at a reference temperature, final angle/radius/
velocity, including the asymptotic velocity implied by the geodesic
criterion when `u_t` is available). See `analysis/README.md` for full
option documentation for each.

`test.py` at the repo root is an old, hand-edited script that predates
`run_pipeline.py` and served the same purpose (invoke the pipeline with a
fixed set of parameters) before this CLI existed; it's kept only for
reference; prefer `run_pipeline.py` for anything new.

## How it fits together

```
run_pipeline.py                     (CLI entry point)
        |
        +-- src/file.py             FileHandler (abstract): chunked
        |     |                     shared-memory snapshot loading, common
        |     |                     to every data source
        |     +-- src/reduced_surface.py  --format reduced_surface
        |     +-- src/athenak.py          --format athenak
        |     +-- src/athdf_spherical.py   --format athdf (spherical-grid athdf only)
        |     +-- src/gra_surface.py      raw GR-Athena++, not on the CLI
        |
        +-- src/interpolators/      PchipInterpolator3D, RegularInterpolator3D,
        |                           MeshblockPchipInterpolator (octree block
        |                           lookup for --format athdf_spherical)
        |                           (+ CartesianToSpherical wrapper, radial
        |                           log/asinh coordinate transforms,
        |                           shared-memory-backed field data)
        |
        +-- src/integrators/        ExplicitTrapezoid, ImplicitTrapezoid, RK4
        |                           (advance one tracer batch by one timestep,
        |                           given interpolator(s) for the surrounding
        |                           snapshot(s))
        |
        +-- src/seeds.py            spherical_by_volume, spherical_surface_by_area
        |                           (build initial tracer positions/times/masses)
        |
        +-- src/tracers.py          Tracer, Tracers: drive the chunk-by-chunk
                                    integration loop across a multiprocessing.Pool,
                                    write each Tracer's history to ASCII
```

`src/trajectory.py`'s `Trajectory` is the read-side counterpart used by the
analysis scripts.

`transform_files.py` sits upstream of all of this and is specific to
GR-Athena++ surface output (raw `.hdf5` -> transformed `.hdf5`); see
[docs/formats/gr_athena.md](docs/formats/gr_athena.md).

## Supporting a new data format

Reading a new code's output means implementing one `FileHandler` subclass
(four methods) and adding an entry to `FORMATS` in `run_pipeline.py`.
Seeding, integration, interpolation and output need no changes.

[docs/writing_a_reader.md](docs/writing_a_reader.md) is the guide: the
contract, the conventions that aren't visible in the method signatures, how
to handle non-uniform grids and polar/periodic ghost zones, and a checklist
of the things that fail silently if you get them wrong.

## Tests and examples

- `tests/` -- fast (< 1s), self-contained pytest unit tests for the
  integrators, interpolators, and seeding helpers, using synthetic data
  (no real snapshots, no MPI). Run with `python -m pytest tests/ -v`. See
  `tests/README.md` for the full breakdown.
- `examples/` -- slower, exploratory end-to-end scripts that run the full
  pipeline against synthetic analytic velocity fields (a Sedov-Taylor
  blast wave and a disk-wind ejecta analogue) to compare integrators and
  interpolators against each other and, where possible, an exact
  reference solution. Useful for building intuition about the tuning
  choices above without needing real simulation data. See
  `examples/README.md`.

## Troubleshooting

- **`MemoryError: Required total memory for loading data (... GB) exceeds
  90% of available shared memory`**: raised up front, before any snapshots
  are loaded. Lower `--files-per-step` (or `--max-tot-memory-gb`), lower
  `--n-cpu`, trim `--keys` to only what you need, or free up `/dev/shm`
  (check with `df -h /dev/shm`) -- e.g. from a previous crashed run that
  didn't clean up (see the next point).
- **Leftover shared memory after a crash**: `FileHandler` registers an
  `atexit` cleanup handler and `ReducedSurfaceFileHandler` also catches
  `SIGINT` to free shared memory on Ctrl-C, but a hard crash (`SIGKILL`,
  OOM-killer) can still leave segments behind in `/dev/shm`. Check
  `ls /dev/shm` after a failed run; stale segments are named like
  `psm_<random>` and can be removed with `rm /dev/shm/psm_...` once you've
  confirmed no other run is using them.
- **`ValueError: n_files_per_step (...) must be at least n_snap (...)`**:
  `--files-per-step` (or the value implied by `--max-tot-memory-gb`) is
  smaller than the integrator's snapshot stencil (2 for the trapezoid
  schemes, 4 for `rk4`). Raise `--files-per-step`, or raise
  `--max-tot-memory-gb`'s implied snapshot count by giving it a bigger
  budget or asking for fewer `--keys`.
- **`WARNING: ... samples non-finite or |value| > X ... replaced with 0.0`**:
  a field in your snapshots carries values it cannot mean -- typically a
  quantity derived as a ratio, which blows up wherever its denominator
  vanishes. The reader substitutes zero there and carries on. Check the
  reported fraction against where you expect that field to be meaningful;
  if it is being replaced somewhere it matters, the dump needs fixing, not
  the pipeline.
- **`KeyError: Not every requested key is available at every snapshot time
  in ...`**: a `--keys` entry is missing from at least one snapshot. The
  message lists the offending times and the keys that *were* found
  anywhere. If it's missing everywhere, it's a typo -- check your format's
  field table (linked from [Step 1](#step-1-point-it-at-your-data)). If
  it's missing from only some times, that variable was dumped at a
  different cadence than the rest; either drop it from `--keys` or point
  `--data-dir` at only the times that carry everything.
- **`ValueError: Some seed times do not coincide with available file
  times`**: seeding always happens exactly at a snapshot time.
  `spherical_by_volume` uses `--start-t` directly (not rounded to the
  nearest snapshot) -- if it doesn't exactly match one of your snapshot
  times, seeding fails. Use one of the printed/known snapshot times.
- **Resource-tracker "leaked shared_memory object" warnings at exit**:
  should not occur in normal operation (worker pools shut down gracefully
  and interpolators release their shared-memory handles on unload) -- if
  you see these, it usually means a run was interrupted uncleanly; check
  the previous two points.
