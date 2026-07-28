# Analysis

Post-processing scripts that operate on *real* tracer output (from
`run_pipeline.py`), not the synthetic-velocity-field validation scripts in
`examples/`. Each one is a standalone, fully argparse-driven CLI tool --
run `--help` on any of them for the authoritative option list.

```bash
PYTHONPATH=. python analysis/plot_trajectories.py --tracer-dirs ... --output trajectories.png
PYTHONPATH=. python analysis/animate.py           --tracer-dirs ... --output animation.mp4
PYTHONPATH=. python analysis/mass_conservation.py --volume-dirs ... --surface-dirs ... --r-check ...
PYTHONPATH=. python analysis/histograms.py         --tracer-dirs ... --output histograms.png
```

`PYTHONPATH=.` (repo root) is needed so `analysis._common` -- the shared
loading/units helper the four scripts import -- resolves; it's not meant
to be run standalone. If you've run `pip install -e .` (see the root
`README.md`), the package is on your import path permanently and none of
these scripts need `PYTHONPATH` set at all, from any directory.

## Requirements

Beyond the core pipeline's dependencies (see the root `README.md`):

- `matplotlib` for all four scripts.
- **`tabulatedEOS`** (external sibling package, not distributed with this
  repo -- put it on `PYTHONPATH`) for physical-unit conversion (length,
  time, temperature, density). All four scripts need it.
- `ffmpeg` on `PATH`, for `animate.py` only.
- **`surface`** (external sibling package, optional) for
  `mass_conservation.py`'s `--raw-sim-dir` ground-truth cross-check only;
  everything else works without it.

## `plot_trajectories.py`

Randomly samples `--n-sample` tracers and plots each requested `--fields`
entry vs time, one line per tracer, colour-coded by its average
`--color-by` value. Good first look at a new run: are trajectories smooth,
do fields evolve as expected, are there obvious outliers? Available
fields (see `FIELD_REGISTRY` in the script): `r`, `theta`, `phi`,
`rho_r3` (`rho*r**3`, a mass proxy -- see the root README's "Units"
section for why this comes out in `M_sun` directly from raw code-unit
`rho`/`r`), `s`, `Ye`, `T`, `rho`, `v` (velocity magnitude), `u_t`.

## `animate.py`

Renders one frame per snapshot time (3-D scatter of tracer positions,
colour-coded by `--color-by`, default `r_0` = `Ye`) in parallel across
`--frame-workers` processes, then encodes them to mp4 with `ffmpeg`.
Frames are rendered into a temporary directory that's cleaned up
afterward by default -- pass `--frames-dir` or `--keep-frames` to keep
them (e.g. for debugging a bad frame). `--drop-below-r` (code units,
default 200) discards tracers that ever come within that radius of the
origin, filtering out atmosphere/near-remnant contamination that would
otherwise dominate the colour scale or clutter the animation; set it to
`0` to disable.

## `mass_conservation.py`

Checks tracer-derived cumulative ejected mass against a raw-simulation
ground truth -- **this ground-truth comparison (`--raw-sim-dir`) is the
actual validation this script performs.** `volume`- and `surface`-seeded
tracers represent two genuinely distinct, non-overlapping parts of the
ejecta (see the root `README.md`'s "volume and surface seeding are
complementary" section): `volume` catches whatever is still inside the
domain at the shared reference time, `surface` catches whatever already
crossed `--r-check` earlier. They are **not** expected to match each
other -- there's nothing to check by comparing them alone. What's
meaningful is their *sum*, checked against an independent measurement.
Without `--raw-sim-dir`, the script has no ground truth to compare
against and just plots the combined/surface-only breakdown for
inspection, not a pass/fail check.

Both populations are advected *backward* from a late seed time toward the
merger epoch, so every tracer's radius decreases going backward in time
-- meaning every tracer sweeps downward through any given `--r-check` as
long as `--r-check` is smaller than its own seed radius. Takes
separately-loaded `--volume-dirs` and `--surface-dirs` populations, and
plots the combined tracer mass currently at radius >= `--r-check` (and
its time derivative) -- the tracer-derived cumulative mass/rate that
crossed `--r-check` by a given time, in exactly the sense a raw
simulation's own surface-flux diagnostic at that radius measures it.

**`--r-check` is an observation radius, not a seeding radius** -- see
[Choosing `--r-check`](#choosing---r-check) below; getting this backward
(e.g. setting it equal to `--r-max`/`--r-surf`) silently makes the
`volume` contribution ~0 always, since volume tracers are seeded *at*
`--r-max` and only move to smaller radii going backward from there.

`--raw-sim-dir` is the path to the *raw*, untransformed simulation output
-- the same directory `transform_files.py` reads its input from (often
called something like `data/`, as distinct from the `transformed/`
directory `run_pipeline.py` reads). Given this, the script computes an
independent ground-truth mass flux/total directly from the simulation's
own surface diagnostics, via the external `surface` package (guarded
import -- only attempted if `--raw-sim-dir` is given, so the script still
runs without that package installed if you skip this). This is a
genuine cross-check: it catches systematic errors that would affect
*both* tracer populations identically (e.g. a wrong density field, or a
seeding bug that's wrong for both `volume` and `surface` in the same
way), which comparing the two tracer populations against each other never
could.

`--irad` (which raw-diagnostic radius index to use) defaults to
auto-inferred from `--r-check`, by reading a sample raw file's radius
grid (`coordinates/<key>/R`, the same layout `transform_files.py`
expects) and picking the closest match -- pass it explicitly only if that
guess is wrong or you want a specific index regardless of `--r-check`.

`--min-innermost-r` is an optional sanity filter (disabled by default):
drop tracers whose innermost (earliest-time) radius never gets below the
given value, e.g. to exclude tracers that didn't integrate back far
enough to reach a reference epoch you care about.

### Choosing `--r-check`

Pick whatever radius you're curious about the cumulative crossing
mass/flux at -- e.g. wherever a raw-simulation surface diagnostic exists,
so `--raw-sim-dir` has something to compare against. It should generally
be **smaller** than both seeding runs' `--r-max`/`--r-surf` (often close
to or inside `--r-min`): only then do tracers spend part of the observed
backward-integration window above it and part below, showing the time
dependence this script is meant to visualize. If `--raw-sim-dir` is given
and its surface radius doesn't match `--r-check` to within 0.1%, the
script prints a warning -- the overlay would otherwise silently be
comparing against the wrong radius.

## `histograms.py`

Mass-weighted histograms (`Δm` per bin, not tracer count) of the standard
per-tracer summary quantities used to characterize ejecta composition and
kinematics:

| Panel | What it shows |
|---|---|
| `Tmax` | Peak temperature reached (GK). |
| `Ye_ref` | Electron fraction at the reference ("NSE dropout") temperature `--t-ref-gk` (default 5 GK) -- the standard proxy for a tracer's final nucleosynthesis composition, since weak rates freeze out around there. |
| `s_ref` | Entropy at the same reference temperature. |
| `tau_ref` | Expansion timescale (`rho / |drho/dt|`, ms) at the same reference temperature. Together, `Ye_ref`/`s_ref`/`tau_ref` are the standard triplet of parameters characterizing r-process nucleosynthesis outcome. |
| `theta_final`, `phi_final` | Angular position at the tracer's final recorded time -- where the ejecta ends up. |
| `r_final` | Radius at the final recorded time. |
| `v_final` | Final coordinate speed `|v|`, overlaid with the *asymptotic* velocity implied by two different conserved-energy criteria: geodesic (`-u_t`, gravity only) and, if `hu_t` is present, Bernoulli (`-h*u_t`, also lets thermal/internal energy unbind or accelerate a tracer). Each curve's legend entry reports the mass fraction it represents, since a criterion that leaves most tracers bound will produce a much smaller-looking curve even where its shape is otherwise unremarkable. |

Tracers that never reach `--t-ref-gk` are excluded from the `*_ref`
panels (and reported, both as a tracer count and, for the velocity panel,
as a mass fraction). `v_inf = sqrt(1 - 1/W_inf**2)` wherever the relevant
`W_inf` (`-u_t` or `-h*u_t`) exceeds 1 (unbound under that criterion), NaN
(excluded) otherwise. The two criteria can disagree substantially --
thermal/magnetic energy can unbind a tracer that's bound on the
pure-geodesic criterion alone -- and neither necessarily matches
`v_final`: that's limited by how far the simulation domain/integration
actually followed the tracer, while `v_inf` is the exact terminal value
implied by energy conservation regardless of that (generally *smaller*
than `v_final` for a tracer still deep in the potential well, since it
hasn't yet paid the deceleration cost of climbing the rest of the way
out).

Select a subset with `--panels` (default: all of the above).

**Other panels worth adding, not yet implemented:** a 2-D `Ye`-`s`
histogram (or `theta` vs. `Ye`, to see equatorial/polar composition
differences directly) instead of two separate 1-D ones; a mass-weighted
histogram of injection/seed time (particularly informative for the
`surface` population, showing *when* mass was ejected, complementing
`theta_final`/`phi_final`'s *where*); a `|dYe/dt|` vs. position heatmap
(see the old `t_in_hist.py`, superseded by this directory, for a rough
version) to locate where composition is still changing fastest.

## Superseded scripts

These four replace the root-level `plot_test.py`, `animate_tracer.py`,
`t_in_hist.py`, and the tracer-vs-tracer-surface part of
`tracer_surface_analysis.py`, which had hard-coded paths/parameters and
have been removed. `tracer_surface_analysis.py`'s raw-surface-diagnostic
logic lives on, generalized, in `mass_conservation.py`'s `--raw-sim-dir`
path.
