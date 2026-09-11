# AthenaK spherical-grid output (`--format athenak`)

Spherical-grid output from [AthenaK](https://github.com/IAS-Astrophysics/athenak),
written as legacy binary VTK. `run_pipeline.py` reads these files directly
-- there is no transform step.

## Contents

- [File layout](#file-layout)
- [Running the pipeline](#running-the-pipeline)
- [Grid conventions](#grid-conventions)
- [Fields](#fields)
- [Caveats](#caveats)

## File layout

AthenaK writes **one variable per file**, so a single snapshot is spread
across several files:

```
bhns.r=150.00-4096.00.mhd_w_d.00305.vtk      # dens
bhns.r=150.00-4096.00.mhd_t.00305.vtk        # temperature
bhns.r=150.00-4096.00.mhd_w_s.00305.vtk      # s_00  (Ye)
bhns.r=150.00-4096.00.u_t.00305.vtk          # u_t
bhns.r=150.00-4096.00.rad_m1_e.00305.vtk     # e:0 .. e:3
bhns.r=150.00-4096.00.rad_m1_absF.00305.vtk  # |F|:0 .. |F|:3
```

The handler groups files by the `TIME` record in their headers, not by
filename, so the naming scheme doesn't matter and a variable can move
between files without any change here. Each file's contents are discovered
from its own `SCALARS` records.

Each file is a `STRUCTURED_GRID` whose `POINTS` are spherical
`(r, theta, phi)` triples rather than Cartesian coordinates, big-endian
`float32`, with `r` varying fastest and `phi` slowest:

```
# vtk DataFile Version 3.0
# AthenaK data at time=6100 cycle=97600 nradii=128 rmin=150 rmax=4096 xc=0 yc=0 zc=0
BINARY
DATASET STRUCTURED_GRID
DIMENSIONS 128 128 256          # (n_r, n_theta, n_phi)
POINTS 4194304 float
FIELD FieldData 3               # TIME, CYCLE, RADII
POINT_DATA 4194304
SCALARS weights float 1         # r^2 dmu dphi quadrature weights (unused)
SCALARS dens float 1
```

Only headers are read at parse time; bulk blocks are seeked over, and each
field is read by a single seek when its snapshot is loaded.

## Running the pipeline

```bash
python run_pipeline.py --format athenak \
    --data-dir /path/to/vtk --output-dir data/tracers_out \
    --start-t 6120 --end-t 6080 --adm-mass 2.7 \
    volume-mc --r-min 300 --r-max 1000 --n-tracers 20000 --weight-grid native
```

This format keeps its data on one global spherical grid and implements
`native_cell_weights`, so `--weight-grid native` builds the sampling weights
on the dumped samples themselves, with no interpolation and no helper grid.
`auto` (the default) picks it here anyway; spelling it out makes a run
self-documenting and turns a missing implementation into an error rather than
a silent fallback.

Both polar conventions described above are handled. The edges are built in
whichever coordinate the axis is actually uniform in -- `theta` or
`mu = cos(theta)` -- and only then converted, and a node-centred axis, whose
first and last samples sit *on* the poles and so own half a cell, is handled
by clipping the edges to the physical range. The check that matters is that
the cell measures still sum to `4 pi`, which `tests/test_athenak.py` asserts
for both conventions.

Region limits are cut exactly: `--r-max` and the angular limits clip the
cells they pass through, and `surface-mc` puts its sphere at exactly
`--r-surf`, taking the flux from the shell whose cell contains it. A volume
and a surface seeded at the same radius therefore share one boundary -- see
[the README](../../README.md#where-the-sampling-weights-come-from---weight-grid).
The grid-based `volume` and `surface` modes work unchanged and ignore all of
this.

`--format athenak` sets these defaults:

| Option | Default for this format |
|---|---|
| `--file-pattern` | `*.vtk` |
| `--rad-transform` | `log` (the radial grid is geometrically spaced) |
| `--keys` | `V_u_x V_u_y V_u_z T u_t rho r_0 F_nue F_anue F_nux eps_nue eps_anue eps_nux` |
| `--heavy-neutrinos` | `sum` (see below) |

See the [main README](../../README.md#step-2-run-the-pipeline) for every
other option.

### The fourth neutrino species

AthenaK's M1 radiation evolves 4 species, `[nue, anue, nux, anux]`;
GR-Athena++ evolves only 3, with `nux` already lumping all heavy leptons
together at the evolution level. `--heavy-neutrinos` (athenak format only)
controls how the mismatch is resolved, via `_build_key_specs` in
`src/athenak.py`:

| Mode | Behaviour |
|---|---|
| `sum` *(default)* | `nux`+`anux` folded into one GRA-style `nux`. `F_nux` (extensive) is summed; `eps_nux` (intensive) is a number-flux-weighted average instead, so a vacuum species can't poison it. |
| `drop` | `eps_nux`/`F_nux`/`eps_anux`/`F_anux` are not resolved at all -- omit them from `--keys` under this mode. |
| `separate` | All 4 species kept distinct: `eps_nue/anue/nux/anux`, `F_nue/anue/nux/anux`. |

`sum` is the default so that, with the default `--keys` above, an AthenaK
run produces the same 3-species key set as `reduced_surface`, making the
two data sources directly comparable through the rest of the pipeline
without touching `analysis/`.

## Grid conventions

`r` is **geometrically spaced** from `rmin` to `rmax` inclusive, hence
`--rad-transform log`. Older dumps were linear in `r`; pass
`--rad-transform none` for those. Either way the radial axis may be
non-uniform, so this is an accuracy choice rather than a correctness one.
`phi` is uniform over `[0, 2*pi)`, and may start at `0` or at `dphi/2` --
it makes no difference, `phi` is periodic and the ghost zones cover either
seam.

The **polar axis** is the one that matters, and the reader detects its
convention per dataset rather than assuming one, because AthenaK may write
either of these:

| Convention | Polar nodes | Interpolated in | Nearest node to the axis (`n_th=128`) |
|---|---|---|---|
| `theta`, cell-centred *(current)* | uniform in `theta`, first node at `dtheta/2` | `theta` | 0.70 deg |
| `mu`, node-centred *(older dumps)* | uniform in `cos(theta)`, both poles are nodes, rows descending | `mu = cos(theta)` | 10.18 deg |

`polar_axis()` in `src/athenak.py` decides which by checking whether
`theta` or `cos(theta)` is the uniformly spaced one, whether a node sits on
the pole, and whether the rows ascend or descend. That fixes three things
at once:

- **The interpolation coordinate.** `PchipInterpolator3D` indexes its
  second and third axes as `x0 + dx*i`, so the interpolator has to work in
  whichever coordinate the grid is uniform in.
  `CartesianToSpherical(..., polar=...)` hands it either `arccos(z/r)` or
  `z/r`. The detected nodes are then replaced by exactly uniform ones, so
  the spacing is uniform to machine precision rather than to `float32`.
- **The row order.** A descending polar axis is reversed on load, so the
  interpolator always sees an ascending one.
- **The pole ghost fill.** `utils.fill_spherical_ghosts` continues the
  field across a pole by taking the row `k` cells inside it (node-centred)
  or `k - 1` cells (cell-centred) and rotating by `pi` in `phi`.

A grid uniform in neither coordinate is rejected at start-up rather than
silently mis-indexed.

## Fields

AthenaK's own scalar names are mapped to the canonical names the rest of
`trace` and everything under `analysis/` expects. The mapping lives in
`FIELD_MAP` in `src/athenak.py`; names not listed there pass through
unchanged, so a raw AthenaK name can always be requested directly via
`--keys`.

| `--keys` name | AthenaK scalar | Meaning |
|---|---|---|
| `rho` | `dens` | Rest-mass density. |
| `T` | `temperature` | Temperature (MeV). |
| `r_0` | `s_00` | Passive scalar 0, i.e. `Y_e`. |
| `u_t` | `u_t` | Covariant time component of the four-velocity. |
| `V_u_x`, `V_u_y`, `V_u_z` | `win_Vx`, `win_Vy`, `win_Vz` | Coordinate velocity `dx^i/dt`, Cartesian components. |
| `F_nue`, `F_anue`, `F_nux`, `F_anux` | `\|F\|:0` .. `\|F\|:3` | Absolute neutrino number flux per species. |
| `eps_nue`, `eps_anue`, `eps_nux`, `eps_anux` | `e:0` .. `e:3` | Mean neutrino energy per species. |

The `weights` scalar present in every file is the `r^2 dmu dphi` surface
quadrature weight. `trace` doesn't use it -- `src/seeds.py` builds its own
Gauss-Legendre quadrature over each tracer's cell.

## Caveats

- **Variables dumped at different cadences will be rejected.** Because each
  variable is a separate file, it is easy to end up with (say) `mhd_w_d` at
  one more time than everything else. The pipeline requires every requested
  key at every snapshot time and raises a `KeyError` naming the times and
  keys if not, rather than integrating through a stale buffer. Either trim
  `--keys`, or point `--data-dir` at a directory holding only the times
  that carry all of them.

  The reader replaces anything non-finite or beyond `FIELD_MAX_ABS` (in
  `src/athenak.py`, 1e4 MeV) with zero on load, and prints one warning per
  file and field saying how much it replaced -- this is a no-op on
  already-clean, fixed-build dumps, and only does anything on older data.

  The bound is not trying to tell good samples from bad, which a magnitude
  test cannot do -- on affected dumps a meaningful fraction of the no-flux
  cells carry a perfectly physical-looking value. It does not need to.
  Those are harmless, since the flux they multiply downstream vanishes
  there. What has to go is the extreme tail, because the interpolation
  stencil of a tracer just inside the neutrino-carrying region reaches
  across the boundary and one such neighbour would swamp it. 1e4 MeV leaves
  ~300x headroom above the observed physical maximum (~35 MeV) while
  staying far below any observed garbage floor.

  Guarding the division in the dumps (as the `ene_conv`/`isfinite` fixes
  now do) makes all of this unnecessary going forward; the clamp exists for
  whatever older data is still floating around.
- **Four neutrino species, not three.** See "The fourth neutrino species"
  above -- `--heavy-neutrinos` controls whether `nux`/`anux` are summed
  (default), dropped, or kept separate. Confirm the species ordering of
  `e:0..3` / `|F|:0..3` against your input file before reading physics into
  the labels.
- **No `hu_t`.** It is not in the dumps, so the Bernoulli unbound
  criterion is unavailable. `analysis/` skips it when absent, so nothing
  breaks. Entropy is likewise absent and likewise not needed -- see
  [analysis/README.md](../../analysis/README.md#a-note-on-entropy).
- **Polar accuracy, on the `mu` node-centred grid only.** Interpolation
  loses most of its accuracy inside the polar cap, `theta < 10.2` degrees
  for `n_th = 128` -- about 1.6% of the sphere, both caps together.
  Measured on `f = sin(theta) cos(phi)` sampled on this grid, with the
  field's amplitude normalised to 1:

  | Cell | `theta` | abs. error | rel. error |
  |---|---|---|---|
  | polar cap | 0 -- 10.2 deg | 3.9e-2 | 46% |
  | next row | 10.2 -- 14.4 deg | 1.9e-3 | 0.9% |
  | third row | 14.4 -- 17.7 deg | 1.1e-4 | 0.05% |
  | beyond | > 17.7 deg | < 3e-5 | negligible |

  Two things cause it, both consequences of the grid being uniform in `mu`
  rather than in `theta`. First, a field that is smooth on the sphere is
  *not* smooth in `mu` at the poles: its non-axisymmetric part carries a
  factor `sin(theta) = sqrt(1 - mu^2)`, whose `mu`-derivative is infinite
  at `mu = +-1`. Second, `mu` folds at the pole (both sides of it map to
  the same `mu`), so the ghost nodes past `|mu| = 1` are fictitious and no
  value put there is correct -- even a perfectly axisymmetric `f = cos
  (theta)`, which *is* a polynomial in `mu`, picks up a 2e-3 error in the
  polar cap from the mirror alone, against ~1e-8 everywhere else.

  **A cell-centred uniform-`theta` grid removes both**, which is why one
  has been requested from the AthenaK side. There the mirror *is* the exact
  analytic continuation, `theta` has no branch point, and the polar-cap
  error drops by more than two orders of magnitude
  (`test_cell_centred_theta_grid_is_accurate_at_the_pole`). The reader
  already handles such a grid, so no change here is needed when it lands.

  There is no way to rescue the existing `mu` dumps to the same standard,
  short of teaching `PchipInterpolator3D` to accept non-uniform nodes on
  its second axis (it already does on the first) and interpolating those
  in `theta`. That would recover the scheme but not the *sampling*: an
  equal-solid-angle grid puts its first off-pole row at 10.2 degrees no
  matter how the values in between are reconstructed, and refines only as
  `2/sqrt(n_th)`.
- **A `VECTORS` block would be skipped.** The reader accounts for its bytes
  so later `SCALARS` stay addressable, but doesn't expose its components.
  Should a variable arrive as a vector rather than as separate scalar files,
  `scan_vtk` needs a few lines to record its offset and read strided
  components.
