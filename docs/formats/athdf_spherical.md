# Athena++ athdf meshblock output (`--format athdf_spherical`)

HDF5 output from [Athena++](https://github.com/PrincetonUniversity/athena),
containing the raw octree meshblock structure. `run_pipeline.py` reads these
files directly. Unlike the other formats, the data is *not* restitched onto
one product grid: shared-memory buffers keep the meshblock decomposition
(`(n_blocks, nb1, nb2, nb3)` per key) and interpolation first locates the
meshblock containing each query point, then evaluates a 4-point PCHIP
stencil on that block's own node arrays (`--interpolator meshblock`, the
format default and the only interpolator compatible with this layout).

## Requirements on the dumps

- **Ghost zones must be written** (`ghost_zones = true` in the `<output>`
  block). Each block then carries its neighbours' overlap, so a 4-point
  stencil works anywhere in a block without inter-block communication.
  Two ghost cells per side are needed; dumps without ghosts are rejected.
- Spherical-type coordinates: `schwarzschild` or `spherical_polar`,
  `(x1, x2, x3) = (r, theta, phi)`. The radial spacing may be geometric
  (`RootGridX1` ratio != 1); no `--rad-transform` applies (interpolation
  uses the true per-block node coordinates, and the handler rejects one).
- Unigrid and static AMR octrees both work. The block layout is read once
  from the first file; a layout change between outputs (re-gridding) raises.
  AMR needs the block interior divisible by `2^MaxLevel` (Athena++ default
  block sizes satisfy this).

## File layout and field discovery

A snapshot may be spread over several output series written at the same
times (`out1`, `out2`, ...). Files are grouped by their `Time` attribute
and each file's contents are discovered from its own
`DatasetNames`/`VariableNames` attributes -- which series carries which
variable is never assumed. Files carrying none of the requested keys (e.g.
a cons-only series) are skipped. A variable present in several same-time
files is read once, from the *last* file in sorted path order.

Segmented runs (`output-0000/`, `output-0001/`, ...) are read by pointing
`--data-dir` at the parent with `--file-pattern '**/*.athdf'`. Because the
lexically latest path wins, a segment redone after an error supersedes the
original wherever the two contain the same snapshot time.

## Fields

Canonical key -> raw Athena++ variable name (see `FIELD_MAP` in
`src/athdf_spherical.py`; unmapped keys pass through by raw name, so any variable can
be requested directly via `--keys`):

| key | raw | note |
|---|---|---|
| `rho` | `rho` | rest-mass density |
| `press` | `press` | |
| `r_0` | `rYE` | electron fraction passive scalar |
| `s` | `rENT` | entropy passive scalar |
| `T` | `Temperature` | user output |
| `u_t` | `u_t` | user output |
| `h` | `h` | user output |
| `V_u_x/y/z` | `vel1/2/3` | synthesised, see below |

## Velocity transform

`vel1/vel2/vel3` are the GR primitive `utilde^i = W v^i` (normal-frame
projected 4-velocity, coordinate basis). The handler converts them to the
Cartesian transport velocity at load time, assuming Schwarzschild
coordinates with zero shift:

```
W   = sqrt(1 + utilde_r^2/alpha^2 + r^2 utilde_th^2 + r^2 sin^2(th) utilde_ph^2)
V^i = dx^i/dt = (alpha / W) * utilde^j * dx^i_cart/dx^j_sph,   alpha^2 = 1 - 2M/r
```

The black hole mass `M` is not stored in the athdf metadata; pass it with
`--bh-mass` (default 1.0 code units, `--bh-mass 0` gives flat spherical
coordinates).

## Polar boundary conventions

Verified on real dumps: pole ghost cells hold the field from across the
pole (phi + pi), with `vel2` **and** `vel3` sign-flipped, and their `x2v`
coordinates mirrored back into the domain (non-monotonic rows). The reader
reflects the ghost node coordinates into the extended chart (theta < 0 /
theta > pi) and undoes the `vel3` flip there, which makes the Cartesian
velocity components exactly continuous across the pole.

## Running the pipeline

```sh
python run_pipeline.py --format athdf_spherical \
    --data-dir <dir with *.athdf> \
    --start-t 40.9446 --end-t 0.0 \
    ... volume ...
```

`--start-t` is snapped to the nearest available snapshot time
automatically (a rounded value is fine; the pipeline prints the snap).
Beware when reading times off the files yourself: `Time` is float32, and
its printed repr (e.g. `203015.4`) can lie *below* the exact stored value
(`203015.40625`).

## Octree block lookup

The `Levels`/`LogicalLocations` metadata is compiled once into a dense
lookup table at finest-block granularity (`n_root_blocks * 2^MaxLevel` per
axis): a query point maps to a root-grid cell by binary search on the three
reconstructed root face arrays, then to its meshblock by integer arithmetic
and one table lookup -- O(1) per point, fully vectorised, for any static
octree.

## Native weight grid

The mass-weighted seeding modes (`volume-mc`, `surface-mc`) can build their
sampling weights on this format's own cells, with no interpolation and no
helper grid. `--weight-grid auto` picks it; `--weight-grid native` says so
explicitly and errors rather than falling back.

An octree has no single global grid, but its meshblocks tile the domain, so
the reader hands back every block's cells, pooled. `load_grid` already refuses a file
whose blocks do not tile, which is what makes this safe: cells counted twice
would be mass counted twice.

Two things specific to this format:

- **The surface is the sphere you asked for, under AMR too.** In every
  block the sphere passes through, the cell containing it supplies the flux,
  measured at that cell's own `x1v` as `r**2 rho v_r` and carried to the
  sphere unchanged. Blocks at different levels have different `x1v`, but the
  sphere does not move, so there is no staircase and nothing is lost through
  its risers. It is exactly the outer boundary of a volume seeded out to the
  same radius.
- **Under AMR the angular resolution on that sphere is non-uniform**, since
  blocks at different refinement levels contribute different cell sizes. The
  cells still tile it exactly. That is correct, but it surprises people.

Verified against a real dump (256 blocks, `MeshBlockSize` 20³, 2 ghost cells):
summing `rho*dV` over the returned cells reproduces the same sum computed
straight from the HDF5, over the same 307,200 cells, to five decimal places.
