# Supporting a new data format

`trace` is meant to be agnostic about where its snapshots come from. To read
a new simulation code's output you implement one class, a `FileHandler`
subclass, and register it in `run_pipeline.py`. Nothing else in the pipeline
(seeding, integration, output) needs to know about your format.

Two implementations exist to read alongside this guide:
`src/reduced_surface.py` (HDF5, one file per snapshot) and `src/athenak.py`
(binary VTK, several files per snapshot). Between them they cover most of
the variations you are likely to hit.

## Contents

- [What the pipeline needs from your data](#what-the-pipeline-needs-from-your-data)
- [The `FileHandler` contract](#the-filehandler-contract)
- [Rules that aren't in the signatures](#rules-that-arent-in-the-signatures)
- [Grid conventions and ghost zones](#grid-conventions-and-ghost-zones)
- [Wiring it into the CLI](#wiring-it-into-the-cli)
- [Checklist](#checklist)

## What the pipeline needs from your data

At minimum:

- A **time** per snapshot.
- A **3-D structured grid** whose axes are separable. Both existing readers
  use spherical `(r, polar, phi)` grids, which is what
  `CartesianToSpherical` converts tracer positions into; a Cartesian grid
  works too, by handing the interpolator class straight to the pipeline
  without that wrapper.
- Three **velocity components** in Cartesian coordinates, giving `dx^i/dt`
  in the same time units as the snapshot times. Everything else you carry
  along is passenger data.
- Any other **scalar fields** you want recorded along each trajectory.

The interpolators impose one constraint worth knowing before you start:
`PchipInterpolator3D` (the default) indexes the second and third axes as
`x0 + dx*i`, so **both must be uniformly spaced** in whatever coordinate you
interpolate in. Only the first axis may be non-uniform. If your grid is
uniform in some transformed coordinate rather than the coordinate itself
(e.g. uniform in `cos(theta)`, or geometrically spaced in `r`), interpolate
in that transformed coordinate -- see
[Grid conventions](#grid-conventions-and-ghost-zones).
`RegularInterpolator3D` accepts arbitrary node arrays on all three axes and
is a good escape hatch while you are getting a new reader working.

## The `FileHandler` contract

`src/file.py`'s `FileHandler` handles chunked loading into shared memory,
parallel file parsing, and chunk bookkeeping. You implement four methods.

### `list_files(self, directory) -> list[str]`

Return every file in `directory` that belongs to this dataset. Use
`src/utils.py`'s `glob_files(directory, pattern)`, which sorts, accepts
either a glob or a plain suffix, and raises a clear `FileNotFoundError` if
nothing matches.

This is also where you read the grid, once, from the first file (both
existing handlers call a `load_grid` helper here). `load_grid` must
populate `self.extra_data` with everything `setup_interpolator` will later
need -- the coordinate axes, the array `shape`, and `mem_size`.

### `parse_file(file_path, keys, extra_data) -> (time, path, metadata, mem_size)` *(static)*

Called in parallel over every file. Read *headers only* -- this runs once
per file at start-up, so it must not pull in bulk data.

Return the file's time, its path, an opaque `metadata` object that
`load_step_to_memory` will get back, and the number of bytes one field
occupies. Return `(0.0, "", [], 0)` if the file holds none of the requested
`keys`, and it will be ignored.

Both existing handlers use the list of keys the file actually provides as
their `metadata`.

**Several files per snapshot is supported and needs no special handling.**
`parse_files` groups every file that reports the same time into one dict,
and hands the whole dict to `load_step_to_memory`. A code that writes one
variable per file (as AthenaK does) just works.

### `load_step_to_memory(metadata_dict, shared_memory, extra_data)` *(static)*

Fill one snapshot's shared-memory blocks. `metadata_dict` maps file path ->
whatever `parse_file` returned as metadata, for every file at this time.
`shared_memory` maps field key -> shared-memory segment name.

For each key, attach the segment, wrap it as
`np.ndarray(shape=extra_data['shape'], dtype=np.float64, buffer=shm.buf)`,
write the field (ghost zones included) and close the handle. Always close in
a `finally`.

### `setup_interpolator(shared_memory, extra_data) -> InterpolatorBase` *(static)*

Build an interpolator over one snapshot's shared memory. This runs **in
worker processes**, so it must reconstruct everything it needs from
`extra_data` alone. Do not touch `self` -- there isn't one.

## Rules that aren't in the signatures

- **`extra_data` must be picklable and self-sufficient.** Workers are
  spawned, not forked, so `extra_data` is pickled across. Put coordinate
  arrays and shapes in it; never file handles, open HDF5 objects, or
  lambdas.
- **`extra_data` is created by `FileHandler.__init__`**, which seeds it with
  `interpolator` and `interpolator_kwargs`. Your subclass should therefore
  set its own entries either from `load_grid` (which runs *inside*
  `super().__init__`, via `list_files`) or *after* calling
  `super().__init__`. Setting `self.extra_data = {...}` before
  `super().__init__` silently loses them.
- **Pass `interpolator` through to `super().__init__`.** The usual pattern:

  ```python
  def __init__(self, interpolator, *args, my_option=None, **kwargs):
      self.n_ghosts = interpolator.n_ghosts     # needed by load_grid
      super().__init__(interpolator, *args, **kwargs)
      self.extra_data['my_option'] = my_option  # after: __init__ rebuilds extra_data
  ```

- **`mem_size` must be the same for every key.** One block of that size is
  allocated per key per resident snapshot, so it has to fit the largest
  field. In practice: all fields share one grid, so
  `prod(shape) * 8` for float64.
- **Data in shared memory is always float64**, whatever the file stores.
- **Requested keys are checked, per snapshot time.** `parse_files` raises
  `KeyError` if any requested key is missing from any time, listing the
  offending times and the names it did find. A missing key is never loaded
  into that snapshot's buffer, which then still holds whatever the previous
  chunk left there, so this would otherwise integrate tracers through a
  stale field rather than fail. It also catches a plain typo at start-up
  instead of producing an all-NaN column.
- **Key names are yours to choose.** If your code's names differ from the
  canonical ones the analysis scripts expect (`rho`, `T`, `r_0`,
  `u_t`, `V_u_x/y/z`, `F_*`, `eps_*`), map them inside the handler rather
  than making users pass raw names. `src/athenak.py`'s `FIELD_MAP` does
  this, falling through unchanged for names it doesn't know so raw names
  still work.

## Grid conventions and ghost zones

Tracers are queried at arbitrary positions, including right up against the
grid boundary and across the coordinate singularities of a spherical grid.
The interpolation stencil needs `interpolator.n_ghosts` cells of padding per
side on the angular axes (3 for `PchipInterpolator3D`, 1 for
`RegularInterpolator3D`). Your `load_grid` extends the angular coordinate
arrays by that many nodes on each side, and `load_step_to_memory` fills the
corresponding data.

For a spherical grid the fill is:

- **phi**: periodic. The leading ghosts copy the trailing real columns and
  vice versa.
- **theta**: continue across the pole by taking the row that many cells
  inside the pole and rotating it by `pi` in phi.

`src/utils.py`'s `fill_spherical_ghosts(buf, ar, ng, node_centred)` does
both, on the last two axes of whatever you hand it. Its one flag is the
thing to get right: with a **cell-centred** polar grid (no node on the
pole) ghost row `k` mirrors row `k-1` and the result is the *exact*
analytic continuation of a field smooth on the sphere; with a
**node-centred** one (the first and last rows *are* the poles) it mirrors
row `k` instead. Picking the wrong one does not raise, it just gives
quietly wrong values near the poles.

If your grid is uniform only after a transform, say so rather than
resampling:

- **Radial**: pass `coord_transforms={0: "log"}` for a geometrically spaced
  grid, or `("asinh", scale)` for a lin-log one. Exposed on the CLI as
  `--rad-transform`. A non-uniform radial axis is fine untransformed too,
  since `PchipInterpolator3D` uses the actual node array on axis 0 -- the
  transform is about interpolation accuracy, not indexing.
- **Polar**: `CartesianToSpherical(..., polar="mu")` hands the interpolator
  `mu = cos(theta)` instead of `theta`, for grids built uniform in
  `cos(theta)`. Default is `polar="theta"`.

Validate these assumptions in `load_grid` and raise if they don't hold.
Both are silent-wrong-answer failures otherwise, which is much worse than a
start-up error. `AthenaKFileHandler.load_grid` is the example: it checks the
file's own coordinates against the analytic uniform nodes and then uses the
analytic ones, so the spacing is uniform to machine precision rather than to
float32.

## Wiring it into the CLI

Add an entry to `FORMATS` in `run_pipeline.py`:

```python
FORMATS = {
    'my_code': {
        'handler': MyFileHandler,
        'file_pattern': '*.h5',
        'rad_transform': 'none',
        'keys': ('V_u_x', 'V_u_y', 'V_u_z', 'rho', 'T', ...),
    },
    ...
}
```

Those become the defaults for `--file-pattern`, `--rad-transform` and
`--keys` when the user passes `--format my_code`; anything given explicitly
on the command line still wins. `--vel-keys` and `--density-key` keep their
global defaults (`V_u_x V_u_y V_u_z` and `rho`), so name your velocity and
density keys accordingly and there is nothing else to configure.

## Checklist

Once your handler runs, check these before trusting a production run:

1. **Grid points come back exactly.** Interpolate at the grid nodes and
   compare with the raw file values. This is the check that catches an axis
   order or transpose mistake, and nothing else will.
2. **The poles are finite and smooth.** Query points within one or two cells
   of `theta = 0` and `theta = pi` at several `phi`. NaNs mean the ghost
   zones are too narrow; a jump between neighbouring `phi` means the mirror
   is rolling by the wrong amount, or is off by a row.
3. **phi wraps.** Query either side of `phi = 0`.
4. **Times are right.** `file_handler.times` should be sorted, unique, and
   in the same units as `--start-t`/`--end-t`.
5. **A known velocity field integrates correctly.** Write synthetic
   snapshots with, say, `v = V0 * r_hat` on your real grid, and check the
   tracers follow `r(t) = r0 + V0 * t`. `tests/test_athenak.py` shows the
   pattern.
6. **Shared memory is released.** `ls /dev/shm` after a run; see the
   troubleshooting section of the [main README](../README.md).
