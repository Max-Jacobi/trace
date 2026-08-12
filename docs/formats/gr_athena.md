# GR-Athena++ surface output (`--format reduced_surface`)

Surface output from [GR-Athena++](https://github.com/computationalrelativity/gr-athena):
a set of concentric spherical shells written to HDF5, one file per snapshot.
This was the format `trace` was originally built for.

Raw files are **not** read by `run_pipeline.py` directly -- run
`transform_files.py` over them once per simulation first.

## Contents

- [Raw file layout](#raw-file-layout)
- [Step 1: transform the raw files](#step-1-transform-the-raw-files)
- [Step 2: run the pipeline](#step-2-run-the-pipeline)
- [Grid conventions](#grid-conventions)
- [Fields](#fields)
- [Caveats](#caveats)

## Raw file layout

```
coordinates/NN/{R,T,th,ph}          # one group per radius NN, R and T scalars
fields/NN/<group>/<group>.<...>.<name>
```

`th` and `ph` are stored identically under every radius group, and the field
names carry their full dotted group path. Both are what the transform step
removes.

## Step 1: transform the raw files

```bash
python transform_files.py <input_files...> \
    --output_dir data/transformed \
    --num_workers 8 \
    [--delete]
```

| Flag | Meaning |
|---|---|
| `input_files` | One or more raw `*.surface*.hdf5` files (shell-glob them, e.g. `raw/*.surface1.*.hdf5`). |
| `--output_dir` | Where to write the transformed files (default `transformed_files`). This is what you later pass as `--data-dir`. |
| `--num_workers` | Parallel worker processes (default 4). |
| `--delete` | Remove each raw input file once it's successfully transformed. No undo. |

Concretely it consolidates the per-radius groups into single
`(n_r, n_theta, n_phi)` datasets, stores `r`/`th`/`ph` once instead of once
per radius, shortens field names to their last dotted component, and -- when
the M1 neutrino-transport fields `J_*`, `n_*`, `sc_sqrt_det_g` are present --
reduces them to a number flux and mean energy per species:

```
F_nue,  F_anue,  F_nux    = n_0i / sqrt(det g)
eps_nue, eps_anue, eps_nux = J_0i / n_0i
```

discarding the much larger raw M1 quantities. The result is one flat
dataset per field plus a `coordinates/` group holding `time`, `r`, `th`,
`ph`.

## Step 2: run the pipeline

```bash
python run_pipeline.py --format reduced_surface \
    --data-dir data/transformed --output-dir data/tracers_out \
    --start-t 11600 --end-t 0 \
    volume --r-min 300 --r-max 1000 --n-r 30 --n-th 15 --n-ph 30
```

`reduced_surface` is the default format, so `--format` can be omitted. It
sets these defaults:

| Option | Default for this format |
|---|---|
| `--file-pattern` | `*.hdf5` |
| `--rad-transform` | `log` |
| `--keys` | `V_u_x V_u_y V_u_z T hu_t s u_t rho r_0 F_nue F_anue F_nux eps_nue eps_anue eps_nux` |

See the [main README](../../README.md#step-2-run-the-pipeline) for every
other option.

## Grid conventions

| Axis | Layout |
|---|---|
| `r` | Geometrically spaced (constant `r[i+1]/r[i]`), hence `--rad-transform log`. |
| `theta` | Uniform in `theta`, **cell-centred**: nodes at `dth/2 ... pi - dth/2`, no node on either pole. |
| `phi` | Uniform, cell-centred: `dphi/2 ... 2*pi - dphi/2`. |

Ghost zones are filled by `src/gra_surface.py`'s `_fill_with_ghosts`, which
mirrors across the poles assuming the cell-centred layout above.

## Fields

Time is in `coordinates/time`; lengths and times are geometric
(`G = c = M_sun = 1`).

| Key | Meaning |
|---|---|
| `V_u_x`, `V_u_y`, `V_u_z` | Coordinate velocity `dx^i/dt`. |
| `rho` | Rest-mass density. |
| `T` | Temperature. |
| `s` | Specific entropy. |
| `r_0` | Passive scalar 0, i.e. `Y_e`. |
| `u_t` | Covariant time component of the four-velocity (geodesic unbound criterion). |
| `hu_t` | `h * u_t` (Bernoulli unbound criterion). |
| `F_nue`, `F_anue`, `F_nux` | Neutrino number flux per species. |
| `eps_nue`, `eps_anue`, `eps_nux` | Mean neutrino energy per species. |

## Caveats

- **`transform_files.py` drops `s` and `hu_t`.** `reduce_m1_quantities`
  deletes them along with the raw M1 quantities, even though both appear in
  this format's default `--keys` and both are used by `analysis/`. On a run
  with M1 fields present you will therefore get a `KeyError` at start-up
  listing the keys that were found. Either drop them from `--keys`, or
  remove them from the delete list in `transform_files.py:78`.
- **`src/gra_surface.py`'s `GRASurfaceFileHandler`** reads *raw* surface
  files without the transform step. It is not wired into `run_pipeline.py`
  and is much slower per snapshot, since it reassembles the per-radius
  groups on every load.
