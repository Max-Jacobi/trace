# Examples

These are exploratory, end-to-end scripts -- not pytest tests.  Each one
runs the full tracer-integration pipeline on a prescribed analytic velocity
field, prints a short report, and saves comparison plots (and an
animation) to `examples/plots/`.  They have no pass/fail assertions: a run
"succeeds" as long as it completes without raising.

They're allowed to take longer than a unit test.  Each script parallelises
its own tracer-integration loop across CPU cores (see "Parallelisation"
below), so running one doesn't block on a single core.

```bash
PYTHONPATH=. python examples/blast_wave.py                  # (a)
PYTHONPATH=. python examples/blast_wave_interpolators.py     # (b)
PYTHONPATH=. python examples/disk_wind.py                    # (c)
PYTHONPATH=. python examples/disk_wind_interpolators.py       # (d)
PYTHONPATH=. python examples/trapezoid_comparison.py          # (e)
```

Each pair of scripts ((a)/(b) and (c)/(d)) deliberately isolates one axis
of comparison at a time -- holding the interpolator fixed while varying
the integrator, or vice versa -- rather than conflating both in one chart:

| Script | Comparison | Holds fixed | Varies |
|---|---|---|---|
| `blast_wave.py` | (a) integrators | PCHIP interpolator | Euler / ExplicitTrapezoid / RK4 |
| `blast_wave_interpolators.py` | (b) interpolators | RK4 integrator | Linear / PCHIP |
| `disk_wind.py` | (c) integrators | PCHIP interpolator | Euler / ExplicitTrapezoid / RK4 |
| `disk_wind_interpolators.py` | (d) interpolators | RK4 integrator | Linear / PCHIP |
| `trapezoid_comparison.py` | (e) stability | PCHIP interpolator | ExplicitTrapezoid vs ImplicitTrapezoid |

`disk_wind_interpolators.py` and `blast_wave_interpolators.py` are thin
scripts that just import `run_comparison` and the physics/grid/plotting
machinery from `disk_wind.py` / `blast_wave.py` respectively and supply a
different scheme list; `trapezoid_comparison.py` similarly reuses
`disk_wind.py`'s machinery.

### disk-wind scenario (c, d, e)

Differentially-rotating, homologously-expanding disk-wind outflow
(post-BNS-merger ejecta analogue), sampled on a real-simulation-scale
spherical (r, theta, phi) grid (256 x 128 x 256), dt = 0.1 ms.
`trapezoid_comparison.py` additionally demonstrates ImplicitTrapezoid's
unconditional (A-)stability advantage over ExplicitTrapezoid near the
launch radius.

### blast-wave scenario (a, b)

2-D asymmetric Sedov-Taylor blast wave on a Cartesian grid (256 x 256 x 7,
NX/NY matching disk-wind's radial point count; dt = 0.1 ms matching
disk-wind's timestep -- see `blast_wave.py`'s module docstring for the
resulting grid spacing and how it compares to disk-wind's). Errors are
measured against a `scipy.integrate.solve_ivp` (DOP853) reference
trajectory integrated directly from the exact analytic velocity field (no
spatial interpolation needed for the reference), valid for every tracer --
not just the deep-inside-shock tracers the previous analytic approximation
was restricted to.

Tracer counts are small everywhere (a few radii x a few angles): both
scenarios are rotationally/radially symmetric, so a handful of tracers is
enough to see the schemes' qualitative differences cheaply.

Each script's module docstring has the full physical setup and the list of
plots it produces.

## Parallelisation

Within a single script, each integration step's tracer batch is split into
bunches and dispatched to a persistent `multiprocessing.Pool`, mirroring
the production pattern in `src/tracers.py`:

- Tracers are sorted by their interpolation-grid bin (theta/phi for the
  spherical disk-wind grid, y/z for the Cartesian blast-wave grid) before
  bunching, so that tracers needing the same PCHIP stencil end up in the
  same bunch -- this maximises cache hits in `PchipInterpolator3D`'s
  per-column interpolator cache (see `_sort_order` / `sort_tracers`).
- Workers attach to the velocity snapshot's shared memory by name (it's
  built once in the main process) rather than rebuilding it, and cache the
  wrapped interpolator across calls within a step.
- See `examples/_parallel.py` for the shared pool helper (`worker_pool`),
  which forces the `fork` start method (workers need to inherit the
  calling module's grid constants) and shuts down gracefully so workers can
  clean up their attached shared-memory handles.

By default each script uses `os.cpu_count()` worker processes.
