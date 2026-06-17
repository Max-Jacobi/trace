# Test Suite

All tests live in `tests/` and are run with [pytest](https://docs.pytest.org).

```bash
# Run the full suite
python -m pytest tests/ -v

# Run a single file
python -m pytest tests/test_integrators.py -v
python -m pytest tests/test_seeds.py -v

# Run a single class or test
python -m pytest tests/test_integrators.py::TestRK4 -v
python -m pytest tests/test_seeds.py::TestSphericalByVolume::test_constant_density_full_sphere_mass -v
```

All tests are self-contained: they require no real simulation data, no shared
memory, and no MPI or multiprocessing. The full suite runs in under one second.

---

## `test_integrators.py`

Tests the three integrators in `src/integrators/` using plain Python callables
as mock velocity interpolators. No file I/O or shared memory is involved.

### Design: mock interpolators

Each integrator is called as

```python
x_new = integrator(x, dt, interps, snap_times=st)
```

where `interps` is a tuple of callables `v(x) -> velocity`. Tests pass simple
lambdas such as `lambda pos: -pos` (for the decay ODE `dx/dt = -x`) or
`lambda pos: np.zeros_like(pos)` (zero velocity). Positions are NumPy arrays
of shape `(n_dims, n_tracers)`.

### Design: convergence test helper `_convergence_order`

The function `_convergence_order(integrator, ode_rhs, x0, T, n_snap, tol_order)`
integrates

```
dx/dt = -x,   x(0) = x0
```

from `t=0` to `T=1` at four successive step sizes `dt = T/2^k` for
`k = 3, 4, 5, 6` (i.e. 8 to 64 steps), computes the global error against the
exact solution `x(T) = x0 * exp(-T)`, and checks that each consecutive error
ratio `log2(err_k / err_{k+1})` exceeds `tol_order`. All integrators receive
the same ODE; only the step count and `snap_times` array change per step.

---

### `TestIntegratorBase` — class attributes

| Test | What it checks |
|---|---|
| `test_n_snapshots_defaults` | `IntegratorBase.n_snapshots == 2` (base default) |
| `test_explicit_trapezoid_n_snapshots` | `ExplicitTrapezoid.n_snapshots == 2` |
| `test_implicit_trapezoid_n_snapshots` | `ImplicitTrapezoid.n_snapshots == 2` |
| `test_rk4_n_snapshots` | `RK4.n_snapshots == 4` (needs four snapshots) |

These tests guard against accidental changes to the class-level attribute that
controls how many snapshots the rolling-window loader keeps in memory. Getting
it wrong silently breaks the file loading logic.

---

### `TestExplicitTrapezoid`

The explicit (Adams-Bashforth-style) trapezoid scheme is second-order.

| Test | Velocity field | Expected result |
|---|---|---|
| `test_uniform_velocity_no_motion` | `v = 0` everywhere | Position unchanged: `x_new == x0` |
| `test_constant_velocity` | `v = c = 2.5` (uniform) | `x_new = x0 + c*dt` exactly (any 2nd-order scheme is exact for constant velocity) |
| `test_second_order_convergence` | `v(x) = -x` (decay ODE) | Observed order >= 1.9 at all four refinements |
| `test_batch_of_tracers` | `v = 1` | Input shape `(3, 5)` (3 dims, 5 tracers) preserved; all advance by `dt = 0.5` |
| `test_snap_times_ignored` | `v(x) = -x` | `snap_times` keyword is accepted but does not change the result (trapezoid only uses the two endpoint interpolators) |

---

### `TestImplicitTrapezoid`

The implicit (Crank-Nicolson) trapezoid scheme iterates a fixed-point loop at
each step. It is also second-order but can handle stiffer velocity fields.

| Test | What it checks |
|---|---|
| `test_uniform_velocity_no_motion` | Zero velocity leaves position unchanged |
| `test_constant_velocity` | Constant velocity gives `x0 + c*dt` (rtol 1e-10) |
| `test_second_order_convergence` | Observed order >= 1.9 on `dx/dt = -x` |
| `test_convergence_flag_set_on_success` | After a normal step, `integrator.converged is True` |
| `test_convergence_flag_unset_on_failure` | With `max_iter=1` and a stiff ODE (`v = -100x`, `dt=1`), the fixed-point loop cannot converge; `integrator.converged is False` |
| `test_n_iter_updated` | After a step, `integrator.n_iter >= 1` |
| `test_relax_parameter_accepted` | `relax=0.5` (under-relaxation) runs without error and returns a finite result |
| `test_snap_times_ignored` | Same as explicit: keyword is a no-op for trapezoid schemes |

---

### `TestRK4`

The RK4 integrator uses four simulation snapshots and a cubic time
interpolation (PCHIP or Lagrange) to evaluate velocities at the two RK4
midpoints. It is fourth-order accurate.

#### Convergence

| Test | Mode | Expected order |
|---|---|---|
| `test_fourth_order_convergence_monotone` | `RK4(monotone=True)` — PCHIP | >= 3.9 |
| `test_fourth_order_convergence_lagrange` | `RK4(monotone=False)` — Lagrange | >= 3.9 |

Both modes are tested on `dx/dt = -x` over `T=1` with x0 = [[1.0]].

#### Basic correctness

| Test | What it checks |
|---|---|
| `test_uniform_velocity_no_motion` | `v = 0` -> position unchanged |
| `test_constant_velocity` | `v = 3` -> `x_new = x0 + 3*dt` (exact for RK4) |
| `test_batch_of_tracers` | Shape `(3, 8)` is preserved; `v=1` advances all by `dt=0.5` |

#### `snap_times` (non-uniform snapshot spacing)

| Test | What it checks |
|---|---|
| `test_snap_times_uniform_matches_none` | Passing `snap_times = [-dt, 0, dt, 2*dt]` explicitly gives exactly the same answer as `snap_times=None` (the default, which assumes uniform spacing) |
| `test_snap_times_nonuniform_accepted` | `snap_times = [-0.2, 0.0, 0.1, 0.3]` (non-uniform) does not raise and returns finite values |

#### PCHIP monotonicity (`_pchip_v_mid` internals)

These tests call `_pchip_v_mid` directly to verify that the Fritsch-Carlson
PCHIP formula correctly suppresses oscillations near sharp velocity features —
critical for ejecta from neutron star mergers and supernovae where strong
shocks produce near-discontinuous velocity profiles.

| Test | Velocity profile at the four snapshots | What it verifies |
|---|---|---|
| `test_monotone_clamps_temporal_spike` | `[4, 0, 0, 0]` — large spike at `t_{n-1}`, then zero | PCHIP midpoint velocity is exactly 0: the spike from the past snapshot must not bleed into the current step's midpoint |
| `test_monotone_no_undershoot_u_shape` | `[1, 0, 0, 1]` — high at both ends, zero in the middle | Lagrange interpolation would go negative here (undershoot); PCHIP clamps the midpoint to >= 0 |

#### Lagrange weights (`_lagrange_weights` internals)

| Test | What it checks |
|---|---|
| `test_lagrange_weights_uniform_known_values` | For uniform `st = [-1, 0, 1, 2]`, weights at the midpoint `t=0.5` must equal the analytic values `[-1/16, 9/16, 9/16, -1/16]` |
| `test_lagrange_weights_sum_to_one` | For three different (uniform and non-uniform) spacing arrays, weights sum to 1 (partition of unity) — tested to rtol 1e-12 |

#### PCHIP scale invariance

| Test | What it checks |
|---|---|
| `test_pchip_scale_invariant` | Doubling all time intervals (keeping velocities the same) must not change the interpolated midpoint velocity — a necessary property of any properly time-scaled interpolator |

---

## `test_seeds.py`

Tests `spherical_by_volume` from `src/seeds.py` and the two internal
quadrature helpers it relies on: `_gauss_legendre_3d` and
`_gauss_legendre_surface`.

### Design: `MockFileHandler` and `_MockInterpolator`

The seeder calls `file_handler.setup_interpolator(shm, extra_data)` to
construct an interpolator, then queries it with Cartesian quadrature point
coordinates to get the density at those points. The mock infrastructure
replaces the real file handler and shared-memory interpolator with:

**`_MockInterpolator`** — evaluates an analytic density function
`rho(r)` at arbitrary Cartesian coordinates `(x, y, z)` without any file or
shared-memory access. Given coordinates it computes `r = sqrt(x^2+y^2+z^2)`
and returns the user-supplied `density_fn(r)`.

**`MockFileHandler`** — exposes a single time step (`times = [0.0]`) and
routes `setup_interpolator` to `_MockInterpolator`. The density function is
threaded through the `extra_data` dict that the real seeder already passes to
the setup call.

This means **all seed tests run the full `spherical_by_volume` code path**
(grid construction, quadrature, interpolator calls, tracer object creation,
mass assignment) without reading any file.

### Density profiles used

The tests cover three analytic density profiles for which the exact total mass
can be computed in closed form:

| Profile | `density_fn(r)` | Exact total mass over `[r_min, r_max]`, full sphere |
|---|---|---|
| Uniform | `rho_0` (constant) | `(4/3) * pi * rho_0 * (r_max^3 - r_min^3)` |
| Linear | `r` | `pi * (r_max^4 - r_min^4)` |
| Quadratic | `r^2` | `(4/5) * pi * (r_max^5 - r_min^5)` |

All tests use `r_min=1.0`, `r_max=3.0`, `n_r=4`, `n_th=5`, `n_ph=6` (120
tracers total), `n_quad=4` (4^3 = 64 GL quadrature points per cell), and
`random_shift_in_cell=False` (tracers placed at cell centres, no stochastic
jitter).

### Analytical reference: `_analytical_mass_sphere`

The ground-truth comparison value is computed by a 200-point Gauss-Legendre
quadrature in `r`, combined with the analytic theta and phi integrals:

```
M = [sum_i w_i * rho(r_i) * r_i^2 * r_h]
    * [cos(theta_min) - cos(theta_max)]
    * [phi_max - phi_min]
```

This is essentially machine-precision accurate for any smooth `rho(r)`, so
the bottleneck in all mass comparisons is the coarseness of the 4x5x6 cell
grid plus the n_quad=4 per-cell quadrature.

---

### `TestGaussLegendre3D` — quadrature helper unit tests

These tests exercise `_gauss_legendre_3d(r_lo, r_hi, th_lo, th_hi, ph_lo, ph_hi, n_quad)`
which returns `(pts, wts)` where `pts` is shape `(3, n_quad^3)` (Cartesian)
and `wts` is shape `(n_quad^3,)`.

> **Important background:** GL quadrature is exact only for polynomials. The
> spherical volume element contains `sin(theta)`, which is transcendental.
> Over a large angular span (e.g. `theta` from 0 to pi) a high `n_quad` is
> needed to achieve small relative error. For the typical use case of small
> cells (few degrees per cell) even `n_quad=2` is highly accurate.

| Test | Cell geometry | n_quad | What it verifies | Tolerance |
|---|---|---|---|---|
| `test_nquad1_weight_equals_midpoint_dV` | `r in [1,2]`, `th in [pi/4, 3pi/4]`, `ph in [0, pi]` | 1 | Single GL weight equals `r_c^2 * sin(th_c) * dr * dth * dph` (the midpoint-rule volume, exact by construction) | rtol 1e-14 |
| `test_weights_converge_to_exact_volume` | Same cell | 5, 6 | Sum of weights converges to exact cell volume `(r_hi^3-r_lo^3)/3 * (cos(th_lo)-cos(th_hi)) * (ph_hi-ph_lo)` | rtol 1e-5 |
| `test_constant_density_exact` | Full spherical shell `r in [2,3]`, full sphere | 6 | `dot(wts, ones) == 4*pi/3 * (r_hi^3 - r_lo^3)` | rtol 1e-8 |
| `test_r_squared_density_exact_with_nquad6` | Full spherical shell `r in [1,2]`, full sphere | 6 | `dot(wts, r^2) == 4*pi/5 * (r_hi^5 - r_lo^5)` | rtol 1e-8 |
| `test_output_shapes` | Full sphere, n_quad=2 | 2 | `pts.shape == (3, 8)`, `wts.shape == (8,)` | — |
| `test_points_inside_cell` | `r in [1,3]`, `th in [pi/6, pi/2]`, `ph in [0.3, 1.5]` | 3 | All 27 quadrature points lie inside the cell bounds (converts back to spherical and checks) | 1e-10 padding |

---

### `TestGaussLegendreSupface` — surface quadrature helper unit tests

These tests exercise `_gauss_legendre_surface(r_surf, th_lo, th_hi, ph_lo, ph_hi, n_quad)`
which returns `(x, y, z, wts)` for integrating surface densities on a sphere.

| Test | Geometry | n_quad | What it verifies | Tolerance |
|---|---|---|---|---|
| `test_nquad1_weight_equals_midpoint_area` | `r=2`, `th in [pi/4, 3pi/4]`, `ph in [0, pi]` | 1 | Single weight equals `r^2 * sin(th_c) * dth * dph` (midpoint surface area) | rtol 1e-14 |
| `test_weights_converge_to_exact_area` | Same cell | 5, 6 | Sum converges to `r^2 * (cos(th_lo)-cos(th_hi)) * (ph_hi-ph_lo)` | rtol 1e-5 |
| `test_full_sphere_area` | `r=3`, full sphere | 6 | `sum(wts) == 4*pi*r^2` | rtol 1e-8 |
| `test_output_shapes` | Full sphere, n_quad=2 | 2 | All four arrays have length `n_quad^2 = 4` | — |
| `test_points_on_sphere` | `r=5`, full sphere | 3 | `sqrt(x^2+y^2+z^2) == r_surf` for all 9 points | rtol 1e-12 |

---

### `TestSphericalByVolume` — end-to-end seeding + mass integral tests

These tests run the full `spherical_by_volume` pipeline with `MockFileHandler`.
The shared default geometry is `r in [1, 3]`, 4x5x6 cells, n_quad=4.

#### Constant density (rho = 1)

| Test | What it verifies | Tolerance |
|---|---|---|
| `test_constant_density_full_sphere_mass` | Sum of all tracer masses equals `(4/3)*pi*(3^3-1^3) = 104.72...` | rtol 1e-5 |
| `test_constant_density_number_of_tracers` | Exactly `4*5*6 = 120` tracers are created | exact |
| `test_constant_density_all_masses_positive` | Every tracer mass is strictly positive | strict |
| `test_constant_density_dV_stored` | Every tracer props dict contains a positive `dV` entry | strict |

#### Power-law density

| Test | Profile | Exact mass | Tolerance |
|---|---|---|---|
| `test_power_law_r2_full_sphere_mass` | `rho = r^2` | `4*pi/5 * (3^5-1^5) = 993.5...` | rtol 1e-5 |
| `test_power_law_r1_full_sphere_mass` | `rho = r` | `pi * (3^4-1^4) = 248.0...` | rtol 1e-5 |
| `test_higher_density_gives_higher_mass` | `rho=1` vs `rho=2` | Mass ratio must be exactly 2 | rtol 1e-10 |

#### Partial phi domain

| Test | Domain | What it verifies |
|---|---|---|
| `test_half_phi_half_mass` | `phi in [0, pi]` vs `[0, 2*pi]`, uniform density | Half-phi mass is exactly half of full-phi mass (rtol 1e-8) — tests the `dph` fix in the seeder |
| `test_half_phi_correct_absolute_mass` | `phi in [0, pi]`, uniform density | Total mass matches the analytic `(4/3)*pi*(r_max^3-r_min^3) * pi/2` (rtol 1e-5) |

The `test_half_phi_half_mass` test is particularly important: a previous bug
set `dph = 2*pi/n_ph` regardless of the phi range, which gave the wrong cell
widths, wrong quadrature points, and wrong masses for any non-full-circle phi
domain. This test would have caught that bug.

#### Partial theta domain

| Test | Domain | What it verifies |
|---|---|---|
| `test_northern_hemisphere_mass` | `theta in [0, pi/2]` | Total mass matches `(4/3)*pi*(r_max^3-r_min^3) * cos(0) = (4/3)*pi*(r_max^3-r_min^3)/2` (rtol 1e-5) |
| `test_north_plus_south_hemisphere_equals_full` | North `[0,pi/2]` + south `[pi/2,pi]` | Sum equals full sphere mass (rtol 1e-8) — additivity check |

#### Per-tracer mass vs cell volume

| Test | n_quad | What it verifies |
|---|---|---|
| `test_uniform_density_mass_equals_rho_times_dV` | 1 | For `rho_0=3.7` and `n_quad=1`, each tracer's `mass == rho_0 * dV` to machine precision (rtol 1e-14). With a single GL node the quadrature weight is exactly `r_c^2 * sin(th_c) * dr * dth * dph`, which is the same formula used to compute `dV`. |

Note: for `n_quad > 1` the GL mass uses a more accurate approximation to the
true cell volume than the midpoint `dV`, so they will not agree exactly for
large cells. Only the `n_quad=1` equality is machine-exact.

#### Quadrature convergence

| Test | Profile | What it verifies |
|---|---|---|
| `test_mass_accuracy_improves_with_n_quad` | `rho = r^2` | Absolute mass error with `n_quad=4` is strictly smaller than with `n_quad=1`. |

#### Position bounds

| Test | What it verifies |
|---|---|
| `test_positions_inside_radial_range` | With `r_min=1.5`, `r_max=4.0` and `random_shift_in_cell=False`, every tracer's initial position satisfies `1.5 <= r <= 4.0`. |
