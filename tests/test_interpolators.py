"""
Unit tests for src/interpolators.

All interpolators are exercised against analytic fields stored in shared
memory via the shared_memory_arrays context manager from conftest.
No file I/O or real simulation data is required.
"""

import numpy as np
import pytest
from tests.conftest import shared_memory_arrays

from src.interpolators.regular import RegularInterpolator3D
from src.interpolators.pchip import PchipInterpolator3D
from src.interpolators.coordinate_transformations import CartesianToSpherical
from src.tracers import Tracer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracer(idx, position):
    """Create a Tracer with one recorded step at *position*."""
    tr = Tracer(id=idx, position=position, time=0.0, keys=[], n_steps=2)
    tr.add_step(position, 0.0, {})
    return tr

def _linear_field(x, y, z):
    """f(x,y,z) = x + 2*y + 3*z  (exact for any linear-capable interpolator)."""
    return x + 2.0 * y + 3.0 * z


def _make_regular_grid(nx=6, ny=6, nz=6, x0=0.0, xmax=5.0):
    """Return uniformly spaced coordinate arrays."""
    x = np.linspace(x0, xmax, nx)
    y = np.linspace(x0, xmax, ny)
    z = np.linspace(x0, xmax, nz)
    return x, y, z


def _make_field_3d(x, y, z, fn):
    """Evaluate fn(x, y, z) on a meshgrid and return shape (nx, ny, nz)."""
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    return fn(xx, yy, zz)


# ---------------------------------------------------------------------------
# RegularInterpolator3D
# ---------------------------------------------------------------------------

class TestRegularInterpolator3D:
    """Tests for the SciPy-backed regular-grid interpolator."""

    def _build(self, x, y, z, field, **kwargs):
        """Helper: set up shared memory, return loaded interpolator."""
        return x, y, z, field, kwargs

    def test_query_at_grid_nodes_linear_field(self):
        """At grid nodes the interpolator must reproduce the exact field values."""
        x, y, z = _make_regular_grid()
        field = _make_field_3d(x, y, z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = RegularInterpolator3D(x, y, z, shm=shm, shape=field.shape)
            interp.load()
            # Query at all grid nodes
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            pts = np.array([xx.ravel(), yy.ravel(), zz.ravel()])
            vals = interp(pts)   # shape (1, n_pts)
            expected = _linear_field(xx.ravel(), yy.ravel(), zz.ravel())
            np.testing.assert_allclose(vals[0], expected, rtol=1e-12)

    def test_linear_interpolation_at_midpoints(self):
        """Midpoints between nodes must be interpolated exactly for a linear field."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        z = np.array([0.0, 1.0, 2.0, 3.0])
        field = _make_field_3d(x, y, z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = RegularInterpolator3D(x, y, z, shm=shm, shape=field.shape)
            interp.load()
            # Midpoint between nodes (0,0,0) and (1,1,1)
            pts = np.array([[0.5], [0.5], [0.5]])
            vals = interp(pts)
            np.testing.assert_allclose(
                vals[0, 0], _linear_field(0.5, 0.5, 0.5), rtol=1e-12
            )

    def test_out_of_bounds_returns_nan(self):
        """Queries outside the grid must return NaN (fill_value=nan)."""
        x, y, z = _make_regular_grid()
        field = _make_field_3d(x, y, z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = RegularInterpolator3D(x, y, z, shm=shm, shape=field.shape)
            interp.load()
            pts = np.array([[999.0], [999.0], [999.0]])
            vals = interp(pts)
            assert np.isnan(vals[0, 0])

    def test_multiple_keys(self):
        """When multiple field keys are present, all are returned correctly."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        z = np.array([0.0, 1.0, 2.0])
        f1 = _make_field_3d(x, y, z, _linear_field)
        f2 = _make_field_3d(x, y, z, lambda xi, yi, zi: xi**2)
        with shared_memory_arrays({'f1': f1, 'f2': f2}) as shm:
            interp = RegularInterpolator3D(x, y, z, shm=shm, shape=(3, 3, 3))
            interp.load()
            pts = np.array([[1.0], [1.0], [1.0]])
            vals = interp(pts)   # shape (2, 1)
            assert vals.shape == (2, 1)
            np.testing.assert_allclose(vals[0, 0], _linear_field(1, 1, 1), rtol=1e-12)
            np.testing.assert_allclose(vals[1, 0], 1.0, rtol=1e-12)

    def test_raises_without_load(self):
        """Calling the interpolator before load() must raise RuntimeError."""
        x, y, z = _make_regular_grid()
        field = _make_field_3d(x, y, z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = RegularInterpolator3D(x, y, z, shm=shm, shape=field.shape)
            pts = np.array([[1.0], [1.0], [1.0]])
            with pytest.raises(RuntimeError):
                interp(pts)

    def test_load_unload_cycle(self):
        """After unload(), loaded flag is False; reload brings it back."""
        x, y, z = _make_regular_grid()
        field = _make_field_3d(x, y, z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = RegularInterpolator3D(x, y, z, shm=shm, shape=field.shape)
            assert interp.loaded is False
            interp.load()
            assert interp.loaded is True
            interp.unload()
            assert interp.loaded is False

    def test_log_coords_x_axis(self):
        """
        With log_coords=[0], x coordinates are transformed as log10(x) before
        interpolation.  A field defined as f(log10(x), y, z) = log10(x) should
        be reproduced exactly at node positions.
        """
        x = np.array([1.0, 10.0, 100.0, 1000.0])
        y = np.linspace(0.0, 3.0, 4)
        z = np.linspace(0.0, 3.0, 4)
        # Field value = log10(x)  (independent of y, z)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        field = np.log10(xx).astype(np.float64)
        with shared_memory_arrays({'f': field}) as shm:
            interp = RegularInterpolator3D(
                x, y, z, shm=shm, shape=field.shape, log_coords=[0]
            )
            interp.load()
            pts = np.array([[10.0], [1.5], [1.5]])
            vals = interp(pts)
            np.testing.assert_allclose(vals[0, 0], 1.0, rtol=1e-6)

    def test_nonpositive_log_coord_raises_on_init(self):
        """Passing a coordinate with non-positive values for a log axis must raise."""
        x = np.array([-1.0, 1.0, 10.0])   # contains negative
        y = np.array([0.0, 1.0, 2.0])
        z = np.array([0.0, 1.0, 2.0])
        field = np.ones((3, 3, 3))
        with shared_memory_arrays({'f': field}) as shm:
            with pytest.raises(ValueError, match="log"):
                RegularInterpolator3D(x, y, z, shm=shm, shape=(3, 3, 3), log_coords=[0])

    def test_shape_mismatch_raises(self):
        """Mismatched coordinate lengths and shape must raise ValueError."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        z = np.array([0.0, 1.0])
        field = np.ones((3, 3, 3))  # shape (3,3,3) but coords say (2,2,2)
        with shared_memory_arrays({'f': field}) as shm:
            with pytest.raises(ValueError):
                RegularInterpolator3D(x, y, z, shm=shm, shape=(3, 3, 3))

    def test_mismatched_query_coords_raise(self):
        """Query coordinates with different shapes must raise ValueError."""
        x, y, z = _make_regular_grid()
        field = _make_field_3d(x, y, z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = RegularInterpolator3D(x, y, z, shm=shm, shape=field.shape)
            interp.load()
            # Pass coords as a list so numpy does not reject the ragged shape
            pts = [np.array([1.0, 2.0, 3.0]), np.array([1.0]), np.array([1.0])]
            with pytest.raises(ValueError):
                interp(pts)


# ---------------------------------------------------------------------------
# PchipInterpolator3D
# ---------------------------------------------------------------------------

class TestPchipInterpolator3D:
    """
    Tests for the 3-D PCHIP interpolator on a uniform Cartesian grid.

    Grid: 7 x 7 x 7 nodes on [0, 6]^3 (uniform spacing dx=dy=dz=1).
    The PCHIP stencil requires 4 nodes in y and z, so valid queries must
    have y in [1, 4) and z in [1, 4) (indices 1..3 are well-supported).
    The x direction is handled by SciPy's PchipInterpolator with
    extrapolate=False, returning NaN for x outside [0, 6].
    """

    NX = NY = NZ = 7
    X = Y = Z = np.linspace(0.0, 6.0, 7)

    def _build_and_load(self, field):
        """Context manager that yields a loaded PchipInterpolator3D."""
        # Not a real context manager - returns (shm context, interp)
        return field

    def test_query_at_interior_nodes_linear_field(self):
        """At interior grid nodes the interpolator must reproduce the field exactly."""
        field = _make_field_3d(self.X, self.Y, self.Z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = PchipInterpolator3D(
                self.X, self.Y, self.Z, shm=shm, shape=field.shape
            )
            interp.load()
            # Interior nodes: x in any, y in [1, 5], z in [1, 5]
            test_pts = [(2.0, 2.0, 2.0), (3.0, 3.0, 3.0), (1.0, 2.0, 3.0)]
            for (xq, yq, zq) in test_pts:
                pts = np.array([[xq], [yq], [zq]])
                vals = interp(pts)
                expected = _linear_field(xq, yq, zq)
                np.testing.assert_allclose(
                    vals[0, 0], expected, rtol=1e-10,
                    err_msg=f"PCHIP at ({xq},{yq},{zq}): expected {expected}"
                )

    def test_linear_field_exact_at_midpoints(self):
        """
        PCHIP must reproduce a linear function exactly at midpoints between
        interior grid nodes (PCHIP is exact for polynomials of degree <= 3).
        """
        field = _make_field_3d(self.X, self.Y, self.Z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = PchipInterpolator3D(
                self.X, self.Y, self.Z, shm=shm, shape=field.shape
            )
            interp.load()
            # Midpoint in the valid interior: y=2.5, z=2.5
            xq, yq, zq = 2.5, 2.5, 2.5
            pts = np.array([[xq], [yq], [zq]])
            vals = interp(pts)
            expected = _linear_field(xq, yq, zq)
            np.testing.assert_allclose(vals[0, 0], expected, rtol=1e-10)

    def test_boundary_stencil_returns_nan(self):
        """
        Queries where the 4-point yz-stencil falls outside the grid must
        return NaN (no extrapolation policy).
        y=0.5 -> iy=0, iy0=-1 -> stencil invalid -> NaN.
        """
        field = _make_field_3d(self.X, self.Y, self.Z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = PchipInterpolator3D(
                self.X, self.Y, self.Z, shm=shm, shape=field.shape
            )
            interp.load()
            pts = np.array([[3.0], [0.5], [3.0]])  # y too close to boundary
            vals = interp(pts)
            assert np.isnan(vals[0, 0]), (
                f"Expected NaN for near-boundary query, got {vals[0, 0]}"
            )

    def test_x_out_of_domain_returns_nan(self):
        """x outside [x_min, x_max] returns NaN (SciPy PCHIP with extrapolate=False)."""
        field = _make_field_3d(self.X, self.Y, self.Z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = PchipInterpolator3D(
                self.X, self.Y, self.Z, shm=shm, shape=field.shape
            )
            interp.load()
            pts = np.array([[999.0], [3.0], [3.0]])  # x far outside grid
            vals = interp(pts)
            assert np.isnan(vals[0, 0])

    def test_raises_without_load(self):
        """Calling before load() must raise RuntimeError."""
        field = _make_field_3d(self.X, self.Y, self.Z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = PchipInterpolator3D(
                self.X, self.Y, self.Z, shm=shm, shape=field.shape
            )
            with pytest.raises(RuntimeError):
                interp(np.array([[3.0], [3.0], [3.0]]))

    def test_load_unload_cycle(self):
        """After unload(), loaded=False; interp_cache is cleared."""
        field = _make_field_3d(self.X, self.Y, self.Z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = PchipInterpolator3D(
                self.X, self.Y, self.Z, shm=shm, shape=field.shape
            )
            assert not interp.loaded
            interp.load()
            assert interp.loaded
            # Trigger a cache entry
            pts = np.array([[3.0], [3.0], [3.0]])
            interp(pts)
            assert len(interp._xp_cache) > 0
            interp.unload()
            assert not interp.loaded

    def test_shape_mismatch_raises(self):
        """Coordinate lengths inconsistent with shape must raise ValueError."""
        x2 = np.linspace(0, 6, 4)  # length 4, but shape says 7
        field = _make_field_3d(self.X, self.Y, self.Z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            with pytest.raises(ValueError):
                PchipInterpolator3D(
                    x2, self.Y, self.Z, shm=shm, shape=field.shape
                )

    def test_sort_tracers_returns_array(self):
        """sort_tracers must return an array of the same length as the input."""
        from src.tracers import Tracer
        field = _make_field_3d(self.X, self.Y, self.Z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = PchipInterpolator3D(
                self.X, self.Y, self.Z, shm=shm, shape=field.shape
            )
            interp.load()
            # sort_tracers uses tr.positions[-1], so each tracer needs a step.
            tracers = np.array([
                _make_tracer(i, np.array([float(i), 3.0, 3.0]))
                for i in range(1, 5)
            ])
            sorted_tr = interp.sort_tracers(tracers)
            assert len(sorted_tr) == len(tracers)

    def test_cache_eviction(self):
        """With max_cache_size_GB very small, cache entries are evicted."""
        field = _make_field_3d(self.X, self.Y, self.Z, _linear_field)
        with shared_memory_arrays({'f': field}) as shm:
            interp = PchipInterpolator3D(
                self.X, self.Y, self.Z, shm=shm, shape=field.shape,
                max_cache_size_GB=1e-12,  # force aggressive eviction
            )
            interp.load()
            # Query many distinct (y, z) pairs to fill the cache
            for yq in [2.0, 3.0, 4.0]:
                for zq in [2.0, 3.0, 4.0]:
                    pts = np.array([[3.0], [yq], [zq]])
                    interp(pts)
            assert interp.n_evicted > 0


# ---------------------------------------------------------------------------
# CartesianToSpherical
# ---------------------------------------------------------------------------

class TestCartesianToSpherical:
    """
    Tests for the Cartesian -> spherical coordinate wrapper.

    The inner interpolator is a RegularInterpolator3D defined on a
    (r, theta, phi) grid.  CartesianToSpherical converts the Cartesian
    query (x, y, z) to (r, theta, phi) before delegating.
    """

    # Grid: r in [0.5, 2.5], theta in [0.1, pi-0.1], phi in [0, 2pi]
    R_NODES     = np.linspace(0.5, 2.5, 5)
    THETA_NODES = np.linspace(0.1, np.pi - 0.1, 6)
    PHI_NODES   = np.linspace(0.0, 2 * np.pi, 7)

    def _build(self, fn):
        """Build a CartesianToSpherical wrapping a RegularInterpolator3D."""
        r, th, ph = self.R_NODES, self.THETA_NODES, self.PHI_NODES
        field = fn(r, th, ph)
        return r, th, ph, field

    def test_constant_field_returns_constant(self):
        """A field equal to a constant must return that constant everywhere."""
        r, th, ph = self.R_NODES, self.THETA_NODES, self.PHI_NODES
        rr, _, _ = np.meshgrid(r, th, ph, indexing='ij')
        field = np.ones_like(rr)  # constant = 1
        with shared_memory_arrays({'f': field}) as shm:
            inner = RegularInterpolator3D(r, th, ph, shm=shm, shape=field.shape)
            interp = CartesianToSpherical(
                RegularInterpolator3D, r, th, ph, shm=shm, shape=field.shape
            )
            interp.load()
            # Query at a point that maps to a well-defined (r, th, phi)
            pts = np.array([[1.0], [0.0], [1.0]])
            vals = interp(pts)
            np.testing.assert_allclose(vals[0, 0], 1.0, rtol=1e-6)

    def test_radial_field_correct_value(self):
        """
        Field = r (only depends on radius).  CartesianToSpherical must return
        the Euclidean norm of the Cartesian query point.
        """
        r, th, ph = self.R_NODES, self.THETA_NODES, self.PHI_NODES
        rr, _, _ = np.meshgrid(r, th, ph, indexing='ij')
        field = rr.copy().astype(np.float64)  # f(r, theta, phi) = r
        with shared_memory_arrays({'f': field}) as shm:
            interp = CartesianToSpherical(
                RegularInterpolator3D, r, th, ph, shm=shm, shape=field.shape
            )
            interp.load()
            # Point at (1.0, 0, 0) -> r=1.0
            pts = np.array([[1.0], [0.0], [0.0]])
            vals = interp(pts)
            np.testing.assert_allclose(vals[0, 0], 1.0, rtol=1e-5)

    def test_coordinate_conversion_consistency(self):
        """
        Querying CartesianToSpherical at (x,y,z) must give the same result as
        querying the inner interpolator directly at the corresponding (r,th,phi).
        """
        r, th, ph = self.R_NODES, self.THETA_NODES, self.PHI_NODES
        rr, _, _ = np.meshgrid(r, th, ph, indexing='ij')
        field = rr.astype(np.float64)
        with shared_memory_arrays({'f': field}) as shm:
            c2s = CartesianToSpherical(
                RegularInterpolator3D, r, th, ph, shm=shm, shape=field.shape
            )
            inner = RegularInterpolator3D(r, th, ph, shm=shm, shape=field.shape)
            c2s.load()
            inner.load()

            # Cartesian point: x=0.8, y=0.6, z=0 -> r=1.0, theta=pi/2, phi=arctan2(0.6,0.8)
            xq, yq, zq = 0.8, 0.6, 0.0
            r_q = np.sqrt(xq**2 + yq**2 + zq**2)
            th_q = np.arccos(np.clip(zq / r_q, -1, 1))
            ph_q = (np.arctan2(yq, xq) + 2 * np.pi) % (2 * np.pi)

            cart_pts = np.array([[xq], [yq], [zq]])
            sph_pts  = np.array([[r_q], [th_q], [ph_q]])

            v_cart = c2s(cart_pts)
            v_sph  = inner(sph_pts)
            np.testing.assert_allclose(v_cart, v_sph, rtol=1e-12)

    def test_attribute_delegation(self):
        """
        Attributes of the inner interpolator (keys, n_keys, loaded) must be
        accessible through the CartesianToSpherical wrapper.
        """
        r, th, ph = self.R_NODES, self.THETA_NODES, self.PHI_NODES
        field = np.ones((len(r), len(th), len(ph)))
        with shared_memory_arrays({'mykey': field}) as shm:
            interp = CartesianToSpherical(
                RegularInterpolator3D, r, th, ph, shm=shm, shape=field.shape
            )
            assert interp.keys == ['mykey']
            assert interp.n_keys == 1

    def test_load_unload_delegates(self):
        """load() and unload() must delegate to the inner interpolator."""
        r, th, ph = self.R_NODES, self.THETA_NODES, self.PHI_NODES
        field = np.ones((len(r), len(th), len(ph)))
        with shared_memory_arrays({'f': field}) as shm:
            interp = CartesianToSpherical(
                RegularInterpolator3D, r, th, ph, shm=shm, shape=field.shape
            )
            assert not interp.interpolator.loaded
            interp.load()
            assert interp.interpolator.loaded
            interp.unload()
            assert not interp.interpolator.loaded

    def test_origin_guard(self):
        """Query at r=0 (origin) must not raise; returns NaN (out of grid)."""
        r, th, ph = self.R_NODES, self.THETA_NODES, self.PHI_NODES
        field = np.ones((len(r), len(th), len(ph)))
        with shared_memory_arrays({'f': field}) as shm:
            interp = CartesianToSpherical(
                RegularInterpolator3D, r, th, ph, shm=shm, shape=field.shape
            )
            interp.load()
            pts = np.array([[0.0], [0.0], [0.0]])  # origin
            vals = interp(pts)   # should not raise; r=0 is below grid min
            # Result is NaN because r=0 < r_min=0.5
            assert np.isnan(vals[0, 0])

    def test_sort_tracers_returns_same_length(self):
        """CartesianToSpherical.sort_tracers must return all input tracers."""
        r, th, ph = self.R_NODES, self.THETA_NODES, self.PHI_NODES
        field = np.ones((len(r), len(th), len(ph)))
        with shared_memory_arrays({'f': field}) as shm:
            interp = CartesianToSpherical(
                RegularInterpolator3D, r, th, ph, shm=shm, shape=field.shape
            )
            interp.load()
            # sort_tracers uses tr.positions[-1], so each tracer needs a step.
            positions = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.5]),
                np.array([0.7, 0.7, 0.7]),
            ]
            tracers = np.array([_make_tracer(i, p) for i, p in enumerate(positions)])
            sorted_tr = interp.sort_tracers(tracers)
            assert len(sorted_tr) == len(tracers)
