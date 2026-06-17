"""
Unit tests for src/seeds.py seeding functions.

All tests use a MockFileHandler that bypasses real file I/O and shared
memory, replacing the interpolator with an analytic density function.
The mass sum returned by the seeder is compared to the exact volume
integral of the chosen density profile.
"""

import sys
import numpy as np
import pytest

from src.seeds import (
    spherical_by_volume,
    _gauss_legendre_3d,
    _gauss_legendre_surface,
)
from src.integrators import ExplicitTrapezoid


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

class _MockInterpolator:
    """
    Analytic interpolator that evaluates a density function at arbitrary
    Cartesian coordinates without touching shared memory.

    Parameters
    ----------
    density_fn : callable
        Signature: density_fn(r: ndarray) -> ndarray of same shape.
    keys : sequence of str
        Data keys.  The density is returned for every key so that the
        caller can pick it up by index without separate key logic.
    """

    def __init__(self, density_fn, keys):
        self._density_fn = density_fn
        self.keys = list(keys)
        self.n_keys = len(self.keys)
        self.loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Return (n_keys, n_points) with the analytic density at each point."""
        x, y, z = coords
        r = np.sqrt(x**2 + y**2 + z**2)
        rho = self._density_fn(r)
        return np.tile(rho, (self.n_keys, 1))


class MockFileHandler:
    """
    Minimal FileHandler substitute that exposes a single time step and
    delegates interpolation to a user-supplied density function.

    Parameters
    ----------
    density_fn : callable
        Maps r (ndarray) -> density (ndarray, same shape).
    keys : sequence of str, optional
        Data keys to expose (default: ``('rho',)``).
    time : float, optional
        The single file time (default: 0.0).
    """

    def __init__(self, density_fn, keys=('rho',), time=0.0):
        self._density_fn = density_fn
        self.keys = list(keys)
        self.times = np.array([time])
        self.cur_times = np.array([time])
        # shared_memory is only passed to setup_interpolator which ignores it.
        self.shared_memory = ({k: '' for k in self.keys},)
        # extra_data carries the density function into the static method.
        self.extra_data = {'density_fn': density_fn, 'keys': self.keys}
        self.parallel_kwargs = {
            'n_cpu': 1,
            'verbose': False,
            'file': sys.stdout,
        }

    def load_chunk(self, index: int, forward: bool = True) -> None:
        """No-op: data is always available via the analytic interpolator."""

    @staticmethod
    def setup_interpolator(shm: dict, extra_data: dict) -> _MockInterpolator:
        """Return a fresh MockInterpolator for the density function in extra_data."""
        return _MockInterpolator(extra_data['density_fn'], extra_data['keys'])


# ---------------------------------------------------------------------------
# Analytical reference integrals
# ---------------------------------------------------------------------------

def _analytical_mass_sphere(density_fn, r_min, r_max,
                             theta_min=0.0, theta_max=np.pi,
                             phi_min=0.0, phi_max=2*np.pi,
                             n_quad=200):
    """
    Numerically integrate density_fn(r) * r^2 * sin(theta) over a spherical
    shell using a very fine Gauss-Legendre quadrature as the reference answer.

    The returned value serves as the ground-truth total mass against which the
    seeder's cell-mass sum is compared.
    """
    xi, wi = np.polynomial.legendre.leggauss(n_quad)

    # Radial integral
    r_c, r_h = (r_min + r_max) / 2, (r_max - r_min) / 2
    r_pts = r_c + r_h * xi
    rho_r = density_fn(r_pts)
    I_r = np.dot(wi, rho_r * r_pts**2) * r_h

    # Theta integral: integral of sin(theta) from theta_min to theta_max
    #   = cos(theta_min) - cos(theta_max)
    I_theta = np.cos(theta_min) - np.cos(theta_max)

    # Phi integral
    I_phi = phi_max - phi_min

    return I_r * I_theta * I_phi


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature helper tests
# ---------------------------------------------------------------------------

class TestGaussLegendre3D:
    """Tests for the internal _gauss_legendre_3d quadrature helper."""

    def test_nquad1_weight_equals_midpoint_dV(self):
        # For n_quad=1 the single weight must equal the midpoint-rule cell
        # volume r_c^2 * sin(th_c) * dr * dth * dph (exact by construction).
        r_lo, r_hi = 1.0, 2.0
        th_lo, th_hi = np.pi/4, 3*np.pi/4
        ph_lo, ph_hi = 0.0, np.pi
        r_c  = (r_lo + r_hi) / 2
        th_c = (th_lo + th_hi) / 2
        dV_midpoint = (
            r_c**2 * np.sin(th_c)
            * (r_hi - r_lo) * (th_hi - th_lo) * (ph_hi - ph_lo)
        )
        pts, wts = _gauss_legendre_3d(r_lo, r_hi, th_lo, th_hi, ph_lo, ph_hi, n_quad=1)
        np.testing.assert_allclose(wts.sum(), dV_midpoint, rtol=1e-14)

    def test_weights_converge_to_exact_volume(self):
        # sum(weights) must converge to the exact cell volume as n_quad grows.
        # GL is not polynomial-exact for sin(theta), so we need n_quad >= 5.
        r_lo, r_hi = 1.0, 2.0
        th_lo, th_hi = np.pi/4, 3*np.pi/4
        ph_lo, ph_hi = 0.0, np.pi
        dV_exact = (
            (r_hi**3 - r_lo**3) / 3
            * (np.cos(th_lo) - np.cos(th_hi))
            * (ph_hi - ph_lo)
        )
        for n in [5, 6]:
            pts, wts = _gauss_legendre_3d(r_lo, r_hi, th_lo, th_hi, ph_lo, ph_hi, n)
            np.testing.assert_allclose(
                wts.sum(), dV_exact, rtol=1e-5,
                err_msg=f"n_quad={n}: weight sum does not match cell volume"
            )

    def test_constant_density_exact(self):
        # rho = 1: mass = dV. GL needs n_quad=6 to reach < 1e-8 rel-error
        # for a full-sphere theta span (sin is not polynomial).
        r_lo, r_hi = 2.0, 3.0
        th_lo, th_hi = 0.0, np.pi
        ph_lo, ph_hi = 0.0, 2*np.pi
        pts, wts = _gauss_legendre_3d(r_lo, r_hi, th_lo, th_hi, ph_lo, ph_hi, n_quad=6)
        mass = np.dot(wts, np.ones(len(wts)))
        dV_exact = 4 * np.pi / 3 * (r_hi**3 - r_lo**3)
        np.testing.assert_allclose(mass, dV_exact, rtol=1e-8)

    def test_r_squared_density_exact_with_nquad6(self):
        # rho = r^2: integrand is r^4 * sin(theta). The r^4 part is exact
        # with n_quad=3 (GL exact to degree 2*3-1=5). The sin(theta) part is
        # transcendental; n_quad=6 gives < 1e-8 relative error over full sphere.
        r_lo, r_hi = 1.0, 2.0
        th_lo, th_hi = 0.0, np.pi
        ph_lo, ph_hi = 0.0, 2*np.pi
        pts, wts = _gauss_legendre_3d(r_lo, r_hi, th_lo, th_hi, ph_lo, ph_hi, n_quad=6)
        x, y, z = pts
        r = np.sqrt(x**2 + y**2 + z**2)
        rho = r**2
        mass = np.dot(wts, rho)
        # Exact: 4pi * integral_r_lo^r_hi r^4 dr = 4pi/5 * (r_hi^5 - r_lo^5)
        mass_exact = 4 * np.pi / 5 * (r_hi**5 - r_lo**5)
        np.testing.assert_allclose(mass, mass_exact, rtol=1e-8)

    def test_output_shapes(self):
        pts, wts = _gauss_legendre_3d(1.0, 2.0, 0.0, np.pi, 0.0, 2*np.pi, n_quad=2)
        assert pts.shape == (3, 8), f"Expected (3, 8) got {pts.shape}"
        assert wts.shape == (8,)

    def test_points_inside_cell(self):
        r_lo, r_hi = 1.0, 3.0
        th_lo, th_hi = np.pi/6, np.pi/2
        ph_lo, ph_hi = 0.3, 1.5
        pts, wts = _gauss_legendre_3d(r_lo, r_hi, th_lo, th_hi, ph_lo, ph_hi, n_quad=3)
        x, y, z = pts
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.clip(z / r, -1, 1))
        phi = (np.arctan2(y, x) + 2*np.pi) % (2*np.pi)
        assert np.all(r >= r_lo - 1e-10) and np.all(r <= r_hi + 1e-10)
        assert np.all(theta >= th_lo - 1e-10) and np.all(theta <= th_hi + 1e-10)


class TestGaussLegendreSupface:
    """Tests for the internal _gauss_legendre_surface quadrature helper."""

    def test_nquad1_weight_equals_midpoint_area(self):
        # For n_quad=1 the single weight must equal the midpoint surface element
        # r^2 * sin(th_c) * dth * dph (exact by construction of the GL formula).
        r_surf = 2.0
        th_lo, th_hi = np.pi/4, 3*np.pi/4
        ph_lo, ph_hi = 0.0, np.pi
        th_c = (th_lo + th_hi) / 2
        dA_midpoint = r_surf**2 * np.sin(th_c) * (th_hi - th_lo) * (ph_hi - ph_lo)
        x, y, z, wts = _gauss_legendre_surface(r_surf, th_lo, th_hi, ph_lo, ph_hi, n_quad=1)
        np.testing.assert_allclose(wts.sum(), dA_midpoint, rtol=1e-14)

    def test_weights_converge_to_exact_area(self):
        # sum(weights) converges to exact area r^2*(cos(th_lo)-cos(th_hi))*(ph_hi-ph_lo).
        # GL requires n_quad >= 5 to achieve < 1e-5 for the large angular span used.
        r_surf = 2.0
        th_lo, th_hi = np.pi/4, 3*np.pi/4
        ph_lo, ph_hi = 0.0, np.pi
        area_exact = r_surf**2 * (np.cos(th_lo) - np.cos(th_hi)) * (ph_hi - ph_lo)
        for n in [5, 6]:
            x, y, z, wts = _gauss_legendre_surface(r_surf, th_lo, th_hi, ph_lo, ph_hi, n)
            np.testing.assert_allclose(
                wts.sum(), area_exact, rtol=1e-5,
                err_msg=f"n_quad={n}: weight sum does not match surface area"
            )

    def test_full_sphere_area(self):
        # Full sphere: 4*pi*r^2. n_quad=6 gives < 1e-8 relative error.
        r_surf = 3.0
        x, y, z, wts = _gauss_legendre_surface(
            r_surf, 0.0, np.pi, 0.0, 2*np.pi, n_quad=6
        )
        np.testing.assert_allclose(wts.sum(), 4*np.pi*r_surf**2, rtol=1e-8)

    def test_output_shapes(self):
        x, y, z, wts = _gauss_legendre_surface(1.0, 0.0, np.pi, 0.0, 2*np.pi, n_quad=2)
        assert len(x) == len(y) == len(z) == len(wts) == 4

    def test_points_on_sphere(self):
        r_surf = 5.0
        x, y, z, wts = _gauss_legendre_surface(r_surf, 0.0, np.pi, 0.0, 2*np.pi, n_quad=3)
        r = np.sqrt(x**2 + y**2 + z**2)
        np.testing.assert_allclose(r, r_surf, rtol=1e-12)


# ---------------------------------------------------------------------------
# spherical_by_volume mass integral tests
# ---------------------------------------------------------------------------

class TestSphericalByVolume:
    """
    Tests that verify the seeder correctly integrates the density field to
    assign a mass to each tracer, such that the total matches the analytic
    integral over the seeded domain.
    """

    # Shared geometry - coarse enough to be fast, fine enough to test accuracy
    R_MIN   = 1.0
    R_MAX   = 3.0
    N_R     = 4
    N_TH    = 5
    N_PH    = 6
    N_QUAD  = 4   # 4^3 = 64 quadrature points per cell

    def _seed(self, density_fn, **overrides):
        """Helper: call spherical_by_volume and return the Tracers object."""
        fh = MockFileHandler(density_fn)
        kwargs = dict(
            r_min=self.R_MIN,
            r_max=self.R_MAX,
            n_r=self.N_R,
            n_th=self.N_TH,
            n_ph=self.N_PH,
            start_t=0.0,
            n_quad=self.N_QUAD,
            random_shift_in_cell=False,
            file_handler=fh,
            vel_keys=['vx', 'vy', 'vz'],
            integrator=ExplicitTrapezoid(),
            density_key='rho',
        )
        kwargs.update(overrides)
        return spherical_by_volume(**kwargs)

    def _total_mass(self, tracers):
        return sum(tr.props['mass'] for tr in tracers.tracers)

    # -- constant density -------------------------------------------------

    def test_constant_density_full_sphere_mass(self):
        """
        rho = 1 everywhere.
        Exact total mass = (4/3) * pi * (r_max^3 - r_min^3).
        """
        rho_0 = 1.0
        tracers = self._seed(lambda r: np.full_like(r, rho_0))
        mass_sum = self._total_mass(tracers)
        mass_exact = _analytical_mass_sphere(
            lambda r: np.full_like(r, rho_0), self.R_MIN, self.R_MAX
        )
        np.testing.assert_allclose(mass_sum, mass_exact, rtol=1e-5)

    def test_constant_density_number_of_tracers(self):
        """Number of tracers must equal n_r * n_th * n_ph."""
        tracers = self._seed(lambda r: np.ones_like(r))
        assert len(tracers.tracers) == self.N_R * self.N_TH * self.N_PH

    def test_constant_density_all_masses_positive(self):
        tracers = self._seed(lambda r: np.ones_like(r))
        assert all(tr.props['mass'] > 0 for tr in tracers.tracers)

    def test_constant_density_dV_stored(self):
        """Each tracer props must include a positive dV volume element."""
        tracers = self._seed(lambda r: np.ones_like(r))
        assert all('dV' in tr.props for tr in tracers.tracers)
        assert all(tr.props['dV'] > 0 for tr in tracers.tracers)

    # -- power-law density (rho = r^2) ------------------------------------

    def test_power_law_r2_full_sphere_mass(self):
        """
        rho = r^2.
        Exact total mass = (4/5) * pi * (r_max^5 - r_min^5).
        n_quad=4 should be more than sufficient for a degree-4 radial integrand.
        """
        tracers = self._seed(lambda r: r**2)
        mass_sum = self._total_mass(tracers)
        mass_exact = _analytical_mass_sphere(lambda r: r**2, self.R_MIN, self.R_MAX)
        np.testing.assert_allclose(mass_sum, mass_exact, rtol=1e-5)

    def test_power_law_r1_full_sphere_mass(self):
        """rho = r.  Exact: 4*pi * (r_max^4 - r_min^4) / 4."""
        tracers = self._seed(lambda r: r)
        mass_sum = self._total_mass(tracers)
        mass_exact = _analytical_mass_sphere(lambda r: r, self.R_MIN, self.R_MAX)
        np.testing.assert_allclose(mass_sum, mass_exact, rtol=1e-5)

    def test_higher_density_gives_higher_mass(self):
        """Doubling density must double total mass."""
        t1 = self._seed(lambda r: np.ones_like(r))
        t2 = self._seed(lambda r: np.full_like(r, 2.0))
        np.testing.assert_allclose(
            self._total_mass(t2), 2 * self._total_mass(t1), rtol=1e-10
        )

    # -- partial phi domain -----------------------------------------------

    def test_half_phi_half_mass(self):
        """
        Restricting phi to [0, pi] (half the circle) must give exactly half
        the mass of a [0, 2*pi] domain for uniform density.
        """
        fh = MockFileHandler(lambda r: np.ones_like(r))
        common = dict(
            r_min=self.R_MIN, r_max=self.R_MAX,
            n_r=self.N_R, n_th=self.N_TH, n_ph=self.N_PH,
            start_t=0.0, n_quad=self.N_QUAD, random_shift_in_cell=False,
            file_handler=fh, vel_keys=['vx', 'vy', 'vz'],
            integrator=ExplicitTrapezoid(), density_key='rho',
        )
        full  = spherical_by_volume(phi_min=0.0,  phi_max=2*np.pi, **common)
        half  = spherical_by_volume(phi_min=0.0,  phi_max=np.pi,   **common)
        np.testing.assert_allclose(
            self._total_mass(half), self._total_mass(full) / 2, rtol=1e-8
        )

    def test_half_phi_correct_absolute_mass(self):
        """Half-sphere phi domain: exact mass comparison to analytic integral."""
        tracers = self._seed(
            lambda r: np.ones_like(r),
            phi_min=0.0, phi_max=np.pi,
        )
        mass_sum = self._total_mass(tracers)
        mass_exact = _analytical_mass_sphere(
            lambda r: np.ones_like(r), self.R_MIN, self.R_MAX,
            phi_min=0.0, phi_max=np.pi
        )
        np.testing.assert_allclose(mass_sum, mass_exact, rtol=1e-5)

    # -- partial theta domain ---------------------------------------------

    def test_northern_hemisphere_mass(self):
        """theta in [0, pi/2] must integrate correctly (northern hemisphere)."""
        tracers = self._seed(
            lambda r: np.ones_like(r),
            theta_min=0.0, theta_max=np.pi/2,
        )
        mass_sum = self._total_mass(tracers)
        mass_exact = _analytical_mass_sphere(
            lambda r: np.ones_like(r), self.R_MIN, self.R_MAX,
            theta_min=0.0, theta_max=np.pi/2,
        )
        np.testing.assert_allclose(mass_sum, mass_exact, rtol=1e-5)

    def test_north_plus_south_hemisphere_equals_full(self):
        """North + south hemisphere masses must sum to the full-sphere mass."""
        fh = MockFileHandler(lambda r: np.ones_like(r))
        common = dict(
            r_min=self.R_MIN, r_max=self.R_MAX,
            n_r=self.N_R, n_th=self.N_TH, n_ph=self.N_PH,
            start_t=0.0, n_quad=self.N_QUAD, random_shift_in_cell=False,
            file_handler=fh, vel_keys=['vx', 'vy', 'vz'],
            integrator=ExplicitTrapezoid(), density_key='rho',
        )
        north = spherical_by_volume(theta_min=0.0,    theta_max=np.pi/2, **common)
        south = spherical_by_volume(theta_min=np.pi/2, theta_max=np.pi,   **common)
        full  = spherical_by_volume(theta_min=0.0,    theta_max=np.pi,   **common)
        np.testing.assert_allclose(
            self._total_mass(north) + self._total_mass(south),
            self._total_mass(full),
            rtol=1e-8,
        )

    # -- mass per tracer vs cell volume -----------------------------------

    def test_uniform_density_mass_equals_rho_times_dV(self):
        """
        For rho = rho_0 and n_quad=1, each tracer's mass must exactly equal
        rho_0 * dV. This holds by construction: the single GL node lands at
        the cell centre with weight r_c^2 * sin(th_c) * dr * dth * dph = dV.
        """
        rho_0 = 3.7
        # Force n_quad=1 so GL weight == dV exactly (midpoint rule)
        tracers = self._seed(lambda r: np.full_like(r, rho_0), n_quad=1)
        for tr in tracers.tracers:
            np.testing.assert_allclose(
                tr.props['mass'], rho_0 * tr.props['dV'], rtol=1e-14,
                err_msg=f"Tracer {tr.id}: mass != rho * dV"
            )

    # -- quadrature accuracy vs n_quad ------------------------------------

    def test_mass_accuracy_improves_with_n_quad(self):
        """
        For rho=r^2, error with n_quad=4 must be smaller than with n_quad=1.
        """
        density_fn = lambda r: r**2
        mass_exact = _analytical_mass_sphere(density_fn, self.R_MIN, self.R_MAX)

        t_coarse = self._seed(density_fn, n_quad=1)
        t_fine   = self._seed(density_fn, n_quad=4)

        err_coarse = abs(self._total_mass(t_coarse) - mass_exact)
        err_fine   = abs(self._total_mass(t_fine)   - mass_exact)

        assert err_fine < err_coarse, (
            f"Higher n_quad did not improve accuracy: "
            f"err_coarse={err_coarse:.3e}, err_fine={err_fine:.3e}"
        )

    # -- tracer positions inside the expected domain ----------------------

    def test_positions_inside_radial_range(self):
        """
        All seeded tracer positions must lie within [r_min, r_max] in radius.
        (True regardless of random jitter, since jitter stays within cell.)
        """
        r_min, r_max = 1.5, 4.0
        tracers = self._seed(
            lambda r: np.ones_like(r),
            r_min=r_min, r_max=r_max, random_shift_in_cell=False,
        )
        for tr in tracers.tracers:
            r = float(np.linalg.norm(tr.initial_position))
            assert r_min <= r <= r_max, (
                f"Tracer {tr.id} at r={r:.4f} outside [{r_min}, {r_max}]"
            )
