"""
Unit tests for spherical_surface_by_area (src/seeds.py).

A MockSurfaceFileHandler replaces real file I/O.  The mock interpolator
evaluates analytic density and radial-velocity profiles.  The mass flux
sum is compared to the closed-form surface integral.
"""

import sys
import numpy as np
import pytest

from src.seeds import spherical_surface_by_area
from src.integrators import ExplicitTrapezoid


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

class _MockSurfaceInterpolator:
    """
    Analytic interpolator that returns (rho, v_r) at Cartesian coordinates
    without touching shared memory.

    The keys list must match the order used by spherical_surface_by_area:
    vel_keys first, then density_key -- i.e. ('vx', 'vy', 'vz', 'rho').

    The radial velocity vr_fn(r) is decomposed into Cartesian components
    proportional to (x, y, z)/r so that dot(v, r_hat) = vr_fn(r).
    """

    def __init__(self, rho_fn, vr_fn, keys):
        self._rho_fn = rho_fn
        self._vr_fn = vr_fn
        self.keys = list(keys)
        self.n_keys = len(self.keys)
        self.loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        x, y, z = [np.atleast_1d(c) for c in coords]
        r = np.sqrt(x**2 + y**2 + z**2)
        safe_r = np.where(r > 0, r, 1.0)
        rho = self._rho_fn(r)
        vr_mag = self._vr_fn(r)
        vx = vr_mag * x / safe_r
        vy = vr_mag * y / safe_r
        vz_comp = vr_mag * z / safe_r
        key_map = {'vx': vx, 'vy': vy, 'vz': vz_comp, 'rho': rho}
        result = np.empty((self.n_keys, len(x)))
        for i, k in enumerate(self.keys):
            result[i] = key_map[k]
        return result


class MockSurfaceFileHandler:
    """
    Minimal FileHandler substitute for spherical_surface_by_area.

    Exposes a simple uniform time axis and delegates interpolation to an
    analytic rho(r) and vr(r) profile via _MockSurfaceInterpolator.

    Parameters
    ----------
    rho_fn : callable
        rho_fn(r: ndarray) -> ndarray of same shape.
    vr_fn : callable
        vr_fn(r: ndarray) -> ndarray.  Positive = outward flux.
    times : array-like
        Uniformly spaced file times.
    """

    n_files_per_step = 1

    def __init__(self, rho_fn, vr_fn, times):
        self.times = np.asarray(times, dtype=float)
        self._rho_fn = rho_fn
        self._vr_fn = vr_fn
        self.keys = ['vx', 'vy', 'vz', 'rho']
        self.extra_data = {
            'rho_fn': rho_fn,
            'vr_fn': vr_fn,
            'keys': self.keys,
        }
        # shared_memory: one dict per n_files_per_step slot.
        # The mock interpolator ignores the shm names, so dummies suffice.
        self.shared_memory = tuple(
            {k: 'dummy' for k in self.keys}
            for _ in range(self.n_files_per_step)
        )
        self.cur_times = np.array([])
        self.parallel_kwargs = {
            'n_cpu': 1,
            'verbose': False,
            'file': sys.stdout,
        }

    def load_chunk(self, i_step: int, forward: bool = True) -> None:
        """Set cur_times to the n_files_per_step times starting at i_step."""
        n = self.n_files_per_step
        if forward:
            hi = min(i_step + n, len(self.times))
            self.cur_times = self.times[i_step:hi]
        else:
            lo = max(i_step - n + 1, 0)
            self.cur_times = self.times[lo : i_step + 1][::-1]

    def get_chunk_indices(
        self,
        start_t: float,
        end_t: float,
        overlap: bool = False,
        n_snap: int = 2,
    ) -> tuple:
        """Return chunk indices matching the real FileHandler logic (overlap=False)."""
        file_times = self.times
        n_fps = self.n_files_per_step
        forward = end_t >= start_t
        if forward:
            t_s = float(np.min(file_times[file_times >= start_t]))
            t_e = float(np.max(file_times[file_times <= end_t]))
            i0 = int(np.searchsorted(file_times, t_s, side='left'))
            i1 = int(np.searchsorted(file_times, t_e, side='left'))
            chunk_indices = np.arange(i0, i1 + 1, n_fps)
        else:
            t_s = float(np.max(file_times[file_times <= start_t]))
            t_e = float(np.min(file_times[file_times >= end_t]))
            i0 = int(np.searchsorted(file_times, t_s, side='left'))
            i1 = int(np.searchsorted(file_times, t_e, side='left'))
            chunk_indices = np.arange(i0, i1 - 1, -n_fps)
        return chunk_indices, forward, t_s, t_e

    @staticmethod
    def setup_interpolator(shm: dict, extra_data: dict) -> '_MockSurfaceInterpolator':
        return _MockSurfaceInterpolator(
            extra_data['rho_fn'],
            extra_data['vr_fn'],
            extra_data['keys'],
        )


# ---------------------------------------------------------------------------
# Analytical reference
# ---------------------------------------------------------------------------

def _analytical_surface_mass(
    rho: float,
    vr: float,
    r_surf: float,
    dt_slot: float,
    theta_min: float = 0.0,
    theta_max: float = np.pi,
    phi_min: float = 0.0,
    phi_max: float = 2 * np.pi,
) -> float:
    """
    Closed-form integral of rho * vr over a spherical surface patch,
    multiplied by dt_slot.

    For constant rho and vr:
        M = rho * vr * r^2 * (cos(theta_min) - cos(theta_max)) * (phi_max - phi_min) * dt
    """
    dA = r_surf**2 * (np.cos(theta_min) - np.cos(theta_max)) * (phi_max - phi_min)
    return rho * vr * dA * dt_slot


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestSphericalSurfaceByArea:
    """
    End-to-end tests for spherical_surface_by_area using analytic (rho, vr)
    profiles and the MockSurfaceFileHandler.

    Test geometry
    -------------
    r_surf = 2.0
    times  = [0.0, 1.0]   (2 file times, 1 time slot, dt_slot = 1.0)
    t_start = [0.0, 1.0]  (slot boundaries: 1 slot from t=0 to t=1)
    n_th = 4, n_ph = 6
    n_quad = 5  (surface GL; full-sphere accuracy < 1e-7 with n_quad=5)
    """

    R_SURF   = 2.0
    TIMES    = [0.0, 1.0]
    T_START  = np.array([0.0, 1.0])
    N_TH     = 4
    N_PH     = 6
    N_QUAD   = 5
    DT_SLOT  = 1.0  # |TIMES[1] - TIMES[0]|

    def _seed(self, rho_fn, vr_fn, **overrides):
        """Call spherical_surface_by_area and return the Tracers object."""
        fh = MockSurfaceFileHandler(rho_fn, vr_fn, self.TIMES)
        kwargs = dict(
            r_surf=self.R_SURF,
            t_start=self.T_START,
            n_th=self.N_TH,
            n_ph=self.N_PH,
            n_quad=self.N_QUAD,
            random_shift_in_cell=False,
            file_handler=fh,
            vel_keys=['vx', 'vy', 'vz'],
            integrator=ExplicitTrapezoid(),
            density_key='rho',
        )
        kwargs.update(overrides)
        return spherical_surface_by_area(**kwargs)

    def _total_mass(self, tracers):
        return sum(tr.props['mass'] for tr in tracers.tracers)

    # -- basic structure ---------------------------------------------------

    def test_tracer_count(self):
        """
        Number of tracers = n_t_slots * n_th * n_ph.
        With t_start=[0, 1] there is 1 slot, so count = 1 * 4 * 6 = 24.
        """
        tracers = self._seed(
            lambda r: np.ones_like(r),
            lambda r: np.ones_like(r),
        )
        n_slots = len(self.T_START) - 1
        assert len(tracers.tracers) == n_slots * self.N_TH * self.N_PH

    def test_positions_on_sphere(self):
        """All injected positions must lie exactly on the sphere of radius r_surf."""
        tracers = self._seed(
            lambda r: np.ones_like(r),
            lambda r: np.ones_like(r),
        )
        for tr in tracers.tracers:
            r = float(np.linalg.norm(tr.initial_position))
            np.testing.assert_allclose(
                r, self.R_SURF, rtol=1e-12,
                err_msg=f"Tracer {tr.id}: r={r:.6f} != r_surf={self.R_SURF}",
            )

    def test_props_mass_stored(self):
        """Each tracer must have a 'mass' entry in its props dict."""
        tracers = self._seed(
            lambda r: np.ones_like(r),
            lambda r: np.ones_like(r),
        )
        assert all('mass' in tr.props for tr in tracers.tracers)

    # -- zero flux ---------------------------------------------------------

    def test_zero_radial_velocity_zero_mass(self):
        """
        With v_r = 0, the surface mass flux is zero for every tracer.
        """
        tracers = self._seed(
            lambda r: np.ones_like(r),      # rho = 1
            lambda r: np.zeros_like(r),     # v_r = 0 -> zero flux
        )
        for tr in tracers.tracers:
            assert tr.props['mass'] == pytest.approx(0.0, abs=1e-14), (
                f"Tracer {tr.id}: expected mass=0 for zero v_r, got {tr.props['mass']}"
            )

    def test_zero_density_zero_mass(self):
        """With rho = 0, mass flux is zero for every tracer."""
        tracers = self._seed(
            lambda r: np.zeros_like(r),     # rho = 0
            lambda r: np.ones_like(r),      # v_r = 1
        )
        for tr in tracers.tracers:
            assert tr.props['mass'] == pytest.approx(0.0, abs=1e-14)

    # -- total mass vs analytic --------------------------------------------

    def test_unit_vr_const_rho_total_mass(self):
        """
        rho = 1, v_r = 1 (unit radial), dt = 1:
            Total mass = 4 * pi * r_surf^2 * rho * vr * dt_slot
        """
        rho_0, vr_0 = 1.0, 1.0
        tracers = self._seed(
            lambda r: np.full_like(r, rho_0),
            lambda r: np.full_like(r, vr_0),
        )
        mass_sum = self._total_mass(tracers)
        mass_exact = _analytical_surface_mass(rho_0, vr_0, self.R_SURF, self.DT_SLOT)
        np.testing.assert_allclose(mass_sum, mass_exact, rtol=1e-5)

    def test_scaled_rho_scales_mass(self):
        """Doubling rho must double the total mass."""
        tracers_1 = self._seed(
            lambda r: np.ones_like(r),
            lambda r: np.ones_like(r),
        )
        tracers_2 = self._seed(
            lambda r: np.full_like(r, 2.0),
            lambda r: np.ones_like(r),
        )
        np.testing.assert_allclose(
            self._total_mass(tracers_2),
            2 * self._total_mass(tracers_1),
            rtol=1e-10,
        )

    def test_scaled_vr_scales_mass(self):
        """Doubling v_r must double the total mass."""
        tracers_1 = self._seed(
            lambda r: np.ones_like(r),
            lambda r: np.ones_like(r),
        )
        tracers_2 = self._seed(
            lambda r: np.ones_like(r),
            lambda r: np.full_like(r, 2.0),
        )
        np.testing.assert_allclose(
            self._total_mass(tracers_2),
            2 * self._total_mass(tracers_1),
            rtol=1e-10,
        )

    # -- partial domains ---------------------------------------------------

    def test_half_phi_half_mass(self):
        """
        Restricting phi to [0, pi] must give exactly half the mass of [0, 2*pi]
        for uniform rho and v_r (since dA is proportional to Delta phi).
        """
        fh = MockSurfaceFileHandler(
            lambda r: np.ones_like(r),
            lambda r: np.ones_like(r),
            self.TIMES,
        )
        common = dict(
            r_surf=self.R_SURF, t_start=self.T_START,
            n_th=self.N_TH, n_ph=self.N_PH, n_quad=self.N_QUAD,
            random_shift_in_cell=False, file_handler=fh,
            vel_keys=['vx', 'vy', 'vz'], integrator=ExplicitTrapezoid(),
            density_key='rho',
        )
        full = spherical_surface_by_area(phi_min=0.0, phi_max=2 * np.pi, **common)
        half = spherical_surface_by_area(phi_min=0.0, phi_max=np.pi, **common)
        np.testing.assert_allclose(
            self._total_mass(half),
            self._total_mass(full) / 2,
            rtol=1e-8,
        )

    def test_half_phi_absolute_mass(self):
        """Half-phi domain: mass matches analytic integral for phi in [0, pi]."""
        rho_0, vr_0 = 1.0, 1.0
        tracers = self._seed(
            lambda r: np.full_like(r, rho_0),
            lambda r: np.full_like(r, vr_0),
            phi_min=0.0, phi_max=np.pi,
        )
        mass_exact = _analytical_surface_mass(
            rho_0, vr_0, self.R_SURF, self.DT_SLOT,
            phi_min=0.0, phi_max=np.pi,
        )
        np.testing.assert_allclose(self._total_mass(tracers), mass_exact, rtol=1e-5)

    def test_northern_hemisphere_mass(self):
        """theta in [0, pi/2]: mass matches analytic integral."""
        rho_0, vr_0 = 1.0, 1.0
        tracers = self._seed(
            lambda r: np.full_like(r, rho_0),
            lambda r: np.full_like(r, vr_0),
            theta_min=0.0, theta_max=np.pi / 2,
        )
        mass_exact = _analytical_surface_mass(
            rho_0, vr_0, self.R_SURF, self.DT_SLOT,
            theta_min=0.0, theta_max=np.pi / 2,
        )
        np.testing.assert_allclose(self._total_mass(tracers), mass_exact, rtol=1e-5)

    def test_north_plus_south_equals_full(self):
        """North + south hemisphere masses must sum to full-sphere mass."""
        fh = MockSurfaceFileHandler(
            lambda r: np.ones_like(r),
            lambda r: np.ones_like(r),
            self.TIMES,
        )
        common = dict(
            r_surf=self.R_SURF, t_start=self.T_START,
            n_th=self.N_TH, n_ph=self.N_PH, n_quad=self.N_QUAD,
            random_shift_in_cell=False, file_handler=fh,
            vel_keys=['vx', 'vy', 'vz'], integrator=ExplicitTrapezoid(),
            density_key='rho',
        )
        north = spherical_surface_by_area(theta_min=0.0,      theta_max=np.pi / 2, **common)
        south = spherical_surface_by_area(theta_min=np.pi / 2, theta_max=np.pi,    **common)
        full  = spherical_surface_by_area(theta_min=0.0,      theta_max=np.pi,     **common)
        np.testing.assert_allclose(
            self._total_mass(north) + self._total_mass(south),
            self._total_mass(full),
            rtol=1e-8,
        )

    def test_two_slots_double_mass(self):
        """
        With two equal time slots (dt=1 each), the total mass from a uniform
        rho=1, vr=1 profile must be double that of one slot.
        """
        times_2 = [0.0, 1.0, 2.0]
        t_start_2 = np.array([0.0, 1.0, 2.0])

        fh_1 = MockSurfaceFileHandler(
            lambda r: np.ones_like(r),
            lambda r: np.ones_like(r),
            self.TIMES,
        )
        fh_2 = MockSurfaceFileHandler(
            lambda r: np.ones_like(r),
            lambda r: np.ones_like(r),
            times_2,
        )
        t1 = spherical_surface_by_area(
            r_surf=self.R_SURF, t_start=self.T_START,
            n_th=self.N_TH, n_ph=self.N_PH, n_quad=self.N_QUAD,
            random_shift_in_cell=False, file_handler=fh_1,
            vel_keys=['vx', 'vy', 'vz'], integrator=ExplicitTrapezoid(),
            density_key='rho',
        )
        t2 = spherical_surface_by_area(
            r_surf=self.R_SURF, t_start=t_start_2,
            n_th=self.N_TH, n_ph=self.N_PH, n_quad=self.N_QUAD,
            random_shift_in_cell=False, file_handler=fh_2,
            vel_keys=['vx', 'vy', 'vz'], integrator=ExplicitTrapezoid(),
            density_key='rho',
        )
        np.testing.assert_allclose(
            self._total_mass(t2), 2 * self._total_mass(t1), rtol=1e-8,
        )
