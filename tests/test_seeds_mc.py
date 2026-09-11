"""
Unit tests for the mass-weighted Monte-Carlo seeders in src/seeds.py.

Both seeders are driven by the analytic mock file handlers of the
neighbouring seed tests.  The checks are that the sampled total mass
matches the closed-form integral, that every tracer carries exactly
M_tot / n_tracers, and that the tracer positions really are distributed
like the mass (not like the volume).
"""

import numpy as np
import pytest

from src.seeds import spherical_by_volume_mc, spherical_surface_mc, _auto_grid
from src.utils import cell_measure, tensor_cell_bounds
from src.integrators import ExplicitTrapezoid
from tests.test_seeds import MockFileHandler
from tests.test_seeds_surface import MockSurfaceFileHandler


R_MIN, R_MAX = 100.0, 1000.0


def _masses(tracers):
    return np.array([tr.props['mass'] for tr in tracers.tracers])


def _radii(tracers):
    return np.linalg.norm(np.array([tr.initial_position for tr in tracers.tracers]), axis=1)


class TestAutoGrid:
    def test_cell_count_and_isotropy(self):
        n_r, n_th, n_ph = _auto_grid(8000, (2.0, np.pi, 2 * np.pi))
        # phi spans twice the theta range, so it gets twice the cells.
        assert n_ph == pytest.approx(2 * n_th, rel=0.05)
        assert n_r * n_th * n_ph == pytest.approx(8000, rel=0.15)

    def test_at_least_two_cells_per_axis(self):
        # A thin wedge rounds its short axis below one cell; it is clamped to 2.
        assert _auto_grid(1, (1.0, 1.0)) == (2, 2)
        assert _auto_grid(4000, (2.3, np.pi, np.radians(5)))[2] == 2


class TestVolumeMC:
    """spherical_by_volume_mc against rho = r^-3 (mass uniform in ln r)."""

    @staticmethod
    def _seed(n_tracers=4000, cells_per_tracer=8, seed=12345):
        np.random.seed(seed)
        fh = MockFileHandler(lambda r: r**-3.0, keys=('rho',), time=0.0)
        return spherical_by_volume_mc(
            r_min=R_MIN, r_max=R_MAX, n_tracers=n_tracers, start_t=0.0,
            cells_per_tracer=cells_per_tracer,
            file_handler=fh, integrator=ExplicitTrapezoid(), vel_keys=('rho',),
        )

    def test_total_mass_matches_analytic(self):
        tracers = self._seed()
        # M = 4 pi int r^-3 r^2 dr = 4 pi ln(r_max/r_min)
        expected = 4 * np.pi * np.log(R_MAX / R_MIN)
        assert _masses(tracers).sum() == pytest.approx(expected, rel=1e-3)

    def test_all_masses_equal(self):
        m = _masses(self._seed(n_tracers=500))
        assert np.all(m == m[0])
        assert len(m) == 500

    def test_positions_inside_domain(self):
        r = _radii(self._seed(n_tracers=500))
        assert r.min() >= R_MIN and r.max() <= R_MAX

    def test_radial_distribution_tracks_mass(self):
        # rho ~ r^-3 puts equal mass in every log-radial decade, so the
        # sampled radii must be uniform in ln r (a volume-uniform sampler
        # would pile up at r_max instead).
        r = _radii(self._seed(n_tracers=20000))
        u = np.log(r / R_MIN) / np.log(R_MAX / R_MIN)
        counts, _ = np.histogram(u, bins=10, range=(0, 1))
        assert np.allclose(counts, len(r) / 10, rtol=0.1)


class _HemisphericInterpolator:
    """Constant rho with v_r outward in the north, inward in the south."""

    def __init__(self, rho, v_r, keys):
        self._rho, self._v_r, self.keys = rho, v_r, list(keys)
        self.n_keys = len(self.keys)

    def load(self):
        pass

    def unload(self):
        pass

    def __call__(self, coords):
        x, y, z = coords
        r = np.sqrt(x**2 + y**2 + z**2)
        v_r = np.where(z >= 0, self._v_r, -self._v_r)
        vals = {'rho': np.full_like(r, self._rho),
                'vx': v_r * x / r, 'vy': v_r * y / r, 'vz': v_r * z / r}
        return np.array([vals[k] for k in self.keys])


class _HemisphericFileHandler(MockSurfaceFileHandler):
    def __init__(self, rho, v_r, times):
        super().__init__(rho_fn=lambda r: np.full_like(r, rho),
                         vr_fn=lambda r: np.full_like(r, v_r), times=times)
        self.extra_data = {'rho': rho, 'v_r': v_r, 'keys': self.keys}

    @staticmethod
    def setup_interpolator(shm, extra_data):
        return _HemisphericInterpolator(extra_data['rho'], extra_data['v_r'],
                                        list(shm.keys()))


class TestSurfaceMC:
    """spherical_surface_mc against a constant rho and v_r on the sphere."""

    R_SURF = 300.0
    RHO = 2.0
    V_R = 0.5

    @staticmethod
    def _seed(n_tracers=2000, times=np.linspace(0.0, 10.0, 11), vr=None, seed=999):
        np.random.seed(seed)
        vr_fn = vr if vr is not None else (lambda r: np.full_like(r, TestSurfaceMC.V_R))
        fh = MockSurfaceFileHandler(
            rho_fn=lambda r: np.full_like(r, TestSurfaceMC.RHO),
            vr_fn=vr_fn,
            times=times,
        )
        return spherical_surface_mc(
            r_surf=TestSurfaceMC.R_SURF, t_start=times, n_tracers=n_tracers,
            file_handler=fh, integrator=ExplicitTrapezoid(),
            vel_keys=('vx', 'vy', 'vz'),
        )

    def test_total_mass_matches_analytic(self):
        times = np.linspace(0.0, 10.0, 11)
        tracers = self._seed(times=times)
        # M = rho * v_r * 4 pi r^2 * T, with the trapezoidal dt summing to T.
        expected = self.RHO * self.V_R * 4 * np.pi * self.R_SURF**2 * (times[-1] - times[0])
        assert _masses(tracers).sum() == pytest.approx(expected, rel=1e-3)

    def test_inflow_gives_negative_masses(self):
        # Inflow is sampled with the same probability (the weight magnitude is
        # the same) but its tracers subtract from the net crossing mass.
        out = _masses(self._seed(n_tracers=100))
        inn = _masses(self._seed(n_tracers=100,
                                 vr=lambda r: np.full_like(r, -self.V_R)))
        assert np.all(out > 0) and np.all(inn < 0)
        assert inn.sum() == pytest.approx(-out.sum(), rel=1e-12)

    def test_net_flux_cancels_for_a_hemispheric_wind(self):
        # v_r > 0 on the northern hemisphere and < 0 on the southern one: the
        # net crossing mass is zero, but |D v_r| still seeds the whole sphere
        # and the two hemispheres' tracers cancel in the signed sum.
        np.random.seed(3)
        times = np.linspace(0.0, 10.0, 11)
        fh = _HemisphericFileHandler(self.RHO, self.V_R, times)
        tracers = spherical_surface_mc(
            r_surf=self.R_SURF, t_start=times, n_tracers=20000,
            file_handler=fh, integrator=ExplicitTrapezoid(),
            vel_keys=('vx', 'vy', 'vz'),
        )
        m = _masses(tracers)
        z = np.array([tr.initial_position[2] for tr in tracers.tracers])

        # Signs follow the hemisphere, magnitudes are all equal.  The sign is
        # the drawn *cell's*, so skip the one cell band straddling the equator,
        # into which a tracer can be jittered across z = 0.
        off_equator = np.abs(z) / self.R_SURF > 0.05
        np.testing.assert_array_equal(np.sign(m[off_equator]), np.sign(z[off_equator]))
        np.testing.assert_allclose(np.abs(m), np.abs(m[0]), rtol=1e-12)
        # The total crossing mass is still the full-sphere one ...
        crossing = self.RHO * self.V_R * 4 * np.pi * self.R_SURF**2 * (times[-1] - times[0])
        assert np.abs(m).sum() == pytest.approx(crossing, rel=1e-3)
        # ... while the net cancels, up to the equator-straddling band and the
        # Monte-Carlo noise of N draws.
        assert abs(m.sum()) < 0.05 * crossing

    def test_seeded_on_the_surface_at_snapshot_times(self):
        times = np.linspace(0.0, 10.0, 11)
        tracers = self._seed(n_tracers=500, times=times)
        r = _radii(tracers)
        np.testing.assert_allclose(r, self.R_SURF, rtol=1e-12)
        t_seed = np.array([tr.initial_time for tr in tracers.tracers])
        assert np.all(np.isin(t_seed, times))

    def test_uniform_flux_samples_uniformly_in_costheta(self):
        tracers = self._seed(n_tracers=20000)
        z = np.array([tr.initial_position[2] for tr in tracers.tracers])
        counts, _ = np.histogram(z / self.R_SURF, bins=10, range=(-1, 1))
        assert np.allclose(counts, len(z) / 10, rtol=0.1)


class _TwoKeyInterpolator:
    """Mock interpolator returning rho(r) for 'rho' and a constant for 'u_t'."""

    def __init__(self, rho_fn, u_t, keys):
        self._rho_fn, self._u_t, self.keys = rho_fn, u_t, list(keys)
        self.n_keys = len(self.keys)

    def load(self):
        pass

    def unload(self):
        pass

    def __call__(self, coords):
        r = np.linalg.norm(coords, axis=0)
        vals = {'rho': self._rho_fn(r), 'u_t': np.full_like(r, self._u_t)}
        return np.array([vals[k] for k in self.keys])


class _TwoKeyFileHandler(MockFileHandler):
    """MockFileHandler exposing a separate, physically signed u_t field."""

    def __init__(self, rho_fn, u_t, adm_mass=None):
        super().__init__(rho_fn, keys=('rho', 'u_t'), time=0.0,
                         adm_mass=adm_mass)
        self.extra_data = {'rho_fn': rho_fn, 'u_t': u_t, 'keys': self.keys}

    @staticmethod
    def setup_interpolator(shm, extra_data):
        return _TwoKeyInterpolator(extra_data['rho_fn'], extra_data['u_t'],
                                   list(shm.keys()))


class TestConservativeDensity:
    """With adm_mass the weight and the total use D = sqrt(gamma) W rho."""

    U_T = -1.05
    ADM_MASS = 2.7

    @staticmethod
    def _seed(adm_mass, n_tracers=200):
        np.random.seed(7)
        # The ADM mass is the handler's, not the seeder's: whether the weight
        # is rho or D is a property of the data, and the seeder never asks.
        fh = _TwoKeyFileHandler(lambda r: r**-3.0, TestConservativeDensity.U_T,
                                adm_mass=adm_mass)
        return spherical_by_volume_mc(
            r_min=R_MIN, r_max=R_MAX, n_tracers=n_tracers, start_t=0.0,
            cells_per_tracer=200,
            file_handler=fh, integrator=ExplicitTrapezoid(), vel_keys=('rho',),
        )

    def test_total_mass_is_densitized(self):
        from src.utils import densitization_factor

        m_dens = _masses(self._seed(self.ADM_MASS)).sum()

        r = np.geomspace(R_MIN, R_MAX, 200001)
        integrand = r**-3.0 * densitization_factor(r, self.U_T, self.ADM_MASS) * r**2
        expected = 4 * np.pi * np.trapezoid(integrand, r)
        assert m_dens == pytest.approx(expected, rel=1e-3)

    def test_densitization_raises_the_mass(self):
        # W*sqrt(gamma) > 1, and here it is nearly constant (~ -u_t) over the
        # shell, so the conserved mass exceeds the rho-based one by about that.
        m_plain = _masses(self._seed(None)).sum()
        m_dens = _masses(self._seed(self.ADM_MASS)).sum()
        assert m_dens / m_plain == pytest.approx(-self.U_T, rel=0.05)
        assert m_dens > m_plain


class _SplitShellHandler(MockFileHandler):
    """
    A handler whose native cells arrive as two adjacent radial groups, with the
    density non-zero in only one of them.

    Cells are handed over flat, with a mass each and no field values, so a
    mistake in the bounds-to-mass pairing is invisible to any total: the mass
    still sums correctly, it is simply attributed to the wrong cells. Emptying
    one group turns that into something observable -- every tracer must land in
    the other.
    """

    SPLIT = 400.0
    grid_geometry = 'spherical'

    def native_cell_weights(self, slot, surface_radius=None):
        n = 8
        cth = np.linspace(1.0, -1.0, 5)
        ph = np.linspace(0.0, 2 * np.pi, 9)
        los, his, masses = [], [], []
        for r_lo, r_hi, rho in ((R_MIN, self.SPLIT, 0.0), (self.SPLIT, R_MAX, 1.0)):
            lo, hi = tensor_cell_bounds(np.geomspace(r_lo, r_hi, n + 1), cth, ph)
            los.append(lo)
            his.append(hi)
            masses.append(rho * cell_measure(lo, hi))
        return (np.concatenate(los, axis=1), np.concatenate(his, axis=1),
                np.concatenate(masses))


class TestNativeCellSeeding:
    @staticmethod
    def _seed(n_tracers=2000, seed=7):
        np.random.seed(seed)
        fh = _SplitShellHandler(lambda r: np.ones_like(r), keys=('rho',), time=0.0)
        return spherical_by_volume_mc(
            r_min=R_MIN, r_max=R_MAX, n_tracers=n_tracers, start_t=0.0,
            weight_grid='native',
            file_handler=fh, integrator=ExplicitTrapezoid(), vel_keys=('rho',),
        )

    def test_every_tracer_lands_in_the_populated_shell(self):
        r = _radii(self._seed())
        assert r.min() >= _SplitShellHandler.SPLIT
        assert r.max() <= R_MAX * (1 + 1e-12)

    def test_a_mid_cell_r_max_is_honoured_exactly(self):
        """
        700 falls inside an outer cell. The volume must stop at 700 itself, not
        at that cell's face, so rho = 1 gives exactly 4/3 pi (700^3 - split^3)
        and no tracer lands beyond it.
        """
        np.random.seed(3)
        fh = _SplitShellHandler(lambda r: np.ones_like(r), keys=('rho',), time=0.0)
        tr = spherical_by_volume_mc(
            r_min=R_MIN, r_max=700.0, n_tracers=2000, start_t=0.0,
            weight_grid='native',
            file_handler=fh, integrator=ExplicitTrapezoid(), vel_keys=('rho',),
        )
        expected = 4 / 3 * np.pi * (700.0**3 - _SplitShellHandler.SPLIT**3)
        assert _masses(tr).sum() == pytest.approx(expected, rel=1e-12)
        assert _radii(tr).max() <= 700.0 * (1 + 1e-12)

    def test_total_mass_is_the_populated_shell_only(self):
        """rho = 1 outside the split, so M = 4/3 pi (r_max^3 - split^3)."""
        m = _masses(self._seed()).sum()
        expected = 4 / 3 * np.pi * (R_MAX**3 - _SplitShellHandler.SPLIT**3)
        assert m == pytest.approx(expected, rel=1e-3)
