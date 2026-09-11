"""
Tests for run_pipeline.write_output and the mass-density policy it defers to.

Uses a minimal stand-in for Tracer/Tracers rather than running the pipeline,
since the only logic here is how the represented mass is computed and how the
columns are renamed.
"""

import numpy as np
import pytest

from run_pipeline import write_output
from src.mass import MassDensity
from src.utils import densitization_factor


class _StubTracer:
    def __init__(self, data, props, times, positions=None):
        self.data = data
        self.props = props
        self.times = times
        self.positions = (np.zeros((len(times), 3)) if positions is None
                          else np.asarray(positions, dtype=float))
        self.written = None

    def output_to_ascii(self, coords, filebase):
        self.written = (tuple(coords), filebase)


class _StubTracers:
    def __init__(self, tracers):
        self.tracers = tracers


def _tracer(density_key="rho", rho=(1.0, 2.0, 4.0), dV=10.0):
    return _StubTracer(
        data={density_key: np.array(rho), "x": np.zeros(3)},
        props={"dV": dV},
        times=np.array([0.0, 1.0, 2.0]),
    )


class TestWriteOutput:
    def test_mass_uses_the_handlers_density_key(self, tmp_path):
        """
        The represented mass must come from whichever field the handler's
        MassDensity names, not from a hard-coded "rho". A data source whose
        density field is called something else would otherwise raise KeyError
        here, after the whole integration had already run.
        """
        tr = _tracer(density_key="dens")
        write_output(_StubTracers([tr]), str(tmp_path), MassDensity("dens"))
        # mass = dV * density at the latest time
        assert tr.props["mass"] == pytest.approx(10.0 * 4.0)

    def test_default_density_key_is_rho(self, tmp_path):
        tr = _tracer(density_key="rho")
        write_output(_StubTracers([tr]), str(tmp_path), MassDensity())
        assert tr.props["mass"] == pytest.approx(10.0 * 4.0)

    def test_mass_is_taken_at_the_latest_time(self, tmp_path):
        """Seeding happens at max(times), whichever end of the array it is."""
        tr = _tracer()
        tr.times = np.array([2.0, 1.0, 0.0])
        write_output(_StubTracers([tr]), str(tmp_path), MassDensity())
        assert tr.props["mass"] == pytest.approx(10.0 * 1.0)

    def test_seeded_masses_are_not_touched(self, tmp_path):
        """
        Without 'dV' the mass was set by the seeder -- the surface and the two
        '-mc' modes all do -- and it is already the right one, so write_output
        must leave it exactly as it is.
        """
        tr = _tracer()
        tr.props = {"mass": 1.234}
        write_output(_StubTracers([tr]), str(tmp_path),
                     MassDensity(adm_mass=2.746))
        assert tr.props["mass"] == pytest.approx(1.234)

    def test_only_one_mass_is_written(self, tmp_path):
        """
        One mass per tracer, whichever density it came from. An extra 'mass_D'
        would make downstream code guess which of the two to believe.
        """
        tr = self._tracer_at(r=400.0, u_t=-1.02)
        write_output(_StubTracers([tr]), str(tmp_path),
                     MassDensity(adm_mass=2.746))
        assert [k for k in tr.props if k.startswith("mass")] == ["mass"]

    def test_dotted_group_prefixes_are_stripped(self, tmp_path):
        tr = _StubTracer(
            data={"rho": np.ones(3), "tracer.hydro.aux.T": np.full(3, 5.0)},
            props={},
            times=np.array([0.0, 1.0, 2.0]),
        )
        write_output(_StubTracers([tr]), str(tmp_path), MassDensity())
        assert "T" in tr.data and "tracer.hydro.aux.T" not in tr.data
        assert tr.written == (("x", "y", "z"), f"{tmp_path}/tracer_")

    @staticmethod
    def _tracer_at(r, u_t, dV=10.0, rho=4.0):
        n = 3
        pos = np.zeros((n, 3))
        pos[-1] = (0.0, 0.0, r)          # seed sample is the latest time
        return _StubTracer(
            data={"rho": np.full(n, rho), "u_t": np.full(n, u_t)},
            props={"dV": dV},
            times=np.arange(float(n)),
            positions=pos,
        )

    def test_an_adm_mass_makes_the_one_mass_the_conserved_one(self, tmp_path):
        tr = self._tracer_at(r=400.0, u_t=-1.02)
        write_output(_StubTracers([tr]), str(tmp_path),
                     MassDensity(adm_mass=2.746))
        expected = 40.0 * densitization_factor(400.0, -1.02, 2.746)
        assert tr.props["mass"] == pytest.approx(expected)
        assert tr.props["mass"] > 40.0

    def test_without_one_it_is_the_plain_rest_mass(self, tmp_path):
        tr = self._tracer_at(r=400.0, u_t=-1.02)
        write_output(_StubTracers([tr]), str(tmp_path), MassDensity())
        assert tr.props["mass"] == pytest.approx(40.0)


class TestMassDensity:
    """
    The policy object itself. Which fields it asks for is as much part of the
    contract as what it returns: a caller loads exactly those and passes them
    back in that order.
    """

    def test_plain_density_asks_for_one_field(self):
        md = MassDensity("rho", ("vx", "vy", "vz"))
        assert md.density_keys == ("rho",)
        assert md.flux_keys == ("rho", "vx", "vy", "vz")

    def test_densitizing_adds_u_t_but_only_once(self):
        md = MassDensity("rho", ("vx", "vy", "vz"), adm_mass=2.7, ut_key="u_t")
        assert md.density_keys == ("rho", "u_t")
        assert md.flux_keys == ("rho", "u_t", "vx", "vy", "vz")
        assert md.flux_keys.count("u_t") == 1

    def test_a_velocity_that_is_also_the_density_is_not_repeated(self):
        """Degenerate, but a duplicated key would mis-align every index."""
        md = MassDensity("rho", ("rho", "vy", "vz"))
        assert md.flux_keys == ("rho", "vy", "vz")

    def test_radial_flux_is_signed(self):
        """
        Inflow must come back negative: the surface seeders rely on the sign to
        subtract returning material from the net crossing mass.
        """
        md = MassDensity("rho", ("vx", "vy", "vz"))
        pos = np.array([[400.0], [0.0], [0.0]])
        out = np.array([[2.0], [0.1], [0.0], [0.0]])
        inn = np.array([[2.0], [-0.1], [0.0], [0.0]])
        assert md.radial_flux(out, pos)[0] == pytest.approx(0.2)
        assert md.radial_flux(inn, pos)[0] == pytest.approx(-0.2)

    def test_radial_flux_densitizes_too(self):
        md = MassDensity("rho", ("vx", "vy", "vz"), adm_mass=2.746)
        pos = np.array([[400.0], [0.0], [0.0]])
        vals = np.array([[2.0], [-1.02], [0.1], [0.0], [0.0]])
        want = 2.0 * densitization_factor(400.0, -1.02, 2.746) * 0.1
        assert md.radial_flux(vals, pos)[0] == pytest.approx(want)

    def test_a_flux_without_velocities_says_so(self):
        with pytest.raises(ValueError, match="three velocity keys"):
            MassDensity("rho").radial_flux(np.ones((1, 1)), np.ones((3, 1)))


class TestDensitizationFactor:
    def test_factor_matches_W_times_sqrt_gamma(self):
        """
        Spelled out the long way: W = -u_t/alpha and sqrt(gamma) = psi**6 for
        isotropic Schwarzschild, so the factor must be their product.
        """
        r, u_t, m = 400.0, -1.02118, 2.746
        psi = 1.0 + m / (2.0 * r)
        alpha = (1.0 - m / (2.0 * r)) / psi
        assert densitization_factor(r, u_t, m) == pytest.approx(
            (-u_t / alpha) * psi ** 6)

    def test_far_field_limit_is_minus_u_t(self):
        """At large r the metric factors vanish and only W is left."""
        assert densitization_factor(1e12, -1.05, 2.746) == pytest.approx(1.05, rel=1e-10)

    def test_refuses_inside_the_horizon_scale(self):
        with pytest.raises(ValueError, match="r > adm_mass"):
            densitization_factor(1.0, -1.0, 2.746)


class TestDensitizationProvenance:
    """
    With one mass per tracer, nothing in the file says whether it is the
    rho-based or the conserved one -- and analysis/mass_conservation.py can
    still densitize a mass after the fact. 'adm_mass' in the header is what
    stops it doing that twice.
    """

    @staticmethod
    def _tr():
        return _StubTracer(
            data={"rho": np.full(3, 4.0), "u_t": np.full(3, -1.02)},
            props={"dV": 10.0},
            times=np.arange(3.0),
            positions=np.tile((0.0, 0.0, 400.0), (3, 1)),
        )

    def test_recorded_when_the_mass_is_densitized(self, tmp_path):
        tr = self._tr()
        write_output(_StubTracers([tr]), str(tmp_path), MassDensity(adm_mass=2.746))
        assert tr.props["adm_mass"] == pytest.approx(2.746)

    def test_absent_when_it_is_not(self, tmp_path):
        tr = self._tr()
        write_output(_StubTracers([tr]), str(tmp_path), MassDensity())
        assert "adm_mass" not in tr.props
