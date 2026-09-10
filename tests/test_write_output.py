"""
Tests for run_pipeline.write_output.

Uses a minimal stand-in for Tracer/Tracers rather than running the
pipeline, since the only logic here is which key the represented mass is
computed from and how the columns are renamed.
"""

import numpy as np
import pytest

from run_pipeline import densitization_factor, write_output


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
    def test_mass_uses_the_density_key(self, tmp_path):
        """
        The represented mass must come from --density-key, not from a
        hard-coded "rho".  A data source whose density field is called
        something else would otherwise raise KeyError here, after the
        whole integration had already run.
        """
        tr = _tracer(density_key="dens")
        write_output(_StubTracers([tr]), str(tmp_path), density_key="dens")
        # mass = dV * density at the latest time
        assert tr.props["mass"] == pytest.approx(10.0 * 4.0)

    def test_default_density_key_is_rho(self, tmp_path):
        tr = _tracer(density_key="rho")
        write_output(_StubTracers([tr]), str(tmp_path))
        assert tr.props["mass"] == pytest.approx(10.0 * 4.0)

    def test_mass_is_taken_at_the_latest_time(self, tmp_path):
        """Seeding happens at max(times), whichever end of the array it is."""
        tr = _tracer()
        tr.times = np.array([2.0, 1.0, 0.0])
        write_output(_StubTracers([tr]), str(tmp_path))
        assert tr.props["mass"] == pytest.approx(10.0 * 1.0)

    def test_surface_tracers_keep_their_flux_mass(self, tmp_path):
        """Without 'dV' the mass was set by the seeder and must not be touched."""
        tr = _tracer()
        tr.props = {"mass": 1.234}
        write_output(_StubTracers([tr]), str(tmp_path))
        assert tr.props["mass"] == pytest.approx(1.234)

    def test_dotted_group_prefixes_are_stripped(self, tmp_path):
        tr = _StubTracer(
            data={"rho": np.ones(3), "tracer.hydro.aux.T": np.full(3, 5.0)},
            props={},
            times=np.array([0.0, 1.0, 2.0]),
        )
        write_output(_StubTracers([tr]), str(tmp_path))
        assert "T" in tr.data and "tracer.hydro.aux.T" not in tr.data
        assert tr.written == (("x", "y", "z"), f"{tmp_path}/tracer_")


class TestDensitizedMass:
    """
    `--adm-mass` adds 'mass_D', the conserved rest mass, alongside the
    rho-based 'mass'. Both must be present and 'mass' must be untouched,
    so downstream code that predates the flag keeps working.
    """

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

    def test_both_masses_are_written(self, tmp_path):
        tr = self._tracer_at(r=400.0, u_t=-1.02)
        write_output(_StubTracers([tr]), str(tmp_path), adm_mass=2.746)
        assert tr.props["mass"] == pytest.approx(40.0)
        expected = 40.0 * densitization_factor(400.0, -1.02, 2.746)
        assert tr.props["mass_D"] == pytest.approx(expected)
        assert tr.props["mass_D"] > tr.props["mass"]

    def test_absent_without_the_flag(self, tmp_path):
        tr = self._tracer_at(r=400.0, u_t=-1.02)
        write_output(_StubTracers([tr]), str(tmp_path))
        assert "mass_D" not in tr.props

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


class TestNoDoubleDensitization:
    """
    The '-mc' seeding modes weight their sampling with D directly, so their
    'mass' is already the conserved one and they say so by emitting 'mass_D'
    themselves. write_output must leave that alone -- recomputing it would
    apply W*sqrt(gamma) a second time.
    """

    def test_existing_mass_D_is_not_recomputed(self, tmp_path):
        tr = _StubTracer(
            data={"rho": np.full(3, 4.0), "u_t": np.full(3, -1.02)},
            props={"mass": 7.0, "mass_D": 7.0},      # as the MC seeders write it
            times=np.arange(3.0),
            positions=np.tile((0.0, 0.0, 400.0), (3, 1)),
        )
        write_output(_StubTracers([tr]), str(tmp_path), adm_mass=2.746)
        assert tr.props["mass_D"] == pytest.approx(7.0)
        assert tr.props["mass"] == pytest.approx(7.0)
