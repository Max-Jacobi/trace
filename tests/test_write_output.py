"""
Tests for run_pipeline.write_output.

Uses a minimal stand-in for Tracer/Tracers rather than running the
pipeline, since the only logic here is which key the represented mass is
computed from and how the columns are renamed.
"""

import numpy as np
import pytest

from run_pipeline import write_output


class _StubTracer:
    def __init__(self, data, props, times):
        self.data = data
        self.props = props
        self.times = times
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
