"""
Tests for analysis._common.tracer_masses.

The sign of a tracer's mass is physical: a surface-seeded tracer in an
inflowing cell carries a negative flux integral and must subtract from the
net crossing mass.
"""

import numpy as np
import pytest

from analysis._common import tracer_masses


class _StubTraj:
    def __init__(self, **props):
        self.props = props


class TestTracerMasses:
    def test_signs_are_preserved(self):
        """
        Inflowing surface cells carry negative mass. Taking the magnitude
        would count material crossing the sphere inward as ejecta, which is
        exactly backwards, and the mass-weighted '-mc' seeding samples such
        cells in proportion to their flux like any other.
        """
        trajs = [_StubTraj(mass=3.0), _StubTraj(mass=-1.0)]
        assert tracer_masses(trajs).tolist() == [3.0, -1.0]
        assert tracer_masses(trajs).sum() == pytest.approx(2.0)

    def test_mass_D_wins_when_present(self):
        """
        The conserved rest mass is the one to weight by whenever the seeding
        recorded it, whether it came from '--adm-mass' or from '-mc' sampling.
        """
        trajs = [_StubTraj(mass=1.0, mass_D=1.04), _StubTraj(mass=2.0)]
        assert tracer_masses(trajs).tolist() == pytest.approx([1.04, 2.0])

    def test_returns_a_float_array(self):
        assert tracer_masses([_StubTraj(mass=1.0)]).dtype == np.float64
