"""
Tests for the spherical ghost-zone fill in src/gra_surface.py.

Tracers are queried at arbitrary positions, including right up against
the poles and either side of the phi = 0 seam, so the interpolation
stencil needs ``interpolator.n_ghosts`` cells of padding there.  Getting
the padding wrong does not raise, it just returns quietly wrong values in
those regions, which is what these tests are here to catch.
"""

import numpy as np
import pytest

from src.gra_surface import _fill_with_ghosts


def smooth_sphere_field(theta, phi):
    """A field smooth on the sphere, with l = 0, 1 and 2 content."""
    return (np.sin(theta) * np.cos(phi)
            + 0.5 * np.cos(theta)
            + 0.25 * np.sin(theta) ** 2 * np.sin(2 * phi))


@pytest.mark.parametrize("ng", [1, 2, 3])
class TestFillWithGhosts:
    """
    ``ng = 1`` is what RegularInterpolator3D asks for and ``ng = 3`` what
    PchipInterpolator3D (the default) does, so both must hold.  Several
    plausible off-by-one mistakes are invisible at ng = 1.
    """

    n_th, n_ph = 16, 32

    def _grid(self):
        d_th = np.pi / self.n_th
        # GR-Athena++'s polar grid is cell-centred: no node on either pole.
        th = (np.arange(self.n_th) + 0.5) * d_th
        ph = (np.arange(self.n_ph) + 0.5) * (2 * np.pi / self.n_ph)
        return d_th, th, ph

    def _filled(self, ng, ar):
        buf = np.zeros((self.n_th + 2 * ng, self.n_ph + 2 * ng))
        _fill_with_ghosts(buf, {"f": ar}, "f", ng=ng)
        return buf

    def test_interior_is_untouched(self, ng):
        ar = np.arange(self.n_th * self.n_ph, dtype=float).reshape(self.n_th, self.n_ph)
        buf = self._filled(ng, ar)
        np.testing.assert_array_equal(buf[ng:-ng, ng:-ng], ar)

    def test_polar_ghosts_are_the_exact_continuation(self, ng):
        """
        Every ghost row must match the analytic continuation across the
        pole, not just the middle one -- filling the rows in reverse order
        leaves exactly the middle row correct.
        """
        d_th, th, ph = self._grid()
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        buf = self._filled(ng, smooth_sphere_field(TH, PH))

        for k in range(1, ng + 1):
            # ghost k cells beyond theta = 0, and k cells beyond theta = pi
            th_lo = -(k - 0.5) * d_th
            th_hi = np.pi + (k - 0.5) * d_th
            np.testing.assert_allclose(
                buf[ng - k, ng:-ng], smooth_sphere_field(-th_lo, ph + np.pi), atol=1e-12
            )
            np.testing.assert_allclose(
                buf[-ng + k - 1, ng:-ng],
                smooth_sphere_field(2 * np.pi - th_hi, ph + np.pi), atol=1e-12
            )

    def test_phi_ghosts_wrap_in_order(self, ng):
        ar = np.arange(self.n_th * self.n_ph, dtype=float).reshape(self.n_th, self.n_ph)
        buf = self._filled(ng, ar)
        np.testing.assert_array_equal(buf[ng:-ng, :ng], ar[:, -ng:])
        np.testing.assert_array_equal(buf[ng:-ng, -ng:], ar[:, :ng])

    def test_corners_agree_with_both_directions(self, ng):
        """
        The theta-ghost x phi-ghost corners must be the periodic image of
        the polar ghost rows, i.e. consistent with filling in either order.
        """
        _, th, ph = self._grid()
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        buf = self._filled(ng, smooth_sphere_field(TH, PH))
        np.testing.assert_array_equal(buf[:, :ng], buf[:, -2 * ng:-ng])
        np.testing.assert_array_equal(buf[:, -ng:], buf[:, ng:2 * ng])

    def test_axisymmetric_field_stays_axisymmetric(self, ng):
        """A field independent of phi must remain so in the ghost zones."""
        _, th, _ = self._grid()
        ar = np.broadcast_to(np.cos(th)[:, None], (self.n_th, self.n_ph)).copy()
        buf = self._filled(ng, ar)
        np.testing.assert_allclose(buf - buf[:, :1], 0.0, atol=1e-14)
