"""
Tests for the optional native_cell_weights hook and --weight-grid selection.

The hook lets a format hand the mass-weighted seeders its own grid, so the
weights are built on the data as written instead of interpolated onto a helper
grid first. Formats without one global grid must keep working unchanged.
"""

import numpy as np
import pytest

from src.file import FileHandler
from src.seeds import _Cells, _native_cells, _resolve_weight_grid
from src.utils import tensor_cell_bounds


class _NoHook:
    """A handler that cannot enumerate its own cells."""


class _WithHook:
    grid_geometry = 'spherical'

    def native_cell_weights(self, slot, surface_radius=None):
        raise AssertionError("not called in these tests")


class _CartesianHook(_WithHook):
    """A format with native cells, but in coordinates the seeders do not use."""
    grid_geometry = 'cartesian'


class _UndeclaredHook(_WithHook):
    """A format that implements the hook but never says what its axes are."""
    grid_geometry = None


class TestWeightGridResolution:
    def test_auto_falls_back_when_unsupported(self):
        assert _resolve_weight_grid(_NoHook(), 'auto') is False

    def test_auto_takes_native_when_available(self):
        assert _resolve_weight_grid(_WithHook(), 'auto') is True

    def test_helper_never_asks(self):
        assert _resolve_weight_grid(_WithHook(), 'helper') is False

    def test_native_refuses_rather_than_falling_back(self):
        """
        Silently using a different grid than the one asked for would make a
        run unreproducible, so 'native' is a demand, not a preference.
        """
        with pytest.raises(NotImplementedError, match="does not implement"):
            _resolve_weight_grid(_NoHook(), 'native')

    def test_inheriting_the_stub_does_not_count_as_support(self):
        """
        FileHandler defines the method so the contract has somewhere to live,
        but a subclass that has not overridden it still has no native grid.
        Comparing against the base implementation is what distinguishes them.
        """
        class _Bare(FileHandler):
            # only here to satisfy the ABC; nothing calls them
            def list_files(self, directory): ...
            @staticmethod
            def parse_file(file_path, keys, extra_data=None): ...
            @staticmethod
            def load_step_to_memory(metadata_dict, shared_memory, extra_data=None): ...
            @staticmethod
            def setup_interpolator(shared_memory, extra_data=None): ...

        assert _resolve_weight_grid(_Bare.__new__(_Bare), 'auto') is False


class TestGridGeometry:
    """
    The seeders read a format's native cell bounds as (r, cos theta, phi). A
    format whose cells are in anything else would have its x read as r and so
    on -- every mass and position wrong, and nothing downstream would notice.
    The declared geometries have to agree before any native cell is used.
    """

    def test_every_shipped_format_declares_spherical(self):
        from src.athdf_spherical import SphericalAthdfFileHandler
        from src.athenak import AthenaKFileHandler
        from src.reduced_surface import ReducedSurfaceFileHandler
        from src.seeds import GRID_GEOMETRY
        for cls in (ReducedSurfaceFileHandler, AthenaKFileHandler,
                    SphericalAthdfFileHandler):
            assert cls.grid_geometry == GRID_GEOMETRY

    def test_native_refuses_a_mismatch(self):
        with pytest.raises(ValueError, match="'cartesian' coordinates"):
            _resolve_weight_grid(_CartesianHook(), 'native')

    def test_native_refuses_an_undeclared_geometry(self):
        """Not saying is not the same as saying 'spherical'."""
        with pytest.raises(ValueError, match="None coordinates"):
            _resolve_weight_grid(_UndeclaredHook(), 'native')

    def test_auto_falls_back_to_the_helper_grid(self, capsys):
        """
        The helper grid never reads the format's cells, so it is exact in any
        geometry: 'auto' has a correct route to take and says it took it.
        """
        assert _resolve_weight_grid(_CartesianHook(), 'auto') is False
        assert "helper grid" in capsys.readouterr().out

    def test_helper_does_not_look(self):
        assert _resolve_weight_grid(_CartesianHook(), 'helper') is False

    def test_the_base_class_declares_nothing(self):
        assert FileHandler.grid_geometry is None


class _Bounds:
    """
    A handler whose cells are handed over one by one, as the contract asks:
    bounds and a mass each, with no field values and no say for the caller in
    what "mass" means.
    """

    grid_geometry = 'spherical'

    def __init__(self, lo, hi, weights=None):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.weights = (np.ones(self.lo.shape[1]) if weights is None
                        else np.asarray(weights, float))

    def native_cell_weights(self, slot, surface_radius=None):
        return self.lo, self.hi, self.weights


class TestRegionRestriction:
    """
    Cells are kept by their centre, one flat mask over the lot. There is no
    block structure left to get wrong, but the axis order and the direction an
    axis runs in still are.
    """

    @staticmethod
    def _radial_cells():
        # four radial cells [0,1] [1,2] [2,3] [3,4], one cell each in angle
        lo = np.array([[0.0, 1.0, 2.0, 3.0], [-1.0] * 4, [0.0] * 4])
        hi = np.array([[1.0, 2.0, 3.0, 4.0], [1.0] * 4, [2 * np.pi] * 4])
        return _Bounds(lo, hi, weights=np.arange(4.0))

    def test_cells_inside_are_kept_whole(self):
        cells, weights = _native_cells(
            self._radial_cells(), 0,
            ranges=((1.0, 3.0), (-1.0, 1.0), (0.0, 2 * np.pi)))
        assert cells.n_cells == 2
        # centres 1.5 and 2.5, i.e. the middle two cells and their masses
        assert weights.tolist() == [1.0, 2.0]

    def test_handles_a_descending_axis(self):
        """
        The requested range is given low-to-high while cos(theta) may run
        either way; comparing on the centres is what makes both work.
        """
        lo = np.array([[1.0] * 4, [0.5, 0.0, -0.5, -1.0], [0.0] * 4])
        hi = np.array([[2.0] * 4, [1.0, 0.5, 0.0, -0.5], [1.0] * 4])
        cells, _ = _native_cells(_Bounds(lo, hi), 0,
                                 ranges=((1.0, 2.0), (-0.5, 0.5), (0.0, 1.0)))
        assert cells.n_cells == 2

    def test_an_empty_region_is_an_error_the_user_can_act_on(self):
        with pytest.raises(ValueError, match="No native cell"):
            _native_cells(self._radial_cells(), 0,
                          ranges=((10.0, 20.0), (-1.0, 1.0), (0.0, 2 * np.pi)))

    def test_a_surface_stays_at_the_radius_asked_for(self):
        """Nothing is snapped: the sphere is exactly the one requested."""
        lo = np.array([[-1.0], [0.0]])
        hi = np.array([[1.0], [2 * np.pi]])
        cells, _ = _native_cells(_Bounds(lo, hi), 0,
                                 ranges=((-1.0, 1.0), (0.0, 2 * np.pi)), r_surf=7.0)
        assert cells.r_surf == 7.0


class TestExactLimits:
    """
    A limit that passes through a cell cuts it there, and the cell keeps the
    fraction of its mass that lies inside. This is what makes the edge of the
    seeded region the limit itself rather than the nearest cell face -- and so
    what lets a volume seeded to R and a surface at R share one boundary.
    """

    @staticmethod
    def _shell(weight=7.0):
        # one radial cell [1, 2] covering the whole sphere
        lo = np.array([[1.0], [-1.0], [0.0]])
        hi = np.array([[2.0], [1.0], [2 * np.pi]])
        return _Bounds(lo, hi, weights=[weight])

    def test_a_radial_cut_keeps_the_r_cubed_fraction(self):
        cells, w = _native_cells(self._shell(), 0,
                                 ranges=((1.0, 1.5), (-1.0, 1.0), (0.0, 2 * np.pi)))
        assert cells.hi[0, 0] == 1.5
        assert w[0] == pytest.approx(7.0 * (1.5**3 - 1.0) / (2.0**3 - 1.0))

    def test_the_cut_mass_is_the_cut_measure_times_the_density(self):
        """Constant density across the cell makes the rescaling exact."""
        lo = np.array([[1.0], [-1.0], [0.0]])
        hi = np.array([[2.0], [1.0], [2 * np.pi]])
        rho = 3.0
        full = _Cells(lo, hi).measure()[0]
        cells, w = _native_cells(_Bounds(lo, hi, weights=[rho * full]), 0,
                                 ranges=((1.2, 1.7), (-1.0, 1.0), (0.0, 2 * np.pi)))
        assert w[0] == pytest.approx(rho * cells.measure()[0])

    def test_angular_cuts_are_linear(self):
        cells, w = _native_cells(self._shell(), 0,
                                 ranges=((1.0, 2.0), (0.0, 1.0), (0.0, np.pi)))
        # half the cos(theta) range, half the phi range
        assert w[0] == pytest.approx(7.0 * 0.5 * 0.5)
        assert (cells.lo[1, 0], cells.hi[1, 0]) == (0.0, 1.0)

    def test_a_surface_cut_is_linear_in_cos_theta_not_cubic(self):
        """On a surface the first axis is cos(theta); r is not there to cube."""
        lo = np.array([[-1.0], [0.0]])
        hi = np.array([[1.0], [2 * np.pi]])
        _, w = _native_cells(_Bounds(lo, hi, weights=[4.0]), 0,
                             ranges=((0.0, 1.0), (0.0, 2 * np.pi)), r_surf=7.0)
        assert w[0] == pytest.approx(2.0)

    def test_draws_land_inside_the_cut(self):
        cells, _ = _native_cells(self._shell(), 0,
                                 ranges=((1.0, 1.5), (-1.0, 1.0), (0.0, 2 * np.pi)))
        u = np.random.default_rng(0).uniform(size=(3, 5000))
        pos = cells.sample_positions(np.zeros(5000, dtype=int), u)
        r = np.linalg.norm(pos, axis=0)
        assert r.min() >= 1.0 - 1e-12
        assert r.max() <= 1.5 + 1e-12

    def test_a_cell_touching_the_limit_from_outside_is_dropped(self):
        """Zero measure inside is not a cell to sample from."""
        with pytest.raises(ValueError, match="No native cell"):
            _native_cells(self._shell(), 0,
                          ranges=((2.0, 3.0), (-1.0, 1.0), (0.0, 2 * np.pi)))


class TestCellPlacement:
    """
    Every tracer must land inside the cell it was drawn from. Cells of
    different sizes sitting next to each other is the case that catches a
    bounds mixup: the totals stay right either way, the positions do not.
    """

    @staticmethod
    def _uneven():
        # adjacent radial cells [1,2] and [2,4], one cell each in angle
        lo = np.array([[1.0, 2.0], [-1.0, -1.0], [0.0, 0.0]])
        hi = np.array([[2.0, 4.0], [1.0, 1.0], [2 * np.pi, 2 * np.pi]])
        return _Cells(lo, hi)

    def test_measure_is_per_cell(self):
        cells = self._uneven()
        assert cells.n_cells == 2
        want = 4 * np.pi / 3 * np.array([2.0 ** 3 - 1.0, 4.0 ** 3 - 2.0 ** 3])
        assert cells.measure() == pytest.approx(want)

    def test_each_draw_stays_inside_its_own_cell(self):
        cells = self._uneven()
        for frac in (0.0, 0.5, 1.0):
            pos = cells.sample_positions(np.array([0, 1]), np.full((3, 2), frac))
            r = np.sqrt((pos ** 2).sum(axis=0))
            assert 1.0 - 1e-12 <= r[0] <= 2.0 + 1e-12
            assert 2.0 - 1e-12 <= r[1] <= 4.0 + 1e-12

    def test_a_surface_cell_set_sits_at_its_radius(self):
        lo = np.array([[-1.0], [0.0]])
        hi = np.array([[1.0], [2 * np.pi]])
        cells = _Cells(lo, hi, r_surf=3.0)
        assert cells.measure() == pytest.approx([4 * np.pi * 9.0])
        pos = cells.sample_positions(np.array([0]), np.full((2, 1), 0.5))
        assert np.sqrt((pos ** 2).sum(axis=0))[0] == pytest.approx(3.0)


class TestTensorCellBounds:
    def test_a_separable_grid_flattens_c_ordered(self):
        lo, hi = tensor_cell_bounds(np.array([0.0, 1.0, 2.0]),
                                    np.array([0.0, 10.0]))
        assert lo.shape == (2, 2)
        assert lo[0].tolist() == [0.0, 1.0]
        assert hi[0].tolist() == [1.0, 2.0]

    def test_a_descending_axis_comes_back_sorted(self):
        """
        AthenaK's cos(theta) runs downward. Sorting here is what lets every
        caller write hi - lo without an abs.
        """
        lo, hi = tensor_cell_bounds(np.array([1.0, 0.0, -1.0]))
        assert np.all(hi >= lo)
        assert lo[0].tolist() == [0.0, -1.0]
