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
    def native_cell_weights(self, slot, keys, surface_radius=None):
        raise AssertionError("not called in these tests")


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


class _Bounds:
    """A handler whose cells are handed over one by one, as the contract asks."""

    def __init__(self, lo, hi, values=None, r_used=None):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.values = (np.zeros((1, self.lo.shape[1])) if values is None
                       else np.asarray(values, float))
        self.r_used = r_used

    def native_cell_weights(self, slot, keys, surface_radius=None):
        return self.lo, self.hi, self.values, self.r_used


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
        return _Bounds(lo, hi, values=np.arange(4.0)[None, :])

    def test_selects_cells_by_centre(self):
        cells, vals = _native_cells(
            self._radial_cells(), 0, ('rho',),
            ranges=((1.0, 3.0), (-1.0, 1.0), (0.0, 2 * np.pi)))
        assert cells.n_cells == 2
        # centres 1.5 and 2.5, i.e. the middle two cells and their values
        assert vals[0].tolist() == [1.0, 2.0]

    def test_handles_a_descending_axis(self):
        """
        The requested range is given low-to-high while cos(theta) may run
        either way; comparing on the centres is what makes both work.
        """
        lo = np.array([[1.0] * 4, [0.5, 0.0, -0.5, -1.0], [0.0] * 4])
        hi = np.array([[2.0] * 4, [1.0, 0.5, 0.0, -0.5], [1.0] * 4])
        cells, _ = _native_cells(_Bounds(lo, hi), 0, ('rho',),
                                 ranges=((1.0, 2.0), (-0.5, 0.5), (0.0, 1.0)))
        assert cells.n_cells == 2

    def test_an_empty_region_is_an_error_the_user_can_act_on(self):
        with pytest.raises(ValueError, match="No native cell"):
            _native_cells(self._radial_cells(), 0, ('rho',),
                          ranges=((10.0, 20.0), (-1.0, 1.0), (0.0, 2 * np.pi)))

    def test_an_unsnapped_radius_stays_the_one_that_was_asked_for(self):
        lo = np.array([[-1.0], [0.0]])
        hi = np.array([[1.0], [2 * np.pi]])
        cells, _ = _native_cells(_Bounds(lo, hi), 0, ('rho',),
                                 ranges=((-1.0, 1.0), (0.0, 2 * np.pi)), r_surf=7.0)
        assert cells.r_surf == 7.0

    def test_a_snapped_radius_overrides_it(self):
        lo = np.array([[-1.0], [0.0]])
        hi = np.array([[1.0], [2 * np.pi]])
        cells, _ = _native_cells(_Bounds(lo, hi, r_used=6.5), 0, ('rho',),
                                 ranges=((-1.0, 1.0), (0.0, 2 * np.pi)), r_surf=7.0)
        assert cells.r_surf == 6.5


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
