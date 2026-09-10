"""
Tests for the optional native_cell_weights hook and --weight-grid selection.

The hook lets a format hand the mass-weighted seeders its own grid, so the
weights are built on the data as written instead of interpolated onto a helper
grid first. Formats without one global grid must keep working unchanged.
"""

import numpy as np
import pytest

from src.file import FileHandler
from src.seeds import _CellSet, _axis_keep, _resolve_weight_grid


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


class TestAxisKeep:
    def test_selects_cells_by_centre(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])   # centres 0.5 1.5 2.5 3.5
        assert _axis_keep(edges, 1.0, 3.0) == slice(1, 3)

    def test_handles_a_descending_axis(self):
        """cos(theta) runs downward; the range must still be found."""
        edges = np.array([1.0, 0.5, 0.0, -0.5, -1.0])  # centres .75 .25 -.25 -.75
        assert _axis_keep(edges, -0.5, 0.5) == slice(1, 3)

    def test_reports_no_overlap_rather_than_raising(self):
        """
        Once cells arrive in blocks, an axis that does not reach the requested
        range is ordinary -- a block off to the side of the seeding region
        contributes nothing. Raising here would make that an error.
        """
        edges = np.array([0.0, 1.0, 2.0])
        assert _axis_keep(edges, 1.01, 1.02) is None
        assert _axis_keep(edges, 5.0, 6.0) is None


class TestCellSetIndexing:
    """
    The flat index has to resolve to the right cell of the right block. This
    is the part no measure-sum test can catch: getting it wrong scatters
    tracers into the wrong block while every total stays correct.
    """

    @staticmethod
    def _two_blocks():
        # adjacent radial blocks, [1,2] and [2,4], one cell each in angle
        cth = np.array([1.0, -1.0])
        ph = np.array([0.0, 2 * np.pi])
        return _CellSet([(np.array([1.0, 2.0]), cth, ph),
                         (np.array([2.0, 4.0]), cth, ph)])

    def test_offsets_span_every_block(self):
        cells = self._two_blocks()
        assert cells.n_cells == 2
        assert cells.measure().shape == (2,)

    def test_each_index_lands_in_its_own_block(self):
        cells = self._two_blocks()
        u = np.full((3, 2), 0.5)
        pos = cells.sample_positions(np.array([0, 1]), u)
        r = np.sqrt((pos ** 2).sum(axis=0))
        assert 1.0 <= r[0] <= 2.0
        assert 2.0 <= r[1] <= 4.0

    def test_draws_stay_inside_their_cell_at_both_extremes(self):
        cells = self._two_blocks()
        for frac in (0.0, 1.0):
            pos = cells.sample_positions(np.array([0, 1]), np.full((3, 2), frac))
            r = np.sqrt((pos ** 2).sum(axis=0))
            assert 1.0 <= r[0] <= 2.0 + 1e-12
            assert 2.0 <= r[1] <= 4.0 + 1e-12
