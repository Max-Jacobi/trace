"""
Tests for the optional native_cell_weights hook and --weight-grid selection.

The hook lets a format hand the mass-weighted seeders its own grid, so the
weights are built on the data as written instead of interpolated onto a helper
grid first. Formats without one global grid must keep working unchanged.
"""

import numpy as np
import pytest

from src.file import FileHandler
from src.seeds import _axis_slice, _resolve_weight_grid


class _NoHook:
    """A handler that never implements the hook (e.g. octree meshblocks)."""


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


class TestAxisSlice:
    def test_selects_cells_by_centre(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])   # centres 0.5 1.5 2.5 3.5
        assert _axis_slice(edges, 1.0, 3.0) == slice(1, 3)

    def test_handles_a_descending_axis(self):
        """cos(theta) runs downward; the range must still be found."""
        edges = np.array([1.0, 0.5, 0.0, -0.5, -1.0])  # centres .75 .25 -.25 -.75
        assert _axis_slice(edges, -0.5, 0.5) == slice(1, 3)

    def test_refuses_a_range_narrower_than_a_cell(self):
        edges = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="narrower than one cell"):
            _axis_slice(edges, 1.01, 1.02)
