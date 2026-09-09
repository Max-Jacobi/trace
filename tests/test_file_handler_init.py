"""
Tests for constructing FileHandler subclasses.

These cover the failure path rather than the happy one: a handler whose
``__init__`` raises must report *why*, and must not turn that into a
different, unrelated error while being torn down.
"""

import gc

import numpy as np
import pytest

from src.file import FileHandler
from src.gra_surface import GRASurfaceFileHandler
from src.interpolators import PchipInterpolator3D


class TestGRASurfaceFileHandlerInit:
    def test_constructor_reaches_the_directory(self, tmp_path):
        """
        ``GRASurfaceFileHandler.__init__`` must pass ``interpolator``
        through to ``FileHandler.__init__``.  Without it the base class
        never receives the argument, and construction fails with a
        TypeError before the handler does anything at all.

        An empty directory is the cheapest way to prove construction got
        as far as looking for files: reaching FileNotFoundError means
        every argument was wired up correctly.
        """
        with pytest.raises(FileNotFoundError, match="surface1"):
            GRASurfaceFileHandler(
                interpolator=PchipInterpolator3D,
                directory=str(tmp_path),
                keys=["rho"],
                n_cpu=1,
                files_per_step=2,
            )

    def test_failed_construction_tears_down_cleanly(self, tmp_path):
        """
        ``__del__`` runs even on a half-built handler, so
        ``free_shared_memory`` must tolerate ``__init__`` having raised
        before any shared memory was allocated -- otherwise the
        AttributeError from teardown masks the real error.
        """
        with pytest.raises(FileNotFoundError):
            GRASurfaceFileHandler(
                interpolator=PchipInterpolator3D,
                directory=str(tmp_path),
                keys=["rho"],
                n_cpu=1,
                files_per_step=2,
            )
        # Force the half-built instance's __del__ to run now, and fail if
        # it raises anything (pytest surfaces this as an unraisable
        # exception warning).
        gc.collect()


class _StubHandler(FileHandler):
    """
    FileHandler with the file system stubbed out.

    ``parse_files`` normally walks a directory; here it just reports a
    fixed per-key memory size, which is all the memory arithmetic in
    ``__init__`` depends on.
    """

    MEM_SIZE = 1 << 20  # bytes per key per snapshot

    def list_files(self, directory):
        return []

    @staticmethod
    def parse_file(file_path, keys, extra_data=None):
        raise NotImplementedError

    @staticmethod
    def load_step_to_memory(metadata_dict, shared_memory, extra_data=None):
        raise NotImplementedError

    @staticmethod
    def setup_interpolator(shared_memory, extra_data=None):
        raise NotImplementedError

    def parse_files(self, directory):
        self.memory_size = self.MEM_SIZE
        self.times = np.arange(10.0)
        self.files = np.array([{} for _ in self.times])


@pytest.fixture
def stub_handler():
    """Build _StubHandler instances and unlink their shared memory afterwards."""
    handlers = []

    def make(**kwargs):
        handler = _StubHandler(
            interpolator=PchipInterpolator3D,
            directory="unused",
            keys=[f"key{i}" for i in range(13)],
            n_cpu=1,
            **kwargs,
        )
        handlers.append(handler)
        return handler

    yield make
    for handler in handlers:
        handler.free_shared_memory()


class TestSharedMemoryBudget:
    """
    ``allocate_memory`` creates one segment of ``memory_size`` per key per
    snapshot slot, so ``tot_memory`` -- the number its /dev/shm guard is
    checked against -- has to carry the ``len(keys)`` factor too.
    """

    def test_tot_memory_counts_every_key(self, stub_handler):
        handler = stub_handler(files_per_step=3)
        assert handler.n_files_per_step == 3
        assert handler.tot_memory == 3 * handler.memory_size * len(handler.keys)
        # The guard must not understate what was actually allocated.
        allocated = sum(len(sh) for sh in handler.shared_memory) * handler.memory_size
        assert handler.tot_memory == allocated

    def test_max_tot_memory_allocation_fits_the_budget(self, stub_handler):
        keys = 13
        budget = 40 * _StubHandler.MEM_SIZE  # room for 3 slots of 13 keys
        handler = stub_handler(max_tot_memory=budget)
        assert handler.n_files_per_step == budget // (_StubHandler.MEM_SIZE * keys)
        allocated = sum(len(sh) for sh in handler.shared_memory) * handler.memory_size
        assert allocated <= budget
        # ...and it is the largest count that does fit.
        assert allocated + handler.memory_size * keys > budget
