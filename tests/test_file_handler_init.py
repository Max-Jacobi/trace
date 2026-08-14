"""
Tests for constructing FileHandler subclasses.

These cover the failure path rather than the happy one: a handler whose
``__init__`` raises must report *why*, and must not turn that into a
different, unrelated error while being torn down.
"""

import gc

import pytest

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
