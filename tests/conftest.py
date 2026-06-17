"""
Shared fixtures and helpers for the trace test suite.
"""

import contextlib
import numpy as np
import pytest
from multiprocessing.shared_memory import SharedMemory


@contextlib.contextmanager
def shared_memory_arrays(data_dict: dict) -> dict:
    """
    Context manager that creates shared memory blocks, writes float64 data into
    them, yields a dict mapping key -> shared memory name, then cleans up.

    Parameters
    ----------
    data_dict : dict[str, np.ndarray]
        Arrays to expose as shared memory.

    Yields
    ------
    dict[str, str]
        Mapping from key to shared memory block name, suitable for passing
        to an interpolator constructor.
    """
    shm_blocks = {}
    shm_names = {}
    try:
        for key, arr in data_dict.items():
            arr = np.asarray(arr, dtype=np.float64)
            shm = SharedMemory(create=True, size=max(arr.nbytes, 1))
            np.ndarray(arr.shape, dtype=np.float64, buffer=shm.buf)[:] = arr
            shm_blocks[key] = shm
            shm_names[key] = shm.name
        yield shm_names
    finally:
        for shm in shm_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
