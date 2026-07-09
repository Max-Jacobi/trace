"""
examples/_parallel.py

Shared multiprocessing helpers for the example scripts.  Each example
parallelises its own tracer-integration loop (splitting a step's tracer
batch into bunches dispatched to a worker pool), following the same
sort-for-cache-locality pattern used by the production driver in
src/tracers.py.  This module only holds example-specific config (worker
count, bunch count); the generic graceful-shutdown pool helper lives in
src/utils.py (shared with src/tracers.py) since it isn't specific to these
examples.  Each example defines its own worker function and cache-locality
sort key, since those depend on the example's specific grid/interpolator
setup.
"""

import os

from src.utils import worker_pool  # noqa: F401  (re-exported for examples)

N_CPU = os.cpu_count() or 1
N_BUNCHES = 3 * N_CPU
