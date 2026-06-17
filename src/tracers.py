import numpy as np
from tqdm import tqdm
from typing import Any, Optional, Callable
from multiprocessing import Pool

from .integrators.base import IntegratorBase, InterpolatorCallable
from .file import FileHandler
from .utils import do_parallel_star, do_parallel_star_pool

# ---------------------------------------------------------------------------
# Worker-process globals — set once by _init_worker_persistent, then updated
# lazily by _ensure_interps each time the interpolator data changes.
# ---------------------------------------------------------------------------

# Constants (set at pool init, never change):
_integrator:       Optional[IntegratorBase]   = None
_setup_interp_fn:  Optional[Callable]         = None  # staticmethod ref: FileHandler.setup_interpolator
_extra_data:       Optional[Any]              = None
_vel_keys:         Optional[tuple[str, ...]]  = None
_data_keys:        Optional[tuple[str, ...]]  = None

# Per-step interpolator state (updated lazily):
_chunk_id:        Optional[int]               = None
_vel_interp_0:    Optional[InterpolatorCallable] = None  # loaded for t
_vel_interp_1:    Optional[InterpolatorCallable] = None  # loaded for t+dt
_data_interp:     Optional[InterpolatorCallable] = None  # loaded for t+dt
_shm_names_vel_0: Optional[dict]              = None
_shm_names_vel_1: Optional[dict]              = None
_shm_names_data:  Optional[dict]              = None

class Tracer:
    """
    Holds the integration history of a single tracer.

    Storage uses pre-allocated numpy arrays sized to the total number of
    integration steps, avoiding per-step Python object allocations entirely.
    """

    def __init__(
        self,
        id: int,
        position: np.ndarray,
        time: float,
        keys: list[str],
        n_steps: int,
        props: dict[str, Any] | None = None,
        ) -> None:
        self.id = id
        self.initial_position = position.copy()
        self.initial_time = time
        self._keys = list(keys)
        self.props = props if props is not None else {}
        self.active = False
        self.done = False
        self.failed = False
        self._n_steps: int = 0
        self._positions = np.empty((n_steps+1, 3), dtype=np.float64)
        self._times     = np.empty(n_steps+1,      dtype=np.float64)
        self._data      = {k: np.empty(n_steps+1, dtype=np.float64) for k in keys}

    def add_step(
       self,
       position: np.ndarray,
       time: float,
       data: dict[str, float],
       ) -> None:
       i = self._n_steps
       self._positions[i] = position
       self._times[i]     = time
       for k, v in data.items():
           self._data[k][i] = v
       self._n_steps += 1

    @property
    def positions(self) -> np.ndarray:
        return self._positions[:self._n_steps]

    @property
    def times(self) -> np.ndarray:
        return self._times[:self._n_steps]

    @property
    def data(self) -> dict[str, np.ndarray]:
        return {k: self._data[k][:self._n_steps] for k in self._keys}

    def output_to_ascii(self, coords: list[str], filebase: str) -> str:
        keys = self._keys

        if self.failed:
            status = "failed"
        elif self.done:
            status = "done"
        elif self.active:
            status = "active"
        else:
            status = "not started"
        self.props['status'] = status

        props = "; ".join((f"{key}={val}" for key, val in self.props.items()))
        legend = "{:>23s}" + "{:>26s}"* (4 + len(keys) - 1)
        legend = legend.format("time", *coords, *keys)
        header = f"{props}\n{legend}"

        filename = f"{filebase}{self.id:06d}.dat"

        tsort = np.argsort(self.times)

        for key in keys:
            if len(self._data[key][:self._n_steps]) != len(tsort):
                return "failed"

        data = [self.times[tsort]]
        data += [self.positions[tsort, i] for i in range(len(coords))]
        data += [self._data[key][:self._n_steps][tsort] for key in keys]
        data = np.column_stack(data)

        np.savetxt(filename, data, header=header, fmt="%25.16e")
        return filename

class Tracers:

    def __init__(
        self,
        positions: np.ndarray,
        times: np.ndarray,
        vel_keys: list[str],
        integrator: IntegratorBase,
        file_handler: FileHandler,
        pbar_pos: int = 0,
        props: list[dict[str, list]] | None = None,
        ):
        self.vel_keys = vel_keys
        self.integrator = integrator
        self.file_handler = file_handler
        self.pbar_pos = pbar_pos

        file_times = self.file_handler.times

        unmatched = [t for t in times if not np.any(np.isclose(t, file_times))]
        if unmatched:
            raise ValueError(
                f"Some seed times do not coincide with available file times: {unmatched[:5]}"
            )

        n_steps = len(file_times) - 1  # maximum number of integration steps per tracer

        if props is None:
            props = [{} for _ in range(positions.shape[0])]

        self.tracers = np.array([
            Tracer(id=i, position=pos, time=t, keys=self.file_handler.keys,
                   n_steps=n_steps, props=prop)
            for i, (pos, t, prop) in enumerate(zip(positions, times, props))
        ])

        self._chunk_id = 0

        n_cpu = self.file_handler.parallel_kwargs["n_cpu"]
        init_args = (
            integrator,
            type(file_handler).setup_interpolator,
            file_handler.extra_data,
            tuple(vel_keys),
            tuple(file_handler.keys),
        )
        if n_cpu > 1:
            self._pool: Optional[Pool] = Pool(
                n_cpu,
                initializer=_init_worker_persistent,
                initargs=init_args,
            )
        else:
            self._pool = None
            _init_worker_persistent(*init_args)

    def __del__(self) -> None:
        if getattr(self, "_pool", None) is not None:
            self._pool.terminate()
            self._pool.join()

    def integrate_loaded_chunk(self) -> None:
        times = self.file_handler.cur_times
        dts = np.diff(times)
        shm_all = self.file_handler.shared_memory[:len(times)]
        extra_data = self.file_handler.extra_data

        self._chunk_id += 1
        chunk_id = self._chunk_id

        n_cpu = self.file_handler.parallel_kwargs["n_cpu"]
        n_bunches = 3 * n_cpu

        # One unloaded interpolator on the main process — only used for sort_tracers
        # (which only needs the grid coordinates, not the loaded data arrays).
        sort_interp = type(self.file_handler).setup_interpolator(
            {k: shm_all[0][k] for k in self.vel_keys}, extra_data
        ) if n_bunches > 1 else None

        pbar_kwargs = {k: v for k, v in self.file_handler.parallel_kwargs.items() if k != "n_cpu"}
        pbar_kwargs["disable"] = not pbar_kwargs.pop('verbose', False)
        for i, (dt, time) in tqdm(
            enumerate(zip(dts, times)),
            desc="Integrating tracers",
            total=len(dts),
            position=self.pbar_pos,
            unit="time step",
            ncols=0,
            **pbar_kwargs,
        ):
            shm_vel_0 = {k: shm_all[i][k]   for k in self.vel_keys}
            shm_vel_1 = {k: shm_all[i+1][k] for k in self.vel_keys}
            shm_data  = shm_all[i+1]  # all keys, for data interpolation

            new_tracers = np.array([tr for tr in self.tracers
                                    if len(tr.positions) == 0 and np.isclose(tr.initial_time, time)])

            if len(new_tracers) > 0:
                n_init_bunches = max(1, min(n_bunches, len(new_tracers)))
                init_pos  = np.array([tr.initial_position for tr in new_tracers]).T  # (3, n_new)
                shm_init  = {k: shm_all[i][k] for k in self.file_handler.keys}

                init_bunches = [b for b in np.array_split(new_tracers, n_init_bunches) if len(b) > 0]
                pos_splits   = np.cumsum([len(b) for b in init_bunches[:-1]])
                pos_bunches  = np.split(init_pos, pos_splits, axis=1)

                init_task_args = [
                    (j, pos, shm_init)
                    for j, pos in enumerate(pos_bunches)
                ]
                init_results = do_parallel_star_pool(
                    self._pool,
                    _initialize_bundle,
                    init_task_args,
                    chunksize=1,
                    position=self.pbar_pos+1,
                    leave=False,
                    unit="bunches",
                    ncols=0,
                    desc=f"Initializing {len(new_tracers)} new tracers",
                    **pbar_kwargs,
                )
                init_results.sort(key=lambda r: r[0])

                for (_, init_data), init_bunch in zip(init_results, init_bunches):
                    for j, tr in enumerate(init_bunch):
                        tr.add_step(
                            position=tr.initial_position,
                            time=tr.initial_time,
                            data=dict(zip(self.file_handler.keys, init_data[:, j])),
                        )
                        tr.active = True

            active_tracers = np.array([tr for tr in self.tracers if tr.active])

            if len(active_tracers) == 0:
                continue

            if sort_interp is not None and hasattr(sort_interp, "sort_tracers"):
                active_tracers = sort_interp.sort_tracers(active_tracers)

            # Extract only the current positions — O(n_tracers) pickle data,
            # independent of integration history, eliminating the memory growth.
            positions = np.array([tr.positions[-1] for tr in active_tracers]).T  # (3, n_active)

            tracer_bunches = [b for b in np.array_split(active_tracers, n_bunches) if len(b) > 0]
            split_sizes = np.cumsum([len(b) for b in tracer_bunches[:-1]])
            position_bunches = np.split(positions, split_sizes, axis=1)

            task_args = [
                (j, pos, time, dt, chunk_id, shm_vel_0, shm_vel_1, shm_data)
                for j, pos in enumerate(position_bunches)
            ]

            results = do_parallel_star_pool(
                self._pool,
                _integrate_positions,
                task_args,
                chunksize=1,
                position=self.pbar_pos+1,
                leave=False,
                unit="bunches",
                desc=f"Integrating {len(active_tracers)} active tracers",
                ncols=0,
                **pbar_kwargs,
            )

            # imap_unordered returns results in arbitrary order; sort by bundle
            # index so each result aligns with the correct tracer_bunches entry.
            results.sort(key=lambda r: r[0])

            for (_, new_pos, new_data, valid_mask), tracer_bunch in zip(results, tracer_bunches):
                for j, tr in enumerate(tracer_bunch):
                    if not valid_mask[j]:
                        tr.done = True
                        tr.active = False
                    else:
                        tr.add_step(
                            position=new_pos[:, j],
                            time=time + dt,
                            data=dict(zip(self.file_handler.keys, new_data[:, j])),
                        )

        print(f"{sum(tr.done for tr in self.tracers)} tracers done. "
              f"{sum(tr.active for tr in self.tracers)} tracers active.")

    def integrate(self, start_t: float, end_t: float) -> None:
        chunk_indices, forward, t_start, t_end = self.file_handler.get_chunk_indices(start_t, end_t)

        print(f"Integrating from t={t_start} to t={t_end} with {len(chunk_indices)} chunks.")

        for i_step in chunk_indices:
            self.file_handler.load_chunk(i_step, forward=forward)
            self.integrate_loaded_chunk()
            if all(tr.done for tr in self.tracers):
                print("All tracers done. Stopping integration.")
                break

def _init_worker_persistent(
    integrator: IntegratorBase,
    setup_interp_fn: Callable,
    extra_data: Any,
    vel_keys: tuple[str, ...],
    data_keys: tuple[str, ...],
) -> None:
    """
    Pool initializer — called once per worker when the pool is first created.

    Sets only the constants that never change across the entire run.
    Interpolator state is zero-initialized and populated lazily by
    _ensure_interps the first time a task arrives.
    """
    global _integrator, _setup_interp_fn, _extra_data, _vel_keys, _data_keys
    global _chunk_id, _vel_interp_0, _vel_interp_1, _data_interp
    global _shm_names_vel_0, _shm_names_vel_1, _shm_names_data

    _integrator       = integrator
    _setup_interp_fn  = setup_interp_fn
    _extra_data       = extra_data
    _vel_keys         = vel_keys
    _data_keys        = data_keys

    # Interpolator state starts empty; populated on first task.
    _chunk_id        = None
    _vel_interp_0    = None
    _vel_interp_1    = None
    _data_interp     = None
    _shm_names_vel_0 = None
    _shm_names_vel_1 = None
    _shm_names_data  = None


def _ensure_interps(
    chunk_id: int,
    shm_vel_0: dict,
    shm_vel_1: dict,
    shm_data: dict,
) -> None:
    """
    Lazily (re-)load worker interpolators to match the current step.

    Three cases:
      1. chunk_id changed — full reinit: old shared-memory slot names are reused
         with new data between chunks, so _xp_cache entries are stale.
      2. shm_vel_0 changed but chunk is same — step transition within chunk:
         roll vel_1 → vel_0 (preserving its valid _xp_cache), then load new
         vel_1 and data_interp.
      3. Same chunk, same step — nothing to do.
    """
    global _chunk_id, _vel_interp_0, _vel_interp_1, _data_interp
    global _shm_names_vel_0, _shm_names_vel_1, _shm_names_data

    if chunk_id != _chunk_id:
        # ── Full reinit ──────────────────────────────────────────────────────
        if _vel_interp_0 is not None:
            _vel_interp_0.unload()
        if _vel_interp_1 is not None:
            _vel_interp_1.unload()
        if _data_interp is not None:
            _data_interp.unload()

        _vel_interp_0 = _setup_interp_fn(shm_vel_0, _extra_data)
        _vel_interp_1 = _setup_interp_fn(shm_vel_1, _extra_data)
        _data_interp  = _setup_interp_fn(shm_data,  _extra_data)

        _vel_interp_0.load()
        _vel_interp_1.load()
        _data_interp.load()

        _chunk_id        = chunk_id
        _shm_names_vel_0 = shm_vel_0
        _shm_names_vel_1 = shm_vel_1
        _shm_names_data  = shm_data

    elif shm_vel_0 != _shm_names_vel_0:
        # ── Step transition within chunk ─────────────────────────────────────
        # vel_interp_1 from previous step now plays the role of vel_interp_0.
        # Its _xp_cache entries are valid because the underlying shm slot still
        # holds the same data.
        _vel_interp_0.unload()
        _vel_interp_0 = _vel_interp_1   # roll: preserves _xp_cache
        _vel_interp_1 = _setup_interp_fn(shm_vel_1, _extra_data)
        _vel_interp_1.load()

        _data_interp.unload()
        _data_interp = _setup_interp_fn(shm_data, _extra_data)
        _data_interp.load()

        _shm_names_vel_0 = shm_vel_0
        _shm_names_vel_1 = shm_vel_1
        _shm_names_data  = shm_data
    # else: same chunk, same step — all interpolators already current.


def _integrate_positions(
    bundle_idx: int,
    positions: np.ndarray,
    time: float,
    dt: float,
    chunk_id: int,
    shm_vel_0: dict,
    shm_vel_1: dict,
    shm_data: dict,
) -> tuple:
    """
    Compute new positions and interpolated data for a bundle of tracers.

    Parameters
    ----------
    bundle_idx : int
        Index of this bundle (used to re-order results from imap_unordered).
    positions : ndarray, shape (3, n)
        Current Cartesian positions of the tracers in this bundle.
    time : float
        Current integration time (unused here, kept for clarity).
    dt : float
        Integration timestep.
    chunk_id : int
        Monotonically incrementing counter identifying the current data chunk.
    shm_vel_0, shm_vel_1, shm_data : dict
        Shared-memory name mappings for the two velocity snapshots and the
        data snapshot at t+dt.

    Returns
    -------
    bundle_idx, new_pos (3, n), new_data (n_keys, n), valid_mask (n,)
        new_pos / new_data are NaN for tracers that left the domain.
    """
    _ensure_interps(chunk_id, shm_vel_0, shm_vel_1, shm_data)

    n = positions.shape[1]
    new_pos = _integrator(xn=positions, dt=dt, interps=[_vel_interp_0, _vel_interp_1])
    valid_mask = np.isfinite(new_pos).all(axis=0)

    new_data = np.full((len(_data_keys), n), np.nan)
    if valid_mask.any():
        new_data[:, valid_mask] = _data_interp(new_pos[:, valid_mask])

    return bundle_idx, new_pos, new_data, valid_mask


def _initialize_bundle(
    bundle_idx: int,
    init_positions: np.ndarray,
    shm_init: dict,
) -> tuple:
    """
    Worker function: interpolate initial field data for a bundle of tracers.

    Parameters
    ----------
    bundle_idx : int
        Bundle index used to re-order results from imap_unordered.
    init_positions : ndarray, shape (3, n_tracers)
        Cartesian positions of the tracers to initialise.
    shm_init : dict[str, str]
        Shared-memory name mappings for all field keys at the init timestep.

    Returns
    -------
    bundle_idx, init_data (n_keys, n_tracers)
    """
    global _setup_interp_fn, _extra_data

    interp = _setup_interp_fn(shm_init, _extra_data)
    interp.load()
    try:
        init_data = interp(init_positions)  # (n_keys, n_tracers)
    finally:
        interp.unload()

    return bundle_idx, init_data
