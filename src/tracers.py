import numpy as np
from tqdm import tqdm
from typing import Any, Optional

from .integrators.base import IntegratorBase, InterpolatorCallable
from .file import FileHandler
from .utils import do_parallel, do_parallel_star

_time: Optional[float] = None
_dt: Optional[float] = None
_integrator: Optional[IntegratorBase] = None
_vel_interpolators: Optional[list[InterpolatorCallable]] = None
_data_interpolator: Optional[InterpolatorCallable] = None
_keys: Optional[tuple[str, ...]] = None

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
        self._positions = np.empty((n_steps, 3), dtype=np.float64)
        self._times     = np.empty(n_steps,      dtype=np.float64)
        self._data      = {k: np.empty(n_steps, dtype=np.float64) for k in keys}

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

        self.tracers = np.array([Tracer(id=i, position=pos, time=t, keys=self.file_handler.keys, n_steps=n_steps, props=prop)
                        for i, (pos, t, prop) in enumerate(zip(positions, times, props))])


    def integrate_loaded_chunk(self) -> None:
        times = self.file_handler.cur_times
        dts = np.diff(times)

        kwargs = dict(shared_memory=self.file_handler.shared_memory[:len(times)], extra_data=self.file_handler.extra_data)
        vel_interpolators = self.file_handler.setup_interpolators(self.vel_keys, **kwargs)
        data_interpolators = self.file_handler.setup_interpolators(self.file_handler.keys, **kwargs)

        n_bunches = 20*self.file_handler.parallel_kwargs["n_cpu"]

        for i, (dt, time) in tqdm(
            enumerate(zip(dts, times)),
            desc="Integrating tracers",
            total=len(dts),
            position=self.pbar_pos,
            unit="time step",
            ncols=0,
        ):

            initargs = (
                time,
                dt,
                self.integrator,
                vel_interpolators[i:i+2],
                data_interpolators[i+1],
                self.file_handler.keys,
            )

            new_tracers = np.array([tr for tr in self.tracers
                                    if len(tr.positions) == 0 and np.isclose(tr.initial_time, time)])

            if len(new_tracers) > 0:
                init_pos = np.array([tracer.initial_position for tracer in new_tracers]).T
                data_interpolators[i].load()
                initial_data = data_interpolators[i](init_pos) #shape (n_keys, n_tracers)
                data_interpolators[i].unload()
                for tr, data in zip(new_tracers, initial_data.T):
                    tr.add_step(
                        position=tr.initial_position,
                        time=tr.initial_time,
                        data=dict(zip(self.file_handler.keys, data)),
                    )
                    tr.active = True

            active_tracers = np.array([tr for tr in self.tracers if tr.active])

            if len(active_tracers) == 0:
                continue

            if hasattr(vel_interpolators[i], "sort_tracers"):
                active_tracers = vel_interpolators[i].sort_tracers(active_tracers)

            # Extract only the current positions — O(n_tracers) pickle data,
            # independent of integration history, eliminating the memory growth.
            positions = np.array([tr.positions[-1] for tr in active_tracers]).T  # (3, n_active)

            tracer_bunches = [b for b in np.array_split(active_tracers, n_bunches) if len(b) > 0]
            split_sizes = np.cumsum([len(b) for b in tracer_bunches[:-1]])
            position_bunches = np.split(positions, split_sizes, axis=1)

            results = do_parallel_star(
                _integrate_positions,
                [(j, pos, time) for j, pos in enumerate(position_bunches)],
                desc=f"  Step {i}",
                total=len(tracer_bunches),
                initializer=_init_worker,
                initargs=initargs,
                position=self.pbar_pos+1,
                leave=False,
                unit="bunches",
                ncols=0,
                **self.file_handler.parallel_kwargs,
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
            # print free memory after each step
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            mem_free = int(meminfo.split("MemFree:")[1].split()[0])
            est_int_cache = sum(interp._estimate_cached_memory()
                                for interp in (*data_interpolators, *vel_interpolators))
            est_tracer_memory = sum(
                tr._positions.nbytes + tr._times.nbytes +
                sum(arr.nbytes for arr in tr._data.values())
                for tr in self.tracers
            )

            print()
            print(f"Free memory: {mem_free/1024**2:.2f} GB")
            print(f"Estimated interp. cache: {est_int_cache/1024**3:.2e} GB")
            print(f"Estimated tracer data: {est_tracer_memory/1024**3:.2e} GB")


            print(f"{sum(tr.done for tr in self.tracers)} tracers done. "
                  f"{sum(tr.active for tr in self.tracers)} tracers active.")

    def integrate(self, start_t: float, end_t: float) -> None:

        file_times = self.file_handler.times
        n_files_per_step = self.file_handler.n_files_per_step
        forward = end_t > start_t

        if forward:
            t_start = np.min(file_times[file_times >= start_t])
            t_end = np.max(file_times[file_times <= end_t])
            i_start = file_times.searchsorted(t_start, side='left')
            i_end = file_times.searchsorted(t_end, side='left')
            chunk_indices = np.arange(i_start, i_end+1, n_files_per_step-1)
        else:
            t_start = np.max(file_times[file_times <= start_t])
            t_end = np.min(file_times[file_times >= end_t])
            i_start = file_times.searchsorted(t_start, side='left')
            i_end = file_times.searchsorted(t_end, side='left')
            chunk_indices = np.arange(i_start, i_end-1, -n_files_per_step+1)
        print(f"Integrating from t={t_start} to t={t_end} with {len(chunk_indices)} chunks.")

        for i_step in chunk_indices:
            self.file_handler.load_chunk(i_step, forward=forward)
            self.integrate_loaded_chunk()
            if all(tr.done for tr in self.tracers):
                print("All tracers done. Stopping integration.")
                break


def _init_worker(
    time: float,
    dt: float,
    integrator: IntegratorBase,
    vel_interpolators: list[InterpolatorCallable],
    data_interpolator: InterpolatorCallable,
    keys: tuple[str, ...],
    ) -> None:
    global _time, _dt, _vel_interpolators, _data_interpolator, _keys, _integrator

    _time = time
    _dt = dt

    if _vel_interpolators is not None:
        for interp in (*_vel_interpolators, _data_interpolator):
            if hasattr(interp, "_xp_cache"):
                 interp._xp_cache.clear()
            interp.unload()

    _vel_interpolators = vel_interpolators
    _data_interpolator = data_interpolator
    _keys = keys
    _integrator = integrator
    for interp in (*_vel_interpolators, _data_interpolator):
        interp.load()

def _integrate_positions(
    bundle_idx: int,
    positions: np.ndarray,
    time: float,
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
        Current integration time.

    Returns
    -------
    bundle_idx, new_pos (3, n), new_data (n_keys, n), valid_mask (n,)
        new_pos / new_data are NaN for tracers that left the domain.
    """
    global _time, _dt, _vel_interpolators, _data_interpolator, _keys, _integrator
    assert _time is not None and _time == time, "Worker not initialized with correct time"

    n = positions.shape[1]
    new_pos = _integrator(xn=positions, dt=_dt, interps=_vel_interpolators)
    valid_mask = np.isfinite(new_pos).all(axis=0)

    new_data = np.full((len(_keys), n), np.nan)
    if valid_mask.any():
        new_data[:, valid_mask] = _data_interpolator(new_pos[:, valid_mask])

    return bundle_idx, new_pos, new_data, valid_mask
