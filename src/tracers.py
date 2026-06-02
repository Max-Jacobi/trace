import pickle, time
import numpy as np
from itertools import repeat
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
      Simple class that holds the history of the position, interpolated data and time for a single tracer.
    """

    def __init__(
        self,
        id: int,
        position: np.ndarray,
        time: float,
        keys: list[str],
        props: dict[str, Any] | None = None,
        ) -> None:
        self.id = id
        self.initial_position = position.copy()
        self.initial_time = time
        self.positions = []
        self.times = []
        self.props = props if props is not None else {}
        self.data = {k: [] for k in keys}
        self.active = False
        self.done = False
        self.failed = False

    def add_step(
       self,
       position: np.ndarray,
       time: float,
       data: dict[str, float],
       ) -> None:

       self.positions.append(position.copy())
       self.times.append(time)
       for k, v in data.items():
           self.data[k].append(v)

    def output_to_ascii(self, coords: list[str], filebase: str) -> str:
        keys = list(self.data.keys())

        props = "; ".join((f"{key}={val}" for key, val in self.props.items()))
        legend = "{:>23s}" + "{:>26s}"* (4 + len(keys) - 1)
        legend = legend.format("time", *coords, *keys)
        header = f"{props}\n{legend}"

        filename = f"{filebase}{self.id:06d}.dat"

        tsort = np.argsort(self.times)

        incmp = False
        for key, dd in self.data.items():
            if len(dd) != len(tsort):
                incmp = True
        if incmp:
            return "failed"

        data = [np.array(self.times)[tsort]]
        data += [np.array(self.positions)[tsort, i] for i in range(len(coords))]
        data += [np.array(self.data[key])[tsort] for key in keys]
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

        if props is None:
            props = [{} for _ in range(positions.shape[0])]

        self.tracers = np.array([Tracer(id=i, position=pos, time=t, keys=self.file_handler.keys, props=prop)
                        for i, (pos, t, prop) in enumerate(zip(positions, times, props))])


    def integrate_loaded_chunk(self) -> None:
        times = self.file_handler.cur_times
        dts = np.diff(times)

        kwargs = dict(shared_memory=self.file_handler.shared_memory[:len(times)], extra_data=self.file_handler.extra_data)
        vel_interpolators = self.file_handler.setup_interpolators(self.vel_keys, **kwargs)
        data_interpolators = self.file_handler.setup_interpolators(self.file_handler.keys, **kwargs)

        n_bunches = 5*self.file_handler.parallel_kwargs["n_cpu"]

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
                initial_data = data_interpolators[i](init_pos) #shape (n_keys, n_tracers)
                for tr, data in zip(new_tracers, initial_data.T):
                    tr.add_step(
                        position=tr.initial_position,
                        time=tr.initial_time,
                        data=dict(zip(self.file_handler.keys, data)),
                    )
                    tr.active = True

            active_tracers = [tr for tr in self.tracers if tr.active]
            active_idxs = [i for i, tr in enumerate(self.tracers) if tr.active]

            if len(active_tracers) == 0:
                continue

            bunches = np.array_split(active_tracers, n_bunches)
            bunches = [b for b in bunches if len(b) > 0]

            bunches = do_parallel_star(
                _integrate_vectorized,
                zip(bunches, repeat(time)),
                desc=f"  Step {i}",
                total=len(bunches),
                initializer=_init_worker,
                initargs=initargs,
                position=self.pbar_pos+1,
                leave=False,
                unit="bunches",
                ncols=0,
                **self.file_handler.parallel_kwargs,
            )

            self.tracers[active_idxs] = np.concatenate(bunches)


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
        print(f"Chunk times: {file_times[chunk_indices]}")

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
    _vel_interpolators = vel_interpolators
    _data_interpolator = data_interpolator
    _keys = keys
    _integrator = integrator

    for interp in (*_vel_interpolators, _data_interpolator):
        interp.load()


def _integrate_vectorized(tracers: np.ndarray, time) -> np.ndarray:
    global _time, _dt, _vel_interpolators, _data_interpolator, _keys, _integrator
    assert _time is not None and _time == time, "Worker not initialized with correct time"

    positions = np.transpose([tracer.positions[-1] for tracer in tracers])

    new_pos = _integrator(
            xn=positions,
            dt=_dt,
            interps=_vel_interpolators,
    )
    valid_mask = np.isfinite(new_pos).all(axis=0)
    for tracer in tracers[~valid_mask]:
        # tracer has reached domain boundary
        tracer.done = True
        tracer.active = False

    valid_pos = new_pos[:, valid_mask]
    valid_tracers = tracers[valid_mask]

    new_data = _data_interpolator(valid_pos)

    for tracer, new_x, data in zip(
            valid_tracers,
            valid_pos.T,
            new_data.T
        ):
        tracer.add_step(
            position=new_x,
            time=time + _dt,
            data=dict(zip(_keys, data)),
        )

    return tracers
