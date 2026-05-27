import numpy as np
from tqdm import tqdm
from typing import Any

from .integrators.base import IntegratorBase, InterpolatorCallable
from .file import FileHandler
from .utils import do_parallel

# module level variables loaded on worker initialization
_dt: float | None = None
_time: float | None = None
_integrator: IntegratorBase | None = None
_vel_interpolators: list[InterpolatorCallable] | None = None
_data_interpolator: InterpolatorCallable | None = None
_keys: tuple[str, ...] | None = None

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


    def integrate(self) -> None:
        vel_interpolators = self.file_handler.setup_interpolators(self.vel_keys)
        data_interpolators = self.file_handler.setup_interpolators(self.file_handler.keys)

        times = self.file_handler.cur_times
        dts = np.diff(times)
        forward = dts[0] > 0

        n_chunks = 5*self.file_handler.parallel_kwargs["n_cpu"]

        for i, (dt, time) in tqdm(
            enumerate(zip(dts, times)),
            desc="Integrating tracers",
            total=len(dts),
            position=self.pbar_pos,
            unit="step",
            ncols=0,
        ):
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

            chunks = np.array_split(active_tracers, n_chunks)
            chunks = [ch for ch in chunks if ch.size>0]

            chunks = do_parallel(
                _do_integrate_vectorized,
                chunks,
                desc=f"  Step {i}",
                total=len(chunks),
                position=self.pbar_pos+1,
                leave=False,
                unit="chunks",
                ncols=0,
                initializer=_init_worker,
                initargs=(
                    time,
                    dt,
                    self.integrator,
                    vel_interpolators[i:i+2],
                    data_interpolators[i+1],
                    self.file_handler.keys,
                ),
                **self.file_handler.parallel_kwargs,
            )

            self.tracers[active_idxs] = np.concatenate(chunks)

        print(f"{sum(tr.done for tr in self.tracers)} tracers done. "
              f"{sum(tr.active for tr in self.tracers)} tracers active.")

def _init_worker(
    time: float,
    dt: float,
    integrator: IntegratorBase,
    vel_interpolators: list[InterpolatorCallable],
    data_interpolator: InterpolatorCallable,
    keys: tuple[str, ...],
):
    global _dt, _time, _integrator, _vel_interpolators, _data_interpolator, _keys, _dt

    _dt = dt
    _time = time
    _integrator = integrator
    _vel_interpolators = vel_interpolators
    _data_interpolator = data_interpolator
    _keys = keys


def _do_integrate(tracer: Tracer):
    global _dt, _integrator, _vel_interpolators, _data_interpolator, _keys, _dt
    new_x = _integrator(
            xn=tracer.positions[-1],
            dt=_dt,
            interps=_vel_interpolators,
    )

    if np.isnan(new_x).any():
        # tracer has reached domain boundary
        tracer.done = True
        tracer.active = False
        return

    raw_data = _data_interpolator(new_x)
    new_data = {k: raw_data[idx] for idx, k in enumerate(_keys)}

    tracer.add_step(
        position=new_x,
        time=_time + _dt,
        data={k: new_data[k] for k in _keys},
    )

    return tracer

def _do_integrate_vectorized(tracers: list[Tracer]):
    global _dt, _integrator, _vel_interpolators, _data_interpolator, _keys, _dt
    tracers = np.array(tracers)

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
            time=_time + _dt,
            data=dict(zip(_keys, data)),
        )

    return tracers
