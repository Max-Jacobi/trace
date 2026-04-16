import numpy as np
from tqdm import tqdm
from typing import Any

from .integrators.base import IntegratorBase, InterpolatorCallable
from .file import FileHandler


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

        file_times = self.file_handler.times

        unmatched = [t for t in times if not np.any(np.isclose(t, file_times))]
        if unmatched:
            raise ValueError(
                f"Some seed times do not coincide with available file times: {unmatched[:5]}"
            )

        if props is None:
            props = [{} for _ in range(positions.shape[0])]

        self.tracers = [Tracer(id=i, position=pos, time=t, keys=self.file_handler.keys, props=prop)
                        for i, (pos, t, prop) in enumerate(zip(positions, times, props))]

        self.tqdm_kwargs = {
            "unit": "step",
            "position": pbar_pos,
            "disable": pbar_pos < 0,
            "ncols": 0,
        }

    def interpolate_data(
        self,
        positions: np.ndarray,
        interpolator: InterpolatorCallable,
        ) -> dict[str, np.ndarray]:

        data = interpolator(positions)
        return {k: data[idx] for idx, k in enumerate(self.file_handler.keys)}


    def integrate(self) -> None:
        vel_interpolators = self.file_handler.setup_interpolators(self.vel_keys)
        data_interpolators = self.file_handler.setup_interpolators(self.file_handler.keys)

        times = self.file_handler.cur_times
        dts = np.diff(times)
        forward = dts[0] > 0

        for i, (dt, time) in tqdm(
            enumerate(zip(dts, times)),
            desc="Integrating tracers",
            total=len(dts),
            **self.tqdm_kwargs
        ):
            new_tracers = [tr for tr in self.tracers
                           if len(tr.positions) == 0 and np.isclose(tr.initial_time, time)]
            if new_tracers:
                new_pos = np.array([tr.initial_position for tr in new_tracers]).T
                data = self.interpolate_data(new_pos, data_interpolators[i])
                for j, tr in enumerate(new_tracers):
                   tr.add_step(
                       position=tr.initial_position,
                       time=tr.initial_time,
                       data={k: data[k][j] for k in self.file_handler.keys},
                   )

            for tr in self.tracers:
                tr.active = (len(tr.times) > 0) and not tr.done and (
                    (forward and tr.times[-1] <= time) or
                    (not forward and tr.times[-1] >= time)
                )
            active_tracers = [tr for tr in self.tracers if tr.active]
            if not any(active_tracers):
                continue

            pos = np.array([tracer.positions[-1] for tracer in active_tracers]).T


            new_x = self.integrator(xn=pos, dt=dt, interps=vel_interpolators[i:i+2],)
            nan_mask = np.isnan(new_x).any(axis=0)
            for j, tracer in enumerate(active_tracers):
                if nan_mask[j]:
                    tracer.done = True
                    tracer.active = False

            new_data = {k: np.full(len(active_tracers), np.nan) for k in self.file_handler.keys}
            for k, data in self.interpolate_data(new_x[:, ~nan_mask], data_interpolators[i+1]).items():
                new_data[k][~nan_mask] = data

            for j, tracer in enumerate(active_tracers):
                if nan_mask[j]:
                    continue
                tracer.add_step(
                    position=new_x[:, j],
                    time=time + dt,
                    data={k: new_data[k][j] for k in self.file_handler.keys},
                )
        print(f"{sum(tr.done for tr in self.tracers)} tracers done. "
              f"{sum(tr.active for tr in self.tracers)} tracers active.")
