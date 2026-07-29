import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from src.reduced_surface import ReducedSurfaceFileHandler
from src.interpolators import PchipInterpolator3D, RegularInterpolator3D
from src.interpolators.coordinate_transformations import CartesianToSpherical
from src.integrators import ImplicitTrapezoid, ExplicitTrapezoid
from src.tracers import Tracers
from src.seeds import spherical_by_volume

def main():
    mp.set_start_method("spawn")
    tracer_path = "data"
    data_path = f"{tracer_path}/transformed/"

    start_t = 11600
    end_t   = 0
    rmin    = 300
    rmax    = 1000
    n_r     = 3
    n_th    = 2
    n_ph    = 3

    keys = ('V_u_x', 'V_u_y', 'V_u_z',
            'T', 'hu_t', 's',
            'u_t', 'rho', 'r_0',
            'F_nue', 'F_anue', 'F_nux',
            'eps_nue', 'eps_anue', 'eps_nux')

    vel_keys = ('V_u_x', 'V_u_y', 'V_u_z')

    n_cpu = int(sys.argv[1])

    # get free memory from meminfo
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                free_mem_GB = int(line.split()[1]) / 1024**2  # convert from kB to bytes
                break
    files_per_step = 10 #max(2, n_cpu)
    n_interpolators = (len(keys)+len(vel_keys))*n_cpu
    mem_avail_per_interp_GB = 0.8 * free_mem_GB / n_interpolators

    # interpolator = RegularInterpolator3D
    # interpolator_kwargs = {"method": "pchip"}
    interpolator = PchipInterpolator3D
    print(f"Using {n_interpolators} {interpolator.__name__}s with cache size of {mem_avail_per_interp_GB:.2f} GB")
    print(f"  resulting in {n_interpolators*mem_avail_per_interp_GB} GB total cache size. Free memory = {free_mem_GB} GB")
    interpolator_kwargs = {"max_cache_size_GB": mem_avail_per_interp_GB}

    # integrator = ExplicitTrapezoid()
    integrator = ImplicitTrapezoid(max_iter=3, relax=0.8)

    output_dir = f"{tracer_path}/test_"
    output_dir += f"nr{n_r}_nth{n_th}_nph{n_ph}_"
    output_dir += f"{integrator.__class__.__name__[:4]}_".lower()
    output_dir += f"{interpolator.__name__[:3]}".lower()
    os.makedirs(output_dir, exist_ok=True)

    filebase = f"{output_dir}/tracer_"

    file_handler = ReducedSurfaceFileHandler(
        interpolator=interpolator,
        directory=data_path,
        keys=keys,
        n_cpu=n_cpu,
        verbose=True,
        files_per_step=files_per_step,
        interpolator_kwargs=interpolator_kwargs,
       )

    ##

    tracers = spherical_by_volume(
        r_min=rmin,
        r_max=rmax,
        n_r=n_r,
        n_th=n_th,
        n_ph=n_ph,
        start_t=start_t,
        integrator=integrator,
        file_handler=file_handler,
        vel_keys=vel_keys,
    )

    tracers.integrate(start_t, end_t)

    for tr in tracers.tracers:
        i_tmax = np.argmax(tr.times)
        tr.props['mass'] = tr.props['dV'] * tr.data['rho'][i_tmax]

        short_keys = {key: key.split(".")[-1] for key in tr.data.keys()}
        for key, short in short_keys.items():
            if short == key:
                continue
            tr.data[short] = tr.data[key]
            del tr.data[key]

        tr.output_to_ascii(coords=['x', 'y', 'z'], filebase=filebase)
if __name__ == "__main__":
     main()
