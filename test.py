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
    #mp.set_start_method("spawn")
    tracer_path = "/beegfs/ho54hof/simulations/Lam300_1_LR/"
    data_path = f"{tracer_path}/transformed/"
    
    start_t = 16080
    end_t   = 15500
    rmin    = 300
    rmax    = 1000
    n_r     = 20
    n_th    = 10
    n_ph    = 20
    
    n_cpu = int(sys.argv[1])
    files_per_step = max(2, n_cpu)
    
    interpolator = RegularInterpolator3D
    # interpolator = PchipInterpolator3D
    
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
        log_rad=True,
        keys=[
            'V_u_x',
            'V_u_y',
            'V_u_z',
            'T',
            'hu_t',
            's',
            'u_t',
            'rho',
            'r_0',
            'F_nue',
            'F_nua',
            'F_nux',
            'eps_nue',
            'eps_nua',
            'eps_nux',
        ],
        n_cpu=n_cpu,
        files_per_step=files_per_step,
        verbose=True,
        interpolator_kwargs={"method": "linear"},
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
        vel_keys=(
            'V_u_x',
            'V_u_y',
            'V_u_z',
        ),
    )
    
    tracers.integrate(start_t, end_t)
    
    for tr in tracers.tracers:
        tr.props['mass'] = tr.props['dV'] * tr.data['tracer.hydro.prim.rho'][0]
    
        short_keys = {key: key.split(".")[-1] for key in tr.data.keys()}
        for key, short in short_keys.items():
            if short == key:
                continue
            tr.data[short] = tr.data[key]
            del tr.data[key]
    
        tr.output_to_ascii(coords=['x', 'y', 'z'], filebase=filebase)
if __name__ == "__main__":

     main()
