import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from src.gra_surface import GRASurfaceFileHandler
from src.interpolators import PchipInterpolator3D, LinearInterpolator3D
from src.interpolators.coordinate_transformations import CartesianToSpherical
from src.integrators import ImplicitTrapezoid, ExplicitTrapezoid
from src.tracers import Tracers
from src.seeds import spherical_by_volume

def main():
    #mp.set_start_method("spawn")
    tracer_path = "/scratch2/11245/mjacobi/tracers_edu"
    data_path = f"{tracer_path}/data/"
    
    start_t = 11500
    end_t   = 3500
    rmin    = 300
    rmax    = 1000
    n_r     = 2
    n_th    = 2
    n_ph    = 2
    
    n_cpu = 8
    files_per_step = min(15, max(2, n_cpu))
    
    # interpolator = LinearInterpolator3D
    interpolator = PchipInterpolator3D
    
    # integrator = ExplicitTrapezoid()
    integrator = ImplicitTrapezoid(max_iter=5, relax=0.8)
    
    output_dir = f"{tracer_path}/test_"
    output_dir += f"nr{n_r}_nth{n_th}_nph{n_ph}_"
    output_dir += f"{integrator.__class__.__name__[:4]}_".lower()
    output_dir += f"{interpolator.__name__[:3]}".lower()
    os.makedirs(output_dir, exist_ok=True)
    
    filebase = f"{output_dir}/tracer_"
    
    
    file_handler = GRASurfaceFileHandler(
        interpolator=interpolator,
        surface_num=2,
        directory=data_path,
        log_rad=True,
        keys=[
            'tracer.hydro.aux.T',
            'tracer.hydro.aux.hu_t',
            'tracer.hydro.aux.s',
            'tracer.hydro.aux.u_t',
            'tracer.hydro.prim.rho',
            'tracer.passive_scalars.r_0',
            'tracer.hydro.aux.V_u_x',
            'tracer.hydro.aux.V_u_y',
            'tracer.hydro.aux.V_u_z',
            'M1.geom.sc_sqrt_det_g',
            'M1.rad.J_00',
            'M1.rad.J_01',
            'M1.rad.J_02',
            'M1.rad.n_00',
            'M1.rad.n_01',
            'M1.rad.n_02',
            'M1.rad.st_H_u_t_00',
            'M1.rad.st_H_u_t_01',
            'M1.rad.st_H_u_t_02',
            'M1.rad.st_H_u_x_00',
            'M1.rad.st_H_u_x_01',
            'M1.rad.st_H_u_x_02',
            'M1.rad.st_H_u_y_00',
            'M1.rad.st_H_u_y_01',
            'M1.rad.st_H_u_y_02',
            'M1.rad.st_H_u_z_00',
            'M1.rad.st_H_u_z_01',
            'M1.rad.st_H_u_z_02',
        ],
        n_cpu=n_cpu,
        files_per_step=files_per_step,
        verbose=True,
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
            'tracer.hydro.aux.V_u_x',
            'tracer.hydro.aux.V_u_y',
            'tracer.hydro.aux.V_u_z',
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
