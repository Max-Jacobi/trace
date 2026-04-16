import signal
import os
import numpy as np
import matplotlib.pyplot as plt

from src.gra_surface import GRASurfaceFileHandler
from src.interpolators import PchipInterpolator3D, LinearInterpolator3D
from src.integrators import ImplicitTrapezoid, ExplicitTrapezoid
from src.tracers import Tracers

##

tracer_path = "/home/ho54hof/repos/trace"

start_t = 6500
end_t = 3000
n_r = 3
n_th = 3
n_ph = 3
# interpolator = LinearInterpolator3D
interpolator = PchipInterpolator3D
# integrator = ExplicitTrapezoid()
integrator = ImplicitTrapezoid(max_iter=5, relax=0.8)

output_dir = f"{tracer_path}/test_"
output_dir += f"{integrator.__class__.__name__[:4]}_".lower()
output_dir += f"{interpolator.__name__[:3]}".lower()
os.makedirs(output_dir, exist_ok=True)

filebase = f"{output_dir}/tracer_"


file_handler = GRASurfaceFileHandler(
    interpolator=interpolator,
    #interpolator=PchipInterpolator3D,
    surface_num=2,
    directory=f"{tracer_path}/data/PL_LR/Lam400_0_LR/combine",
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
    ],
    n_cpu=12,
    files_per_step=15,
    verbose=True,
    )

def handler(signum, frame):
    if signum == signal.SIGINT:
        print("Received interrupt signal. Exiting gracefully...")
        file_handler.free_shared_memory()

signal.signal(signal.SIGINT, handler)

##

n_tracers = n_r * n_th * n_ph
t_start = np.full(n_tracers, 6000)
r_start = np.geomspace(400, 800, n_r+1)
dr = np.diff(r_start)
r_start = r_start[:-1] + dr/2
th_start = np.linspace(0, np.pi, n_th+1)
dth = np.diff(th_start)
th_start = th_start[:-1] + dth/2
ph_start = np.linspace(0, 2*np.pi, n_ph, endpoint=False)
dph = np.full(n_ph, 2*np.pi/n_ph)
ph_start = ph_start + dph/2

r_start, th_start, ph_start = np.meshgrid(r_start, th_start, ph_start, indexing='ij')
dr, dth, dph = np.meshgrid(dr, dth, dph, indexing='ij')
dV = r_start**2 * np.sin(th_start) * dr * dth * dph
r_start = r_start.flatten()
th_start = th_start.flatten()
ph_start = ph_start.flatten()
dV = dV.flatten()
props = [{'dV': v} for v in dV]

# n_tracers = 3
# t_start = np.full(n_tracers, start_t)
# r_start = np.full(n_tracers, 1000.)
# th_start = np.full(n_tracers, np.pi/2)
# ph_start = np.linspace(0, 2*np.pi, n_tracers, endpoint=False)

x_start = r_start*np.sin(th_start)*np.cos(ph_start)
y_start = r_start*np.sin(th_start)*np.sin(ph_start)
z_start = r_start*np.cos(th_start)

tracers = Tracers(
    positions=np.array([x_start, y_start, z_start]).T,
    times=t_start,
    props=props,
    vel_keys=[
        'tracer.hydro.aux.V_u_x',
        'tracer.hydro.aux.V_u_y',
        'tracer.hydro.aux.V_u_z',
    ],
    integrator=integrator,
    file_handler=file_handler,
    )

##

times = t_start
file_times = file_handler.times
n_files_per_step = file_handler.n_files_per_step
forward = end_t > start_t

if forward:
    t_start = np.min(times)
    t_end = np.max(file_times)
    i_start = np.where(file_times == t_start)[0][0]
    i_end = np.where(file_times == t_end)[0][0]
    i_end = min(len(file_times) - 1, i_end)
    chunk_indices = np.arange(i_start, i_end+1, n_files_per_step-1)
else:
    t_start = np.max(times)
    t_end = np.max(file_times[file_times <= end_t])
    i_start = np.where(file_times == t_start)[0][0]
    i_end = max(0, np.where(file_times == t_end)[0][0])
    chunk_indices = np.arange(i_start, i_end-1, -n_files_per_step+1)
    t_end = file_times[chunk_indices[-1]-n_files_per_step]

print(f"Integrating from t={t_start} to t={t_end} with {len(chunk_indices)} chunks.")
print(f"Chunk times: {file_times[chunk_indices]}")


##
for i_step in chunk_indices:
    file_handler.load_chunk(i_step, forward=forward)
    tracers.integrate()
    if all(tr.done for tr in tracers.tracers):
        print("All tracers done. Stopping integration.")
        break

fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
keys = ['T', 'rho', 'ye']

for ax, key in zip(axs, keys):
    ax.set_xlabel("t")
    ax.set_ylabel(key)
axs[1].set_yscale("log")

for tr in tracers.tracers:
    t = np.array(tr.times) * 0.004925502303934785
    T = np.array(tr.data['tracer.hydro.aux.T']) * 11.604522060401004
    rho = np.array(tr.data['tracer.hydro.prim.rho']) * 6.176003576200146e+17
    ye = tr.data['tracer.passive_scalars.r_0']
    axs[0].plot(t, T, lw=.4)
    axs[1].plot(t, rho, lw=.4)
    axs[2].plot(t, ye, lw=.4)
axs[0].set_ylim(0, 20)
axs[1].set_ylim(1e6, 1e12)
axs[2].set_ylim(0, 0.5)
plt.tight_layout()
plt.savefig(f"{output_dir}/tracers.png")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))

for a in ax:
    a.set_xlabel("t")
ax[0].set_ylabel("r")
ax[0].set_ylim(0, 900)
ax[1].set_ylabel("theta")
ax[1].set_ylim(0, np.pi)
ax[2].set_ylabel("phi")
ax[2].set_ylim(0, 2*np.pi)

print(np.array(tracers.tracers[0].positions).shape)
print(max(np.array(tr.positions).max() for tr in tracers.tracers))
print(min(np.array(tr.positions).min() for tr in tracers.tracers))
for tr in tracers.tracers:
    t = np.array(tr.times) * 0.004925502303934785
    pos = np.array(tr.positions)
    x, y, z = pos.T
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    ax[0].plot(t, r, lw=.4)
    ax[1].plot(t, theta, lw=.4)
    ax[2].plot(t, phi, lw=.4)

plt.tight_layout()
plt.savefig(f"{output_dir}/tracers_pos.png")

for tr in tracers.tracers:
    tr.props['mass'] = tr.props['dV'] * tr.data['tracer.hydro.prim.rho'][0]
    tr.output_to_ascii(coords=['x', 'y', 'z'], filebase=filebase)
