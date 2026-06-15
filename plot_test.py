import pathlib as pl
import numpy as np
from multiprocessing import Pool
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from tqdm import tqdm

from src.trajectory import Trajectory
import tabulatedEOS.unit_system as us

lfac = us.GeometricSolar.LengthConversion(us.CGS)*1e-5
tfac = us.GeometricSolar.TimeConversion(us.CGS)*1000
Tfac = us.GeometricSolar.TemperatureConversion(us.CGS)/1e9
rhofac = us.GeometricSolar.MassDensityConversion(us.CGS)

##

PATH_IN     = "data/test_nr30_nth15_nph30_impl_pch"
PATH_OUT    = "data/test_surf_nth15_nph30_impl_pch"
OUTPUT_PATH = "data"
N_CPU       = 12

prefix = "tracer_"

files = []
#files += sorted(pl.Path(PATH_IN).glob("tracer*"))
files += sorted(pl.Path(PATH_OUT).glob("tracer*"))
file_paths = [str(f) for f in files]

with Pool(N_CPU) as pool:
    trajs = list(tqdm(
        pool.imap_unordered(Trajectory.from_ascii, file_paths),
        total=len(file_paths),
        ncols=0, unit="tracers", desc="Loading tracers", leave=False,
    ))


print(f"Loaded {len(trajs)} trajectories.")
trajs = [t for t in trajs if t.data['T'][0] >= 4/Tfac]
print(f"{len(trajs)} trajectories left after filtering.")


for traj in trajs:
    traj.data["r"] = np.sqrt(traj.data["x"]**2 + traj.data["y"]**2 + traj.data["z"]**2)
    phi = np.arctan2(traj.data["y"], traj.data["x"])
    phi[phi < 0] += 2*np.pi
    phi = np.unwrap(phi) * 180/np.pi
    theta = np.arccos(traj.data["z"]/traj.data["r"]) * 180/np.pi
    traj.data["phi"] = phi
    traj.data["theta"] = theta

##

fig, ax = plt.subplots(2, 3, figsize=(15, 10))

ye_norm = Normalize(vmin=0.1, vmax=0.6)
theta_norm = Normalize(vmin=0, vmax=90)
cmap = plt.get_cmap("jet_r")
for traj in trajs[::5]:
    t = traj.data["time"] * tfac
    r = traj.data["r"] * lfac
    theta = traj.data["theta"]
    phi = traj.data["phi"]

    T = traj.data["T"] * Tfac
    rho = traj.data["rho"] * rhofac
    ye = traj.data["r_0"]
    # color = cmap(ye_norm(np.average(ye)))
    color = cmap(theta_norm(np.abs(np.average(theta)-90)))

    kw = dict(lw=.3, alpha=.5, c=color)
    ax[0, 0].plot(t, r, **kw)
    ax[0, 1].plot(t, phi, **kw)
    ax[0, 2].plot(t, theta, **kw)
    ax[1, 0].plot(t, rho, **kw)
    ax[1, 1].plot(t, T, **kw)
    ax[1, 2].plot(t, ye, **kw)
for a, yl in zip(ax.flat, ["r (km)", "phi (deg)", "theta (deg)",
                           r"$\rho$ (g/cm$^3$)", "T (GK)", r"$Y_e$"]):
    a.set_xlabel("time (ms)")
    a.set_ylabel(yl)
ax[1, 0].set_yscale("log")
plt.colorbar(plt.cm.ScalarMappable(norm=theta_norm, cmap=cmap), label="theta (deg)", ax=ax[0, 2])
plt.gca().set_rasterization_zorder(-1)
plt.savefig(f"{OUTPUT_PATH}/test_traj.png", dpi=300, bbox_inches="tight")

##

def at_6GK(traj, key):
    T = traj.data["T"] * Tfac
    return np.interp(6, T[::-1], traj.data[key][::-1])

mm = np.abs([traj.props['mass'] for traj in trajs])
ye6 = np.array([at_6GK(traj, 'r_0') for traj in trajs])
s6 = np.array([at_6GK(traj, 's') for traj in trajs])
th = np.array([traj.data['theta'][-1] for traj in trajs])
T0 = np.array([traj.data['T'][0] for traj in trajs]) * Tfac

#t_ej = np.array([traj.data['time'][0] for traj in trajs]) * us.GeometricSolar.TimeConversion(us.CGS)*1000

fig, ax = plt.subplots(2, 2, figsize=(13, 10))
ax = ax.flatten()

ax[0].hist(ye6, bins=30, weights=mm, histtype="step");
ax[0].set_xlabel(r"$Y_e$ at 6 GK")

ax[1].hist(s6, bins=np.linspace(0, 120, 30), weights=mm, histtype="step");
ax[1].set_xlabel(r"$s$ at 6 GK ($k_{\rm B}$)")

ax[2].hist(T0, bins=30, weights=mm, histtype="step");
ax[2].set_xlabel(r"$T_{\rm max}$ (GK)")

ax[3].hist(th, bins=30, weights=mm, histtype="step");
ax[3].set_xlabel(r"$\theta$ (rad)")

for a in ax:
    a.set_ylabel(r"$\Delta m$ (M$_\odot$)")
plt.savefig(f"{OUTPUT_PATH}/test_traj_hists.png", dpi=300, bbox_inches="tight")
# plt.hist(t_ej, bins=30, weights=mm, histtype="step", label="t_ej")

##

x_grid = np.linspace(150, 1000, 50)
dye_grid = np.geomspace(1e-7, 2e-3, 50)
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
norm = LogNorm(1, 1e4)
for ax, direc in zip(axs, 'xyz'):
    img = np.zeros((len(dye_grid), len(x_grid)))
    for traj in trajs:
        dyedt = np.gradient(traj.data['r_0'], traj.data['time'])
        # plt.plot(traj.data['r'], np.abs(dyedt), lw=1, c='k', alpha=0.05)
        i_x = np.digitize(traj.data[direc], x_grid) - 1
        i_dye = np.digitize(np.abs(dyedt), dye_grid) - 1
        for ix, idye in zip(i_x, i_dye):
            if 0 <= ix < len(x_grid) and 0 <= idye < len(dye_grid):
                img[idye, ix] += 1
    im = ax.pcolormesh(x_grid*lfac, dye_grid/tfac, img, norm=norm, cmap="nipy_spectral")
    ax.set_xlabel(f"{direc} (km)")
    # ax.set_xlim(150, 1000)
    ax.set_yscale("log")
plt.colorbar(im, label="Number of tracers", ax=axs)
axs[0].set_ylabel(r"$|\dot{Y_e}|$ [1/ms]")
# plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/test_traj_dye.png", dpi=300, bbox_inches="tight")
