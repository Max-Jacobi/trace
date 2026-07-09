#!/bin/env python3
################################################################################
import argparse
import os
from datetime import datetime
from typing import Any
import pathlib as pl
from multiprocessing import Pool

import numpy as np
from h5py import File
from tqdm import tqdm
import matplotlib.pyplot as plt

from surface import Surfaces
from surface import surface_func as sf
import tabulatedEOS.unit_system as us

from src.trajectory import Trajectory


parser = argparse.ArgumentParser(
    description="Postprocessing for surface outputs"
)
parser.add_argument("simpath", type=str, help="Path to simulation")
parser.add_argument("-t", "--tracer_dirs", default=[], type=str, nargs='+', help="Path to simulation")
parser.add_argument("-b", "--batchtools",  action="store_true",
                    help="assume batchtools subdirectory structure (output-0000...)")
parser.add_argument("-o", "--outputpath", default=None,
                    help="Directory to output to")
parser.add_argument("-s", "--isurf", default=1, type=int,
                    help="Index of surface to use")
parser.add_argument("-r", "--irad", default=0, type=int,
                    help="Index of radius to use")
parser.add_argument("-n", "--ncpu", default=1, type=int,
                    help="Number of cores to use")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Print progress")

################################################################################

args = parser.parse_args()

if args.outputpath is None:
    today = datetime.today().strftime("%Y-%m-%d")
    args.outputpath = f"mdot_{today}"

os.makedirs(args.outputpath, exist_ok=True)

lfac = us.GeometricSolar.LengthConversion(us.CGS)*1e-5
tfac = us.GeometricSolar.TimeConversion(us.CGS)
Tfac = us.GeometricSolar.TemperatureConversion(us.CGS)/1e9
rhofac = us.GeometricSolar.MassDensityConversion(us.CGS)


################################################################################

if args.batchtools:
    paths = [f"{args.simpath}/{d}" for d in os.listdir(args.simpath)
             if (os.path.isdir(f"{args.simpath}/{d}")
                 and d.startswith("output-"))]
else:
    paths = [args.simpath]

s = Surfaces(
    paths,
    args.isurf,
    args.irad,
    n_cpu=args.ncpu,
    verbose=args.verbose,
    )

mdot_sf = sf.integrate_flux_classical(
    flux=("tracer.hydro.aux.V_u_x", "tracer.hydro.aux.V_u_y", "tracer.hydro.aux.V_u_z"),
    weight="tracer.hydro.prim.rho",
)

data = s.process_h5_parallel((mdot_sf,), ordered=True)
mdot = np.array([d[0] for d in data])

mtot = np.cumsum(mdot * s.dts)

################################################################################

files = []
for path in args.tracer_dirs:
    files += sorted(pl.Path(path).glob("tracer_*"))
file_paths = [str(f) for f in files]

with Pool(args.ncpu) as pool:
    trajs = list(tqdm(
        pool.imap_unordered(Trajectory.from_ascii, file_paths),
        total=len(file_paths),
        ncols=0, unit="tracers", desc="Loading tracers", leave=False,
    ))

print(f"Loaded {len(trajs)} trajectories.")
rmin = 200
trajs = [t for t in trajs if
         #t.data['T'][0] >= 4/Tfac and
         t.props['status'] != 'active' and
         (t.data['x'][0]**2+t.data['y'][0]**2 + t.data['z'][0]**2) < rmin**2]

srf = np.array(['surf' in t.props['filename'] for t in trajs])
vol = ~srf
print(f"{len(trajs)} trajectories left after filtering.")

times = np.unique(np.concatenate([tr.data["time"] for tr in trajs]))
mm = np.abs([traj.props['mass'] for traj in trajs])
n = len(times)
tr_r = np.zeros((len(times), len(trajs)))
for i, tracer in enumerate(tqdm(
    trajs,
    desc="Postprocessing tracers",
    ncols=0, unit="tracers", leave=False,
)):
    t_msk = np.isin(times, tracer.data["time"])
    x = tracer.data["x"]
    y = tracer.data["y"]
    z = tracer.data["z"]
    tr_r[ t_msk, i] = np.sqrt(x*x + y*y + z*z)
    if t_msk.any():
        last_i = n - 1 - np.argmax(t_msk[::-1])
        tr_r[last_i:, i] = tr_r[last_i, i]

tr_mtot_vol = np.array([np.sum(mm[vol][r[vol]>=s.r]) for r in tr_r])
tr_mtot_srf = np.array([np.sum(mm[srf][r[srf]>=s.r]) for r in tr_r])
tr_mtot_vol += tr_mtot_srf

tr_mdot_vol = np.zeros_like(tr_mtot_vol)
tr_mdot_srf = np.zeros_like(tr_mtot_srf)
tr_mdot_vol[1:] = np.diff(tr_mtot_vol) / np.diff(times)
tr_mdot_srf[1:] = np.diff(tr_mtot_srf) / np.diff(times)


# m = 5
# kern = np.ones(m)/m
# tr_mdot_vol = np.convolve(tr_mdot_vol, kern, mode='same')
# tr_mdot_srf = np.convolve(tr_mdot_srf, kern, mode='same')

################################################################################

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(times, tr_mdot_vol/tfac, label='tracers')
ax[0].plot(times, tr_mdot_srf/tfac, label='surface tracers')
ax[1].plot(times, tr_mtot_vol, label='tracers')
ax[1].plot(times, tr_mtot_srf, label='surface tracers')

ax[0].plot(s.times, mdot/tfac, label='surface', c='k')
ax[1].plot(s.times, mtot, label='surface', c='k')

ax[0].set_xlabel("time (ms)")
ax[1].set_xlabel("time (ms)")
ax[0].set_xlim(3200, 12000)
ax[1].set_xlim(3200, 12000)
ax[0].set_ylabel(r"$\dot{M}$ (M$_\odot$/s)")
ax[1].set_ylabel(r"$M$ (M$_\odot$)")
ax[0].set_ylim(0, mdot.max()/tfac*1.1)
ax[1].set_ylim(0, mtot.max()*1.1)
# ax[0].set_ylim(1e-3, mdot.max()/tfac*1.1)
# ax[1].set_ylim(1e-5, mtot.max()*1.1)
# ax[0].set_yscale('log')
# ax[1].set_yscale('log')
fig.suptitle(f"R={s.r:.2f}M")
ax[1].legend()

plt.savefig(f"{args.outputpath}/mdot_r{args.irad}.png", dpi=300)
