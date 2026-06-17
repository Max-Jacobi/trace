import os
import pathlib as pl
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool
from tqdm import tqdm
from src.trajectory import Trajectory

N_CPU       = 10
PATH_IN     = "test_nr30_nth15_nph20_impl_pch"
PATH_OUT    = "test_surf_nth15_nph20_impl_pch"
OUTPUT_PATH = "frames_tmp"
VIDEO_OUT   = "tracer_animation.mp4"
FPS         = 30

##  worker initializer
# Called once per worker at pool creation. Stores shared readonly arrays as
# modulelevel globals so tasks never pickle the data.
def _worker_init(arr, t):
    global tr_pos_ye, times
    tr_pos_ye = arr
    times     = t

##  plotting task
def make_plot(i):
    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_subplot(111, projection="3d")

    t = times[i]
    ax.view_init(elev=20, azim=180 * (t - times[0])/(times[-1] - times[0]))
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    ax.set_zlim(-1500, 1500)

    ax.set_xlabel('x [km]')
    ax.set_ylabel('x [km]')
    ax.set_zlabel('x [km]')
    ax.set_title(f"t = {t*0.004925502303934785:.0f}ms")

    dots = ax.scatter(
        tr_pos_ye[:, 0, i],
        tr_pos_ye[:, 1, i],
        tr_pos_ye[:, 2, i],
        s=10,
        c=tr_pos_ye[:, 3, i],
        cmap="seismic_r",
        vmin=0.1, vmax=0.55,
        edgecolors="none",
    )
    ax.set_aspect("equal")
    plt.colorbar(dots, label="$Y_e$")
    plt.savefig(f"{OUTPUT_PATH}/frame_{i:04d}.png", dpi=300)
    plt.close(fig)

##  main
if __name__ == "__main__":

## 1. load tracers
    files = sorted(pl.Path(PATH_IN).glob("tracer*"))
    files += sorted(pl.Path(PATH_OUT).glob("tracer*"))
    file_paths = [str(f) for f in files]

    ctx = __import__("multiprocessing").get_context("forkserver")
    with ctx.Pool(N_CPU) as pool:
        tracers = list(tqdm(
            pool.imap_unordered(Trajectory.from_ascii, file_paths),
            total=len(file_paths),
            ncols=0, unit="tracers", desc="Loading tracers", leave=False,
        ))

## 2. drop atmosphere tracers
    def is_atmosphere(tracer):
        r2 = tracer.data["x"]**2 + tracer.data["y"]**2 + tracer.data["z"]**2
        return r2.min() < 200**2

    tracers = [t for t in tqdm(
        tracers,
        desc="Removing atmosphere tracers",
        ncols=0, unit="tracers", leave=False,
    ) if not is_atmosphere(t)]

## 3. build position/Ye array
    times = np.unique(np.concatenate([tr.data["time"] for tr in tracers]))
    tr_pos_ye = np.full((len(tracers), 4, len(times)), np.nan)

    for i, tracer in enumerate(tqdm(
        tracers,
        desc="Postprocessing tracers",
        ncols=0, unit="tracers", leave=False,
    )):
        t_msk = np.isin(times, tracer.data["time"])
        tr_pos_ye[i, 0, t_msk] = tracer.data["x"]*1.4766284425812721
        tr_pos_ye[i, 1, t_msk] = tracer.data["y"]*1.4766284425812721
        tr_pos_ye[i, 2, t_msk] = tracer.data["z"]*1.4766284425812721
        tr_pos_ye[i, 3, t_msk] = tracer.data["r_0"]

    # free tracer objects before forking
    del tracers

## 4. render frames
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # initializer pushes the big arrays into each worker once at startup,
    # avoiding pertask pickling of tr_pos_ye (can be gigabytes)
    with Pool(
        N_CPU,
        initializer=_worker_init,
        initargs=(tr_pos_ye, times),
    ) as pool:
        list(tqdm(
            pool.imap_unordered(make_plot, range(len(times))),
            total=len(times),
            ncols=0, unit="frames", desc="Rendering frames",
        ))

## 5. encode video
    subprocess.run(
        f"ffmpeg -framerate {FPS} "
        f"-i {OUTPUT_PATH}/frame_%04d.png "
        f"-c:v libx264 -pix_fmt yuv420p {VIDEO_OUT}",
        shell=True, check=True,
    )
