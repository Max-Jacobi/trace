#!/usr/bin/env python3
"""
Render a 3-D animation of tracer positions over time, colour-coded by a
chosen field (default: electron fraction Ye), and encode it to an mp4 with
ffmpeg.

Example
-------
    python analysis/animate.py \\
        --tracer-dirs data/tracers_out data/tracers_out_surf \\
        --output tracer_animation.mp4
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis._common import add_tracer_loading_args, load_trajectories, get_units

FIELD_LABELS = {
    'r_0': r"$Y_e$",
    'T': "T (GK)",
    's': r"s ($k_{\rm B}$)",
    'rho': r"$\rho$ (g/cm$^3$)",
}

# Worker-process globals for frame rendering, set once by _init_worker so
# the (potentially large) position/colour array isn't repickled per frame.
_pos_color = None
_times = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 3-D animation of tracer positions colour-coded by a chosen field.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_tracer_loading_args(parser)
    parser.add_argument('--drop-below-r', type=float, default=200.0,
                         help="Drop tracers that ever come within this radius (code units) of the origin "
                              "-- filters out atmosphere/near-remnant contamination. Set to 0 to disable.")
    parser.add_argument('--color-by', default='r_0', help="Data key to colour points by (default: r_0 = Ye).")
    parser.add_argument('--color-min', type=float, default=0.1, help="Colour scale lower bound.")
    parser.add_argument('--color-max', type=float, default=0.55, help="Colour scale upper bound.")
    parser.add_argument('--cmap', default='seismic_r', help="Matplotlib colormap.")
    parser.add_argument('--box-half-width-km', type=float, default=1500.0,
                         help="Half-width of the (cubic) plot box, in km.")
    parser.add_argument('--elev', type=float, default=20.0, help="Camera elevation angle (degrees).")
    parser.add_argument('--azim-start', type=float, default=0.0, help="Camera azimuth at the first frame (degrees).")
    parser.add_argument('--azim-end', type=float, default=180.0, help="Camera azimuth at the last frame (degrees).")
    parser.add_argument('--marker-size', type=float, default=10.0, help="Scatter marker size.")
    parser.add_argument('--fps', type=int, default=30, help="Output video frame rate.")
    parser.add_argument('--dpi', type=int, default=300, help="Per-frame render DPI.")
    parser.add_argument('--frame-workers', type=int, default=os.cpu_count() or 1,
                         help="Parallel workers for rendering frames.")
    parser.add_argument('--frames-dir', default=None,
                         help="Directory to render frame PNGs into (default: a temporary directory, "
                              "removed afterward unless --keep-frames is given).")
    parser.add_argument('--keep-frames', action='store_true',
                         help="Don't delete the rendered frame PNGs after encoding.")
    parser.add_argument('--output', default='tracer_animation.mp4', help="Output video path.")
    return parser.parse_args()


def _init_worker(pos_color: np.ndarray, times: np.ndarray) -> None:
    global _pos_color, _times
    _pos_color = pos_color
    _times = times


def _make_frame(args: tuple) -> None:
    i, frames_dir, half_width, elev, azim_start, azim_end, cmap, vmin, vmax, marker_size, dpi, color_label = args
    t = _times[i]
    t0, t1 = _times[0], _times[-1]
    azim = azim_start + (azim_end - azim_start) * (t - t0) / (t1 - t0) if t1 != t0 else azim_start

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)
    ax.set_zlim(-half_width, half_width)
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    ax.set_title(f"t = {t:.0f} ms")

    dots = ax.scatter(
        _pos_color[:, 0, i], _pos_color[:, 1, i], _pos_color[:, 2, i],
        s=marker_size, c=_pos_color[:, 3, i], cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="none",
    )
    ax.set_aspect("equal")
    plt.colorbar(dots, label=color_label)
    fig.savefig(f"{frames_dir}/frame_{i:04d}.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; required to encode the animation.")

    units = get_units()
    trajs = load_trajectories(args.tracer_dirs, args.glob_pattern, args.n_cpu, tuple(args.status))

    if args.drop_below_r > 0:
        def is_atmosphere(tr):
            r2 = tr.data["x"]**2 + tr.data["y"]**2 + tr.data["z"]**2
            return r2.min() < args.drop_below_r**2
        trajs = [t for t in trajs if not is_atmosphere(t)]
    print(f"Animating {len(trajs)} tracers.")

    times = np.unique(np.concatenate([tr.data["time"] for tr in trajs])) * units.time_ms
    pos_color = np.full((len(trajs), 4, len(times)), np.nan)
    for i, tr in enumerate(trajs):
        t_msk = np.isin(times, tr.data["time"] * units.time_ms)
        pos_color[i, 0, t_msk] = tr.data["x"] * units.length_km
        pos_color[i, 1, t_msk] = tr.data["y"] * units.length_km
        pos_color[i, 2, t_msk] = tr.data["z"] * units.length_km
        pos_color[i, 3, t_msk] = tr.data[args.color_by]
    del trajs  # free before forking workers

    color_label = FIELD_LABELS.get(args.color_by, args.color_by)

    frames_dir = args.frames_dir or tempfile.mkdtemp(prefix="trace_animate_frames_")
    os.makedirs(frames_dir, exist_ok=True)
    try:
        frame_args = [
            (i, frames_dir, args.box_half_width_km, args.elev, args.azim_start, args.azim_end,
             args.cmap, args.color_min, args.color_max, args.marker_size, args.dpi, color_label)
            for i in range(len(times))
        ]
        with Pool(args.frame_workers, initializer=_init_worker, initargs=(pos_color, times)) as pool:
            list(pool.imap_unordered(_make_frame, frame_args))

        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps),
             "-i", f"{frames_dir}/frame_%04d.png",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", args.output],
            check=True,
        )
    finally:
        if args.keep_frames or args.frames_dir is not None:
            print(f"Frames kept in {frames_dir}")
        else:
            shutil.rmtree(frames_dir, ignore_errors=True)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
