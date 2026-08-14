"""
AthenaK Module

  This module provides the file handling functionality for AthenaK spherical
  grid output (legacy binary VTK, big-endian).
  See github.com/IAS-Astrophysics/athenak for more details.

  The output is a STRUCTURED_GRID whose POINTS are spherical (r, theta, phi)
  triples rather than Cartesian coordinates, with r varying fastest.  The
  grid is linear in r and uniform in phi; its polar axis may be uniform in
  theta or in cos(theta), node- or cell-centred, and is detected from the
  file rather than assumed.  AthenaK writes one variable per file, so a
  single time step is assembled from several files -- which
  ``FileHandler.parse_files`` supports by merging all files that report the
  same time.
"""

import signal
from typing import Any
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from .file import FileHandler
from .utils import fill_spherical_ghosts, glob_files
from .interpolators.base import InterpolatorBase
from .interpolators.coordinate_transformations import CartesianToSpherical

# AthenaK's own scalar names -> the canonical key names the rest of trace
# (and everything under analysis/) expects.  Keys that are not listed here
# pass through unchanged, so a raw AthenaK name can always be requested
# directly via --keys.
FIELD_MAP = {
    'rho': 'dens',
    'T': 'temperature',
    'r_0': 's_00',            # passive scalar 0 = Ye
    'u_t': 'u_t',
    'V_u_x': 'win_Vx',
    'V_u_y': 'win_Vy',
    'V_u_z': 'win_Vz',
    'F_nue': '|F|:0', 'F_anue': '|F|:1', 'F_nux': '|F|:2', 'F_anux': '|F|:3',
    'eps_nue': 'e:0', 'eps_anue': 'e:1', 'eps_nux': 'e:2', 'eps_anux': 'e:3',
}

_VTK_DTYPES = {b'float': '>f4', b'double': '>f8', b'int': '>i4'}


def scan_vtk(file_path: str) -> dict[str, Any]:
    """
    Read the headers of a legacy binary VTK file without reading bulk data.

    Every binary block is seeked over rather than read, so this is cheap
    enough to call once per file during parsing and again per load.

    Parameters
    ----------
    file_path : str
        Path to the ``.vtk`` file.

    Returns
    -------
    dict
        ``dims`` (n_r, n_theta, n_phi), ``n_points``, ``time``,
        ``points_offset`` (byte offset of the POINTS block) and ``scalars``,
        a mapping from scalar name to its ``(byte offset, numpy dtype)``.
    """
    info: dict[str, Any] = {'scalars': {}, 'time': None}
    n_point_data = 0

    with open(file_path, 'rb') as f:
        def next_line() -> bytes:
            """Next non-empty line (binary blocks are followed by a newline)."""
            while True:
                line = f.readline()
                if not line:
                    return b''
                line = line.strip()
                if line:
                    return line

        while True:
            words = next_line().split()
            if not words:
                break
            keyword = words[0]

            if keyword.startswith(b'#'):
                continue
            elif keyword == b'BINARY':
                pass
            elif keyword == b'ASCII':
                raise ValueError(f"{file_path} is ASCII VTK; only BINARY is supported.")
            elif keyword == b'DATASET':
                if words[1] != b'STRUCTURED_GRID':
                    raise ValueError(
                        f"{file_path} is a {words[1].decode()} dataset; "
                        "this handler only reads STRUCTURED_GRID."
                    )
            elif keyword == b'DIMENSIONS':
                info['dims'] = tuple(int(v) for v in words[1:4])
            elif keyword == b'POINTS':
                info['n_points'] = int(words[1])
                info['points_offset'] = f.tell()
                f.seek(info['n_points'] * 3 * np.dtype(_VTK_DTYPES[words[2]]).itemsize, 1)
            elif keyword == b'FIELD':
                for _ in range(int(words[2])):
                    name, n_comp, n_tup, dtype = next_line().split()
                    n_bytes = int(n_comp) * int(n_tup) * np.dtype(_VTK_DTYPES[dtype]).itemsize
                    if name == b'TIME':
                        info['time'] = float(np.frombuffer(f.read(n_bytes),
                                                           dtype=_VTK_DTYPES[dtype])[0])
                    else:
                        f.seek(n_bytes, 1)
            elif keyword == b'POINT_DATA':
                n_point_data = int(words[1])
            elif keyword == b'SCALARS':
                n_comp = int(words[3]) if len(words) > 3 else 1
                dtype = _VTK_DTYPES[words[2]]
                next_line()  # LOOKUP_TABLE
                info['scalars'][words[1].decode()] = (f.tell(), dtype)
                f.seek(n_point_data * n_comp * np.dtype(dtype).itemsize, 1)
            elif keyword == b'VECTORS':
                # Not used by the tracer pipeline, but skipped with correct
                # byte accounting so any SCALARS after it stay addressable.
                f.seek(n_point_data * 3 * np.dtype(_VTK_DTYPES[words[2]]).itemsize, 1)
            else:
                raise ValueError(
                    f"Unsupported VTK keyword {keyword.decode()!r} in {file_path}."
                )

    for required in ('dims', 'n_points', 'time'):
        if info.get(required) is None:
            raise ValueError(f"{file_path} has no {required.upper()} record; not AthenaK output?")
    return info


def read_block(file_path: str, offset: int, count: int, dtype: str) -> np.ndarray:
    """Read ``count`` values of ``dtype`` at byte ``offset``, as float64."""
    with open(file_path, 'rb') as f:
        f.seek(offset)
        raw = f.read(count * np.dtype(dtype).itemsize)
    return np.frombuffer(raw, dtype=dtype, count=count).astype(np.float64)


def read_grid(file_path: str, info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the POINTS block and return the separable ``(r, theta, phi)`` axes.

    Raises if the point coordinates are not separable, i.e. if the file is
    not the spherical product grid this handler assumes.
    """
    n_r, n_th, n_ph = info['dims']
    pts = read_block(file_path, info['points_offset'], info['n_points'] * 3, '>f4')
    pts = pts.reshape(n_ph, n_th, n_r, 3)

    r = pts[0, 0, :, 0]
    th = pts[0, :, 0, 1]
    ph = pts[:, 0, 0, 2]
    if not (np.allclose(pts[..., 0], r)
            and np.allclose(pts[..., 1], th[:, None])
            and np.allclose(pts[..., 2], ph[:, None, None])):
        raise ValueError(
            f"{file_path}'s POINTS are not a separable (r, theta, phi) product grid."
        )
    return r, th, ph


def polar_axis(th: np.ndarray) -> tuple[str, np.ndarray, bool, bool]:
    """
    Work out which coordinate AthenaK's polar axis is uniform in.

    The interpolators require the polar axis to be uniformly spaced in
    whatever coordinate they interpolate in, so that coordinate has to be
    whichever one this grid was built in.  Both conventions AthenaK writes
    are recognised, and each is described in docs/formats/athenak.md:
    uniform in ``mu = cos(theta)`` (equal solid angle) or uniform in
    ``theta``.

    Parameters
    ----------
    th : ndarray
        The file's polar coordinates, in whatever order it stores them.

    Returns
    -------
    name : {"theta", "mu"}
        The coordinate to interpolate in.
    nodes : ndarray
        Ascending node positions in that coordinate, made exactly uniform
        (the file's own float32 values are only uniform to ~1e-7, and the
        interpolators index this axis arithmetically).
    flip : bool
        Whether the file's theta rows must be reversed to match ``nodes``.
    node_centred : bool
        Whether the first and last rows sit exactly on the poles.
    """
    for name, y, lo, hi in (("theta", np.asarray(th), 0.0, np.pi),
                            ("mu", np.cos(th), -1.0, 1.0)):
        flip = bool(y[0] > y[-1])
        ys = y[::-1] if flip else y
        steps = np.diff(ys)
        if not np.allclose(steps, steps.mean(), rtol=1e-3, atol=0.0):
            continue

        dy = (ys[-1] - ys[0]) / (len(ys) - 1)
        offset_lo, offset_hi = abs(ys[0] - lo), abs(hi - ys[-1])
        node_centred = offset_lo < 0.25 * dy
        if not node_centred and abs(offset_lo - 0.5 * dy) > 0.25 * dy:
            raise ValueError(
                f"Polar axis is uniform in {name} but its first node sits "
                f"{offset_lo / dy:.3f} cells from the pole; expected 0 "
                "(node-centred) or 0.5 (cell-centred)."
            )
        if abs(offset_hi - offset_lo) > 0.25 * dy:
            raise ValueError(
                f"Polar axis is not symmetric about the equator: first node "
                f"{offset_lo / dy:.3f} cells from one pole, last node "
                f"{offset_hi / dy:.3f} from the other."
            )
        return name, ys[0] + dy * np.arange(len(ys)), flip, node_centred

    raise ValueError(
        "Polar axis is uniform in neither theta nor cos(theta); the "
        "interpolators require a uniformly spaced second axis."
    )


class AthenaKFileHandler(FileHandler):
    """
    FileHandler implementation for AthenaK spherical-grid VTK output.

    The polar grid convention is detected per dataset rather than assumed
    (see :func:`polar_axis`), since ``PchipInterpolator3D`` requires a
    uniformly spaced second axis and AthenaK may write a grid uniform in
    ``theta`` or in ``mu = cos(theta)``, node- or cell-centred.  Whichever
    it is decides the interpolation coordinate, the row ordering and the
    polar ghost fill.  Accuracy differs between them near the poles --
    see docs/formats/athenak.md.
    """

    extra_data: dict[str, Any]

    def __init__(
        self,
        interpolator: type[InterpolatorBase],
        *args,
        rad_transform: str | tuple | None = None,
        file_pattern: str = "*.vtk",
        **kwargs,
    ) -> None:
        self.file_pattern = file_pattern
        self.n_ghosts = interpolator.n_ghosts
        super().__init__(interpolator, *args, **kwargs)
        # Transform spec for the radial axis: "log", ("asinh", scale), or None.
        # AthenaK's radial grid is linear, so None is the right default.
        self.extra_data["rad_transform"] = rad_transform
        signal.signal(signal.SIGINT, self.handler)

    def list_files(self, directory: str) -> list[str]:
        files = glob_files(directory, self.file_pattern)
        self.load_grid(files[0])
        return files

    def load_grid(self, file_path: str) -> None:
        """
        Load the grid from an AthenaK VTK file and work out its conventions.
        """
        info = scan_vtk(file_path)
        n_r, n_th, n_ph = info['dims']
        r, th, ph = read_grid(file_path, info)

        try:
            polar, nodes, flip, node_centred = polar_axis(th)
        except ValueError as err:
            raise ValueError(f"{file_path}: {err}") from None

        # phi must be uniform too, but where it starts is free: node-centred
        # at phi = 0 or cell-centred at dphi/2 both work, since phi is
        # periodic and the ghost zones cover either seam.
        d_ph = 2 * np.pi / n_ph
        if not (np.all(np.diff(ph) > 0) and np.allclose(np.diff(ph), d_ph, atol=1e-4)):
            raise ValueError(
                f"{file_path}'s phi axis is not ascending and uniformly spaced "
                f"by 2*pi/{n_ph}; the interpolators require it to be."
            )
        ph = ph[0] + d_ph * np.arange(n_ph)

        ng = self.n_ghosts
        d_polar = nodes[1] - nodes[0]
        g = np.arange(1, ng + 1)
        # For a mu grid the polar ghost nodes run past |mu| = 1.  That is
        # intentional: they hold the field mirrored across the pole, which
        # is an extension rather than a continuation there (see
        # docs/formats/athenak.md).  For a theta grid they are the exact
        # mirrored positions and no such caveat applies.
        nodes = np.concatenate((nodes[0] - d_polar * g[::-1], nodes, nodes[-1] + d_polar * g))
        ph = np.concatenate((ph[0] - d_ph * g[::-1], ph, ph[-1] + d_ph * g))

        self.extra_data['r'] = r
        self.extra_data['polar'] = polar
        self.extra_data['polar_nodes'] = nodes
        self.extra_data['flip_polar'] = flip
        self.extra_data['node_centred'] = node_centred
        self.extra_data['ph'] = ph
        self.extra_data['grid_shape'] = (n_r, n_th, n_ph)
        self.extra_data['shape'] = (n_r, n_th + 2 * ng, n_ph + 2 * ng)
        self.extra_data['mem_size'] = int(np.prod(self.extra_data['shape'])) * 8

    @staticmethod
    def parse_file(
        file_path: str,
        keys: list[str],
        extra_data: Any = None,
    ) -> tuple[float, str, list[str], int]:
        info = scan_vtk(file_path)
        avail_keys = [key for key in keys if FIELD_MAP.get(key, key) in info['scalars']]
        if not avail_keys:
            return 0.0, "", [], 0
        return info['time'], file_path, avail_keys, extra_data['mem_size']

    @staticmethod
    def load_step_to_memory(
        metadata_dict: dict[str, list[str]],
        shared_memory: dict[str, str],
        extra_data: Any = None,
    ) -> None:
        ng = extra_data['interpolator'].n_ghosts
        shape = extra_data['shape']
        n_r, n_th, n_ph = extra_data['grid_shape']
        n_cells = n_r * n_th * n_ph

        for path, keys in metadata_dict.items():
            scalars = scan_vtk(path)['scalars']
            for key in keys:
                offset, dtype = scalars[FIELD_MAP.get(key, key)]
                # File order is phi-slowest, r-fastest; the interpolator
                # wants (r, polar, phi) with the polar axis ascending.
                ar = read_block(path, offset, n_cells, dtype)
                # Checked here rather than several layers down inside
                # scipy.interpolate, which reports only "`y` must contain
                # only finite values" and names neither the field nor the
                # file.  A ratio-derived field (a mean energy, say) is an
                # easy way to end up with inf wherever its denominator
                # vanishes.
                n_bad = int(np.count_nonzero(~np.isfinite(ar)))
                if n_bad:
                    raise ValueError(
                        f"{path}: field '{FIELD_MAP.get(key, key)}' has {n_bad} of "
                        f"{n_cells} values non-finite ({100 * n_bad / n_cells:.1f}%). "
                        f"Interpolation needs finite data everywhere, so either fix "
                        f"the dump or drop '{key}' from --keys."
                    )
                ar = ar.reshape(n_ph, n_th, n_r).transpose(2, 1, 0)
                if extra_data['flip_polar']:
                    ar = ar[:, ::-1, :]
                shm = SharedMemory(name=shared_memory[key])
                try:
                    buf = np.ndarray(shape=shape, dtype=np.float64, buffer=shm.buf)
                    fill_spherical_ghosts(buf, ar, ng,
                                          node_centred=extra_data['node_centred'])
                finally:
                    shm.close()

    @staticmethod
    def setup_interpolator(
        shared_memory: dict[str, str],
        extra_data: Any = None,
    ) -> InterpolatorBase:
        spec = extra_data['rad_transform']
        return CartesianToSpherical(
            extra_data['interpolator'],
            extra_data['r'],
            extra_data['polar_nodes'],
            extra_data['ph'],
            shm=shared_memory,
            coord_transforms={0: spec} if spec else {},
            shape=extra_data['shape'],
            polar=extra_data['polar'],
            **extra_data.get("interpolator_kwargs", {}),
        )

    def handler(self, signum, frame):
        """
        Signal handler for graceful shutdown on interrupt signal.
        """
        if signum == signal.SIGINT:
            print("Received interrupt signal. Exiting gracefully...")
            self.free_shared_memory()
