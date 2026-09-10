"""
AthenaK Module

  This module provides the file handling functionality for AthenaK spherical
  grid output (legacy binary VTK, big-endian).
  See github.com/IAS-Astrophysics/athenak for more details.

  The output is a STRUCTURED_GRID whose POINTS are spherical (r, theta, phi)
  triples rather than Cartesian coordinates, with r varying fastest.  The
  grid is geometrically (log) spaced in r and uniform in phi; its polar
  axis may be uniform in theta or in cos(theta), node- or cell-centred, and
  is detected from the file rather than assumed.  AthenaK writes one
  variable per file, so a single time step is assembled from several files
  -- which ``FileHandler.parse_files`` supports by merging all files that
  report the same time.

  AthenaK's M1 radiation evolves 4 species, [nue, anue, nux, anux].
  GR-Athena++ evolves only 3, with nux already lumping all heavy species
  together at the evolution level.  ``AthenaKFileHandler``'s
  ``heavy_neutrinos`` constructor option controls how the extra species is
  handled:

    - "sum" (default): combine nux and anux into a single GRA-style "nux".
      Number flux is extensive and is summed; average energy is intensive
      and cannot be, so it is combined as a number-flux-weighted average
      instead (see ``_build_key_specs``).
    - "drop": nux/anux keys are simply not resolved -- omit them from
      ``--keys``/``keys`` under this mode.
    - "separate": all 4 species are kept distinct (eps_nue/anue/nux/anux,
      F_nue/anue/nux/anux).
"""

import os
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

# scalar name -> canonical key, for keys that map 1:1. Used to recover the
# canonical name of whichever key a raw scalar "belongs to" regardless of
# which output key(s) actually consume it (see _build_key_specs / the
# heavy_neutrinos combination modes below) -- e.g. FIELD_MAX_ABS is always
# looked up under "eps_nux" for scalar "e:2", whether it ends up read
# directly (heavy_neutrinos="separate") or as one half of a weighted
# average (heavy_neutrinos="sum").
_REVERSE_FIELD_MAP = {raw: key for key, raw in FIELD_MAP.items()}

# Per-species radiation moments: trace-key prefix -> raw AthenaK scalar
# prefix. Component i of "<scalar prefix>:i" is species _RAD_SPECIES[i].
_RAD_PREFIXES = {"eps": "e", "F": "|F|"}
_RAD_SPECIES = ("nue", "anue", "nux", "anux")
_HEAVY_SPECIES = ("nux", "anux")
_HEAVY_NEUTRINO_MODES = ("sum", "drop", "separate")

_VTK_DTYPES = {b'float': '>f4', b'double': '>f8', b'int': '>i4'}


def _species_of(key: str) -> tuple[str, str, int] | None:
    """
    If ``key`` is a per-species radiation key (``eps_<species>`` or
    ``F_<species>``), return ``(prefix, species, index)``; otherwise None.
    """
    for prefix in _RAD_PREFIXES:
        marker = f"{prefix}_"
        if key.startswith(marker):
            species = key[len(marker):]
            if species in _RAD_SPECIES:
                return prefix, species, _RAD_SPECIES.index(species)
    return None


def _build_key_specs(keys: list[str], heavy_neutrinos: str) -> dict[str, tuple]:
    """
    Resolve each requested trace key into a spec describing how to compute
    it from raw AthenaK VTK scalars:

      ("direct", scalar)
          Read one scalar block as-is.
      ("sum", scalar_a, scalar_b)
          Read two scalar blocks and add them (F_nux under
          heavy_neutrinos="sum").
      ("weighted_avg", scalar_val_a, scalar_val_b, scalar_weight_a, scalar_weight_b)
          Number-flux-weighted average of two scalar blocks, weighted by
          two scalar blocks of the corresponding F key (eps_nux under
          heavy_neutrinos="sum").

    A key absent from the returned dict is not resolved at all (dropped).
    """
    if heavy_neutrinos not in _HEAVY_NEUTRINO_MODES:
        raise ValueError(
            f"heavy_neutrinos must be one of {_HEAVY_NEUTRINO_MODES}, got {heavy_neutrinos!r}"
        )

    specs: dict[str, tuple] = {}
    for key in keys:
        species_info = _species_of(key)
        if species_info is None:
            specs[key] = ("direct", FIELD_MAP.get(key, key))
            continue

        prefix, species, idx = species_info
        scalar_prefix = _RAD_PREFIXES[prefix]

        if species not in _HEAVY_SPECIES or heavy_neutrinos == "separate":
            specs[key] = ("direct", f"{scalar_prefix}:{idx}")
        elif heavy_neutrinos == "drop":
            pass
        elif heavy_neutrinos == "sum":
            if species == "anux":
                continue  # folded into "nux" below; no standalone key
            i_nux, i_anux = _RAD_SPECIES.index("nux"), _RAD_SPECIES.index("anux")
            if prefix == "F":
                specs[key] = (
                    "sum", f"{scalar_prefix}:{i_nux}", f"{scalar_prefix}:{i_anux}",
                )
            else:  # "eps": average energy is intensive, weight by number flux
                f_prefix = _RAD_PREFIXES["F"]
                specs[key] = (
                    "weighted_avg",
                    f"{scalar_prefix}:{i_nux}", f"{scalar_prefix}:{i_anux}",
                    f"{f_prefix}:{i_nux}", f"{f_prefix}:{i_anux}",
                )
    return specs


def _invert_key_specs(key_specs: dict[str, tuple]) -> dict[str, list[str]]:
    """Raw AthenaK scalar name -> sorted list of trace keys that need it."""
    by_scalar: dict[str, set[str]] = {}
    for key, spec in key_specs.items():
        kind = spec[0]
        if kind == "direct":
            scalars = (spec[1],)
        elif kind == "sum":
            scalars = (spec[1], spec[2])
        elif kind == "weighted_avg":
            scalars = (spec[1], spec[2], spec[3], spec[4])
        else:
            raise ValueError(f"Unknown key spec kind {kind!r}")
        for scalar in scalars:
            by_scalar.setdefault(scalar, set()).add(key)
    return {scalar: sorted(keys) for scalar, keys in by_scalar.items()}


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


# (file, field) pairs already reported, so a field is not re-reported on
# every chunk load.  Per process, so a worker pool may repeat it once each.
_warned_nonfinite: set[tuple[str, str]] = set()


SANE_FILL = 0.0

# Largest magnitude a field can physically take, by canonical key.  Samples
# outside this (and any non-finite ones) are replaced by SANE_FILL on load.
#
# Only the neutrino mean energies need this, because they are computed as
# J/n and so carry no information wherever the number density vanishes --
# there they come out as inf, or as a finite number that can run up to
# whatever the denominator underflowed against, depending on the AthenaK
# build.
#
# The bound is not trying to separate the good samples from the bad, which
# a magnitude test cannot do -- on affected dumps a meaningful fraction of
# the no-flux cells carry a value that looks perfectly physical. It does
# not need to. Those are harmless, since the flux they are multiplied by
# downstream is vanishing there. What has to go is the extreme tail, because
# the interpolation stencil of a tracer just inside the neutrino-carrying
# region reaches across the boundary, and one such neighbour would swamp
# it. Capping the magnitude caps that bleed.
#
# 1e4 MeV leaves ~300x headroom above the ~35 MeV observed physical
# maximum while staying orders of magnitude below any observed garbage
# floor (1e10 and up), so it can't be confused with real physics on
# fixed-build dumps.
FIELD_MAX_ABS = {
    'eps_nue': 1e4,
    'eps_anue': 1e4,
    'eps_nux': 1e4,
    'eps_anux': 1e4,
}


def _sanitise(path: str, key: str, field: str, values: np.ndarray) -> np.ndarray:
    """
    Replace unusable samples with ``SANE_FILL``, reporting once per field.

    Unusable means non-finite, or beyond this key's entry in
    ``FIELD_MAX_ABS`` if it has one.  Substituting rather than masking is
    deliberate: these samples sit where the field carries no information
    and is multiplied by a vanishing flux downstream, so masking would only
    hand a tracer near the boundary a real flux with a NaN energy to go
    with it, which is worse than a value on its way to zero along with the
    flux.
    """
    bad = ~np.isfinite(values)
    limit = FIELD_MAX_ABS.get(key)
    if limit is not None:
        bad |= np.abs(values) > limit
    n_bad = int(np.count_nonzero(bad))
    if not n_bad:
        return values

    seen = (path, field)
    if seen not in _warned_nonfinite:
        _warned_nonfinite.add(seen)
        bound = "non-finite" if limit is None else f"non-finite or |value| > {limit:g}"
        print(
            f"WARNING: {os.path.basename(path)}: field '{field}' has {n_bad} of "
            f"{values.size} samples {bound} ({100 * n_bad / values.size:.1f}%), "
            f"replaced with {SANE_FILL}. Check that this field is only unusable "
            f"where its value cannot matter.",
            flush=True,
        )
    return np.where(bad, SANE_FILL, values)


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
        rad_transform: str | tuple | None = "log",
        file_pattern: str = "*.vtk",
        heavy_neutrinos: str = "sum",
        **kwargs,
    ) -> None:
        if heavy_neutrinos not in _HEAVY_NEUTRINO_MODES:
            raise ValueError(
                f"heavy_neutrinos must be one of {_HEAVY_NEUTRINO_MODES}, got {heavy_neutrinos!r}"
            )
        self.file_pattern = file_pattern
        self.n_ghosts = interpolator.n_ghosts
        self.heavy_neutrinos = heavy_neutrinos
        super().__init__(interpolator, *args, **kwargs)
        # Transform spec for the radial axis: "log", ("asinh", scale), or None.
        # AthenaK's radial grid is geometrically (log) spaced from rmin to
        # rmax, so "log" is the right default -- see docs/formats/athenak.md.
        # Older, linearly-spaced dumps need rad_transform="none" explicitly.
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

        key_specs = _build_key_specs(self.keys, self.heavy_neutrinos)
        self.extra_data['key_specs'] = key_specs
        self.extra_data['keys_by_scalar'] = _invert_key_specs(key_specs)

    @staticmethod
    def parse_file(
        file_path: str,
        keys: list[str],
        extra_data: Any = None,
    ) -> tuple[float, str, list[str], int]:
        info = scan_vtk(file_path)
        keys_by_scalar = extra_data['keys_by_scalar']
        avail_keys = sorted({
            key for scalar in info['scalars'] for key in keys_by_scalar.get(scalar, ())
        })
        if not avail_keys:
            return 0.0, "", [], 0
        return info['time'], file_path, avail_keys, extra_data['mem_size']

    def native_cell_weights(
        self,
        slot: int,
        keys: tuple[str, ...],
        surface_radius: float | None = None,
        ) -> tuple[list[tuple[tuple[np.ndarray, ...], np.ndarray]], float | None]:
        """
        See :meth:`~src.file.FileHandler.native_cell_weights`.

        AthenaK writes a single global spherical grid, so the cells are the
        dumped samples. Three conventions have to be honoured, all detected in
        :func:`polar_axis` and recorded in ``extra_data``:

        * the radial axis is geometric, so edges sit at the midpoints in
          ``ln r``, extrapolated half a step at each end;
        * the polar axis is uniform in ``theta`` *or* in ``mu = cos(theta)``,
          so edges are built in whichever one it is and only then converted;
        * a node-centred polar axis puts its first and last samples *on* the
          poles, where the cell is a half cell. Clipping the edges to the
          physical range handles that on its own.

        The returned ``cos(theta)`` edges are monotone but may ascend or
        descend, depending on the convention; callers must not assume a
        direction.
        """
        ng = self.n_ghosts
        r = np.asarray(self.extra_data['r'], dtype=float)
        nodes = np.asarray(self.extra_data['polar_nodes'], dtype=float)[ng:-ng]
        ph = np.asarray(self.extra_data['ph'], dtype=float)[ng:-ng]
        shape = self.extra_data['shape']

        d_polar = nodes[1] - nodes[0]
        polar_edges = np.concatenate((nodes - d_polar / 2, [nodes[-1] + d_polar / 2]))
        if self.extra_data['polar'] == 'mu':
            cth_edges = np.clip(polar_edges, -1.0, 1.0)
        else:
            cth_edges = np.cos(np.clip(polar_edges, 0.0, np.pi))

        d_ph = ph[1] - ph[0]
        ph_edges = np.concatenate((ph - d_ph / 2, [ph[-1] + d_ph / 2]))

        if surface_radius is None:
            ln_r = np.log(r)
            mid = (ln_r[:-1] + ln_r[1:]) / 2
            r_edges = np.exp(np.concatenate((
                [2 * ln_r[0] - mid[0]], mid, [2 * ln_r[-1] - mid[-1]])))
            i_r = slice(None)
            edges: tuple[np.ndarray, ...] = (r_edges, cth_edges, ph_edges)
            r_used = None
            n_r_sel = len(r)
        else:
            j = int(np.argmin(np.abs(r - surface_radius)))
            i_r = slice(j, j + 1)
            edges = (cth_edges, ph_edges)
            r_used = float(r[j])
            n_r_sel = 1

        values = np.empty((len(keys), n_r_sel * (len(cth_edges) - 1) * len(ph)))
        for i_k, key in enumerate(keys):
            shm = SharedMemory(name=self.shared_memory[slot][key])
            try:
                buf = np.ndarray(shape=shape, dtype=np.float64, buffer=shm.buf)
                values[i_k] = buf[i_r, ng:-ng, ng:-ng].ravel()
            finally:
                shm.close()

        # One global grid, so a single block.
        return [(edges, values)], r_used

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
        key_specs = extra_data['key_specs']

        # Locate every raw scalar this step's files carry, whichever file
        # each one lives in -- AthenaK writes one variable per file, but a
        # combined key (see _build_key_specs) may need scalars from two.
        scalar_locations: dict[str, tuple[str, int, str]] = {}
        for path in metadata_dict:
            for name, (offset, dtype) in scan_vtk(path)['scalars'].items():
                scalar_locations[name] = (path, offset, dtype)

        raw_cache: dict[str, np.ndarray] = {}

        def get_raw(scalar: str) -> np.ndarray:
            if scalar not in raw_cache:
                path, offset, dtype = scalar_locations[scalar]
                # File order is phi-slowest, r-fastest; the interpolator
                # wants (r, polar, phi) with the polar axis ascending.
                flat = read_block(path, offset, n_cells, dtype)
                ar = flat.reshape(n_ph, n_th, n_r).transpose(2, 1, 0)
                canonical_key = _REVERSE_FIELD_MAP.get(scalar, scalar)
                raw_cache[scalar] = _sanitise(path, canonical_key, scalar, ar)
            return raw_cache[scalar]

        for key in shared_memory:
            spec = key_specs.get(key)
            if spec is None:
                continue

            kind = spec[0]
            if kind == "direct":
                ar = get_raw(spec[1])
            elif kind == "sum":
                ar = get_raw(spec[1]) + get_raw(spec[2])
            elif kind == "weighted_avg":
                _, val_a, val_b, weight_a, weight_b = spec
                va, vb = get_raw(val_a), get_raw(val_b)
                wa, wb = get_raw(weight_a), get_raw(weight_b)
                # A vacuum species (near-zero number flux) can carry a
                # garbage average energy from an internal 0/0 -- already
                # sanitised by get_raw above, so it contributes zero rather
                # than poisoning the average.
                denom = wa + wb
                ar = np.divide(
                    wa * va + wb * vb, denom,
                    out=np.zeros_like(denom), where=denom > 0,
                )
            else:
                raise ValueError(f"Unknown key spec kind {kind!r}")

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
