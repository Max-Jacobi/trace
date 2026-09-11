"""
Athena++ ``.athdf`` (HDF5) file handler for spherical-grid GR output.

The module and class are named for their limitation: only spherical-type
coordinates (``schwarzschild``/``spherical_polar``, zero shift) are
supported, because the Cartesian query mapping and the velocity transform
below assume them.  The meshblock/octree machinery itself is
coordinate-agnostic; a Cartesian or Kerr variant would branch on the file's
``Coordinates`` attribute in ``load_grid`` and swap ``_transport_velocity``.

Athena++ writes each snapshot as the raw octree meshblock structure: one
array per output dataset shaped ``(n_var, n_blocks, nx3, nx2, nx1)``, plus
``Levels``/``LogicalLocations`` describing where each block sits in the
octree and per-block coordinate arrays ``x1f/x1v`` etc.  This handler
requires the dumps to be written with ghost zones (``ghost_zones = true`` in
the ``<output>`` block), so every meshblock is self-contained for a 4-point
interpolation stencil and no inter-block ghost exchange is needed at load
time.

Shared-memory buffers keep the meshblock decomposition: one
``(n_blocks, nb1, nb2, nb3)`` float64 array per key.  Interpolation is done
by :class:`~.interpolators.meshblock.MeshblockPchipInterpolator`, which
locates the block containing each query point through a dense lookup table
built from the octree metadata.

The data is assumed to be in spherical-type coordinates (x1, x2, x3) =
(r, theta, phi).  Velocities ``vel1/vel2/vel3`` are the GR primitive
``utilde^i = W v^i`` (normal-frame projected 4-velocity, coordinate basis)
and are converted to the Cartesian transport velocity
``V^i = dx^i/dt = alpha * utilde^i / W`` at load time (Schwarzschild lapse,
zero shift; ``bh_mass=0`` reduces to flat spherical coordinates).
"""

import signal
from typing import Any
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import h5py

from .file import FileHandler
from .utils import glob_files, tensor_cell_bounds
from .athenak import _sanitise
from .interpolators.base import InterpolatorBase
from .interpolators.coordinate_transformations import CartesianToSpherical

# Canonical trace key -> raw Athena++ variable name.  Keys not listed here
# pass through unchanged, so any raw variable can be requested directly via
# --keys.  Which output series (out1, out2, ...) carries a variable is never
# assumed: every file self-describes through its VariableNames attrs.
FIELD_MAP = {
    'rho': 'rho',
    'press': 'press',
    'r_0': 'rYE',           # electron fraction passive scalar
    's': 'rENT',            # entropy passive scalar
    'T': 'Temperature',
    'u_t': 'u_t',
    'h': 'h',
}

_VEL_KEYS = ('V_u_x', 'V_u_y', 'V_u_z')
_RAW_VEL = ('vel1', 'vel2', 'vel3')

_KNOWN_COORDS = ('schwarzschild', 'spherical_polar')


def _build_key_specs(keys: list[str]) -> dict[str, tuple]:
    """
    Map each requested canonical key to a load spec.

    ``('direct', raw_name)`` reads one raw variable; ``('vel', i)`` is
    component ``i`` of the Cartesian transport velocity synthesised from
    ``vel1/vel2/vel3``.
    """
    specs = {}
    for key in keys:
        if key in _VEL_KEYS:
            specs[key] = ('vel', _VEL_KEYS.index(key))
        else:
            specs[key] = ('direct', FIELD_MAP.get(key, key))
    return specs


def _spec_requirements(spec: tuple) -> tuple[str, ...]:
    """Raw variable names a spec needs, all of which must be in one file."""
    return _RAW_VEL if spec[0] == 'vel' else (spec[1],)


def _variable_table(f: h5py.File) -> dict[str, tuple[str, int]]:
    """Raw variable name -> (dataset name, index within dataset) for a file."""
    names = [n.decode() for n in f.attrs['VariableNames']]
    dsets = [d.decode() for d in f.attrs['DatasetNames']]
    n_vars = np.atleast_1d(f.attrs['NumVariables']).astype(int)
    table = {}
    pos = 0
    for dset, nv in zip(dsets, n_vars):
        for i in range(nv):
            table[names[pos + i]] = (dset, i)
        pos += nv
    return table


def _root_faces(spec: np.ndarray, n: int) -> np.ndarray:
    """
    Reconstruct the ``n + 1`` root-grid cell faces of one axis.

    ``spec`` is the file's ``RootGridX?`` attribute ``(min, max, ratio)``
    where ``ratio`` is the cell-to-cell size ratio of Athena++'s geometric
    grid generator (``1`` for uniform spacing).
    """
    lo, hi, ratio = (float(v) for v in spec)
    if np.isclose(ratio, 1.0):
        return np.linspace(lo, hi, n + 1)
    dx0 = (hi - lo) * (ratio - 1.0) / (ratio**n - 1.0)
    return lo + dx0 * (ratio**np.arange(n + 1) - 1.0) / (ratio - 1.0)


def _fix_polar_ghost_nodes(
    x2v: np.ndarray, x2f: np.ndarray, ng: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make polar-boundary ghost node coordinates monotone.

    At the poles Athena++ stores ghost cell coordinates mirrored back into
    the domain (the ghost *data* is the correct across-pole field), so the
    per-block ``x2v`` rows of pole-touching blocks are non-monotonic.
    Reflect those ghost nodes across the boundary face instead -- to
    negative theta below 0 and past pi above -- which is the extended chart
    the across-pole ghost data lives in.

    Returns the fixed nodes plus the per-block masks of blocks whose
    leading / trailing ghosts were mirrored (needed to undo the polar
    boundary's vel3 sign flip, see ``load_step_to_memory``).
    """
    x2v = x2v.copy()
    pole_lo = np.diff(x2v[:, :ng + 1], axis=1).min(axis=1) <= 0
    x2v[pole_lo, :ng] = (2 * x2f[pole_lo, ng, None]
                         - x2v[pole_lo, 2 * ng - 1:ng - 1:-1])
    pole_hi = np.diff(x2v[:, -ng - 1:], axis=1).min(axis=1) <= 0
    x2v[pole_hi, -ng:] = (2 * x2f[pole_hi, -ng - 1, None]
                          - x2v[pole_hi, -ng - 1:-2 * ng - 1:-1])
    return x2v, pole_lo, pole_hi


def _transport_velocity(
    u1: np.ndarray, u2: np.ndarray, u3: np.ndarray,
    r: np.ndarray, theta: np.ndarray, phi: np.ndarray,
    bh_mass: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cartesian transport velocity from GR primitives on a spherical grid.

    ``u1/u2/u3`` are ``utilde^i = W v^i`` in the coordinate basis
    (Schwarzschild coordinates, zero shift).  Returns
    ``V^i = dx^i_cart/dt = (alpha/W) * utilde^j * dx^i_cart/dx^j_sph``.
    Written chart-covariantly with signed ``sin(theta)``, so it is also
    correct in polar ghost cells whose reflected coordinates have
    ``theta < 0`` or ``theta > pi``.
    """
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)

    alpha2 = 1.0 - 2.0 * bh_mass / r
    W = np.sqrt(1.0 + u1 * u1 / alpha2 + (r * u2)**2 + (r * st * u3)**2)
    fac = np.sqrt(alpha2) / W

    vr = fac * u1                 # dr/dt
    vth = fac * r * u2            # r * dtheta/dt
    vph = fac * r * st * u3       # r*sin(theta) * dphi/dt

    return (
        st * cp * vr + ct * cp * vth - sp * vph,
        st * sp * vr + ct * sp * vth + cp * vph,
        ct * vr - st * vth,
    )


def _write_to_shm(shm_name: str, shape: tuple, values: np.ndarray) -> None:
    shm = SharedMemory(name=shm_name)
    try:
        np.ndarray(shape, dtype=np.float64, buffer=shm.buf)[:] = values
    finally:
        shm.close()


class SphericalAthdfFileHandler(FileHandler):
    """
    FileHandler implementation for Athena++ athdf meshblock output.

    Requires dumps written with ghost zones.  Handles unigrid and static
    AMR octrees; the block layout is read once from the first file and every
    loaded step is checked against it (regridding between outputs raises).
    """

    extra_data: dict[str, Any]

    def __init__(
        self,
        interpolator: type[InterpolatorBase],
        *args,
        rad_transform: str | tuple | None = None,
        file_pattern: str = "*.athdf",
        bh_mass: float = 1.0,
        **kwargs,
    ) -> None:
        if rad_transform is not None:
            raise ValueError(
                "The athdf handler interpolates on each block's true node "
                "coordinates; --rad-transform is meaningless here, pass 'none'."
            )
        self.file_pattern = file_pattern
        self.n_ghosts = interpolator.n_ghosts
        self.bh_mass = float(bh_mass)
        super().__init__(interpolator, *args, **kwargs)
        signal.signal(signal.SIGINT, self.handler)

    def list_files(self, directory: str) -> list[str]:
        files = glob_files(directory, self.file_pattern)
        self.load_grid(files[0])
        return files

    def load_grid(self, file_path: str) -> None:
        """
        Read the meshblock geometry and build the octree lookup tables.
        """
        with h5py.File(file_path, 'r') as f:
            coords = f.attrs['Coordinates']
            coords = coords.decode() if isinstance(coords, bytes) else str(coords)
            if coords not in _KNOWN_COORDS:
                raise ValueError(
                    f"{file_path}: coordinates {coords!r} not supported; the "
                    f"velocity transform assumes one of {_KNOWN_COORDS} "
                    "(zero shift)."
                )
            root_size = f.attrs['RootGridSize'].astype(int)
            root_spec = [f.attrs[f'RootGridX{a}'] for a in (1, 2, 3)]
            mb_size = f.attrs['MeshBlockSize'].astype(int)
            max_level = int(f.attrs['MaxLevel'])
            n_blocks = int(f.attrs['NumMeshBlocks'])
            levels = f['Levels'][:].astype(int)
            locations = f['LogicalLocations'][:].astype(int)
            x1f = f['x1f'][:].astype(np.float64)
            x2f = f['x2f'][:].astype(np.float64)
            x3f = f['x3f'][:].astype(np.float64)
            nodes = [f[f'x{a}v'][:].astype(np.float64) for a in (1, 2, 3)]

        # Ghost count: the innermost block's face closest to the domain
        # minimum is its first interior face; everything before it is ghosts.
        inner = int(np.argmin(x1f[:, 0]))
        file_ng = int(np.argmin(np.abs(x1f[inner] - float(root_spec[0][0]))))
        if file_ng < self.n_ghosts:
            raise ValueError(
                f"{file_path}: meshblocks carry {file_ng} ghost cells but the "
                f"interpolator needs {self.n_ghosts}; write the athdf output "
                "with ghost_zones = true."
            )

        interior = mb_size - 2 * file_ng
        if np.any(interior <= 0) or np.any(root_size % interior):
            raise ValueError(
                f"{file_path}: derived block interior {tuple(interior)} does "
                f"not tile the root grid {tuple(root_size)} "
                f"(ghost count {file_ng})."
            )
        if np.any(interior % 2**max_level):
            raise ValueError(
                f"{file_path}: block interior {tuple(interior)} is not "
                f"divisible by 2^MaxLevel = {2**max_level}; the octree "
                "lookup table cannot be built at finest-block granularity."
            )

        nodes[1], pole_lo, pole_hi = _fix_polar_ghost_nodes(nodes[1], x2f, file_ng)
        for a, x in enumerate(nodes):
            if np.any(np.diff(x, axis=1) <= 0):
                raise ValueError(
                    f"{file_path}: x{a + 1}v node rows are not strictly "
                    "monotone after the polar ghost fix."
                )

        # Dense octree lookup at finest-block granularity: any point's
        # block id in O(1) from integer coordinates (see the interpolator).
        n_fine = (root_size // interior) * 2**max_level
        block_map = np.full(tuple(n_fine), -1, dtype=np.int32)
        for b in range(n_blocks):
            span = 2**(max_level - levels[b])
            s1, s2, s3 = locations[b] * span
            block_map[s1:s1 + span, s2:s2 + span, s3:s3 + span] = b
        if np.any(block_map < 0):
            raise ValueError(f"{file_path}: meshblocks do not tile the domain.")

        shape = (n_blocks, *(int(n) for n in mb_size))
        self.extra_data.update(
            x1v=nodes[0], x2v=nodes[1], x3v=nodes[2],
            # Cell faces per block, kept for native_cell_weights: cell volumes
            # need real edges and cannot be recovered from the centres, since
            # the radial spacing inside a block need not be uniform.
            x1f=x1f, x2f=x2f, x3f=x3f,
            block_map=block_map,
            root_faces=tuple(
                _root_faces(spec, n) for spec, n in zip(root_spec, root_size)
            ),
            max_level=max_level,
            interior=tuple(int(i) for i in interior),
            file_ng=file_ng,
            pole_lo=pole_lo,
            pole_hi=pole_hi,
            levels=levels,
            bh_mass=self.bh_mass,
            shape=shape,
            mem_size=int(np.prod(shape)) * 8,
            key_specs=_build_key_specs(self.keys),
        )

    def native_cell_weights(
        self,
        slot: int,
        surface_radius: float | None = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
        """
        See :meth:`~src.file.FileHandler.native_cell_weights`.

        Every meshblock is a small separable grid of its own, so the cells are
        each block's cells, concatenated in block order to match the value
        array's own layout. The blocks are leaves and tile the domain --
        ``load_grid`` refuses a file where they do not -- so the concatenation
        is a partition and nothing is double counted.

        Ghosts are stripped with ``file_ng``, the count the file actually
        carries, not ``self.n_ghosts``, which is only the interpolator's
        minimum.

        Bounds come from ``x1f`` rather than from midpoints of ``x1v`` because
        the radial spacing inside a block is generally geometric. ``x2f`` needs
        no polar correction: ``_fix_polar_ghost_nodes`` only rewrites ghost
        entries of ``x2v``, and interior faces are always within ``[0, pi]``.
        """
        ng = self.extra_data['file_ng']
        i1, i2, i3 = self.extra_data['interior']
        shape = self.extra_data['shape']
        # Slice by explicit extent rather than ng:-ng. file_ng is only
        # guaranteed >= the interpolator's requirement, and a future
        # interpolator needing none would turn ng:-ng into an empty 0:0.
        r_f = np.asarray(self.extra_data['x1f'])[:, ng:ng + i1 + 1]
        cth_f = np.cos(np.asarray(self.extra_data['x2f'])[:, ng:ng + i2 + 1])
        ph_f = np.asarray(self.extra_data['x3f'])[:, ng:ng + i3 + 1]

        if surface_radius is None:
            blocks = np.arange(shape[0])
            i_r = None
        else:
            # Half-open on purpose. A block whose lower interior face equals
            # r_surf owns it; its inner neighbour, whose upper face equals it,
            # does not. So exactly one radial layer is taken even when the
            # sphere lands on a block boundary.
            blocks = np.flatnonzero((r_f[:, 0] <= surface_radius)
                                    & (surface_radius < r_f[:, -1]))
            if blocks.size == 0:
                raise ValueError(
                    f"No meshblock spans r = {surface_radius:g}; the requested "
                    "surface lies outside the domain."
                )
            # Same cell as searchsorted(..., 'right') - 1, row by row. The
            # bounds test above already guarantees it lands in [0, i1).
            i_r = (r_f[blocks] <= surface_radius).sum(axis=1) - 1

        faces = ((r_f, cth_f, ph_f) if i_r is None else (cth_f, ph_f))
        bounds = [tensor_cell_bounds(*(f[b] for f in faces)) for b in blocks]
        lo = np.concatenate([b[0] for b in bounds], axis=1)
        hi = np.concatenate([b[1] for b in bounds], axis=1)

        keys = (self.mass_density.density_keys if surface_radius is None
                else self.mass_density.flux_keys)
        shms = []
        try:
            rows = []
            for key in keys:
                shm = SharedMemory(name=self.shared_memory[slot][key])
                shms.append(shm)
                buf = np.ndarray(shape=shape, dtype=np.float64, buffer=shm.buf)
                interior = buf[:, ng:ng + i1, ng:ng + i2, ng:ng + i3]
                rows.append((interior[blocks] if i_r is None
                             else interior[blocks, i_r]).reshape(-1))
            values = np.stack(rows)
        finally:
            for shm in shms:
                shm.close()

        weights = self.cell_weights_from_values(lo, hi, values, surface_radius)

        # r_used is None: nothing was snapped. Unlike a format storing discrete
        # shells, a cell here has radial *extent* containing the request, and
        # its value is the field across that extent, so the sphere the caller
        # asked for is the one to attribute the flux to.
        #
        # Using per-block cell-centre radii instead would be wrong, not merely
        # different: under AMR the blocks meeting r_surf sit at different
        # levels with different radial cells, so there is no single radius, and
        # the "sphere" becomes a ragged staircase whose areas do not sum to
        # 4*pi*r**2 -- biasing the total flux by whatever the level
        # distribution happens to be. The residual, a cell's sample radius
        # differing from the surface by up to half a radial cell, is the same
        # O(dr/r) the nearest-shell formats already accept, and is unsigned
        # across blocks so it does not accumulate.
        return lo, hi, weights, None

    @staticmethod
    def parse_file(
        file_path: str,
        keys: list[str],
        extra_data: Any = None,
    ) -> tuple[float, str, list[str], int]:
        with h5py.File(file_path, 'r') as f:
            raw_names = set(_variable_table(f))
            time = float(f.attrs['Time'])
        avail_keys = sorted(
            key for key, spec in extra_data['key_specs'].items()
            if raw_names.issuperset(_spec_requirements(spec))
        )
        if not avail_keys:
            return 0.0, "", [], 0
        return time, file_path, avail_keys, extra_data['mem_size']

    @staticmethod
    def load_step_to_memory(
        metadata_dict: dict[str, Any],
        shared_memory: dict[str, str],
        extra_data: Any = None,
    ) -> None:
        key_specs = extra_data['key_specs']
        shape = tuple(extra_data['shape'])
        done: set[str] = set()

        # When a variable appears in several same-time files, the file with
        # the lexically LATEST path wins: a redone segment (output-0001/...)
        # supersedes the original at duplicated snapshot times. Within one
        # segment the order is irrelevant -- output series carry disjoint
        # requested keys, and a raw variable duplicated across series holds
        # identical data from the same dump.
        for path in sorted(metadata_dict, reverse=True):
            with h5py.File(path, 'r') as f:
                if (int(f.attrs['NumMeshBlocks']) != shape[0]
                        or not np.array_equal(f['Levels'][:], extra_data['levels'])):
                    # ponytail: static mesh assumed; move per-time grids into
                    # parse_file metadata if regridding dumps ever appear.
                    raise ValueError(
                        f"{path}: meshblock layout differs from the grid file; "
                        "re-gridding between outputs is not supported."
                    )
                table = _variable_table(f)

                def read_raw(name: str) -> np.ndarray:
                    dset, ivar = table[name]
                    # file order (block, x3, x2, x1) -> (block, x1, x2, x3)
                    ar = f[dset][ivar].transpose(0, 3, 2, 1).astype(np.float64)
                    return _sanitise(path, name, name, ar)

                for key in shared_memory:
                    spec = key_specs.get(key)
                    if key in done or spec is None or spec[0] != 'direct':
                        continue
                    if spec[1] in table:
                        _write_to_shm(shared_memory[key], shape, read_raw(spec[1]))
                        done.add(key)

                vel_requested = [
                    key for key in shared_memory
                    if key not in done and key_specs.get(key, (None,))[0] == 'vel'
                ]
                if vel_requested and all(name in table for name in _RAW_VEL):
                    u1, u2, u3 = (read_raw(name) for name in _RAW_VEL)
                    # The polar boundary fills mirrored theta ghosts with
                    # vel2 AND vel3 sign-flipped (verified on real dumps).
                    # In the extended chart (theta < 0 / > pi) the correct
                    # continuation flips only vel2, so undo the vel3 flip.
                    ng = extra_data['file_ng']
                    u3[extra_data['pole_lo'], :, :ng, :] *= -1.0
                    u3[extra_data['pole_hi'], :, -ng:, :] *= -1.0
                    v_cart = _transport_velocity(
                        u1, u2, u3,
                        extra_data['x1v'][:, :, None, None],
                        extra_data['x2v'][:, None, :, None],
                        extra_data['x3v'][:, None, None, :],
                        extra_data['bh_mass'],
                    )
                    for key in vel_requested:
                        _write_to_shm(shared_memory[key], shape,
                                      v_cart[key_specs[key][1]])
                        done.add(key)

    @staticmethod
    def setup_interpolator(
        shared_memory: dict[str, str],
        extra_data: Any = None,
    ) -> InterpolatorBase:
        return CartesianToSpherical(
            extra_data['interpolator'],
            extra_data['x1v'],
            extra_data['x2v'],
            extra_data['x3v'],
            extra_data['block_map'],
            extra_data['root_faces'],
            extra_data['max_level'],
            extra_data['interior'],
            extra_data['file_ng'],
            shm=shared_memory,
            shape=extra_data['shape'],
            polar='theta',
            **extra_data.get("interpolator_kwargs", {}),
        )

    def handler(self, signum, frame):
        """
        Signal handler for graceful shutdown on interrupt signal.
        """
        if signum == signal.SIGINT:
            print("Received interrupt signal. Exiting gracefully...")
            self.free_shared_memory()
