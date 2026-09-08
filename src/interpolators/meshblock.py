import numpy as np

from .base import InterpolatorBase
from .pchip import _pchip4_eval


class MeshblockPchipInterpolator(InterpolatorBase):
    """
    Monotone-cubic (PCHIP) interpolator on meshblock-decomposed data.

    Data is stored as one ``(n_blocks, nb1, nb2, nb3)`` array per key, where
    each meshblock carries its own ghost zones (as written by Athena++ athdf
    outputs with ``ghost_zones = true``).  A query point is first assigned to
    the meshblock containing it via a dense octree lookup table built from
    the file's ``Levels``/``LogicalLocations`` metadata, then evaluated with
    a 4-point Fritsch-Carlson stencil per axis on that block's own node
    arrays.  The two file ghost cells are exactly what a 4-point stencil
    needs for interior queries anywhere in a block, so interpolation is
    seamless across block faces.

    Coordinates are the grid's native (x1, x2, x3) -- e.g. (r, theta, phi)
    for spherical data, in which case this is wrapped in
    :class:`~.coordinate_transformations.CartesianToSpherical`.
    """

    n_ghosts = 2  # ghost cells required per block per side (4-point stencil)

    def __init__(
        self,
        x1v: np.ndarray,
        x2v: np.ndarray,
        x3v: np.ndarray,
        block_map: np.ndarray,
        root_faces: tuple[np.ndarray, np.ndarray, np.ndarray],
        max_level: int,
        interior: tuple[int, int, int],
        file_ng: int,
        *,
        shm: dict[str, str],
        shape: tuple[int, int, int, int],
        ):
        """
        Initialize the meshblock interpolator.

        Parameters
        ----------
        x1v, x2v, x3v : ndarray, shape (n_blocks, nb)
            Per-block cell-centre coordinates including ghost cells, strictly
            monotone per row (polar ghost coordinates already reflected to
            negative theta / beyond pi by the file handler).
        block_map : ndarray of int, 3-D
            Dense octree lookup: axis ``a`` has ``n_root_blocks[a] *
            2**max_level`` entries; ``block_map[s1, s2, s3]`` is the id of
            the meshblock covering that finest-granularity slot.
        root_faces : tuple of 3 ndarray
            Root-grid cell face coordinates per axis (lengths ``N_a + 1``).
        max_level : int
            Deepest refinement level present (0 for unigrid).
        interior : tuple of 3 int
            Interior (non-ghost) cells per block per axis.
        file_ng : int
            Ghost cells per side stored in the file (>= ``n_ghosts``).
        shm : dict[str, str]
            Mapping from field keys to shared-memory segment names.
        shape : tuple
            ``(n_blocks, nb1, nb2, nb3)`` shape of each field array.
        """
        super().__init__(shm_names=shm, shape=shape)

        self._nodes = tuple(
            np.ascontiguousarray(x, dtype=np.float64) for x in (x1v, x2v, x3v)
        )
        self.block_map = np.ascontiguousarray(block_map)
        self.root_faces = tuple(np.asarray(f, dtype=np.float64) for f in root_faces)
        self.max_level = int(max_level)
        self.interior = tuple(int(i) for i in interior)
        self.file_ng = int(file_ng)

        if self.file_ng < self.n_ghosts:
            raise ValueError(
                f"Blocks carry {self.file_ng} ghost cells but the 4-point "
                f"stencil requires {self.n_ghosts} per side."
            )
        for a, nodes in enumerate(self._nodes):
            if nodes.shape != (shape[0], shape[1 + a]):
                raise ValueError(
                    f"Axis {a + 1} node array has shape {nodes.shape}, "
                    f"expected {(shape[0], shape[1 + a])} from shape {shape}."
                )

        # Used by CartesianToSpherical.sort_tracers to bin tracers for
        # locality; root-level faces group tracers by grid column.
        self._y_nodes = self.root_faces[1]
        self._z_nodes = self.root_faces[2]

    def _find_blocks(self, q1, q2, q3):
        """
        Vectorized point -> meshblock id lookup via the octree table.

        Returns ``(bid, valid)``; ``bid`` entries for invalid (out-of-domain
        or NaN) points are 0 and must be masked with ``valid``.
        """
        valid = np.ones(q1.shape, dtype=bool)
        slots = []
        for a, qa in enumerate((q1, q2, q3)):
            faces = self.root_faces[a]
            with np.errstate(invalid="ignore"):
                valid &= np.isfinite(qa) & (qa >= faces[0]) & (qa <= faces[-1])
            cell = np.searchsorted(faces, qa, side="right") - 1
            cell = np.clip(cell, 0, len(faces) - 2)
            slots.append(cell * 2**self.max_level // self.interior[a])
        bid = self.block_map[slots[0], slots[1], slots[2]]
        return np.where(valid, bid, 0), valid

    def _stencil(self, axis, bid, qa):
        """
        Per-point 4-node stencil on axis ``axis`` within block ``bid``.

        Returns ``(indices (n, 4), nodes (n, 4))`` with the query lying in
        ``[nodes[:, 1], nodes[:, 2]]`` for every in-domain point.
        """
        rows = self._nodes[axis][bid]  # (n, nb)
        ic = np.sum(qa[:, None] >= rows, axis=1) - 1
        # In-domain queries give ic in [ng - 1, ng + interior - 1]; the clip
        # only catches float round-off at the outermost domain faces.
        ic = np.clip(ic, self.file_ng - 1, self.file_ng + self.interior[axis] - 1)
        idx = ic[:, None] + np.arange(-1, 3)
        return idx, np.take_along_axis(rows, idx, axis=1)

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate all keys at ``coords`` (native grid coordinates, (3, ...))."""
        if not self.loaded:
            raise RuntimeError("Interpolator data not loaded; call load() before querying.")

        q1, q2, q3 = (np.asarray(c, dtype=np.float64).ravel() for c in coords)
        n_points = q1.size
        out = np.full((self.n_keys, n_points), np.nan)
        if n_points == 0:
            return out.reshape((self.n_keys, *np.shape(coords[0])))

        bid, valid = self._find_blocks(q1, q2, q3)
        sel = np.nonzero(valid)[0]
        if sel.size:
            b = bid[sel]
            i_idx, i_nodes = self._stencil(0, b, q1[sel])
            j_idx, j_nodes = self._stencil(1, b, q2[sel])
            k_idx, k_nodes = self._stencil(2, b, q3[sel])

            for i_key, key in enumerate(self.keys):
                V = self.data[key][
                    b[:, None, None, None],
                    i_idx[:, :, None, None],
                    j_idx[:, None, :, None],
                    k_idx[:, None, None, :],
                ]  # (n, 4, 4, 4)
                # Contract one axis at a time; per-point node columns
                # broadcast through _pchip4_eval's elementwise arithmetic.
                V = _pchip4_eval(
                    i_nodes.T[:, :, None, None], V.transpose(1, 0, 2, 3),
                    q1[sel][:, None, None],
                )  # (n, 4, 4)
                V = _pchip4_eval(
                    j_nodes.T[:, :, None], V.transpose(1, 0, 2),
                    q2[sel][:, None],
                )  # (n, 4)
                out[i_key, sel] = _pchip4_eval(k_nodes.T, V.T, q3[sel])

        return out.reshape((self.n_keys, *np.shape(coords[0])))
