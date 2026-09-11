from contextlib import contextmanager
from multiprocessing import Pool, get_context
from itertools import count, repeat
from pathlib import Path
from sys import stdout
from typing import Optional, Callable, Iterable
from tqdm import tqdm
import atexit

import numpy as np


def fill_spherical_ghosts(
    buf,
    ar,
    ng: int,
    node_centred: bool = False,
) -> None:
    """
    Copy ``ar`` into ``buf``'s interior and fill ``buf``'s ghost zones.

    Ghost zones continue the field across both poles (mirroring a row from
    the other side of the pole and rotating it by pi in phi) and
    periodically in phi.  The last two axes of both arrays are the polar
    and azimuthal ones; any leading axes (e.g. radius) are carried along
    untouched, so this works on a single shell or a whole 3-D block.

    Parameters
    ----------
    buf : ndarray, shape (..., n_polar + 2*ng, n_phi + 2*ng)
        Destination buffer.
    ar : ndarray, shape (..., n_polar, n_phi)
        Field data on the bare grid.
    ng : int
        Number of ghost zones per side.
    node_centred : bool, optional
        Whether the first and last polar rows of ``ar`` sit exactly *on*
        the poles (as AthenaK's spherical output does).  If ``False`` (the
        default, and what GR-Athena++ writes) they are half a cell away
        from them, and the mirror source is offset by one row.

    Notes
    -----
    For a cell-centred grid the polar mirror is the exact analytic
    continuation of a field smooth on the sphere.  For a node-centred one
    it is only an extension, since the row on the pole is its own mirror
    image -- see docs/formats/athenak.md.
    """
    half = ar.shape[-1] // 2
    buf[..., ng:-ng, ng:-ng] = ar
    for k in range(1, ng + 1):
        # Ghost row k cells outside the pole mirrors the row k (node-centred)
        # or k - 1 (cell-centred) cells inside it, rotated by pi in phi.
        src = k if node_centred else k - 1
        buf[..., ng - k, ng:-ng] = np.roll(ar[..., src, :], half, axis=-1)
        buf[..., -ng + k - 1, ng:-ng] = np.roll(ar[..., -1 - src, :], half, axis=-1)
    # phi is periodic; done after the polar rows so the corners come out
    # consistent with them.
    buf[..., :ng] = buf[..., -2 * ng:-ng]
    buf[..., -ng:] = buf[..., ng:2 * ng]


def glob_files(directory: str, pattern: str) -> list[str]:
    """
    List files in ``directory`` matching ``pattern``, sorted by name.

    ``pattern`` may be a glob (``*.hdf5``) or a plain suffix (``.hdf5``).

    Raises
    ------
    FileNotFoundError
        If nothing matches, so a mistyped pattern or data directory fails
        immediately instead of looking like an empty simulation.
    """
    path = Path(directory)
    if any(ch in pattern for ch in "*?[]"):
        files = sorted(str(f) for f in path.glob(pattern) if f.is_file())
    else:
        files = sorted(str(f) for f in path.iterdir()
                       if f.is_file() and f.name.endswith(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found in directory: {directory}"
        )
    return files

# Workers that only attach to shared memory owned by the main process rely
# on inheriting the calling module's already-computed state via
# copy-on-fork; "fork" is forced explicitly so this holds regardless of the
# platform/Python default start method (e.g. "forkserver", whose workers
# are spawned from a server process snapshotted independently of the
# calling module's import-time state).
_MP_CONTEXT = get_context("fork")


def close_pool_gracefully(pool: Pool) -> None:
    """
    Shut a ``Pool`` down gracefully (``close()`` + ``join()``), falling back
    to an immediate ``terminate()`` if that raises -- including
    ``KeyboardInterrupt`` from Ctrl-C, so an impatient user can force a fast
    shutdown rather than waiting for in-flight tasks to finish, and a stuck
    pool can't hang the calling process forever.

    ``terminate()`` (used unconditionally by plain ``Pool.__exit__``/typical
    ad-hoc cleanup) kills workers immediately (SIGTERM) before they reach
    normal interpreter shutdown, which is what runs the ``__del__``-triggered
    ``unload()`` on each worker's cached (attached) ``SharedMemory`` handles
    -- skipping it leaves them registered with the resource tracker, which
    complains about "leaked" shared memory at exit even though the memory
    itself was already cleaned up by its owner.
    """
    try:
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()


@contextmanager
def worker_pool(n_cpu: Optional[int] = None, initializer=None, initargs=()):
    """
    Create a ``Pool`` (forced ``fork`` start method) as a context manager
    that shuts down via ``close_pool_gracefully`` on exit instead of the
    default ``Pool.__exit__``'s ``terminate()``.
    """
    pool = _MP_CONTEXT.Pool(n_cpu, initializer=initializer, initargs=initargs)
    try:
        yield pool
    finally:
        close_pool_gracefully(pool)

def _pack_args(args_list, func):
    """Normalize mixed positional and keyword call specifications."""
    packed_args = []
    for item in args_list:
        if isinstance(item, dict):
            packed_args.append((func, (), item))
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], dict):
            packed_args.append((func, item[0], item[1]))
        else:
            packed_args.append((func, item, {}))
    return packed_args

def _unpack_args(packed):
    """Execute one normalized ``(func, args, kwargs)`` call tuple."""
    func, args, kwargs = packed
    return func(*args, **kwargs)

def do_parallel(
    func: Callable,
    args: Iterable,
    n_cpu: int,
    initializer: Optional[Callable] = None,
    initargs: Optional[Iterable] = None,
    verbose: bool = False,
    chunksize: int = 1,
    **kwargs
):
    """
    Execute func over args in parallel using n_cpu processes.
    Parameters
    ----------
    func : callable
        A function to be executed in parallel.
    args : iterable
        An iterable of arguments to pass to func.
    n_cpu : int
        Number of CPUs to use.
    initializer : callable, optional
        A function to initialize each worker process. It will be called with the arguments in initargs
    initargs : iterable, optional
        An iterable of arguments to pass to the initializer function for each worker process.
    verbose : bool
        Whether to show a progress bar.
    **kwargs
        Additional arguments passed to tqdm.
    Returns
    -------
    list
        Results from all function calls.
    """
    try:
        kwargs.setdefault("total", len(args))
    except TypeError:
        ...
    kwargs.setdefault("disable", not verbose)
    kwargs.setdefault("ncols", 0)
    kwargs.setdefault("file", stdout)

    if initargs is None:
        initargs = ()

    if n_cpu == 1:
        if initializer is not None:
            initializer(*initargs)
        return list(tqdm(map(func, args), **kwargs))

    with Pool(n_cpu, initializer=initializer, initargs=initargs) as pool:
        return list(tqdm(pool.imap_unordered(func, args, chunksize=chunksize), **kwargs))


def do_parallel_star(
    func: Callable,
    args_list: Iterable[tuple],
    n_cpu: int,
    verbose: bool = False,
    **kwargs
):
    """
    Wrapper around do_parallel that supports multi-argument functions.

    Parameters
    ----------
    func : callable
        A function that takes multiple arguments.
    args_list : list of tuples
        Each tuple contains (args, kwargs) or just args for each call.
        - If element is a tuple/list of (args, kwargs): func(*args, **kwargs)
        - If element is a tuple/list of args: func(*args)
        - If element is a dict: func(**element)
    n_cpu : int
        Number of CPUs to use.
    verbose : bool
        Whether to show progress bar.
    **kwargs
        Additional arguments passed to do_parallel.

    Returns
    -------
    list
        Results from all function calls.
    """
    packed_args = _pack_args(args_list, func)
    return do_parallel(_unpack_args, packed_args, n_cpu, verbose=verbose, **kwargs)


def do_parallel_star_pool(
    pool: Optional[Pool],
    func: Callable,
    args_list: Iterable[tuple],
    verbose: bool = False,
    chunksize: int = 1,
    **kwargs
):
    """
    Like do_parallel_star but dispatches onto a pre-created persistent Pool.

    If pool is None (i.e. n_cpu == 1), the tasks are executed serially in the
    calling process instead, which keeps the single-process debug path working.

    Parameters
    ----------
    pool : Pool or None
        A multiprocessing.Pool created externally. Pass None to run serially.
    func : callable
        A function that takes multiple arguments.
    args_list : list of tuples
        Same semantics as do_parallel_star.
    verbose : bool
        Whether to show a tqdm progress bar.
    chunksize : int
        imap_unordered chunksize (ignored for serial path).
    **kwargs
        Additional arguments forwarded to tqdm.
    """
    packed_args = _pack_args(args_list, func)
    try:
        kwargs.setdefault("total", len(packed_args))
    except TypeError:
        pass
    kwargs.setdefault("disable", not verbose)
    kwargs.setdefault("ncols", 0)
    kwargs.setdefault("file", stdout)

    if pool is None:
        return list(tqdm(map(_unpack_args, packed_args), **kwargs))

    return list(tqdm(pool.imap_unordered(_unpack_args, packed_args, chunksize=chunksize), **kwargs))


def densitization_factor(r, u_t, adm_mass: float):
    """
    D/rho = W*sqrt(gamma) at radius `r`, from the local `u_t`.

    The rest-mass density rho is not the conserved density; the conserved
    rest mass is the integral of D = rho*W*sqrt(gamma), so a rho-based mass
    is short by that factor. Both pieces are recoverable without any metric
    in the snapshot data:

      * W = -u_t/alpha, exact wherever the shift is negligible (checked against
        a dumped W at r = 400 M in a BNS merger: agreement to 5 decimals).
      * for isotropic Schwarzschild with psi = 1 + M/2r,
        alpha = (1 - M/2r)/psi and sqrt(gamma) = psi**6,

    giving D/rho = (-u_t) * psi**7 / (1 - M/2r). Against a dumped conformal
    metric that was accurate to 0.008% at r = 400 M, and a 0.1 M_sun error in
    `adm_mass` moves it by under 0.1%. It degrades close to the remnant, where
    the shift stops being negligible and the metric stops being Schwarzschild.

    Accepts scalars or arrays; the return has the broadcast shape.
    """
    r = np.asarray(r, dtype=float)
    if np.any(r <= adm_mass):
        r_bad = float(np.min(r))
        raise ValueError(
            f"densitization needs r > adm_mass; got r={r_bad:g}, adm_mass={adm_mass:g}. "
            "The isotropic-Schwarzschild form is meaningless that deep in."
        )
    psi = 1.0 + adm_mass / (2.0 * r)
    factor = -np.asarray(u_t, dtype=float) * psi ** 7 / (1.0 - adm_mass / (2.0 * r))
    return factor if factor.ndim else float(factor)


def sph_to_cart(r, cos_theta, phi) -> np.ndarray:
    """Cartesian coordinates from (r, cos(theta), phi); any argument may be an array."""
    sin_theta = np.sqrt(np.clip(1 - np.asarray(cos_theta, dtype=float)**2, 0.0, None))
    return np.array([r * sin_theta * np.cos(phi),
                     r * sin_theta * np.sin(phi),
                     r * cos_theta])


def cell_measure(lo: np.ndarray, hi: np.ndarray,
                 r_surf: float | np.ndarray | None = None) -> np.ndarray:
    """
    Volume of every cell, or its area when it lies on the sphere at `r_surf`
    (one radius, or one per cell).

    `lo` and `hi` are the ``(D, n_cells)`` bounds of
    :meth:`~src.file.FileHandler.native_cell_weights`, sorted, so the angular
    extents are plain differences. The last two axes are always
    ``(cos(theta), phi)``; a volume carries ``r`` in front of them.
    """
    d_ang = (hi[-2] - lo[-2]) * (hi[-1] - lo[-1])
    if r_surf is not None:
        return r_surf**2 * d_ang
    return (hi[0]**3 - lo[0]**3) / 3 * d_ang


def cell_centres(lo: np.ndarray, hi: np.ndarray,
                 r_surf: float | np.ndarray | None = None) -> np.ndarray:
    """Cartesian centre of every cell, shape (3, n_cells)."""
    # cos(theta) and phi take the midpoint of the measure dV is written in.
    # The radial one deliberately does NOT: it is the arithmetic midpoint in r,
    # not the midpoint of r**3.
    #
    # The formally consistent choice would be ((r_lo**3 + r_hi**3)/2)**(1/3),
    # exact for constant rho. But a radial grid is geometric precisely because
    # the ejecta falls off roughly as rho ~ r**-3, and expanding both rules
    # about a cell of ratio 1+e against the exact ln(r_hi/r_lo):
    #
    #   arithmetic midpoint : e - e**2/2 + e**3/3        (the exact series)
    #   r**3      midpoint : e - e**2/2 - 0.42 e**3
    #
    # so the arithmetic one is third-order accurate on that profile while the
    # measure-consistent one is not. Measured on an analytic rho = r**-3 shell
    # (tests/test_seeds_mc.py) the r**3 midpoint comes out 0.98% low; the
    # arithmetic one is within 0.1%. Do not "fix" this without rerunning that
    # test.
    r = r_surf if r_surf is not None else (lo[0] + hi[0]) / 2
    return sph_to_cart(r, (lo[-2] + hi[-2]) / 2, (lo[-1] + hi[-1]) / 2)


def tensor_cell_bounds(*edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Flat per-cell coordinate bounds for one separable grid.

    ``edges`` gives the cell faces along each axis. The result is the
    ``(lo, hi)`` pair of ``(D, n_cells)`` arrays that
    :meth:`~src.file.FileHandler.native_cell_weights` is defined in terms of,
    C-ordered over the axes so it lines up with a C-ordered value array.

    Bounds come back sorted per cell, so an axis whose faces descend --
    ``cos(theta)`` on an equal-solid-angle grid -- needs no special case here
    or in the caller.
    """
    lo = np.array([a.ravel() for a in
                   np.meshgrid(*(e[:-1] for e in edges), indexing='ij')])
    hi = np.array([a.ravel() for a in
                   np.meshgrid(*(e[1:] for e in edges), indexing='ij')])
    return np.minimum(lo, hi), np.maximum(lo, hi)
