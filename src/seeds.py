"""Provides functions to seed initial tracer positions and times for use with Tracers."""

import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
from .tracers import Tracers
from .integrators import IntegratorBase
from .file import FileHandler
from .utils import do_parallel_star_pool, densitization_factor


# ---------------------------------------------------------------------------
# Worker globals and functions for parallel surface mass calculation
# ---------------------------------------------------------------------------

_smass_setup_interp_fn = None
_smass_extra_data      = None
_smass_d_idx:    int   = 0
_smass_v_indices: tuple = ()
_smass_r_surf:   float = 1.0


def _init_mass_worker(setup_interp_fn, extra_data, d_idx, v_indices, r_surf):
    """Pool initializer: store constants in worker globals to avoid per-task pickling."""
    global _smass_setup_interp_fn, _smass_extra_data
    global _smass_d_idx, _smass_v_indices, _smass_r_surf
    _smass_setup_interp_fn = setup_interp_fn
    _smass_extra_data      = extra_data
    _smass_d_idx           = d_idx
    _smass_v_indices       = v_indices
    _smass_r_surf          = r_surf


def _mass_flux_bundle(
    bundle_idx: int,
    pos_q_flat: np.ndarray,   # (3, n_q2 * n_tr_bundle)
    w_q: np.ndarray,          # (n_q2, n_tr_bundle)
    shm_needed: dict,
) -> tuple:
    """
    Evaluate rho*v_r at all quadrature points for a bundle of tracers and return
    the surface-integrated dm for each tracer in the bundle.

    Pickled per task: pos_q_flat, w_q, shm_needed (string dict).
    Passed via worker globals: setup_interp_fn, extra_data, key indices, r_surf.
    """
    interp = _smass_setup_interp_fn(shm_needed, _smass_extra_data)
    interp.load()
    try:
        vals = interp(pos_q_flat)          # (n_keys, n_q2 * n_tr_bundle)
        n_q2, n_tr = w_q.shape
        pos_q = pos_q_flat.reshape(3, n_q2, n_tr)
        rho = vals[_smass_d_idx      ].reshape(n_q2, n_tr)
        vx  = vals[_smass_v_indices[0]].reshape(n_q2, n_tr)
        vy  = vals[_smass_v_indices[1]].reshape(n_q2, n_tr)
        vz  = vals[_smass_v_indices[2]].reshape(n_q2, n_tr)
        vr  = (pos_q[0]*vx + pos_q[1]*vy + pos_q[2]*vz) / _smass_r_surf
        return bundle_idx, np.sum(w_q * rho * vr, axis=0)
    finally:
        interp.unload()



# ---------------------------------------------------------------------------

def _gauss_legendre_3d(
    r_lo: float,  r_hi: float,
    th_lo: float, th_hi: float,
    ph_lo: float, ph_hi: float,
    n_quad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cartesian Gauss-Legendre quadrature nodes and weights for a spherical
    volume cell ``[r_lo, r_hi] x [th_lo, th_hi] x [ph_lo, ph_hi]``.

    The weights absorb the Jacobian ``r^2 sin(theta)`` and the mapping half-widths
    so that ``mass = weights @ rho(points)`` gives the cell-integrated mass.

    With ``n_quad = 1`` the single node is the cell centre and the weight
    equals ``dV = r_c^2 sin(theta_c) Deltar Deltatheta Deltaphi``, recovering the single-point
    approximation.

    Parameters
    ----------
    n_quad : int
        Number of Gauss-Legendre points per dimension; ``n_quad**3`` points
        in total.

    Returns
    -------
    points : ndarray, shape (3, n_quad**3)
        Cartesian coordinates of the quadrature nodes.
    weights : ndarray, shape (n_quad**3,)
        Integration weights (Jacobian + half-widths included).
    """
    xi, wi = np.polynomial.legendre.leggauss(n_quad)

    r_c,  r_h  = (r_lo  + r_hi)  / 2, (r_hi  - r_lo)  / 2
    th_c, th_h = (th_lo + th_hi) / 2, (th_hi - th_lo) / 2
    ph_c, ph_h = (ph_lo + ph_hi) / 2, (ph_hi - ph_lo) / 2

    r_pts  = r_c  + r_h  * xi
    th_pts = th_c + th_h * xi
    ph_pts = ph_c + ph_h * xi

    r_g, th_g, ph_g = np.meshgrid(r_pts, th_pts, ph_pts, indexing='ij')
    wr, wth, wph    = np.meshgrid(wi,    wi,      wi,     indexing='ij')

    W = wr * wth * wph * r_h * th_h * ph_h * r_g**2 * np.sin(th_g)

    x = r_g * np.sin(th_g) * np.cos(ph_g)
    y = r_g * np.sin(th_g) * np.sin(ph_g)
    z = r_g * np.cos(th_g)

    return np.array([x.flatten(), y.flatten(), z.flatten()]), W.flatten()


def _gauss_legendre_surface(
    r_surf: float,
    th_lo: float, th_hi: float,
    ph_lo: float, ph_hi: float,
    n_quad: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cartesian Gauss-Legendre quadrature nodes and weights for a 2-D spherical
    surface cell at radius ``r_surf`` over ``[th_lo, th_hi] x [ph_lo, ph_hi]``.

    The weights absorb ``r_surf^2 sin(theta)`` and the half-widths so that
    ``quantity = weights @ f(points)`` gives the surface-integrated quantity.

    Parameters
    ----------
    n_quad : int
        Number of Gauss-Legendre points per dimension; ``n_quad**2`` points in
        total.

    Returns
    -------
    x : ndarray, shape (n_quad**2,)
        Flattened x-coordinates of the quadrature nodes.
    y : ndarray, shape (n_quad**2,)
        Flattened y-coordinates of the quadrature nodes.
    z : ndarray, shape (n_quad**2,)
        Flattened z-coordinates of the quadrature nodes.
    W : ndarray, shape (n_quad**2,)
        Flattened surface-integration weights.
    """
    xi, wi = np.polynomial.legendre.leggauss(n_quad)

    th_c, th_h = (th_lo + th_hi) / 2, (th_hi - th_lo) / 2
    ph_c, ph_h = (ph_lo + ph_hi) / 2, (ph_hi - ph_lo) / 2

    th_pts = th_c + th_h * xi
    ph_pts = ph_c + ph_h * xi

    th_g, ph_g = np.meshgrid(th_pts, ph_pts, indexing='ij')
    wth, wph   = np.meshgrid(wi,     wi,     indexing='ij')

    W = wth * wph * th_h * ph_h * r_surf**2 * np.sin(th_g)

    x = r_surf * np.sin(th_g) * np.cos(ph_g)
    y = r_surf * np.sin(th_g) * np.sin(ph_g)
    z = r_surf * np.cos(th_g)

    return x.flatten(), y.flatten(), z.flatten(), W.flatten()


# ---------------------------------------------------------------------------
# Helpers for computing mass directly from interpolated field data
# ---------------------------------------------------------------------------

def spherical_by_volume(
    r_min: float,
    r_max: float,
    n_r: int,
    n_th: int,
    n_ph: int,
    start_t: float,
    phi_min: float = 0.0,
    phi_max: float = 2*np.pi,
    theta_min: float = 0.0,
    theta_max: float = np.pi,
    random_shift_in_cell: bool = True,
    n_quad: int = 2,
    density_key: str = 'rho',
    **kwargs
    ) -> Tracers:
    """
    Seed tracers in spherical cells built for roughly equal represented mass
    (not volume) under a typical homologous outflow.

    Radial bins are geometrically spaced, phi bins are uniformly spaced, and
    theta bins are built from equally spaced ``cos(theta)`` edges so each
    angular strip spans equal solid angle. Geometric radial spacing makes
    cell volume grow as ``r**3`` outward (``dr`` grows with ``r``, and
    ``dV ~ r**2 * dr``), which roughly cancels a homologous outflow's
    ``rho ~ r**-3`` density falloff -- giving tracers of roughly equal mass,
    not equal volume. Each tracer's actual mass is still integrated from the
    density field directly (see ``density_key``), so this holds only
    approximately and only for flows resembling that density profile.

    Parameters
    ----------
    r_min : float
        Minimum radius.
    r_max : float
        Maximum radius.
    n_r : int
        Number of radial bins.
    n_th : int
        Number of theta bins.
    n_ph : int
        Number of phi bins.
    start_t : float
        Initial time assigned to every tracer.
    phi_min : float, optional
        Minimum azimuthal angle in radians.
    phi_max : float, optional
        Maximum azimuthal angle in radians.
    theta_min : float, optional
        Minimum polar angle in radians.
    theta_max : float, optional
        Maximum polar angle in radians.
    random_shift_in_cell : bool, optional
        If True, randomly jitter each tracer position within its spherical cell.
    n_quad : int, optional
        Number of Gauss-Legendre quadrature points per dimension used to
        integrate density over each cell when estimating tracer mass.
    density_key : str, optional
        Field key for the density used in the cell-mass integral.
    **kwargs
        Additional keyword arguments forwarded to ``Tracers``.

    Returns
    -------
    Tracers
        Tracer collection initialized at the seeded positions and times.
    """

    n_tracers = n_r * n_th * n_ph
    times = np.full(n_tracers, start_t)

    r_edges = np.geomspace(r_min, r_max, n_r+1)
    dr = np.diff(r_edges)
    r_start = r_edges[:-1] + dr/2

    costheta_start = np.linspace(np.cos(theta_min), np.cos(theta_max), n_th+1)
    th_edges = np.sort(np.arccos(costheta_start))
    dth = np.diff(th_edges)
    th_start = th_edges[:-1] + dth/2

    ph_start = np.linspace(phi_min, phi_max, n_ph, endpoint=False)
    dph = np.full(n_ph, (phi_max - phi_min) / n_ph)
    ph_start = ph_start + dph/2
    ph_edges = np.linspace(phi_min, phi_max, n_ph + 1)

    r_start, th_start, ph_start = np.meshgrid(r_start, th_start, ph_start, indexing='ij')
    dr, dth, dph = np.meshgrid(dr, dth, dph, indexing='ij')
    dV = r_start**2 * np.sin(th_start) * dr * dth * dph

    if random_shift_in_cell:
        r_start += np.random.uniform(-dr/2, dr/2)
        th_start += np.random.uniform(-dth/2, dth/2)
        ph_start += np.random.uniform(-dph/2, dph/2)

    r_start = r_start.flatten()
    th_start = th_start.flatten()
    ph_start = ph_start.flatten()
    dV = dV.flatten()

    x_start = r_start*np.sin(th_start)*np.cos(ph_start)
    y_start = r_start*np.sin(th_start)*np.sin(ph_start)
    z_start = r_start*np.cos(th_start)

    # Build per-tracer props, compute mass directly via file interpolation.
    # Iteration order matches the 'ij' meshgrid flatten: r slowest, ph fastest.
    r_edges_1d  = np.geomspace(r_min, r_max, n_r+1)
    th_edges_1d = np.sort(np.arccos(np.linspace(np.cos(theta_min), np.cos(theta_max), n_th+1)))

    quads = []   # (pts, wts) per tracer - built alongside props
    props = []
    for ir in range(n_r):
        for ith in range(n_th):
            for iph in range(n_ph):
                idx = ir * n_th * n_ph + ith * n_ph + iph
                pts, wts = _gauss_legendre_3d(
                    r_edges_1d[ir],  r_edges_1d[ir+1],
                    th_edges_1d[ith], th_edges_1d[ith+1],
                    ph_edges[iph],   ph_edges[iph+1],
                    n_quad,
                )
                quads.append((pts, wts))
                props.append({'dV': float(dV[idx])})

    # Load the single snapshot at start_t and batch-evaluate all quadrature points.
    file_handler = kwargs.get('file_handler')
    if file_handler is not None:
        file_times = file_handler.times
        i_ft = int(np.argmin(np.abs(file_times - start_t)))
        file_handler.load_chunk(i_ft, forward=True)
        # find which slot in the loaded chunk matches start_t
        i_loc = int(np.argmin(np.abs(file_handler.cur_times - start_t)))
        shm = file_handler.shared_memory[i_loc]
        key_list = list(file_handler.keys)
        d_idx = key_list.index(density_key)

        interp = type(file_handler).setup_interpolator(shm, file_handler.extra_data)
        interp.load()
        try:
            all_pts = np.concatenate([pts for pts, _ in quads], axis=1)
            all_vals = interp(all_pts)          # (n_keys, total_quad_pts)
            rho_all  = all_vals[d_idx]
            offset = 0
            for prop, (pts, wts) in zip(props, quads):
                n = pts.shape[1]
                prop['mass'] = float(np.dot(wts, rho_all[offset:offset + n]))
                offset += n
        finally:
            interp.unload()

    return Tracers(
        positions=np.array([x_start, y_start, z_start]).T,
        times=times,
        props=props,
        **kwargs
        )

def spherical_surface_by_area(
    r_surf: float,
    t_start: np.ndarray,
    n_th: int,
    n_ph: int,
    phi_min: float = 0.0,
    phi_max: float = 2*np.pi,
    theta_min: float = 0.0,
    theta_max: float = np.pi,
    random_shift_in_cell: bool = True,
    n_quad: int = 3,
    density_key: str = 'rho',
    **kwargs
    ) -> Tracers:
    """
    Seed tracers on a spherical surface, with uniform spacing in theta and phi,
    such that each tracer represents the same area element.

    Each slot in ``t_start`` corresponds to a time window.  For each tracer:

    * One ``MassStep`` is registered per file time in the slot's window so
      that ``Tracers.initialize_masses`` accumulates the full mass flux
      ``dm = dt_k * int rho v_r dA`` over the entire window.
    * When ``random_shift_in_cell=True`` (default), the tracer's injection
      time is drawn uniformly from the file times inside the window, giving
      smoother temporal coverage.

    Parameters
    ----------
    r_surf : float
        Radius of the spherical surface.
    t_start : array-like
        Nominal injection times, one per time slot (e.g. ``t_files[::every_t]``).
    n_th, n_ph : int
        Number of angular bins in theta and phi.
    phi_min, phi_max, theta_min, theta_max : float
        Angular domain (radians).
    random_shift_in_cell : bool
        If True (default), randomly jitter angular positions within each cell
        and randomly pick the injection time from the window file times.
    n_quad : int
        Gauss-Legendre quadrature points per angular dimension (``n_quad**2``
        total). Use ``n_quad=1`` for the single-point approximation.
    density_key : str
        Field key for mass density (default ``'rho'``).
    **kwargs
        Passed to ``Tracers``; must include ``file_handler`` and ``vel_keys``.

    Returns
    -------
    Tracers
        Tracer collection initialized on the requested spherical surface.
    """

    forward = t_start[0] < t_start[1]
    direc = 1 if forward else -1

    t_start = np.asarray(t_start)
    dt_g = np.zeros_like(t_start)
    dt_g[1:-1] = (np.diff(t_start[:-1]) + np.diff(t_start[1:])) / 2
    dt_g[0] = np.diff(t_start[:2])[0] / 2
    dt_g[-1] = np.diff(t_start[-2:])[0] / 2
    dt_g = np.abs(dt_g)
    n_t = len(t_start) - 1

    vel_keys: tuple[str, ...] = kwargs['vel_keys']
    file_handler: FileHandler = kwargs['file_handler']
    _remember_verbose = file_handler.parallel_kwargs['verbose']
    file_handler.parallel_kwargs['verbose'] = False  # silence per-file load messages during mass integration


    # -----------------------------------------------------------------------
    # Angular grid
    # -----------------------------------------------------------------------
    costheta_start = np.linspace(np.cos(theta_min), np.cos(theta_max), n_th+1)
    th_edges = np.sort(np.arccos(costheta_start))
    dth = np.diff(th_edges)
    th_mid = th_edges[:-1] + dth/2

    ph_edges = np.linspace(phi_min, phi_max, n_ph + 1)
    dph = np.full(n_ph, (phi_max - phi_min) / n_ph)
    ph_mid = ph_edges[:-1] + dph/2


    # -----------------------------------------------------------------------
    # Build slot -> file-time windows using the FileHandler's time axis
    # -----------------------------------------------------------------------
    file_times = np.asarray(file_handler.times)
    ft_bins = np.zeros(len(file_times) + 1)
    ft_bins[1:-1] = (file_times[:-1] + file_times[1:]) / 2
    ft_bins[0] = file_times[0]
    ft_bins[-1] = file_times[-1]
    # Index of the file time closest to each slot's nominal start
    i_start = np.minimum(np.digitize(t_start, ft_bins) - 1, len(file_times) - 1)
    if np.any(i_start < 0) or np.any(i_start >= len(file_times)):
        raise ValueError("Some t_start values are out of the file times range.")
    # check if slot starts are unique. if not we are oversampling some file times -> raise
    if len(set(i_start)) != len(i_start):
        raise ValueError("Some t_start values are too close together, leading to duplicate file time assignments. "
                         "Ensure t_start values are spaced sufficiently apart.")
    slot_i_start = i_start[:-1].copy()
    slot_i_end = i_start[1:].copy()

    n_slot = np.abs(slot_i_end - slot_i_start)
    dt_slot = np.abs(file_times[slot_i_end] - file_times[slot_i_start])

    if forward:
        slot_i_end -= 1
    else:
        slot_i_end += 1

    slot_i_start, th_mid, ph_mid = np.meshgrid(slot_i_start, th_mid, ph_mid, indexing='ij')
    slot_i_end, _, _ = np.meshgrid(slot_i_end, dth, dph, indexing='ij')
    dt_g, dth_g, dph_g = np.meshgrid(dt_slot, dth, dph, indexing='ij')
    n_g, _, _ = np.meshgrid(n_slot, dth, dph, indexing='ij')
    slot_i_start = slot_i_start.flatten()
    slot_i_end = slot_i_end.flatten()
    th_mid = th_mid.flatten()
    ph_mid = ph_mid.flatten()
    n_g = n_g.flatten()
    dt_g = dt_g.flatten()
    dth_g = dth_g.flatten()
    dph_g = dph_g.flatten()
    n_tr = len(slot_i_start)

    n_q2 = n_quad**2
    # -----------------------------------------------------------------------
    # Build per-tracer quadrature points
    # -----------------------------------------------------------------------
    pos_quads = np.empty((3, n_q2, n_tr))
    w_quads = np.empty((n_q2, n_tr))
    for it in range(n_t):
        for ith in range(n_th):
            for iph in range(n_ph):
                i_tr = it * n_th * n_ph + ith * n_ph + iph
                x_q, y_q, z_q, w_q = _gauss_legendre_surface(
                    r_surf,
                    th_edges[ith], th_edges[ith+1],
                    ph_edges[iph], ph_edges[iph+1],
                    n_quad
                )
                pos_quads[0, :, i_tr] = x_q
                pos_quads[1, :, i_tr] = y_q
                pos_quads[2, :, i_tr] = z_q
                w_quads[:, i_tr] = w_q

    if random_shift_in_cell:
        th_inject = th_mid + np.random.uniform(-dth_g/2, dth_g/2)
        ph_inject = ph_mid + np.random.uniform(-dph_g/2, dph_g/2)
        t_inject = file_times[slot_i_start + direc*np.random.randint(0, n_g)]
    else:
        th_inject = th_mid
        ph_inject = ph_mid
        t_inject = file_times[slot_i_start]

    x = r_surf * np.sin(th_inject) * np.cos(ph_inject)
    y = r_surf * np.sin(th_inject) * np.sin(ph_inject)
    z = r_surf * np.cos(th_inject)
    pos_inject = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    pos_quads = pos_quads.reshape((3, n_q2, n_tr))
    # _gauss_legendre_surface weights already include r_surf^2 sin(theta) dtheta dphi;
    # only scale by the per-file time step dt/n_g to get dm = int r^2 sin(theta) rho v_r dtheta dphi dt.
    w_quads = w_quads.reshape((n_q2, n_tr)) * (dt_g / n_g).reshape((1, n_tr))
    t_inject = t_inject.flatten()

    dm = np.zeros_like(t_inject)  # to be filled in by mass integration

    # -----------------------------------------------------------------------
    # Compute masses by iterating over unique file times across all windows
    # -----------------------------------------------------------------------
    chunk_indices, forward, t_start, t_end = file_handler.get_chunk_indices(
        file_times[slot_i_start[0]],
        file_times[slot_i_end[-1]],
        overlap=False,
    )

    n_cpu = file_handler.parallel_kwargs["n_cpu"]
    n_bundles_per_step = max(1, 20 * n_cpu)
    needed_keys = (*vel_keys, density_key)
    d_idx    = needed_keys.index(density_key)
    v_indices = tuple(needed_keys.index(vk) for vk in vel_keys)
    pbar_kwargs = {k: v for k, v in file_handler.parallel_kwargs.items() if k != "n_cpu"}
    pbar_kwargs["disable"] = not pbar_kwargs.pop('verbose', False)

    # Precompute slot bounds (constant across all timesteps)
    min_slot = np.minimum(slot_i_start, slot_i_end)
    max_slot = np.maximum(slot_i_start, slot_i_end)

    init_args = (
        type(file_handler).setup_interpolator,
        file_handler.extra_data,
        d_idx,
        v_indices,
        r_surf,
    )

    pbar_kwargs = {k: v for k, v in file_handler.parallel_kwargs.items() if k != "n_cpu"}
    pbar_kwargs.pop('verbose', None)  # we'll handle this separately to avoid tqdm messages from each worker
    pbar_kwargs["disable"] = not _remember_verbose
    if n_cpu > 1:
        mass_pool = Pool(n_cpu, initializer=_init_mass_worker, initargs=init_args)
    else:
        _init_mass_worker(*init_args)
        mass_pool = None

    try:
        for i_step in tqdm(
            chunk_indices,
            desc="Calculating tracer masses",
            ncols=0,
            unit="time step chunk",
            **pbar_kwargs
        ):
            file_handler.load_chunk(i_step, forward=forward)

            for i_loc, time in enumerate(file_handler.cur_times):
                i_t = file_times.searchsorted(time)
                t_msk = (min_slot <= i_t) & (i_t <= max_slot)
                if not np.any(t_msk):
                    continue

                tr_indices   = np.where(t_msk)[0]
                pos_q_active = pos_quads[:, :, tr_indices]  # (3, n_q2, n_active)
                w_q_active   = w_quads[:, tr_indices]        # (n_q2, n_active)
                shm_needed   = {k: file_handler.shared_memory[i_loc][k] for k in needed_keys}

                n_active = len(tr_indices)
                splits = [s for s in np.array_split(np.arange(n_active),
                                                     min(n_bundles_per_step, n_active))
                          if len(s) > 0]
                args_list = [
                    (j, pos_q_active[:, :, s].reshape(3, -1), w_q_active[:, s], shm_needed)
                    for j, s in enumerate(splits)
                ]

                results = do_parallel_star_pool(
                    mass_pool, _mass_flux_bundle, args_list,
                    disable=True, ncols=0,
                )
                results.sort(key=lambda r: r[0])

                for (_, dm_bundle), s in zip(results, splits):
                    dm[tr_indices[s]] += dm_bundle
    finally:
        if mass_pool is not None:
            mass_pool.terminate()
            mass_pool.join()
        file_handler.parallel_kwargs['verbose'] = _remember_verbose

    props = [{'mass': m } for m in dm]

    print(f"Seeded {len(pos_inject)} tracers at {len(np.unique(t_inject))} unique times.")
    return Tracers(
        positions=pos_inject,
        times=t_inject,
        props=props,
        **kwargs
    )


# ---------------------------------------------------------------------------
# Mass-weighted Monte-Carlo seeding
# ---------------------------------------------------------------------------

def _auto_grid(n_target: int, extents: tuple[float, ...]) -> tuple[int, ...]:
    """
    Split ``n_target`` cells over the given axis extents so cells are roughly
    isotropic in those coordinates.  At least 2 cells per axis.
    """
    e = np.asarray(extents, dtype=float)
    scale = (n_target / np.prod(e)) ** (1 / len(e))
    return tuple(max(2, int(round(x))) for x in e * scale)


def _sampling_interpolator(file_handler: FileHandler, i_loc: int, keys: tuple[str, ...]):
    """Interpolator on the loaded snapshot in slot ``i_loc``, restricted to ``keys``."""
    shm = {k: file_handler.shared_memory[i_loc][k] for k in keys}
    return type(file_handler).setup_interpolator(shm, file_handler.extra_data)


def _conservative_density(
    vals: np.ndarray,
    keys: tuple[str, ...],
    pos: np.ndarray,
    density_key: str,
    ut_key: str,
    adm_mass: float | None,
) -> np.ndarray:
    """
    D = sqrt(gamma) W rho at the query points, or plain rho if ``adm_mass`` is
    None.  ``vals`` is the interpolator output for ``keys`` at ``pos``.
    """
    rho = vals[keys.index(density_key)]
    if adm_mass is None:
        return rho
    r = np.linalg.norm(pos, axis=0)
    return rho * densitization_factor(r, vals[keys.index(ut_key)], adm_mass)


def _sample_cells(
    weights: np.ndarray,
    n_tracers: int,
    what: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Draw ``n_tracers`` cell indices with probability proportional to
    ``|weights|``, and return them along with the sign of each drawn cell's
    weight and the total ``sum(|weights|)``.

    Sampling on the magnitude while the estimator carries ``w / p``, with
    ``p = |w| / sum|w|``, leaves each draw worth ``sign(w) * sum|w| / N``.
    Every tracer therefore has the same mass magnitude, the sign of its own
    cell, and the signed sum over tracers is an unbiased estimate of the net
    (signed) total.
    """
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    abs_weights = np.abs(weights)
    total = float(abs_weights.sum())
    if total <= 0:
        raise ValueError(
            f"The {what} weight vanishes everywhere on the sampling grid; "
            "nothing to sample from.  Check the radial/angular range and --density-key."
        )
    idx = np.random.choice(len(weights), size=n_tracers, p=abs_weights / total)
    return idx, np.sign(weights[idx]), total


def _axis_keep(edges: np.ndarray, lo: float, hi: float) -> slice | None:
    """
    Cells of `edges` whose centre lies in [lo, hi], as a contiguous slice, or
    None if the axis does not reach that range at all.

    `edges` may run either way; the comparison is on the centres so a
    descending cos(theta) axis needs no special case. Returning None rather
    than raising matters once cells arrive in blocks: a block lying wholly
    outside the requested region is ordinary, not an error.
    """
    centres = (edges[:-1] + edges[1:]) / 2
    keep = np.flatnonzero((centres >= min(lo, hi)) & (centres <= max(lo, hi)))
    if keep.size == 0:
        return None
    return slice(int(keep[0]), int(keep[-1]) + 1)


class _CellSet:
    """
    A flat set of cells assembled from one or more separable blocks.

    A format with one global grid contributes a single block, and so does the
    interpolated helper grid, so the sampling, the measure and the in-cell
    placement below are written once and serve every path. Blocks are assumed
    to tile without overlapping; that is the hook's contract, and it is what
    stops a cell's mass being counted twice.

    Cells are indexed by one flat integer running block by block, so a drawn
    index is resolved by finding its block and unravelling the remainder with
    that block's own shape.
    """

    def __init__(self, blocks: list[tuple[np.ndarray, ...]], r_surf: float | None = None):
        if not blocks:
            raise ValueError("No cells to sample from in the requested region.")
        self.blocks = blocks
        self.r_surf = r_surf
        self.shapes = [tuple(len(e) - 1 for e in edges) for edges in blocks]
        sizes = [int(np.prod(sh)) for sh in self.shapes]
        self.offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)

    @property
    def n_cells(self) -> int:
        return int(self.offsets[-1])

    def measure(self) -> np.ndarray:
        """Cell volume, or cell area at ``r_surf``, flat over all blocks."""
        out = []
        for edges in self.blocks:
            if self.r_surf is None:
                r_e, cth_e, ph_e = edges
                m = ((np.diff(r_e**3) / 3)[:, None, None]
                     * np.abs(np.diff(cth_e))[None, :, None]
                     * np.diff(ph_e)[None, None, :])
            else:
                cth_e, ph_e = edges
                m = (self.r_surf**2
                     * np.abs(np.diff(cth_e))[:, None]
                     * np.diff(ph_e)[None, :])
            out.append(m.ravel())
        return np.concatenate(out)

    def centres(self) -> np.ndarray:
        """Cartesian cell centres, shape (3, n_cells)."""
        out = []
        for edges in self.blocks:
            if self.r_surf is None:
                r_e, cth_e, ph_e = edges
                # See the note below on why r uses the arithmetic midpoint.
                r_c = (r_e[:-1] + r_e[1:]) / 2
            else:
                cth_e, ph_e = edges
                r_c = np.array([self.r_surf])
            th_c = np.arccos(np.clip((cth_e[:-1] + cth_e[1:]) / 2, -1.0, 1.0))
            ph_c = (ph_e[:-1] + ph_e[1:]) / 2
            r_g, th_g, ph_g = np.meshgrid(r_c, th_c, ph_c, indexing='ij')
            out.append(np.array([
                (r_g * np.sin(th_g) * np.cos(ph_g)).ravel(),
                (r_g * np.sin(th_g) * np.sin(ph_g)).ravel(),
                (r_g * np.cos(th_g)).ravel(),
            ]))
        return np.concatenate(out, axis=1)

    def sample_positions(self, idx: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Cartesian positions for the drawn cells `idx`, placed inside their own
        cell by the uniform deviates `u` (shape (D, len(idx))).

        Bounds are resolved only for the cells actually drawn. With ~1e4
        tracers against ~1e7 cells that is the difference between a loop over a
        few hundred blocks and materialising six arrays the size of the grid.
        """
        idx = np.asarray(idx)
        b_of = np.searchsorted(self.offsets, idx, side='right') - 1
        local = idx - self.offsets[b_of]

        r_s = np.empty(idx.size)
        cth_s = np.empty(idx.size)
        ph_s = np.empty(idx.size)
        for b in np.unique(b_of):
            sel = b_of == b
            edges = self.blocks[b]
            if self.r_surf is None:
                r_e, cth_e, ph_e = edges
                i_r, i_th, i_ph = np.unravel_index(local[sel], self.shapes[b])
                # Uniform in r**3, so uniform in volume rather than in radius.
                r_s[sel] = (r_e[i_r]**3 + u[0, sel] * np.diff(r_e**3)[i_r]) ** (1 / 3)
            else:
                cth_e, ph_e = edges
                i_th, i_ph = np.unravel_index(local[sel], self.shapes[b])
                r_s[sel] = self.r_surf
            cth_s[sel] = cth_e[i_th] + u[-2, sel] * np.diff(cth_e)[i_th]
            ph_s[sel] = ph_e[i_ph] + u[-1, sel] * np.diff(ph_e)[i_ph]

        sth_s = np.sqrt(np.clip(1 - cth_s**2, 0.0, None))
        return np.array([
            r_s * sth_s * np.cos(ph_s),
            r_s * sth_s * np.sin(ph_s),
            r_s * cth_s,
        ])

    def describe(self) -> str:
        if len(self.blocks) == 1:
            return "x".join(str(n) for n in self.shapes[0])
        return f"{self.n_cells} cells in {len(self.blocks)} blocks"


def _native_cellset(file_handler, slot, keys, ranges, r_surf=None):
    """
    Ask the format for its own cells and keep those inside `ranges`.

    `ranges` is one (lo, hi) pair per axis of the returned edges. Blocks with
    no cell in range drop out; if every block does, the region is empty and
    _CellSet says so.
    """
    blocks, r_used = file_handler.native_cell_weights(slot, keys, surface_radius=r_surf)
    # None means the format did not snap: keep the radius the caller asked for.
    if r_used is None:
        r_used = r_surf
    kept_edges, kept_vals = [], []
    for edges, vals in blocks:
        slices = [_axis_keep(e, *rng) for e, rng in zip(edges, ranges)]
        if any(sl is None for sl in slices):
            continue
        shape = tuple(len(e) - 1 for e in edges)
        sub = vals.reshape(len(keys), *shape)[(slice(None), *slices)]
        kept_edges.append(tuple(e[sl.start:sl.stop + 1] for e, sl in zip(edges, slices)))
        kept_vals.append(sub.reshape(len(keys), -1))
    if not kept_edges:
        raise ValueError(
            "No native cell falls in the requested region; check --r-min/--r-max "
            "and the angular limits against the data's extent."
        )
    return _CellSet(kept_edges, r_surf=r_used), np.concatenate(kept_vals, axis=1)


def _resolve_weight_grid(file_handler, weight_grid: str) -> bool:
    """
    Whether to weight on the format's own grid.

    'native' insists and lets the NotImplementedError through; 'auto' asks and
    falls back quietly; 'helper' never asks.
    """
    if weight_grid == 'helper':
        return False
    # getattr, not attribute access: handlers are duck-typed in places and need
    # not derive from FileHandler at all.
    impl = getattr(type(file_handler), 'native_cell_weights', None)
    if impl is None or impl is FileHandler.native_cell_weights:
        if weight_grid == 'native':
            raise NotImplementedError(
                f"--weight-grid native: {type(file_handler).__name__} does not "
                "implement native_cell_weights."
            )
        return False
    return True


def spherical_by_volume_mc(
    r_min: float,
    r_max: float,
    n_tracers: int,
    start_t: float,
    phi_min: float = 0.0,
    phi_max: float = 2*np.pi,
    theta_min: float = 0.0,
    theta_max: float = np.pi,
    density_key: str = 'rho',
    ut_key: str = 'u_t',
    adm_mass: float | None = None,
    cells_per_tracer: int = 8,
    weight_grid: str = 'auto',
    **kwargs
    ) -> Tracers:
    """
    Seed tracers by sampling the spherical shell with the mass density as
    weight, so every tracer carries the same mass ``M_tot / n_tracers``.

    A helper grid (geometric in r, equal solid angle in theta, uniform in phi)
    is laid over the shell and the density is evaluated at each cell centre.
    Cells are then drawn with replacement with probability proportional to
    their mass ``D * dV``, and each tracer is placed uniformly (in volume)
    inside its drawn cell.  The tracer number density therefore tracks the
    mass density, which under-resolves low-density regions but tracks the
    global mass flow optimally.

    The grid resolution is picked automatically from ``n_tracers`` (see
    ``cells_per_tracer``), split over the three axes so cells are roughly
    isotropic in ``(ln r, theta, phi)``.

    Parameters
    ----------
    r_min, r_max : float
        Radial range of the seeding shell.
    n_tracers : int
        Number of tracers to draw.
    start_t : float
        Initial time assigned to every tracer; must be a snapshot time.
    phi_min, phi_max, theta_min, theta_max : float
        Angular domain (radians).
    density_key : str, optional
        Field key for the rest-mass density.
    ut_key : str, optional
        Field key for ``u_t``, needed only when ``adm_mass`` is given.
    adm_mass : float or None, optional
        ADM mass (code units).  When given, the sampling weight and the total
        mass use the conserved density ``D = sqrt(gamma) W rho`` instead of
        ``rho`` (see ``src.utils.densitization_factor``), and the resulting
        ``mass`` prop is the conserved rest mass per tracer.
    cells_per_tracer : int, optional
        Helper-grid cells per tracer.  Higher values resolve the density
        field better at the cost of more interpolator evaluations.
    **kwargs
        Passed to ``Tracers``; must include ``file_handler``.

    Returns
    -------
    Tracers
        Tracer collection with ``n_tracers`` equal-mass tracers.
    """
    file_handler: FileHandler = kwargs['file_handler']
    native = _resolve_weight_grid(file_handler, weight_grid)
    needed_keys = (density_key,) if adm_mass is None else (density_key, ut_key)

    i_ft = int(np.argmin(np.abs(np.asarray(file_handler.times) - start_t)))
    file_handler.load_chunk(i_ft, forward=True)
    i_loc = int(np.argmin(np.abs(file_handler.cur_times - start_t)))

    if native:
        cells, vals = _native_cellset(
            file_handler, i_loc, needed_keys,
            ranges=((r_min, r_max),
                    (np.cos(theta_max), np.cos(theta_min)),
                    (phi_min, phi_max)),
        )
        pos_c = cells.centres()
    else:
        n_r, n_th, n_ph = _auto_grid(
            cells_per_tracer * n_tracers,
            (np.log(r_max / r_min), theta_max - theta_min, phi_max - phi_min),
        )
        cells = _CellSet([(
            np.geomspace(r_min, r_max, n_r + 1),
            np.linspace(np.cos(theta_min), np.cos(theta_max), n_th + 1),
            np.linspace(phi_min, phi_max, n_ph + 1),
        )])
        # cos(theta) and phi cell centres are the midpoint of the measure dV is
        # written in. The radial one deliberately is NOT: it is the arithmetic
        # midpoint in r, not the midpoint of r**3.
        #
        # The formally consistent choice would be ((r_lo**3 + r_hi**3)/2)**(1/3),
        # exact for constant rho. But this grid is geometric precisely because
        # the ejecta falls off roughly as rho ~ r**-3, and expanding both rules
        # about a cell of ratio 1+e against the exact ln(r_hi/r_lo) gives
        #
        #   arithmetic midpoint : e - e**2/2 + e**3/3        (the exact series)
        #   r**3      midpoint : e - e**2/2 - 0.42 e**3
        #
        # so the arithmetic one is third-order accurate on that profile while
        # the measure-consistent one is not. Measured on an analytic rho = r**-3
        # shell (tests/test_seeds_mc.py) the r**3 midpoint comes out 0.98% low;
        # the arithmetic one is within 0.1%. Do not "fix" this without rerunning
        # that test.
        pos_c = cells.centres()
        interp = _sampling_interpolator(file_handler, i_loc, needed_keys)
        interp.load()
        try:
            vals = interp(pos_c)
        finally:
            interp.unload()

    dens = _conservative_density(vals, needed_keys, pos_c, density_key, ut_key, adm_mass)

    weights = dens * cells.measure()
    idx, signs, m_tot = _sample_cells(weights, n_tracers, "mass")

    positions = cells.sample_positions(
        idx, np.random.uniform(size=(3, n_tracers))).T

    masses = signs * (m_tot / n_tracers)
    print(f"Sampled {n_tracers} tracers of mass {m_tot/n_tracers:.4e} from a "
          f"{cells.describe()} {'native' if native else 'interpolated'} weight grid "
          f"(total mass {masses.sum():.4e} of {np.nansum(weights):.4e} on the grid).")
    return Tracers(
        positions=positions,
        times=np.full(n_tracers, start_t),
        props=[{'mass': float(m)} if adm_mass is None
               else {'mass': float(m), 'mass_D': float(m)}
               for m in masses],
        **kwargs
    )


def spherical_surface_mc(
    r_surf: float,
    t_start: np.ndarray,
    n_tracers: int,
    phi_min: float = 0.0,
    phi_max: float = 2*np.pi,
    theta_min: float = 0.0,
    theta_max: float = np.pi,
    density_key: str = 'rho',
    ut_key: str = 'u_t',
    adm_mass: float | None = None,
    cells_per_tracer: int = 8,
    weight_grid: str = 'auto',
    **kwargs
    ) -> Tracers:
    """
    Seed tracers by sampling the (theta, phi, t) surface-time space with the
    mass flux as weight, so every tracer carries the same mass
    ``M_tot / n_tracers``.

    An angular helper grid is laid over the sphere at ``r_surf`` and, for each
    time in ``t_start``, the flux ``D v_r r^2 dcos(theta) dphi dt`` is evaluated
    at every cell centre.  Cells of that 3-D (theta, phi, t) grid are drawn with
    replacement proportional to the *magnitude* of that flux; the angles are
    then jittered uniformly inside the drawn cell and the injection time is the
    cell's exact snapshot time (tracers only activate on snapshot times).

    Inflowing cells (``v_r < 0``) are therefore sampled just like outflowing
    ones, but their tracers carry a *negative* mass, so they subtract from the
    net ejected mass exactly as their surface elements do.  Every tracer's mass
    has the same magnitude ``M_tot / n_tracers``, with ``M_tot`` the mass
    crossing the sphere either way, and the signed sum over tracers is an
    unbiased estimate of the net crossing mass.

    Parameters
    ----------
    r_surf : float
        Radius of the seeding surface.
    t_start : array-like
        Snapshot times forming the sampling time axis (e.g. ``t_files[::every_t]``).
        Each contributes ``dt`` from the midpoints of its neighbours.
    n_tracers : int
        Number of tracers to draw.
    phi_min, phi_max, theta_min, theta_max : float
        Angular domain (radians).
    density_key, ut_key, adm_mass, cells_per_tracer
        As in ``spherical_by_volume_mc``.
    **kwargs
        Passed to ``Tracers``; must include ``file_handler`` and ``vel_keys``.

    Returns
    -------
    Tracers
        Tracer collection with ``n_tracers`` tracers of equal mass magnitude
        and the sign of their own flux.
    """
    vel_keys: tuple[str, ...] = tuple(kwargs['vel_keys'])
    file_handler: FileHandler = kwargs['file_handler']
    file_times = np.asarray(file_handler.times)

    t_start = np.asarray(t_start, dtype=float)
    n_t = len(t_start)
    # Trapezoidal dt around each sampled time, as in spherical_surface_by_area.
    dt = np.empty(n_t)
    dt[1:-1] = (np.diff(t_start[:-1]) + np.diff(t_start[1:])) / 2
    dt[0] = np.diff(t_start[:2])[0] / 2
    dt[-1] = np.diff(t_start[-2:])[0] / 2
    dt = np.abs(dt)

    native = _resolve_weight_grid(file_handler, weight_grid)
    needed_keys = (*vel_keys, density_key)
    if adm_mass is not None and ut_key not in needed_keys:
        needed_keys += (ut_key,)

    ranges = ((np.cos(theta_max), np.cos(theta_min)), (phi_min, phi_max))

    if native:
        # The data's own cells set the angular resolution, and the format
        # decides whether r_surf needs snapping. One probe load to fix the cell
        # layout; the values are re-read per time step below.
        i_probe = int(np.argmin(np.abs(np.asarray(file_handler.times) - t_start[0])))
        file_handler.load_chunk(i_probe, forward=True)
        cells, _ = _native_cellset(
            file_handler, 0, needed_keys, ranges, r_surf=r_surf)
        if not np.isclose(cells.r_surf, r_surf, rtol=1e-3):
            print(f"--r-surf {r_surf:g} snapped to the grid shell at {cells.r_surf:g}.")
        r_surf = cells.r_surf
    else:
        n_th, n_ph = _auto_grid(
            max(cells_per_tracer * n_tracers // n_t, 4),
            (theta_max - theta_min, phi_max - phi_min),
        )
        cells = _CellSet([(
            np.linspace(np.cos(theta_min), np.cos(theta_max), n_th + 1),
            np.linspace(phi_min, phi_max, n_ph + 1),
        )], r_surf=r_surf)

    dA = cells.measure()
    pos_c = cells.centres()
    n_cells = cells.n_cells

    weights = np.zeros((n_t, n_cells))
    chunk_indices, forward, _, _ = file_handler.get_chunk_indices(
        t_start[0], t_start[-1], overlap=False)

    _remember_verbose = file_handler.parallel_kwargs['verbose']
    file_handler.parallel_kwargs['verbose'] = False
    try:
        for i_step in tqdm(chunk_indices, desc="Integrating the mass flux", ncols=0,
                           unit="time step chunk", disable=not _remember_verbose,
                           file=file_handler.parallel_kwargs['file']):
            file_handler.load_chunk(i_step, forward=forward)
            for i_loc, time in enumerate(file_handler.cur_times):
                i_t = np.flatnonzero(np.isclose(t_start, time))
                if len(i_t) == 0:
                    continue
                if native:
                    _, vals = _native_cellset(
                        file_handler, i_loc, needed_keys, ranges, r_surf=r_surf)
                    if vals.shape[1] != n_cells:
                        raise ValueError(
                            f"The cell layout changed between snapshots "
                            f"({vals.shape[1]} cells against {n_cells} at the probe); "
                            "a weight grid can only be built on a static mesh."
                        )
                else:
                    interp = _sampling_interpolator(file_handler, i_loc, needed_keys)
                    interp.load()
                    try:
                        vals = interp(pos_c)
                    finally:
                        interp.unload()
                dens = _conservative_density(
                    vals, needed_keys, pos_c, density_key, ut_key, adm_mass)
                v_r = sum(pos_c[i] * vals[needed_keys.index(vk)]
                          for i, vk in enumerate(vel_keys)) / r_surf
                weights[i_t[0]] = dens * v_r * dA * dt[i_t[0]]
    finally:
        file_handler.parallel_kwargs['verbose'] = _remember_verbose

    idx, signs, m_tot = _sample_cells(weights.ravel(), n_tracers, "mass flux")

    i_t, i_cell = np.divmod(idx, n_cells)
    positions = cells.sample_positions(
        i_cell, np.random.uniform(size=(2, n_tracers))).T
    times = t_start[i_t]

    masses = signs * (m_tot / n_tracers)
    print(f"Sampled {n_tracers} tracers of mass +-{m_tot/n_tracers:.4e} at "
          f"{len(np.unique(times))} unique times from a {cells.describe()} x {n_t} times "
          f"{'native' if native else 'interpolated'} weight grid "
          f"({(signs < 0).sum()} inflowing; net mass {masses.sum():.4e} of "
          f"{np.nansum(weights):.4e} on the grid, {m_tot:.4e} crossing either way).")
    return Tracers(
        positions=positions,
        times=times,
        props=[{'mass': float(m)} if adm_mass is None
               else {'mass': float(m), 'mass_D': float(m)}
               for m in masses],
        **kwargs
    )
