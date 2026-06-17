"""
Methods to seed initial tracer positions and times.
  Should also at least assign a volume element of to each tracer for later use in integration and analysis.
"""

import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
from .tracers import Tracers
from .integrators import IntegratorBase
from .file import FileHandler
from .utils import do_parallel_star_pool


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
    Evaluate ρ·v_r at all quadrature points for a bundle of tracers and return
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
    volume cell ``[r_lo, r_hi] × [th_lo, th_hi] × [ph_lo, ph_hi]``.

    The weights absorb the Jacobian ``r² sin(θ)`` and the mapping half-widths
    so that ``mass = weights @ rho(points)`` gives the cell-integrated mass.

    With ``n_quad = 1`` the single node is the cell centre and the weight
    equals ``dV = r_c² sin(θ_c) Δr Δθ Δφ``, recovering the single-point
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
    surface cell at radius ``r_surf`` over ``[th_lo, th_hi] × [ph_lo, ph_hi]``.

    The weights absorb ``r_surf² sin(θ)`` and the half-widths so that
    ``quantity = weights @ f(points)`` gives the surface-integrated quantity.

    Parameters
    ----------
    n_quad : int
        Number of Gauss-Legendre points per dimension; ``n_quad**2`` points in
        total.

    Returns
    -------
    points : ndarray, shape (3, n_quad**2)
    weights : ndarray, shape (n_quad**2,)
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
    Seed tracers in spherical coordinates, with geometric spacing in r and uniform spacing in theta and phi, such that each tracer represents the same volume element.
      Parameters:
        rmin: minimum radius
        rmax: maximum radius
        n_r: number of radial bins
        n_th: number of theta bins
        n_ph: number of phi bins
        start_t: initial time for all tracers
        phi_min: minimum phi angle (default 0)
        phi_max: maximum phi angle (default 2*pi)
        theta_min: minimum theta angle (default 0)
        theta_max: maximum theta angle (default pi)
        random_shift_in_cell: whether to randomly shift tracer positions within their initial cell to avoid grid artifacts (default True)
        n_quad: number of Gauss-Legendre quadrature points per dimension used
            to integrate rho over each cell and compute tracer mass (default 2,
            giving 2³ = 8 points).  Use n_quad=1 to recover the single-point
            (cell-centre) approximation mass = rho * dV.
        density_key: field key for the mass density used in the mass integral
            (default 'rho').  Must be present in the FileHandler's key list.
        **kwargs: additional keyword arguments to pass to Tracers constructor
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
    dph = np.full(n_ph, 2*np.pi/n_ph)
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

    quads = []   # (pts, wts) per tracer — built alongside props
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
      ``dm = dt_k · ∫ ρ v_r dA`` over the entire window.
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
    # Build slot → file-time windows using the FileHandler's time axis
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
    # _gauss_legendre_surface weights already include r_surf² sin(θ) dθ dφ;
    # only scale by the per-file time step dt/n_g to get dm = ∫ r² sin(θ) ρ v_r dθ dφ dt.
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

    print(pos_inject.shape, t_inject.shape, len(props))
    return Tracers(
        positions=pos_inject,
        times=t_inject,
        props=props,
        **kwargs
    )
