"""Provides functions to seed initial tracer positions and times for use with Tracers."""

import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
from .tracers import Tracers
from .integrators import IntegratorBase
from .file import FileHandler
from .utils import (do_parallel_star_pool, tensor_cell_bounds,
                    cell_centres, cell_measure, sph_to_cart)


# ---------------------------------------------------------------------------
# Worker globals and functions for parallel surface mass calculation
# ---------------------------------------------------------------------------

_smass_setup_interp_fn = None
_smass_extra_data      = None
_smass_mass_density    = None


def _init_mass_worker(setup_interp_fn, extra_data, mass_density):
    """Pool initializer: store constants in worker globals to avoid per-task pickling."""
    global _smass_setup_interp_fn, _smass_extra_data, _smass_mass_density
    _smass_setup_interp_fn = setup_interp_fn
    _smass_extra_data      = extra_data
    # The format's own answer to what the mass density is. It is plain data and
    # pickles, so the decision travels into the workers rather than being
    # reconstructed from key indices here.
    _smass_mass_density    = mass_density


def _mass_flux_bundle(
    bundle_idx: int,
    pos_q_flat: np.ndarray,   # (3, n_q2 * n_tr_bundle)
    w_q: np.ndarray,          # (n_q2, n_tr_bundle)
    shm_needed: dict,
) -> tuple:
    """
    Evaluate the radial mass flux at all quadrature points for a bundle of
    tracers and return the surface-integrated dm for each tracer in the bundle.

    Pickled per task: pos_q_flat, w_q, shm_needed (string dict).
    Passed via worker globals: setup_interp_fn, extra_data, mass_density.
    """
    interp = _smass_setup_interp_fn(shm_needed, _smass_extra_data)
    interp.load()
    try:
        vals = interp(pos_q_flat)          # (n_keys, n_q2 * n_tr_bundle)
        flux = _smass_mass_density.radial_flux(vals, pos_q_flat)
        return bundle_idx, np.sum(w_q * flux.reshape(w_q.shape), axis=0)
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
    density field directly (whichever one ``file_handler.mass_density`` says
    that is), so this holds only approximately and only for flows resembling
    that density profile.

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
    **kwargs
        Additional keyword arguments forwarded to ``Tracers``; must include
        ``file_handler``, whose ``mass_density`` defines the integrand.

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
        # The handler decides what the mass density is; here it is integrated
        # over each cell rather than sampled at a point.
        md = file_handler.mass_density
        d_idx = [key_list.index(k) for k in md.density_keys]

        interp = type(file_handler).setup_interpolator(shm, file_handler.extra_data)
        interp.load()
        try:
            all_pts = np.concatenate([pts for pts, _ in quads], axis=1)
            all_vals = interp(all_pts)          # (n_keys, total_quad_pts)
            rho_all  = md.density(all_vals[d_idx], all_pts)
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
    **kwargs
        Passed to ``Tracers``; must include ``file_handler`` and ``vel_keys``.
        The handler's ``mass_density`` supplies the flux ``rho v_r``, or its
        densitized equivalent.

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
    needed_keys = file_handler.mass_density.flux_keys
    pbar_kwargs = {k: v for k, v in file_handler.parallel_kwargs.items() if k != "n_cpu"}
    pbar_kwargs["disable"] = not pbar_kwargs.pop('verbose', False)

    # Precompute slot bounds (constant across all timesteps)
    min_slot = np.minimum(slot_i_start, slot_i_end)
    max_slot = np.maximum(slot_i_start, slot_i_end)

    init_args = (
        type(file_handler).setup_interpolator,
        file_handler.extra_data,
        file_handler.mass_density,
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


class _Cells:
    """
    A flat set of spherical cells, each given by its own coordinate bounds.

    ``lo`` and ``hi`` are ``(D, n_cells)``: the two bounds of every cell along
    every axis, ``(r, cos(theta), phi)`` for a volume and ``(cos(theta), phi)``
    on the sphere at ``r_surf``. Nothing here knows how the cells are arranged,
    so one global grid, a union of meshblocks and an AMR hierarchy are all the
    same object, and so is the interpolated helper grid. Cells are assumed not
    to overlap; that is the hook's contract, and it is what stops a cell's mass
    being counted twice.
    """

    def __init__(self, lo, hi, r_surf: float | None = None, label: str | None = None):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        if self.lo.shape != self.hi.shape:
            raise ValueError(
                f"Cell bounds disagree in shape: {self.lo.shape} against {self.hi.shape}."
            )
        if self.lo.shape[1] == 0:
            raise ValueError("No cells to sample from in the requested region.")
        self.r_surf = r_surf
        self.label = label

    @property
    def n_cells(self) -> int:
        return self.lo.shape[1]

    def measure(self) -> np.ndarray:
        """Cell volume, or cell area at ``r_surf``."""
        return cell_measure(self.lo, self.hi, self.r_surf)

    def centres(self) -> np.ndarray:
        """Cartesian cell centres, shape (3, n_cells)."""
        return cell_centres(self.lo, self.hi, self.r_surf)

    def sample_positions(self, idx: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Cartesian positions for the drawn cells ``idx``, placed inside their
        own cell by the uniform deviates ``u`` (shape ``(D, len(idx))``).
        """
        lo, hi = self.lo[:, idx], self.hi[:, idx]
        if self.r_surf is not None:
            r = self.r_surf
        else:
            # Uniform in r**3, so uniform in volume rather than in radius.
            r = (lo[0]**3 + u[0] * (hi[0]**3 - lo[0]**3)) ** (1 / 3)
        return sph_to_cart(r,
                           lo[-2] + u[-2] * (hi[-2] - lo[-2]),
                           lo[-1] + u[-1] * (hi[-1] - lo[-1]))

    def describe(self) -> str:
        return self.label or f"{self.n_cells}-cell"


def _grid_cells(*edges: np.ndarray, r_surf: float | None = None) -> _Cells:
    """A ``_Cells`` over one separable grid, labelled by its shape."""
    lo, hi = tensor_cell_bounds(*edges)
    return _Cells(lo, hi, r_surf=r_surf,
                  label="x".join(str(len(e) - 1) for e in edges))


def _native_cells(file_handler, slot, ranges, r_surf=None):
    """
    Ask the format for its own cells and their masses, and cut them to exactly
    ``ranges`` -- one ``(lo, hi)`` pair per axis of the bounds.

    Cells wholly inside are kept, cells wholly outside dropped, and a cell a
    limit passes through is cut at that limit, its mass scaled by the fraction
    of its measure left. The native value is constant across a cell, so that
    fraction is exact. The region's edge is then the limit itself rather than
    whichever cell face happens to lie near it, which is what makes a volume
    seeded out to R and a surface seeded at R share exactly one boundary.

    The spherical measure factorises as ``r**3/3 * cos(theta) * phi``, so the
    fraction is a product over axes: a ratio of ``r**3`` differences on the
    radial one, of plain differences on the angular ones.
    """
    lo, hi, weights = file_handler.native_cell_weights(slot, surface_radius=r_surf)
    lo, hi = lo.copy(), hi.copy()
    frac = np.ones(lo.shape[1])
    for axis, (a, b) in enumerate(ranges):
        a, b = min(a, b), max(a, b)
        cut_lo, cut_hi = np.maximum(lo[axis], a), np.minimum(hi[axis], b)
        # Volume bounds carry r first; surface bounds are angular only.
        p = 3 if (r_surf is None and axis == 0) else 1
        with np.errstate(divide='ignore', invalid='ignore'):
            frac *= np.where(cut_hi > cut_lo,
                             (cut_hi**p - cut_lo**p) / (hi[axis]**p - lo[axis]**p),
                             0.0)
        lo[axis], hi[axis] = cut_lo, cut_hi
    keep = frac > 0
    if not keep.any():
        raise ValueError(
            "No native cell falls in the requested region; check --r-min/--r-max "
            "and the angular limits against the data's extent."
        )
    cells = _Cells(lo[:, keep], hi[:, keep], r_surf=r_surf)
    return cells, weights[keep] * frac[keep]


def _helper_weights(file_handler, i_loc, cells, radial: bool) -> np.ndarray:
    """
    The same per-cell masses, for a format with no native grid: interpolate the
    fields its ``mass_density`` asks for onto the helper cells' centres.

    The density decision still belongs to the handler here. Only the grid the
    weights are built on differs between this and the native route.
    """
    md = file_handler.mass_density
    keys = md.flux_keys if radial else md.density_keys
    pos_c = cells.centres()
    interp = _sampling_interpolator(file_handler, i_loc, keys)
    interp.load()
    try:
        vals = interp(pos_c)
    finally:
        interp.unload()
    dens = md.radial_flux(vals, pos_c) if radial else md.density(vals, pos_c)
    return dens * cells.measure()


# The coordinates every seeder in this module works in: regions are given in
# (r, theta, phi), tracers are placed uniformly in r**3, cos(theta) and phi, and
# a format's native cell bounds are read as those axes. Compared against the
# file handler's own grid_geometry before any native cell is trusted.
GRID_GEOMETRY = 'spherical'


def _resolve_weight_grid(file_handler, weight_grid: str) -> bool:
    """
    Whether to weight on the format's own grid.

    'native' insists, and raises if the format cannot enumerate its cells or
    describes them in other coordinates than these seeders place tracers in;
    'auto' asks and falls back to the helper grid in either case; 'helper' never
    asks. The helper grid is exact in any geometry, since it hands the format's
    interpolator Cartesian points and never reads the format's cells.
    """
    if weight_grid == 'helper':
        return False
    name = type(file_handler).__name__
    # getattr, not attribute access: handlers are duck-typed in places and need
    # not derive from FileHandler at all.
    impl = getattr(type(file_handler), 'native_cell_weights', None)
    if impl is None or impl is FileHandler.native_cell_weights:
        if weight_grid == 'native':
            raise NotImplementedError(
                f"--weight-grid native: {name} does not implement native_cell_weights."
            )
        return False
    geometry = getattr(file_handler, 'grid_geometry', None)
    if geometry != GRID_GEOMETRY:
        if weight_grid == 'native':
            raise ValueError(
                f"--weight-grid native: {name} describes its cells in "
                f"{geometry!r} coordinates, but the seeders place tracers in "
                f"{GRID_GEOMETRY!r} ones, so its cell bounds would be misread. "
                "Use --weight-grid helper, which works in any geometry."
            )
        print(f"{name} describes its cells in {geometry!r} coordinates, not "
              f"{GRID_GEOMETRY!r}; weighting on the interpolated helper grid.")
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
    cells_per_tracer: int = 8,
    weight_grid: str = 'auto',
    **kwargs
    ) -> Tracers:
    """
    Seed tracers by sampling the spherical shell with the mass density as
    weight, so every tracer carries the same mass ``M_tot / n_tracers``.

    What counts as the mass density is the file handler's decision, not this
    function's: ``file_handler.mass_density`` answers it, so a run that asked
    for the conserved ``D = rho W sqrt(gamma)`` and one that wants plain
    ``rho`` take the same path here (see :mod:`src.mass`).

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

    i_ft = int(np.argmin(np.abs(np.asarray(file_handler.times) - start_t)))
    file_handler.load_chunk(i_ft, forward=True)
    i_loc = int(np.argmin(np.abs(file_handler.cur_times - start_t)))

    if native:
        cells, weights = _native_cells(
            file_handler, i_loc,
            ranges=((r_min, r_max),
                    (np.cos(theta_max), np.cos(theta_min)),
                    (phi_min, phi_max)),
        )
    else:
        n_r, n_th, n_ph = _auto_grid(
            cells_per_tracer * n_tracers,
            (np.log(r_max / r_min), theta_max - theta_min, phi_max - phi_min),
        )
        cells = _grid_cells(
            np.geomspace(r_min, r_max, n_r + 1),
            np.linspace(np.cos(theta_min), np.cos(theta_max), n_th + 1),
            np.linspace(phi_min, phi_max, n_ph + 1),
        )
        weights = _helper_weights(file_handler, i_loc, cells, radial=False)

    idx, signs, m_tot = _sample_cells(weights, n_tracers, "mass")

    positions = cells.sample_positions(
        idx, np.random.uniform(size=(3, n_tracers))).T

    masses = signs * (m_tot / n_tracers)
    print(f"Sampled {n_tracers} tracers of mass {m_tot/n_tracers:.4e} from a "
          f"{cells.describe()} {'native' if native else 'interpolated'} weight grid "
          f"(total mass {masses.sum():.4e} of {np.nansum(weights):.4e} on the grid, "
          f"weighted by {file_handler.mass_density.describe()}).")
    return Tracers(
        positions=positions,
        times=np.full(n_tracers, start_t),
        props=[{'mass': float(m)} for m in masses],
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
    cells_per_tracer, weight_grid
        As in ``spherical_by_volume_mc``.
    **kwargs
        Passed to ``Tracers``; must include ``file_handler`` and ``vel_keys``.

    Returns
    -------
    Tracers
        Tracer collection with ``n_tracers`` tracers of equal mass magnitude
        and the sign of their own flux.
    """
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

    ranges = ((np.cos(theta_max), np.cos(theta_min)), (phi_min, phi_max))

    if native:
        # The data's own cells set the angular resolution. The sphere stays at
        # exactly r_surf, so it is the outer boundary of a volume seeded to the
        # same radius. One probe load to fix the cell layout; the fluxes are
        # re-read per time step below.
        i_probe = int(np.argmin(np.abs(np.asarray(file_handler.times) - t_start[0])))
        file_handler.load_chunk(i_probe, forward=True)
        cells, _ = _native_cells(file_handler, 0, ranges, r_surf=r_surf)
    else:
        n_th, n_ph = _auto_grid(
            max(cells_per_tracer * n_tracers // n_t, 4),
            (theta_max - theta_min, phi_max - phi_min),
        )
        cells = _grid_cells(
            np.linspace(np.cos(theta_min), np.cos(theta_max), n_th + 1),
            np.linspace(phi_min, phi_max, n_ph + 1),
            r_surf=r_surf,
        )

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
                    _, flux = _native_cells(
                        file_handler, i_loc, ranges, r_surf=r_surf)
                    if flux.size != n_cells:
                        raise ValueError(
                            f"The cell layout changed between snapshots "
                            f"({flux.size} cells against {n_cells} at the probe); "
                            "a weight grid can only be built on a static mesh."
                        )
                else:
                    flux = _helper_weights(file_handler, i_loc, cells, radial=True)
                weights[i_t[0]] = flux * dt[i_t[0]]
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
          f"{np.nansum(weights):.4e} on the grid, {m_tot:.4e} crossing either way, "
          f"weighted by {file_handler.mass_density.describe()}).")
    return Tracers(
        positions=positions,
        times=times,
        props=[{'mass': float(m)} for m in masses],
        **kwargs
    )
