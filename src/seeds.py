"""
Methods to seed initial tracer positions and times.
  Should also at least assign a volume element of to each tracer for later use in integration and analysis.
"""

import numpy as np
from .tracers import Tracers
from .integrators import IntegratorBase
from .file import FileHandler


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
        **kwargs: additional keyword arguments to pass to Tracers constructor
    """

    n_tracers = n_r * n_th * n_ph
    times = np.full(n_tracers, start_t)
    r_start = np.geomspace(r_min, r_max, n_r+1)
    dr = np.diff(r_start)
    r_start = r_start[:-1] + dr/2

    # th_start = np.linspace(theta_min, theta_max, n_th+1)
    # dth = np.diff(th_start)
    # th_start = th_start[:-1] + dth/2
    costheta_start = np.linspace(np.cos(theta_min), np.cos(theta_max), n_th+1)
    th_start = np.sort(np.arccos(costheta_start))
    dth = np.diff(th_start)
    th_start = th_start[:-1] + dth/2

    ph_start = np.linspace(phi_min, phi_max, n_ph, endpoint=False)
    dph = np.full(n_ph, 2*np.pi/n_ph)
    ph_start = ph_start + dph/2

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
    props = [{'dV': v} for v in dV]

    x_start = r_start*np.sin(th_start)*np.cos(ph_start)
    y_start = r_start*np.sin(th_start)*np.sin(ph_start)
    z_start = r_start*np.cos(th_start)

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
    **kwargs
    ) -> Tracers:
    """
      Seed tracers on a spherical surface, with uniform spacing in theta and phi, such that each tracer represents the same area element.
        Parameters:
            r_surf: radius of the spherical surface
            t_min: minimum initial time
            t_max: maximum initial time
            n_t: number of time bins
            n_th: number of theta bins
            n_ph: number of phi bins
            start_t: initial time for all tracers (overrides t_min and t_max if specified)
            phi_min: minimum phi angle (default 0)
            phi_max: maximum phi angle (default 2*pi)
            theta_min: minimum theta angle (default 0)
            theta_max: maximum theta angle (default pi)
            random_shift_in_cell: whether to randomly shift tracer positions within their initial cell to avoid grid artifacts (default True)
            **kwargs: additional keyword arguments to pass to Tracers constructor
    """

    n_t = len(t_start)
    n_tracers = n_t * n_th * n_ph


    dt = np.zeros_like(t_start)
    ddt = np.diff(t_start)
    dt[1:-1] = (ddt[:-1] + ddt[1:])/2
    dt[0] = ddt[0]/2
    dt[-1] = ddt[-1]/2

    costheta_start = np.linspace(np.cos(theta_min), np.cos(theta_max), n_th+1)
    th_start = np.sort(np.arccos(costheta_start))
    dth = np.diff(th_start)
    th_start = th_start[:-1] + dth/2

    ph_start = np.linspace(phi_min, phi_max, n_ph, endpoint=False)
    dph = np.full(n_ph, 2*np.pi/n_ph)
    ph_start = ph_start + dph/2

    t_start, th_start, ph_start = np.meshgrid(t_start, th_start, ph_start, indexing='ij')
    dt, dth, dph = np.meshgrid(dt, dth, dph, indexing='ij')
    dAdt = r_surf**2 * np.sin(th_start) * dth * dph * dt

    if random_shift_in_cell:
        th_start += np.random.uniform(-dth/2, dth/2)
        ph_start += np.random.uniform(-dph/2, dph/2)

    r_start = np.full_like(t_start, r_surf)

    r_start = r_start.flatten()
    t_start = t_start.flatten()
    th_start = th_start.flatten()
    ph_start = ph_start.flatten()
    dAdt = dAdt.flatten()
    props = [{'dAdt': a} for a in dAdt]

    x_start = r_start*np.sin(th_start)*np.cos(ph_start)
    y_start = r_start*np.sin(th_start)*np.sin(ph_start)
    z_start = r_start*np.cos(th_start)

    return Tracers(
        positions=np.array([x_start, y_start, z_start]).T,
        times=t_start,
        props=props,
        **kwargs
        )
