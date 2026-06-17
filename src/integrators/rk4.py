"""
RK4 integrator with cubic time interpolation.

Uses four consecutive velocity snapshots (t_{n-1}, t_n, t_{n+1}, t_{n+2}) to
construct a cubic-in-time velocity field at the RK4 midpoint substeps.

Two blending modes are available (selected at construction time):

  monotone=False  -- cubic Lagrange interpolation.
                     4th-order accurate in smooth regions; can overshoot at shocks.

  monotone=True   -- PCHIP (Fritsch-Carlson) monotone cubic Hermite interpolation.
                     Guarantees no new extrema are introduced between snapshots,
                     at the cost of reducing to 3rd order near non-smooth features.
                     Recommended for flows with strong shocks (default).

Both modes support non-uniform snapshot spacing via the snap_times argument.

The four RK4 substep velocities are:
    k1  evaluated at t_n          -> interps[1]          (exact)
    k2  evaluated at t_n + dt/2   -> blended interpolant  (approx)
    k3  evaluated at t_n + dt/2   -> blended interpolant  (approx, different pos)
    k4  evaluated at t_{n+1}      -> interps[2]          (exact)

Lagrange weights at t_eval for nodes (t0, t1, t2, t3):
    w_j(t) = prod_{k!=j} (t - t_k) / (t_j - t_k)

PCHIP (Fritsch-Carlson 1980) derivative estimates at t1 and t2:
    h_i = t_{i+1} - t_i,  s_i = (v_{i+1} - v_i) / h_i
    d_j = (h_{j-1} + h_j) / ((2h_j + h_{j-1})/s_{j-1} + (h_j + 2h_{j-1})/s_j)
          when s_{j-1} and s_j share the same sign, else 0.
    Hermite cubic evaluated at alpha = (t_eval - t1) / h1.
"""

import numpy as np

from .base import InterpolatorCallable, IntegratorBase


def _lagrange_weights(snap_times: np.ndarray) -> np.ndarray:
    """
    Compute the four Lagrange basis weights at the midpoint of [t1, t2].

    Parameters
    ----------
    snap_times : array of shape (4,)
        Times [t0, t1, t2, t3] of the four snapshots.

    Returns
    -------
    w : array of shape (4,)
    """
    t0, t1, t2, t3 = snap_times
    t_eval = 0.5 * (t1 + t2)
    w0 = ((t_eval-t1)*(t_eval-t2)*(t_eval-t3)) / ((t0-t1)*(t0-t2)*(t0-t3))
    w1 = ((t_eval-t0)*(t_eval-t2)*(t_eval-t3)) / ((t1-t0)*(t1-t2)*(t1-t3))
    w2 = ((t_eval-t0)*(t_eval-t1)*(t_eval-t3)) / ((t2-t0)*(t2-t1)*(t2-t3))
    w3 = ((t_eval-t0)*(t_eval-t1)*(t_eval-t2)) / ((t3-t0)*(t3-t1)*(t3-t2))
    return np.array([w0, w1, w2, w3])


def _pchip_v_mid(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    snap_times: np.ndarray,
) -> np.ndarray:
    """
    PCHIP (Fritsch-Carlson) monotone cubic Hermite blend at the midpoint of
    [t1, t2], supporting non-uniform snapshot spacing.

    Parameters
    ----------
    v0, v1, v2, v3 : ndarray
        Velocity arrays at t0, t1, t2, t3.
    snap_times : array of shape (4,)
        Times [t0, t1, t2, t3].

    Returns
    -------
    ndarray
        Interpolated velocity at t_eval = (t1 + t2) / 2.
    """
    t0, t1, t2, t3 = snap_times
    h0 = t1 - t0
    h1 = t2 - t1
    h2 = t3 - t2

    # Secant slopes (velocity derivative approximations over each interval).
    s0 = (v1 - v0) / h0
    s1 = (v2 - v1) / h1
    s2 = (v3 - v2) / h2

    # Fritsch-Carlson derivative estimates (eq. 2.9 in Fritsch & Carlson 1980).
    # At t1: uses left interval h0 and right interval h1.
    # At t2: uses left interval h1 and right interval h2.
    def _fc_deriv(sa: np.ndarray, sb: np.ndarray,
                  ha: float, hb: float) -> np.ndarray:
        """
        Derivative estimate at the shared node between intervals ha (left, slope sa)
        and hb (right, slope sb).  Returns 0 wherever sa and sb differ in sign.
        Uses safe division to avoid warnings when a slope is exactly zero.
        """
        same_sign = sa * sb > 0
        safe_sa = np.where(same_sign, sa, 1.0)
        safe_sb = np.where(same_sign, sb, 1.0)
        denom = (2*hb + ha) / safe_sa + (hb + 2*ha) / safe_sb
        return np.where(same_sign, (ha + hb) / denom, 0.0)

    d1 = _fc_deriv(s0, s1, h0, h1)  # derivative at t1
    d2 = _fc_deriv(s1, s2, h1, h2)  # derivative at t2

    # Cubic Hermite basis functions at alpha = 0.5 (midpoint of [t1, t2]).
    # Derivative terms are scaled by h1 because d1, d2 are physical derivatives.
    alpha = 0.5
    h00 =  2*alpha**3 - 3*alpha**2 + 1   # = 0.5
    h10 =    alpha**3 - 2*alpha**2 + alpha  # = 0.125  (scale by h1)
    h01 = -2*alpha**3 + 3*alpha**2          # = 0.5
    h11 =    alpha**3 -   alpha**2          # = -0.125 (scale by h1)

    return h00*v1 + h10*h1*d1 + h01*v2 + h11*h1*d2


class RK4(IntegratorBase):
    """
    Classical RK4 integrator using cubic time interpolation for midpoint substeps.

    Parameters
    ----------
    monotone : bool, optional
        If True (default), use PCHIP monotone cubic Hermite interpolation in
        time.  Prevents velocity overshoot across shocks at the cost of
        dropping to ~3rd order near non-smooth features.
        If False, use cubic Lagrange interpolation (4th-order in smooth flows).

    Requires four velocity snapshots per step:
        interps[0] -> t_{n-1}
        interps[1] -> t_n       (step start)
        interps[2] -> t_{n+1}   (step end)
        interps[3] -> t_{n+2}
    """

    n_snapshots: int = 4

    def __init__(self, monotone: bool = True) -> None:
        self.monotone = monotone

    def __call__(
        self,
        xn: np.ndarray,
        dt: float,
        interps: tuple[InterpolatorCallable, ...],
        snap_times: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Perform a single RK4 step.

        Parameters
        ----------
        xn : ndarray, shape (d {,n})
            Current position(s).
        dt : float
            Integration timestep (from t_n to t_{n+1}).
        interps : tuple of 4 callables
            Velocity interpolators at t_{n-1}, t_n, t_{n+1}, t_{n+2}.
        snap_times : array of shape (4,), optional
            Actual times [t_{n-1}, t_n, t_{n+1}, t_{n+2}].  If None, uniform
            spacing is assumed (all intervals equal to dt).

        Returns
        -------
        x_new : ndarray, shape (d {,n})
        """
        if snap_times is None:
            st = np.array([-dt, 0.0, dt, 2*dt])
        else:
            st = snap_times

        if self.monotone:
            def v_mid(x: np.ndarray) -> np.ndarray:
                return _pchip_v_mid(
                    interps[0](x), interps[1](x), interps[2](x), interps[3](x),
                    st,
                )
        else:
            w = _lagrange_weights(st)

            def v_mid(x: np.ndarray) -> np.ndarray:
                return (w[0]*interps[0](x) + w[1]*interps[1](x)
                      + w[2]*interps[2](x) + w[3]*interps[3](x))

        k1 = interps[1](xn)
        k2 = v_mid(xn + 0.5 * dt * k1)
        k3 = v_mid(xn + 0.5 * dt * k2)
        k4 = interps[2](xn + dt * k3)

        return xn + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
