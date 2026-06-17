"""
RK4 integrator with cubic time interpolation.

Uses four consecutive velocity snapshots (t_{n-1}, t_n, t_{n+1}, t_{n+2}) to
construct a cubic-in-time velocity field at the RK4 midpoint substeps.

Two blending modes are available (selected at construction time):

  monotone=False  — cubic Lagrange interpolation (pre-computed scalar weights).
                    4th-order accurate in smooth regions; can overshoot at shocks.

  monotone=True   — PCHIP (Fritsch-Carlson) monotone cubic Hermite interpolation.
                    Guarantees no new extrema are introduced between snapshots,
                    at the cost of reducing to 3rd order near non-smooth features.
                    Recommended for flows with strong shocks (default).

The four RK4 substep velocities are:
    k1  evaluated at t_n          → interps[1]          (exact)
    k2  evaluated at t_n + dt/2   → blended interpolant  (approx)
    k3  evaluated at t_n + dt/2   → blended interpolant  (approx, different pos)
    k4  evaluated at t_{n+1}      → interps[2]          (exact)

Lagrange weights for α ∈ [0, 1] on the four-point stencil
(τ = -1, 0, 1, 2) normalised so that τ=0 → t_n and τ=1 → t_{n+1}:
    w0(α) = -α(α-1)(α-2)/6
    w1(α) =  (α+1)(α-1)(α-2)/2
    w2(α) = -(α+1)α(α-2)/2
    w3(α) =  (α+1)α(α-1)/6

PCHIP at α = 0.5 (Hermite basis evaluated at midpoint):
    h00 = 0.5,  h10 = 0.125,  h01 = 0.5,  h11 = -0.125
    d1, d2 = Fritsch-Carlson derivative estimates at τ=0 and τ=1
    v_mid = 0.5·v1 + 0.125·d1 + 0.5·v2 - 0.125·d2
"""

import numpy as np

from .base import InterpolatorCallable, IntegratorBase

# ── Lagrange weights at α = 0.5 ───────────────────────────────────────────────
_α = 0.5
_W_MID = np.array([
    -_α * (_α - 1) * (_α - 2) / 6,        # w0:  t_{n-1}
     (_α + 1) * (_α - 1) * (_α - 2) / 2,  # w1:  t_n
    -(_α + 1) * _α * (_α - 2) / 2,        # w2:  t_{n+1}
     (_α + 1) * _α * (_α - 1) / 6,        # w3:  t_{n+2}
])

# ── PCHIP Hermite basis at α = 0.5 ────────────────────────────────────────────
# H(0.5) = h00·y1 + h10·d1 + h01·y2 + h11·d2
_H00 =  0.5    # = 2α³ - 3α² + 1
_H10 =  0.125  # = α³ - 2α² + α
_H01 =  0.5    # = -2α³ + 3α²
_H11 = -0.125  # = α³ - α²


def _pchip_v_mid(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
    """
    PCHIP (Fritsch-Carlson) monotone cubic Hermite blend at α = 0.5.

    Assumes uniform spacing (Δτ = 1) between the four snapshots.
    Operates element-wise on arrays of arbitrary shape.

    Parameters
    ----------
    v0, v1, v2, v3 : ndarray
        Velocity values at τ = -1, 0, 1, 2 (i.e. t_{n-1} … t_{n+2}).

    Returns
    -------
    ndarray
        Interpolated velocity at τ = 0.5 (midpoint of [t_n, t_{n+1}]).
    """
    # Secant slopes over each unit interval.
    s0 = v1 - v0
    s1 = v2 - v1
    s2 = v3 - v2

    # Fritsch-Carlson derivative estimate: harmonic mean of adjacent secants
    # when they share the same sign; zero (flat) otherwise to preserve monotonicity.
    def _fc_deriv(sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
        denom = sa + sb
        # Avoid division by zero; where denom==0 the result is 0 anyway.
        safe = np.where(denom == 0, 1.0, denom)
        return np.where(sa * sb > 0, 2.0 * sa * sb / safe, 0.0)

    d1 = _fc_deriv(s0, s1)  # derivative at t_n
    d2 = _fc_deriv(s1, s2)  # derivative at t_{n+1}

    return _H00 * v1 + _H10 * d1 + _H01 * v2 + _H11 * d2


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
        interps[0] → t_{n-1}
        interps[1] → t_n       (step start)
        interps[2] → t_{n+1}   (step end)
        interps[3] → t_{n+2}
    """

    n_snapshots: int = 4

    def __init__(self, monotone: bool = True) -> None:
        self.monotone = monotone

    def __call__(
        self,
        xn: np.ndarray,
        dt: float,
        interps: tuple[InterpolatorCallable, ...],
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

        Returns
        -------
        x_new : ndarray, shape (d {,n})
        """
        if self.monotone:
            def v_mid(x: np.ndarray) -> np.ndarray:
                """PCHIP monotone cubic velocity at t_n + dt/2."""
                return _pchip_v_mid(
                    interps[0](x), interps[1](x), interps[2](x), interps[3](x)
                )
        else:
            w0, w1, w2, w3 = _W_MID

            def v_mid(x: np.ndarray) -> np.ndarray:
                """Lagrange cubic velocity at t_n + dt/2."""
                return (w0 * interps[0](x)
                      + w1 * interps[1](x)
                      + w2 * interps[2](x)
                      + w3 * interps[3](x))

        k1 = interps[1](xn)
        k2 = v_mid(xn + 0.5 * dt * k1)
        k3 = v_mid(xn + 0.5 * dt * k2)
        k4 = interps[2](xn + dt * k3)

        return xn + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
