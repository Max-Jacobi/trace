"""
RK4 integrator with cubic Lagrange time interpolation.

Uses four consecutive velocity snapshots (t_{n-1}, t_n, t_{n+1}, t_{n+2}) to
construct a cubic-in-time velocity field, enabling true 4th-order accuracy in
both space and time.

The four RK4 substep velocities are:
    k1  evaluated at t_n          → interps[1]
    k2  evaluated at t_n + dt/2   → cubic blend of all four snapshots at α=0.5
    k3  evaluated at t_n + dt/2   → same blend, different position
    k4  evaluated at t_{n+1}      → interps[2]

Lagrange weights for α ∈ [0, 1] on the four-point stencil
(τ = -1, 0, 1, 2) normalised so that τ=0 → t_n and τ=1 → t_{n+1}:
    w0(α) = -α(α-1)(α-2)/6
    w1(α) =  (α+1)(α-1)(α-2)/2
    w2(α) = -(α+1)α(α-2)/2
    w3(α) =  (α+1)α(α-1)/6
"""

import numpy as np

from .base import InterpolatorCallable, IntegratorBase

# Pre-computed Lagrange weights at α = 0.5 (midpoint of [t_n, t_{n+1}])
_α = 0.5
_W_MID = np.array([
    -_α * (_α - 1) * (_α - 2) / 6,        # w0:  t_{n-1}
     (_α + 1) * (_α - 1) * (_α - 2) / 2,  # w1:  t_n
    -(_α + 1) * _α * (_α - 2) / 2,        # w2:  t_{n+1}
     (_α + 1) * _α * (_α - 1) / 6,        # w3:  t_{n+2}
])


class RK4(IntegratorBase):
    """
    Classical RK4 integrator using cubic Lagrange interpolation in time.

    Requires four velocity snapshots per step:
        interps[0] → t_{n-1}
        interps[1] → t_n       (step start)
        interps[2] → t_{n+1}   (step end)
        interps[3] → t_{n+2}
    """

    n_snapshots: int = 4

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
        w0, w1, w2, w3 = _W_MID

        def v_mid(x: np.ndarray) -> np.ndarray:
            """Cubic-in-time velocity at t_n + dt/2."""
            return (w0 * interps[0](x)
                  + w1 * interps[1](x)
                  + w2 * interps[2](x)
                  + w3 * interps[3](x))

        k1 = interps[1](xn)
        k2 = v_mid(xn + 0.5 * dt * k1)
        k3 = v_mid(xn + 0.5 * dt * k2)
        k4 = interps[2](xn + dt * k3)

        return xn + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
