"""
Implicit trapezoid (Crank-Nicolson) fixed-step integrator.

Implements a Picard (fixed-point) iteration per step.

This is a standalone integrator that assumes separate interpolation functions
are provided for the two time levels.

Notes:
- No temporal interpolation at intermediate times is required because the
  scheme only needs velocities at the endpoints t_n and t_{n+1}.
"""
import numpy as np

from .base import InterpolatorCallable, IntegratorBase

class ImplicitTrapezoid(IntegratorBase):
    converged: bool = False
    n_iter: int = 0

    def __init__(self, tol: float = 1e-8, max_iter: int = 20, relax: float = 1.0):
        """
        Initialize the implicit trapezoid integrator.

        Parameters
        ----------
        tol : float, optional
            Relative convergence tolerance based on the Euclidean norm.
        max_iter : int, optional
            Maximum number of Picard iterations before returning the latest
            iterate.
        relax : float, optional
            Relaxation factor in ``(0, 1]`` applied to each Picard update.
        """
        self.tol = tol
        self.max_iter = max_iter
        self.relax = relax

    def __call__(
        self,
        xn: np.ndarray,
        dt: float,
        interps: tuple[InterpolatorCallable, ...],
        snap_times: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Perform a single implicit trapezoid update for a batch of tracers.

        Parameters
        ----------
        xn : ndarray
            Current position array with shape ``(d,)`` or ``(d, n)``.
        dt : float
            Time step from ``t_n`` to ``t_{n+1}``.
        interps : tuple of callable
            Pair of interpolators ``(interp_n, interp_n1)`` evaluated at the
            start and end snapshots.
        snap_times : ndarray or None, optional
            Unused, provided for interface compatibility.

        Returns
        -------
        ndarray
            Updated position array with the same shape as ``xn``.

        Notes
        -----
        Convergence information is stored on ``self.converged`` and
        ``self.n_iter``.
        """

        self.converged = False

        vn = interps[0](xn)

        xk = xn + dt * vn

        # Picard iterations: x_{k+1} = xn + 0.5*dt*(vn + v(tn1, x_k))
        for self.n_iter in range(1, self.max_iter + 1):
            v_k = interps[1](xk)
            x_candidate = xn + 0.5 * dt * (vn + v_k)
            delta = x_candidate - xk
            x_new = xk + self.relax * delta

            if np.linalg.norm(delta) <= self.tol * max(1.0, np.linalg.norm(x_new)):
                self.converged = True
                return x_new
            xk = x_new
        return xk
