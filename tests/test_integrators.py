"""
Unit tests for src/integrators.

Tests are independent of file I/O and shared memory: integrators receive
plain Python callables as interpolators and numpy arrays as positions.
"""

import numpy as np
import pytest

from src.integrators import ExplicitTrapezoid, ImplicitTrapezoid, RK4
from src.integrators.base import IntegratorBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convergence_order(integrator, ode_rhs, x0, T, n_snap, tol_order):
    """
    Integrate dx/dt = ode_rhs(x) from 0 to T at successive halvings of dt and
    return the observed order (log2 ratio of consecutive errors).
    """
    exact = np.exp(-T) * x0  # valid for ode_rhs(x) = -x

    orders = []
    prev_err = None
    for k in range(3, 7):
        dt = T / 2**k
        n = 2**k
        x = x0.copy()
        t = 0.0
        v = lambda pos: -pos
        interps = tuple(v for _ in range(n_snap))
        for _ in range(n):
            st = np.array([t + i * dt for i in range(-1, n_snap - 1)])
            x = integrator(x, dt, interps, snap_times=st)
            t += dt
        err = float(np.max(np.abs(x - exact)))
        if prev_err is not None:
            orders.append(np.log2(prev_err / err))
        prev_err = err

    assert all(o > tol_order for o in orders), (
        f"Convergence orders {orders} below threshold {tol_order}"
    )
    return orders


# ---------------------------------------------------------------------------
# IntegratorBase
# ---------------------------------------------------------------------------

class TestIntegratorBase:
    def test_n_snapshots_defaults(self):
        assert IntegratorBase.n_snapshots == 2

    def test_explicit_trapezoid_n_snapshots(self):
        assert ExplicitTrapezoid.n_snapshots == 2

    def test_implicit_trapezoid_n_snapshots(self):
        assert ImplicitTrapezoid.n_snapshots == 2

    def test_rk4_n_snapshots(self):
        assert RK4.n_snapshots == 4


# ---------------------------------------------------------------------------
# ExplicitTrapezoid
# ---------------------------------------------------------------------------

class TestExplicitTrapezoid:
    def setup_method(self):
        self.integrator = ExplicitTrapezoid()

    def test_uniform_velocity_no_motion(self):
        # v = 0 everywhere -> position unchanged
        v_zero = lambda x: np.zeros_like(x)
        x0 = np.array([[1.0, 2.0, 3.0]])
        x1 = self.integrator(x0, 0.1, (v_zero, v_zero))
        np.testing.assert_allclose(x1, x0)

    def test_constant_velocity(self):
        # v = c -> x_new = x0 + c*dt (exact for any 2nd-order scheme)
        c = 2.5
        v = lambda x: np.full_like(x, c)
        x0 = np.array([[0.0]])
        dt = 0.3
        x1 = self.integrator(x0, dt, (v, v))
        np.testing.assert_allclose(x1, [[c * dt]], rtol=1e-12)

    def test_second_order_convergence(self):
        # dx/dt = -x, exact x(T) = exp(-T)
        x0 = np.array([[1.0]])
        _convergence_order(self.integrator, lambda x: -x, x0, 1.0, 2, tol_order=1.9)

    def test_batch_of_tracers(self):
        # Shape (3, n_tracers) - all tracers should advance independently
        v = lambda x: np.ones_like(x)
        x0 = np.zeros((3, 5))
        dt = 0.5
        x1 = self.integrator(x0, dt, (v, v))
        assert x1.shape == (3, 5)
        np.testing.assert_allclose(x1, np.ones((3, 5)) * dt, rtol=1e-12)

    def test_snap_times_ignored(self):
        # snap_times is accepted but has no effect
        v = lambda x: -x
        x0 = np.array([[1.0]])
        st = np.array([0.0, 0.1])
        x1_with = self.integrator(x0, 0.1, (v, v), snap_times=st)
        x1_without = self.integrator(x0, 0.1, (v, v), snap_times=None)
        np.testing.assert_allclose(x1_with, x1_without)


# ---------------------------------------------------------------------------
# ImplicitTrapezoid
# ---------------------------------------------------------------------------

class TestImplicitTrapezoid:
    def setup_method(self):
        self.integrator = ImplicitTrapezoid(tol=1e-12, max_iter=50)

    def test_uniform_velocity_no_motion(self):
        v_zero = lambda x: np.zeros_like(x)
        x0 = np.array([[1.0, 2.0]])
        x1 = self.integrator(x0, 0.1, (v_zero, v_zero))
        np.testing.assert_allclose(x1, x0)

    def test_constant_velocity(self):
        c = 2.5
        v = lambda x: np.full_like(x, c)
        x0 = np.array([[0.0]])
        dt = 0.3
        x1 = self.integrator(x0, dt, (v, v))
        np.testing.assert_allclose(x1, [[c * dt]], rtol=1e-10)

    def test_second_order_convergence(self):
        x0 = np.array([[1.0]])
        _convergence_order(self.integrator, lambda x: -x, x0, 1.0, 2, tol_order=1.9)

    def test_convergence_flag_set_on_success(self):
        v = lambda x: -x
        x0 = np.array([[1.0]])
        self.integrator(x0, 0.01, (v, v))
        assert self.integrator.converged is True

    def test_convergence_flag_unset_on_failure(self):
        # max_iter=1 and large dt forces non-convergence for a stiff problem
        integrator = ImplicitTrapezoid(tol=1e-15, max_iter=1)
        v = lambda x: -100 * x  # stiff
        x0 = np.array([[1.0]])
        integrator(x0, 1.0, (v, v))
        assert integrator.converged is False

    def test_n_iter_updated(self):
        v = lambda x: -x
        x0 = np.array([[1.0]])
        self.integrator(x0, 0.1, (v, v))
        assert self.integrator.n_iter >= 1

    def test_relax_parameter_accepted(self):
        # Relaxation factor in (0,1] - just check it runs and returns finite result
        integrator = ImplicitTrapezoid(tol=1e-8, max_iter=50, relax=0.5)
        v = lambda x: -x
        x0 = np.array([[1.0]])
        x1 = integrator(x0, 0.1, (v, v))
        assert np.isfinite(x1).all()

    def test_snap_times_ignored(self):
        v = lambda x: -x
        x0 = np.array([[1.0]])
        st = np.array([0.0, 0.1])
        x1_with = self.integrator(x0, 0.1, (v, v), snap_times=st)
        x1_without = self.integrator(x0, 0.1, (v, v), snap_times=None)
        np.testing.assert_allclose(x1_with, x1_without)


# ---------------------------------------------------------------------------
# RK4
# ---------------------------------------------------------------------------

class TestRK4:
    def test_fourth_order_convergence_monotone(self):
        x0 = np.array([[1.0]])
        _convergence_order(RK4(monotone=True), lambda x: -x, x0, 1.0, 4, tol_order=3.9)

    def test_fourth_order_convergence_lagrange(self):
        x0 = np.array([[1.0]])
        _convergence_order(RK4(monotone=False), lambda x: -x, x0, 1.0, 4, tol_order=3.9)

    def test_uniform_velocity_no_motion(self):
        v_zero = lambda x: np.zeros_like(x)
        x0 = np.array([[1.0, 2.0]])
        x1 = RK4()(x0, 0.1, (v_zero,) * 4)
        np.testing.assert_allclose(x1, x0)

    def test_constant_velocity(self):
        c = 3.0
        v = lambda x: np.full_like(x, c)
        x0 = np.array([[0.0]])
        dt = 0.2
        x1 = RK4()(x0, dt, (v,) * 4)
        np.testing.assert_allclose(x1, [[c * dt]], rtol=1e-12)

    def test_snap_times_uniform_matches_none(self):
        # Explicit uniform snap_times should give same result as snap_times=None
        v = lambda x: -x
        x0 = np.array([[1.0]])
        dt = 0.1
        st = np.array([-dt, 0.0, dt, 2 * dt])
        rk4 = RK4(monotone=True)
        x1_none = rk4(x0, dt, (v,) * 4, snap_times=None)
        x1_st   = rk4(x0, dt, (v,) * 4, snap_times=st)
        np.testing.assert_allclose(x1_none, x1_st, rtol=1e-12)

    def test_snap_times_nonuniform_accepted(self):
        # Non-uniform snap_times must not raise and must return finite result
        v = lambda x: -x
        x0 = np.array([[1.0]])
        st = np.array([-0.2, 0.0, 0.1, 0.3])  # non-uniform
        x1 = RK4(monotone=True)(x0, 0.1, (v,) * 4, snap_times=st)
        assert np.isfinite(x1).all()

    def test_monotone_clamps_temporal_spike(self):
        # Velocity that spikes outside [v1, v2] in the outer snapshots.
        # PCHIP must not introduce a midpoint velocity outside [v1, v2].
        from src.integrators.rk4 import _pchip_v_mid
        v0 = np.array([[4.0]])  # large spike at t_{n-1}
        v1 = np.array([[0.0]])
        v2 = np.array([[0.0]])
        v3 = np.array([[0.0]])
        st = np.array([-1.0, 0.0, 1.0, 2.0])
        v_mid = _pchip_v_mid(v0, v1, v2, v3, st)
        assert float(v_mid.flat[0]) == pytest.approx(0.0, abs=1e-12)

    def test_monotone_no_undershoot_u_shape(self):
        from src.integrators.rk4 import _pchip_v_mid
        # U-shape: endpoints high, middle low - Lagrange can go negative
        v0 = np.array([[1.0]])
        v1 = np.array([[0.0]])
        v2 = np.array([[0.0]])
        v3 = np.array([[1.0]])
        st = np.array([-1.0, 0.0, 1.0, 2.0])
        v_mid = _pchip_v_mid(v0, v1, v2, v3, st)
        assert float(v_mid.flat[0]) >= 0.0  # must not go below min(v1, v2) = 0

    def test_lagrange_weights_uniform_known_values(self):
        from src.integrators.rk4 import _lagrange_weights
        st = np.array([-1.0, 0.0, 1.0, 2.0])
        w = _lagrange_weights(st)
        expected = np.array([-1/16, 9/16, 9/16, -1/16])
        np.testing.assert_allclose(w, expected, rtol=1e-12)

    def test_lagrange_weights_sum_to_one(self):
        from src.integrators.rk4 import _lagrange_weights
        # Weights must always sum to 1 (partition of unity)
        for st in [
            np.array([-1.0, 0.0, 1.0, 2.0]),
            np.array([-0.5, 0.0, 0.3, 0.8]),
            np.array([-2.0, 0.0, 1.0, 3.0]),
        ]:
            np.testing.assert_allclose(_lagrange_weights(st).sum(), 1.0, rtol=1e-12)

    def test_pchip_scale_invariant(self):
        # Doubling all time intervals must not change the midpoint value
        from src.integrators.rk4 import _pchip_v_mid
        v0 = np.array([[0.5]])
        v1 = np.array([[1.0]])
        v2 = np.array([[0.7]])
        v3 = np.array([[0.2]])
        st1 = np.array([0.0, 1.0, 2.0, 3.0])
        st2 = st1 * 2.0
        r1 = _pchip_v_mid(v0, v1, v2, v3, st1)
        r2 = _pchip_v_mid(v0, v1, v2, v3, st2)
        np.testing.assert_allclose(r1, r2, rtol=1e-12)

    def test_batch_of_tracers(self):
        v = lambda x: np.ones_like(x)
        x0 = np.zeros((3, 8))
        x1 = RK4()(x0, 0.5, (v,) * 4)
        assert x1.shape == (3, 8)
        np.testing.assert_allclose(x1, 0.5, rtol=1e-12)
