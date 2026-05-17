import numpy as np
import pytest

from swefvm.methods.riemann_solvers import HLLSolver, OsherSolomonSolver
from swefvm.physics.shallow_water import ShallowWater


@pytest.fixture
def physics():
    return ShallowWater()


def test_hll_uniform_state_returns_physics_flux(physics):
    Q = np.array([[1.0, 0.5], [1.0, 0.5], [1.0, 0.5]])
    zb = np.zeros(3)
    F = HLLSolver().solve(Q, Q.copy(), physics, zb)
    np.testing.assert_allclose(F, physics.flux(Q, zb))


def test_hll_output_shape_matches_input(physics):
    Q_L = np.array([[1.0, 0.0], [1.5, 0.2]])
    Q_R = np.array([[1.2, 0.1], [1.5, 0.2]])
    zb = np.zeros(2)
    F = HLLSolver().solve(Q_L, Q_R, physics, zb)
    assert F.shape == Q_L.shape


def test_hll_handles_dry_bed_without_nan(physics):
    Q_L = np.array([[0.0, 0.0]])
    Q_R = np.array([[1.0, 0.0]])
    zb = np.zeros(1)
    F = HLLSolver().solve(Q_L, Q_R, physics, zb)
    assert np.all(np.isfinite(F))


def test_hll_supersonic_left_uses_left_flux(physics):
    Q_L = np.array([[1.0, 50.0]])
    Q_R = np.array([[1.0, 50.0]])
    zb = np.zeros(1)
    F = HLLSolver().solve(Q_L, Q_R, physics, zb)
    np.testing.assert_allclose(F, physics.flux(Q_L, zb))


def test_osher_solomon_uniform_state_returns_physics_flux(physics):
    Q = np.array([[1.0, 0.0], [1.0, 0.0]])
    zb = np.zeros(2)
    F = OsherSolomonSolver().solve(Q, Q.copy(), physics, zb)
    np.testing.assert_allclose(F, physics.flux(Q, zb), atol=1e-12)


def test_osher_solomon_output_shape_matches_input(physics):
    Q_L = np.array([[1.0, 0.0], [1.5, 0.2]])
    Q_R = np.array([[1.2, 0.1], [1.5, 0.2]])
    zb = np.zeros(2)
    F = OsherSolomonSolver().solve(Q_L, Q_R, physics, zb)
    assert F.shape == Q_L.shape