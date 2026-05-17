import numpy as np
import pytest

from swefvm.physics.shallow_water import ShallowWater


class _FakeMesh:
    def __init__(self, zb, dx=1.0):
        self.zb = np.asarray(zb, dtype=float)
        self.directions = (0,)
        self._dx = dx

    def spacing(self, d):
        return self._dx


@pytest.fixture
def physics():
    return ShallowWater()


def test_flux_zero_momentum_flat_bed(physics):
    Q = np.array([[1.0, 0.0], [1.0, 0.0]])
    zb = np.array([0.0, 0.0])
    F = physics.flux(Q, zb)
    np.testing.assert_allclose(F[:, 0], 0.0)
    np.testing.assert_allclose(F[:, 1], 0.5 * 9.81 * 1.0 ** 2)


def test_flux_with_momentum(physics):
    Q = np.array([[2.0, 4.0]])
    zb = np.array([0.0])
    F = physics.flux(Q, zb)
    h, q, u = 2.0, 4.0, 2.0
    np.testing.assert_allclose(F[0, 0], q)
    np.testing.assert_allclose(F[0, 1], q * u + 0.5 * 9.81 * h ** 2)


def test_flux_dry_cell_no_division_error(physics):
    Q = np.array([[0.0, 0.0]])
    zb = np.array([0.0])
    F = physics.flux(Q, zb)
    assert np.all(np.isfinite(F))
    np.testing.assert_allclose(F[0, 0], 0.0)


def test_source_flat_bed_no_friction_zero(physics):
    Q = np.array([[1.0, 0.5], [1.0, 0.5], [1.0, 0.5]])
    mesh = _FakeMesh(zb=[0.0, 0.0, 0.0])
    S = physics.source(Q, mesh, mannings_n=0.0)
    np.testing.assert_allclose(S, 0.0)


def test_source_sloped_bed_produces_momentum_term(physics):
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    zb = 0.1 * x
    Q = np.column_stack([np.full_like(x, 1.0), np.zeros_like(x)])
    mesh = _FakeMesh(zb=zb)
    S = physics.source(Q, mesh, mannings_n=0.0)
    np.testing.assert_allclose(S[:, 0], 0.0)
    np.testing.assert_allclose(S[:, 1], -9.81 * 1.0 * 0.1)


def test_source_friction_opposes_flow(physics):
    Q = np.array([[1.0, 1.0], [1.0, 1.0]])
    mesh = _FakeMesh(zb=[0.0, 0.0])
    S_no = physics.source(Q, mesh, mannings_n=0.0)
    S_fr = physics.source(Q, mesh, mannings_n=0.03)
    assert S_fr[0, 1] < S_no[0, 1]


def test_max_wave_speed_zero_velocity(physics):
    Q = np.array([[1.0, 0.0], [4.0, 0.0]])
    zb = np.array([0.0, 0.0])
    speed = physics.max_wave_speed(Q, zb)
    np.testing.assert_allclose(speed, np.sqrt(9.81 * 4.0))


def test_max_wave_speed_with_flow(physics):
    Q = np.array([[1.0, 5.0]])
    zb = np.array([0.0])
    speed = physics.max_wave_speed(Q, zb)
    expected = 5.0 + np.sqrt(9.81 * 1.0)
    np.testing.assert_allclose(speed, expected)


def test_dynamic_timestep_respects_cfl(physics):
    Q = np.array([[1.0, 0.0], [1.0, 0.0]])
    mesh = _FakeMesh(zb=[0.0, 0.0], dx=1.0)
    dt = physics.dynamic_timestep(Q, mesh)
    max_speed = physics.max_wave_speed(Q, mesh.zb)
    assert dt < mesh.spacing(0) / max_speed
    assert dt == pytest.approx(mesh.spacing(0) / max_speed, rel=1e-10)