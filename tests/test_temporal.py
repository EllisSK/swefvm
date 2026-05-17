import numpy as np
import pytest

from swefvm.core.mesh import Mesh1D
from swefvm.core.boundaries import ReflectiveBoundary
from swefvm.physics.shallow_water import ShallowWater1D
from swefvm.methods.spatial import MUSCL1D
from swefvm.methods.temporal import FirstOrderTemporal, RK2
from swefvm.methods.riemann_solvers import HLLSolver


def _lake_at_rest_mesh():
    return Mesh1D(
        length=10.0,
        resolution=1.0,
        initial_conditions=lambda x: np.column_stack([np.full_like(x, 1.0), np.zeros_like(x)]),
    )


def test_first_order_temporal_preserves_lake_at_rest():
    mesh = _lake_at_rest_mesh()
    physics = ShallowWater1D(dx=mesh.dx)
    spatial = MUSCL1D()
    riemann = HLLSolver()
    integrator = FirstOrderTemporal()
    bcs = [ReflectiveBoundary(0), ReflectiveBoundary(mesh.N + 1)]
    mesh.apply_external_boundary_conditions(bcs)
    Q_before = mesh.Q_array.copy()
    integrator.integrate(mesh, physics, spatial, riemann, bcs, [], dt=0.01)
    np.testing.assert_allclose(mesh.Q_array[1:-1], Q_before[1:-1], atol=1e-12)


def test_rk2_preserves_lake_at_rest():
    mesh = _lake_at_rest_mesh()
    physics = ShallowWater1D(dx=mesh.dx)
    spatial = MUSCL1D()
    riemann = HLLSolver()
    integrator = RK2()
    bcs = [ReflectiveBoundary(0), ReflectiveBoundary(mesh.N + 1)]
    mesh.apply_external_boundary_conditions(bcs)
    Q_before = mesh.Q_array.copy()
    integrator.integrate(mesh, physics, spatial, riemann, bcs, [], dt=0.01)
    np.testing.assert_allclose(mesh.Q_array[1:-1], Q_before[1:-1], atol=1e-12)


def test_first_order_temporal_changes_state_when_non_uniform():
    mesh = Mesh1D(
        length=10.0,
        resolution=1.0,
        initial_conditions=lambda x: np.column_stack([
            np.where(x < 5.0, 2.0, 1.0),
            np.zeros_like(x),
        ]),
    )
    physics = ShallowWater1D(dx=mesh.dx)
    bcs = [ReflectiveBoundary(0), ReflectiveBoundary(mesh.N + 1)]
    mesh.apply_external_boundary_conditions(bcs)
    Q_before = mesh.Q_array.copy()
    FirstOrderTemporal().integrate(mesh, physics, MUSCL1D(), HLLSolver(), bcs, [], dt=0.01)
    assert not np.allclose(mesh.Q_array, Q_before)
