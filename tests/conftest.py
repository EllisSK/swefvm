import numpy as np
import pytest

from swefvm.core.mesh import Mesh1D, Mesh2D
from swefvm.physics.shallow_water import ShallowWater


@pytest.fixture
def flat_initial_conditions():
    def ic(x):
        eta = np.full_like(x, 1.0)
        q = np.zeros_like(x)
        return np.stack([eta, q], axis=1)
    return ic


@pytest.fixture
def linear_bed():
    def bed(x):
        return 0.1 * x
    return bed


@pytest.fixture
def mesh_flat(flat_initial_conditions):
    return Mesh1D(length=10.0, resolution=1.0, initial_conditions=flat_initial_conditions)


@pytest.fixture
def mesh_sloped(flat_initial_conditions, linear_bed):
    return Mesh1D(length=10.0, resolution=1.0, initial_conditions=flat_initial_conditions, bed_function=linear_bed)


@pytest.fixture
def physics_flat(mesh_flat):
    return ShallowWater()


@pytest.fixture
def flat_initial_conditions_2d():
    def ic(X, Y):
        eta = np.full_like(X, 1.0)
        qx = np.zeros_like(X)
        qy = np.zeros_like(X)
        return np.stack([eta, qx, qy], axis=-1)
    return ic


@pytest.fixture
def linear_bed_2d():
    def bed(X, Y):
        return 0.1 * X + 0.05 * Y
    return bed


@pytest.fixture
def mesh2d_flat(flat_initial_conditions_2d):
    return Mesh2D(width=10.0, height=8.0, resolution=1.0, initial_conditions=flat_initial_conditions_2d)


@pytest.fixture
def mesh2d_sloped(flat_initial_conditions_2d, linear_bed_2d):
    return Mesh2D(width=10.0, height=8.0, resolution=1.0, initial_conditions=flat_initial_conditions_2d, bed_function=linear_bed_2d)