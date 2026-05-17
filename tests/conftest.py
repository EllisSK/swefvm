import numpy as np
import pytest

from swefvm.core.mesh import Mesh1D
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