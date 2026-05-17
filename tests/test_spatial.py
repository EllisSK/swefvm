import numpy as np
import pytest

from swefvm.methods.spatial import MUSCL1D, FirstOrder
from swefvm.core.mesh import Mesh1D


def test_muscl_minmod_opposite_signs_returns_zero():
    muscl = MUSCL1D()
    result = muscl._minmod(np.array([1.0, -2.0]), np.array([-1.0, 3.0]))
    np.testing.assert_array_equal(result, [0.0, 0.0])


def test_muscl_minmod_same_sign_returns_smaller_magnitude():
    muscl = MUSCL1D()
    result = muscl._minmod(np.array([2.0, -5.0]), np.array([3.0, -1.0]))
    np.testing.assert_array_equal(result, [2.0, -1.0])


def test_muscl_minmod_zero_input_returns_zero():
    muscl = MUSCL1D()
    result = muscl._minmod(np.array([0.0, 1.0]), np.array([3.0, 0.0]))
    np.testing.assert_array_equal(result, [0.0, 0.0])


def test_muscl_uniform_field_no_reconstruction_gradient():
    mesh = Mesh1D(
        length=10.0,
        resolution=1.0,
        initial_conditions=lambda x: np.column_stack([np.full_like(x, 2.0), np.full_like(x, 0.5)]),
    )
    Q_L, Q_R = MUSCL1D().reconstruct_conserved_variables(mesh)
    np.testing.assert_allclose(Q_L[1:-1], Q_R[1:-1])
    np.testing.assert_allclose(Q_L[1:-1, 0], 2.0)
    np.testing.assert_allclose(Q_L[1:-1, 1], 0.5)


def test_muscl_output_shape():
    mesh = Mesh1D(
        length=10.0,
        resolution=1.0,
        initial_conditions=lambda x: np.zeros((len(x), 2)),
    )
    Q_L, Q_R = MUSCL1D().reconstruct_conserved_variables(mesh)
    assert Q_L.shape == (mesh.N + 1, 2)
    assert Q_R.shape == (mesh.N + 1, 2)


def test_first_order_returns_shifted_arrays():
    mesh = Mesh1D(
        length=4.0,
        resolution=1.0,
        initial_conditions=lambda x: np.column_stack([x, x * 2]),
    )
    Q_L, Q_R = FirstOrder().reconstruct_conserved_variables(mesh)
    np.testing.assert_array_equal(Q_L, mesh.Q_array[:-1])
    np.testing.assert_array_equal(Q_R, mesh.Q_array[1:])
    assert Q_L.shape == (mesh.N + 1, 2)
