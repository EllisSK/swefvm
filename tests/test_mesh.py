import numpy as np
import pytest

from swefvm.core.mesh import Mesh1D


def test_mesh1d_array_shapes(mesh_flat):
    assert mesh_flat.Q_array.shape == (mesh_flat.N + 2, 2)
    assert mesh_flat.F_array.shape == (mesh_flat.N + 2, 2)


def test_mesh1d_cell_count():
    mesh = Mesh1D(length=10.0, resolution=0.5, initial_conditions=lambda x: np.zeros((len(x), 2)))
    assert mesh.N == 20


def test_mesh1d_x_vals_cell_centers(mesh_flat):
    expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
    np.testing.assert_allclose(mesh_flat.x_vals, expected)


def test_mesh1d_initial_conditions_applied_to_interior(mesh_flat):
    np.testing.assert_allclose(mesh_flat.Q_array[1:-1, 0], 1.0)
    np.testing.assert_allclose(mesh_flat.Q_array[1:-1, 1], 0.0)


def test_mesh1d_ghost_cells_initially_zero(mesh_flat):
    np.testing.assert_allclose(mesh_flat.Q_array[0], 0.0)
    np.testing.assert_allclose(mesh_flat.Q_array[-1], 0.0)


def test_mesh1d_no_bed_function_zero_zb(mesh_flat):
    np.testing.assert_allclose(mesh_flat.zb, 0.0)
    np.testing.assert_allclose(mesh_flat.zb_interface, 0.0)
    assert mesh_flat.zb.shape == (mesh_flat.N + 2,)
    assert mesh_flat.zb_interface.shape == (mesh_flat.N + 1,)


def test_mesh1d_with_bed_function(mesh_sloped):
    assert mesh_sloped.zb.shape == (mesh_sloped.N + 2,)
    assert mesh_sloped.zb_interface.shape == (mesh_sloped.N + 1,)
    np.testing.assert_allclose(mesh_sloped.zb_interface, 0.5 * (mesh_sloped.zb[:-1] + mesh_sloped.zb[1:]))


def test_mesh1d_directions_and_spacing(mesh_flat):
    assert mesh_flat.directions == (0,)
    assert mesh_flat.spacing(0) == mesh_flat.dx
    assert mesh_flat.interior_slice == (slice(1, -1),)


def test_mesh1d_interface_bed_matches_zb_interface(mesh_sloped):
    np.testing.assert_array_equal(mesh_sloped.interface_bed(0), mesh_sloped.zb_interface)


def test_mesh1d_spacing_invalid_direction_raises(mesh_flat):
    with pytest.raises(ValueError):
        mesh_flat.spacing(1)


def test_mesh1d_interface_bed_invalid_direction_raises(mesh_flat):
    with pytest.raises(ValueError):
        mesh_flat.interface_bed(1)


def test_mesh1d_resolve_external_left(mesh_flat):
    interior, ghost, normal_idx = mesh_flat._resolve_external(0)
    assert normal_idx == 1
    assert np.shares_memory(interior, mesh_flat.Q_array[1])
    assert np.shares_memory(ghost, mesh_flat.Q_array[0])


def test_mesh1d_resolve_external_right(mesh_flat):
    interior, ghost, normal_idx = mesh_flat._resolve_external(mesh_flat.N + 1)
    assert normal_idx == 1
    assert np.shares_memory(interior, mesh_flat.Q_array[-2])
    assert np.shares_memory(ghost, mesh_flat.Q_array[-1])


def test_mesh1d_resolve_external_invalid_raises(mesh_flat):
    with pytest.raises(ValueError):
        mesh_flat._resolve_external(5)


def test_mesh1d_resolve_internal_valid(mesh_flat):
    F = np.zeros((mesh_flat.N + 1, 2))
    Q_L = np.zeros((mesh_flat.N + 1, 2))
    Q_R = np.zeros((mesh_flat.N + 1, 2))
    f_view, ql_view, qr_view, zb, normal_idx = mesh_flat._resolve_internal(3, F, Q_L, Q_R, 0)
    assert normal_idx == 1
    assert zb == mesh_flat.zb_interface[3]
    assert np.shares_memory(f_view, F)


def test_mesh1d_resolve_internal_out_of_range_raises(mesh_flat):
    F = np.zeros((mesh_flat.N + 1, 2))
    with pytest.raises(ValueError):
        mesh_flat._resolve_internal(mesh_flat.N + 1, F, F.copy(), F.copy(), 0)


def test_mesh1d_resolve_internal_invalid_direction_raises(mesh_flat):
    F = np.zeros((mesh_flat.N + 1, 2))
    with pytest.raises(ValueError):
        mesh_flat._resolve_internal(3, F, F.copy(), F.copy(), 1)