import numpy as np
import pytest

from swefvm.core.mesh import Mesh1D, Mesh2D


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


def test_mesh2d_array_shapes(mesh2d_flat):
    assert mesh2d_flat.Q_array.shape == (mesh2d_flat.Nx + 2, mesh2d_flat.Ny + 2, 3)
    assert mesh2d_flat.F_array.shape == (mesh2d_flat.Nx + 2, mesh2d_flat.Ny + 2, 3)


def test_mesh2d_cell_count():
    mesh = Mesh2D(
        width=10.0,
        height=8.0,
        resolution=0.5,
        initial_conditions=lambda X, Y: np.zeros((*X.shape, 3)),
    )
    assert mesh.Nx == 20
    assert mesh.Ny == 16


def test_mesh2d_x_vals_cell_centers(mesh2d_flat):
    expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
    np.testing.assert_allclose(mesh2d_flat.x_vals, expected)


def test_mesh2d_y_vals_cell_centers(mesh2d_flat):
    expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    np.testing.assert_allclose(mesh2d_flat.y_vals, expected)


def test_mesh2d_initial_conditions_applied_to_interior(mesh2d_flat):
    np.testing.assert_allclose(mesh2d_flat.Q_array[1:-1, 1:-1, 0], 1.0)
    np.testing.assert_allclose(mesh2d_flat.Q_array[1:-1, 1:-1, 1], 0.0)
    np.testing.assert_allclose(mesh2d_flat.Q_array[1:-1, 1:-1, 2], 0.0)


def test_mesh2d_ghost_cells_initially_zero(mesh2d_flat):
    np.testing.assert_allclose(mesh2d_flat.Q_array[0, :], 0.0)
    np.testing.assert_allclose(mesh2d_flat.Q_array[-1, :], 0.0)
    np.testing.assert_allclose(mesh2d_flat.Q_array[:, 0], 0.0)
    np.testing.assert_allclose(mesh2d_flat.Q_array[:, -1], 0.0)


def test_mesh2d_no_bed_function_zero_zb(mesh2d_flat):
    np.testing.assert_allclose(mesh2d_flat.zb, 0.0)
    np.testing.assert_allclose(mesh2d_flat.zb_interface_x, 0.0)
    np.testing.assert_allclose(mesh2d_flat.zb_interface_y, 0.0)
    assert mesh2d_flat.zb.shape == (mesh2d_flat.Nx + 2, mesh2d_flat.Ny + 2)
    assert mesh2d_flat.zb_interface_x.shape == (mesh2d_flat.Nx + 1, mesh2d_flat.Ny + 2)
    assert mesh2d_flat.zb_interface_y.shape == (mesh2d_flat.Nx + 2, mesh2d_flat.Ny + 1)


def test_mesh2d_with_bed_function_shapes(mesh2d_sloped):
    assert mesh2d_sloped.zb.shape == (mesh2d_sloped.Nx + 2, mesh2d_sloped.Ny + 2)
    assert mesh2d_sloped.zb_interface_x.shape == (mesh2d_sloped.Nx + 1, mesh2d_sloped.Ny + 2)
    assert mesh2d_sloped.zb_interface_y.shape == (mesh2d_sloped.Nx + 2, mesh2d_sloped.Ny + 1)


def test_mesh2d_zb_interface_x_averages_neighbours(mesh2d_sloped):
    np.testing.assert_allclose(
        mesh2d_sloped.zb_interface_x,
        0.5 * (mesh2d_sloped.zb[:-1, :] + mesh2d_sloped.zb[1:, :]),
    )


def test_mesh2d_zb_interface_y_averages_neighbours(mesh2d_sloped):
    np.testing.assert_allclose(
        mesh2d_sloped.zb_interface_y,
        0.5 * (mesh2d_sloped.zb[:, :-1] + mesh2d_sloped.zb[:, 1:]),
    )


def test_mesh2d_ghost_zb_matches_adjacent_interior(mesh2d_sloped):
    np.testing.assert_allclose(mesh2d_sloped.zb[0, :], mesh2d_sloped.zb[1, :])
    np.testing.assert_allclose(mesh2d_sloped.zb[-1, :], mesh2d_sloped.zb[-2, :])
    np.testing.assert_allclose(mesh2d_sloped.zb[:, 0], mesh2d_sloped.zb[:, 1])
    np.testing.assert_allclose(mesh2d_sloped.zb[:, -1], mesh2d_sloped.zb[:, -2])


def test_mesh2d_directions_and_spacing(mesh2d_flat):
    assert mesh2d_flat.directions == (0, 1)
    assert mesh2d_flat.spacing(0) == mesh2d_flat.dx
    assert mesh2d_flat.spacing(1) == mesh2d_flat.dy
    assert mesh2d_flat.interior_slice == (slice(1, -1), slice(1, -1))


def test_mesh2d_interface_bed_matches_zb_interface_x(mesh2d_sloped):
    np.testing.assert_array_equal(mesh2d_sloped.interface_bed(0), mesh2d_sloped.zb_interface_x)


def test_mesh2d_interface_bed_matches_zb_interface_y(mesh2d_sloped):
    np.testing.assert_array_equal(mesh2d_sloped.interface_bed(1), mesh2d_sloped.zb_interface_y)


def test_mesh2d_spacing_invalid_direction_raises(mesh2d_flat):
    with pytest.raises(ValueError):
        mesh2d_flat.spacing(2)


def test_mesh2d_interface_bed_invalid_direction_raises(mesh2d_flat):
    with pytest.raises(ValueError):
        mesh2d_flat.interface_bed(2)


def test_mesh2d_resolve_external_left(mesh2d_flat):
    interior, ghost, normal_idx = mesh2d_flat._resolve_external((0, 3))
    assert normal_idx == 1
    assert np.shares_memory(interior, mesh2d_flat.Q_array[1, 3])
    assert np.shares_memory(ghost, mesh2d_flat.Q_array[0, 3])


def test_mesh2d_resolve_external_right(mesh2d_flat):
    interior, ghost, normal_idx = mesh2d_flat._resolve_external((mesh2d_flat.Nx + 1, 3))
    assert normal_idx == 1
    assert np.shares_memory(interior, mesh2d_flat.Q_array[-2, 3])
    assert np.shares_memory(ghost, mesh2d_flat.Q_array[-1, 3])


def test_mesh2d_resolve_external_bottom(mesh2d_flat):
    interior, ghost, normal_idx = mesh2d_flat._resolve_external((4, 0))
    assert normal_idx == 2
    assert np.shares_memory(interior, mesh2d_flat.Q_array[4, 1])
    assert np.shares_memory(ghost, mesh2d_flat.Q_array[4, 0])


def test_mesh2d_resolve_external_top(mesh2d_flat):
    interior, ghost, normal_idx = mesh2d_flat._resolve_external((4, mesh2d_flat.Ny + 1))
    assert normal_idx == 2
    assert np.shares_memory(interior, mesh2d_flat.Q_array[4, -2])
    assert np.shares_memory(ghost, mesh2d_flat.Q_array[4, -1])


def test_mesh2d_resolve_external_interior_raises(mesh2d_flat):
    with pytest.raises(ValueError):
        mesh2d_flat._resolve_external((5, 5))


def test_mesh2d_resolve_external_corner_raises(mesh2d_flat):
    with pytest.raises(ValueError):
        mesh2d_flat._resolve_external((0, 0))


def test_mesh2d_resolve_external_writes_propagate_to_q_array(mesh2d_flat):
    interior, ghost, _ = mesh2d_flat._resolve_external((0, 2))
    ghost[:] = [9.0, 8.0, 7.0]
    np.testing.assert_array_equal(mesh2d_flat.Q_array[0, 2], [9.0, 8.0, 7.0])


def test_mesh2d_resolve_internal_x_direction(mesh2d_flat):
    F = np.zeros((mesh2d_flat.Nx + 1, mesh2d_flat.Ny + 2, 3))
    Q_L = np.zeros_like(F)
    Q_R = np.zeros_like(F)
    f_view, ql_view, qr_view, zb, normal_idx = mesh2d_flat._resolve_internal((3, 4), F, Q_L, Q_R, 0)
    assert normal_idx == 1
    assert zb == mesh2d_flat.zb_interface_x[3, 4]
    assert np.shares_memory(f_view, F)
    assert np.shares_memory(ql_view, Q_L)
    assert np.shares_memory(qr_view, Q_R)


def test_mesh2d_resolve_internal_y_direction(mesh2d_flat):
    F = np.zeros((mesh2d_flat.Nx + 2, mesh2d_flat.Ny + 1, 3))
    Q_L = np.zeros_like(F)
    Q_R = np.zeros_like(F)
    f_view, ql_view, qr_view, zb, normal_idx = mesh2d_flat._resolve_internal((4, 3), F, Q_L, Q_R, 1)
    assert normal_idx == 2
    assert zb == mesh2d_flat.zb_interface_y[4, 3]
    assert np.shares_memory(f_view, F)


def test_mesh2d_resolve_internal_x_out_of_range_raises(mesh2d_flat):
    F = np.zeros((mesh2d_flat.Nx + 1, mesh2d_flat.Ny + 2, 3))
    with pytest.raises(ValueError):
        mesh2d_flat._resolve_internal((mesh2d_flat.Nx + 1, 3), F, F.copy(), F.copy(), 0)


def test_mesh2d_resolve_internal_y_out_of_range_raises(mesh2d_flat):
    F = np.zeros((mesh2d_flat.Nx + 2, mesh2d_flat.Ny + 1, 3))
    with pytest.raises(ValueError):
        mesh2d_flat._resolve_internal((3, mesh2d_flat.Ny + 1), F, F.copy(), F.copy(), 1)


def test_mesh2d_resolve_internal_invalid_direction_raises(mesh2d_flat):
    F = np.zeros((mesh2d_flat.Nx + 1, mesh2d_flat.Ny + 2, 3))
    with pytest.raises(ValueError):
        mesh2d_flat._resolve_internal((3, 3), F, F.copy(), F.copy(), 2)