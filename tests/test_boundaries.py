import numpy as np
import pytest

from swefvm.core.boundaries import (
    ReflectiveBoundary,
    TransmissiveBoundary,
    FixedFlowBoundary,
    FixedDepthBoundary,
    VariableFlowBoundary,
    VariableDepthBoundary,
    VariableConservedBoundary,
    ClosedInterface,
    FixedFluxInterface,
    VariableFluxInterface,
    WeirInterface,
    SluiceGateInterface,
    VariableSluiceGateInterface,
    StageDischargeInterface,
)


def _make_pair(eta=1.0, q=0.5):
    interior = np.array([eta, q])
    ghost = np.zeros(2)
    return interior, ghost


def test_reflective_boundary_negates_momentum():
    interior, ghost = _make_pair(eta=2.0, q=3.0)
    ReflectiveBoundary(0).apply(interior, ghost, normal_idx=1)
    assert ghost[0] == 2.0
    assert ghost[1] == -3.0


def test_transmissive_boundary_copies_state():
    interior, ghost = _make_pair(eta=2.0, q=3.0)
    TransmissiveBoundary(0).apply(interior, ghost, normal_idx=1)
    np.testing.assert_array_equal(ghost, interior)


def test_fixed_flow_boundary_sets_momentum_only():
    interior, ghost = _make_pair(eta=2.0, q=3.0)
    FixedFlowBoundary(0, target_q=7.5).apply(interior, ghost, normal_idx=1)
    assert ghost[0] == 2.0
    assert ghost[1] == 7.5


def test_fixed_depth_boundary_sets_depth_only():
    interior, ghost = _make_pair(eta=2.0, q=3.0)
    FixedDepthBoundary(0, target_h=4.2).apply(interior, ghost, normal_idx=1)
    assert ghost[0] == 4.2
    assert ghost[1] == 3.0


def test_variable_flow_boundary_uses_time_function():
    interior, ghost = _make_pair(eta=2.0, q=3.0)
    bc = VariableFlowBoundary(0, q_t=lambda t: 2 * t)
    bc.apply(interior, ghost, normal_idx=1, t=5.0)
    assert ghost[1] == 10.0


def test_variable_depth_boundary_uses_time_function():
    interior, ghost = _make_pair(eta=2.0, q=3.0)
    bc = VariableDepthBoundary(0, h_t=lambda t: t + 1)
    bc.apply(interior, ghost, normal_idx=1, t=4.0)
    assert ghost[0] == 5.0


def test_variable_conserved_boundary_uses_both_functions():
    interior, ghost = _make_pair(eta=2.0, q=3.0)
    bc = VariableConservedBoundary(0, h_t=lambda t: t, q_t=lambda t: -t)
    bc.apply(interior, ghost, normal_idx=1, t=2.0)
    assert ghost[0] == 2.0
    assert ghost[1] == -2.0


def test_closed_interface_zeroes_flux():
    F = np.array([5.0, 7.0])
    Q_L = np.array([1.0, 2.0])
    Q_R = np.array([1.0, 2.0])
    ClosedInterface(0).apply(F, Q_L, Q_R, zb_interface=0.0, normal_idx=1)
    np.testing.assert_array_equal(F, [0.0, 0.0])


def test_fixed_flux_interface_sets_target():
    F = np.zeros(2)
    Q_L = np.array([1.0, 0.0])
    Q_R = np.array([1.0, 0.0])
    FixedFluxInterface(0, target_flux=[2.5, -1.0]).apply(F, Q_L, Q_R, zb_interface=0.0, normal_idx=1)
    np.testing.assert_array_equal(F, [2.5, -1.0])


def test_variable_flux_interface_uses_time_function():
    F = np.zeros(2)
    Q_L = np.array([1.0, 0.0])
    Q_R = np.array([1.0, 0.0])
    bc = VariableFluxInterface(0, flux_t=lambda t: np.array([t, 2 * t]))
    bc.apply(F, Q_L, Q_R, zb_interface=0.0, normal_idx=1, t=3.0)
    np.testing.assert_array_equal(F, [3.0, 6.0])


def test_weir_interface_no_flow_below_crest():
    F = np.zeros(2)
    Q_L = np.array([0.5, 0.0])
    Q_R = np.array([0.5, 0.0])
    WeirInterface(0, crest_elevation=1.0).apply(F, Q_L, Q_R, zb_interface=0.0, normal_idx=1)
    assert F[0] == 0.0


def test_weir_interface_positive_flow_when_left_higher():
    F = np.zeros(2)
    Q_L = np.array([2.0, 0.0])
    Q_R = np.array([1.5, 0.0])
    WeirInterface(0, crest_elevation=1.0, discharge_coefficient=0.611, g=9.81).apply(
        F, Q_L, Q_R, zb_interface=0.0, normal_idx=1
    )
    H = 1.0
    expected = 0.611 * (2.0 / 3.0) * np.sqrt(2.0 * 9.81) * H ** 1.5
    assert F[0] > 0
    np.testing.assert_allclose(F[0], expected)


def test_weir_interface_negative_flow_when_right_higher():
    F = np.zeros(2)
    Q_L = np.array([1.5, 0.0])
    Q_R = np.array([2.0, 0.0])
    WeirInterface(0, crest_elevation=1.0).apply(F, Q_L, Q_R, zb_interface=0.0, normal_idx=1)
    assert F[0] < 0


def test_sluice_gate_interface_caps_opening_at_water_depth():
    F = np.zeros(2)
    Q_L = np.array([0.3, 0.0])
    Q_R = np.array([0.1, 0.0])
    bc = SluiceGateInterface(0, gate_opening=10.0, discharge_coefficient=0.6, g=9.81)
    bc.apply(F, Q_L, Q_R, zb_interface=0.0, normal_idx=1)
    h_up = 0.3
    expected = 0.6 * h_up * np.sqrt(2.0 * 9.81 * h_up)
    np.testing.assert_allclose(F[0], expected)


def test_variable_sluice_gate_uses_time_function():
    F = np.zeros(2)
    Q_L = np.array([2.0, 0.0])
    Q_R = np.array([1.0, 0.0])
    bc = VariableSluiceGateInterface(0, gate_opening_t=lambda t: 0.5)
    bc.apply(F, Q_L, Q_R, zb_interface=0.0, normal_idx=1, t=1.0)
    assert F[0] > 0


def test_stage_discharge_interface_uses_rating_curve():
    F = np.zeros(2)
    Q_L = np.array([2.0, 0.0])
    Q_R = np.array([1.0, 0.0])
    bc = StageDischargeInterface(0, rating_curve=lambda h: 3.0 * h)
    bc.apply(F, Q_L, Q_R, zb_interface=0.0, normal_idx=1)
    assert F[0] == pytest.approx(3.0 * 2.0)
