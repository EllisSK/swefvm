import numpy as np
import pytest

from swefvm.core.mesh import Mesh1D, Mesh2D
from swefvm.core.boundaries import ReflectiveBoundary
from swefvm.core.simulation import Simulation
from swefvm.core.source import ConstantSource, VariableSource, FieldSource, StateDependentSource
from swefvm.physics.shallow_water import ShallowWater
from swefvm.methods.spatial import MUSCL
from swefvm.methods.temporal import FirstOrderTemporal, RK2
from swefvm.methods.riemann_solvers import HLLSolver


RAIN = 50 / 1000 / 3600  #50 mm/hr in m/s


def _uniform_mesh(eta=1.0):
    return Mesh1D(
        length=10.0,
        resolution=1.0,
        initial_conditions=lambda x: np.column_stack([np.full_like(x, eta), np.zeros_like(x)]),
    )


def _uniform_mesh_2d(eta=1.0):
    return Mesh2D(
        width=5.0,
        height=4.0,
        resolution=1.0,
        initial_conditions=lambda X, Y: np.stack(
            [np.full_like(X, eta), np.zeros_like(X), np.zeros_like(X)], axis=-1
        ),
    )


def _reflective_bcs(mesh):
    return [ReflectiveBoundary(0), ReflectiveBoundary(mesh.N + 1)]


def _reflective_bcs_2d(mesh):
    return (
        [ReflectiveBoundary((0, j)) for j in range(1, mesh.Ny + 1)]
        + [ReflectiveBoundary((mesh.Nx + 1, j)) for j in range(1, mesh.Ny + 1)]
        + [ReflectiveBoundary((i, 0)) for i in range(1, mesh.Nx + 1)]
        + [ReflectiveBoundary((i, mesh.Ny + 1)) for i in range(1, mesh.Nx + 1)]
    )


def _clamped_infiltration(capacity):
    #Infiltration sink capped at the water available in the cell over the step
    def rate(Q_array, mesh, t, dt):
        eta = Q_array[(*mesh.interior_slice, 0)]
        h = np.maximum(eta - mesh.zb[mesh.interior_slice], 0.0)
        return np.maximum(-capacity, -h / dt)
    return rate


#---------------------------------------------------------------------------
#Unit tests: SourceTerm.apply acting on the source array directly
#---------------------------------------------------------------------------

def test_constant_source_adds_rate_to_target_component():
    mesh = _uniform_mesh()
    S = np.zeros_like(mesh.Q_array)
    ConstantSource(component=0, rate=2.5).apply(S, mesh.Q_array, mesh, t=0.0, dt=0.01)
    np.testing.assert_allclose(S[1:-1, 0], 2.5)
    np.testing.assert_allclose(S[1:-1, 1], 0.0)


def test_constant_source_leaves_ghost_cells_untouched():
    mesh = _uniform_mesh()
    S = np.zeros_like(mesh.Q_array)
    ConstantSource(component=0, rate=2.5).apply(S, mesh.Q_array, mesh, t=0.0, dt=0.01)
    np.testing.assert_allclose(S[0], 0.0)
    np.testing.assert_allclose(S[-1], 0.0)


def test_constant_source_can_target_momentum_component():
    mesh = _uniform_mesh()
    S = np.zeros_like(mesh.Q_array)
    ConstantSource(component=1, rate=-0.3).apply(S, mesh.Q_array, mesh, t=0.0, dt=0.01)
    np.testing.assert_allclose(S[1:-1, 1], -0.3)
    np.testing.assert_allclose(S[1:-1, 0], 0.0)


def test_variable_source_evaluates_rate_at_given_time():
    mesh = _uniform_mesh()
    S = np.zeros_like(mesh.Q_array)
    VariableSource(component=0, rate_t=lambda t: 3.0 * t).apply(S, mesh.Q_array, mesh, t=2.0, dt=0.01)
    np.testing.assert_allclose(S[1:-1, 0], 6.0)


def test_field_source_applies_spatially_varying_rate():
    mesh = _uniform_mesh()
    S = np.zeros_like(mesh.Q_array)
    FieldSource(component=0, rate_fn=lambda m, t: 0.1 * m.x_vals).apply(S, mesh.Q_array, mesh, t=0.0, dt=0.01)
    np.testing.assert_allclose(S[1:-1, 0], 0.1 * mesh.x_vals)
    np.testing.assert_allclose(S[0], 0.0)
    np.testing.assert_allclose(S[-1], 0.0)


def test_field_source_broadcasts_scalar_rate():
    mesh = _uniform_mesh()
    S = np.zeros_like(mesh.Q_array)
    FieldSource(component=0, rate_fn=lambda m, t: 4.2).apply(S, mesh.Q_array, mesh, t=0.0, dt=0.01)
    np.testing.assert_allclose(S[1:-1, 0], 4.2)


def test_state_dependent_source_sees_passed_state():
    #The rate must couple to the stage state handed in, not to mesh.Q_array
    mesh = _uniform_mesh(eta=1.0)
    Q_stage = mesh.Q_array.copy()
    Q_stage[1:-1, 0] = 2.0
    S = np.zeros_like(mesh.Q_array)
    term = StateDependentSource(component=0, rate_fn=lambda Q, m, t, dt: -0.1 * Q[(*m.interior_slice, 0)])
    term.apply(S, Q_stage, mesh, t=0.0, dt=0.01)
    np.testing.assert_allclose(S[1:-1, 0], -0.2)


def test_state_dependent_source_receives_time_and_dt():
    mesh = _uniform_mesh()
    seen = {}

    def rate(Q, m, t, dt):
        seen["t"] = t
        seen["dt"] = dt
        return 0.0

    term = StateDependentSource(component=0, rate_fn=rate)
    term.apply(np.zeros_like(mesh.Q_array), mesh.Q_array, mesh, t=3.0, dt=0.25)
    assert seen == {"t": 3.0, "dt": 0.25}


def test_source_terms_superpose_on_shared_component():
    mesh = _uniform_mesh()
    S = np.zeros_like(mesh.Q_array)
    rain = ConstantSource(component=0, rate=2e-5)
    evaporation = ConstantSource(component=0, rate=-1e-5)
    rain.apply(S, mesh.Q_array, mesh, t=0.0, dt=0.01)
    evaporation.apply(S, mesh.Q_array, mesh, t=0.0, dt=0.01)
    np.testing.assert_allclose(S[1:-1, 0], 1e-5)


def test_application_order_is_irrelevant():
    #Terms read the state, never the accumulating source array, so order cannot matter
    mesh = _uniform_mesh()
    rain = ConstantSource(component=0, rate=RAIN)
    sink = StateDependentSource(component=0, rate_fn=_clamped_infiltration(1e-5))

    S_a = np.zeros_like(mesh.Q_array)
    rain.apply(S_a, mesh.Q_array, mesh, t=0.0, dt=0.01)
    sink.apply(S_a, mesh.Q_array, mesh, t=0.0, dt=0.01)

    S_b = np.zeros_like(mesh.Q_array)
    sink.apply(S_b, mesh.Q_array, mesh, t=0.0, dt=0.01)
    rain.apply(S_b, mesh.Q_array, mesh, t=0.0, dt=0.01)

    np.testing.assert_array_equal(S_a, S_b)


def test_constant_source_2d_targets_interior_only():
    mesh = _uniform_mesh_2d()
    S = np.zeros_like(mesh.Q_array)
    ConstantSource(component=0, rate=RAIN).apply(S, mesh.Q_array, mesh, t=0.0, dt=0.01)
    np.testing.assert_allclose(S[1:-1, 1:-1, 0], RAIN)
    np.testing.assert_allclose(S[1:-1, 1:-1, 1:], 0.0)
    np.testing.assert_allclose(S[0], 0.0)
    np.testing.assert_allclose(S[-1], 0.0)
    np.testing.assert_allclose(S[:, 0], 0.0)
    np.testing.assert_allclose(S[:, -1], 0.0)


def test_field_source_2d_spatially_varying_rate():
    mesh = _uniform_mesh_2d()

    def rate_fn(m, t):
        X, Y = np.meshgrid(m.x_vals, m.y_vals, indexing="ij")
        return 0.01 * (X + Y)

    S = np.zeros_like(mesh.Q_array)
    FieldSource(component=0, rate_fn=rate_fn).apply(S, mesh.Q_array, mesh, t=0.0, dt=0.01)
    X, Y = np.meshgrid(mesh.x_vals, mesh.y_vals, indexing="ij")
    np.testing.assert_allclose(S[1:-1, 1:-1, 0], 0.01 * (X + Y))
    np.testing.assert_allclose(S[1:-1, 1:-1, 1:], 0.0)


#---------------------------------------------------------------------------
#Integration tests: source terms through the temporal integrators
#---------------------------------------------------------------------------

def test_first_order_rain_raises_surface_by_rate_dt():
    mesh = _uniform_mesh()
    physics = ShallowWater()
    bcs = _reflective_bcs(mesh)
    mesh.apply_external_boundary_conditions(bcs)
    dt = 0.01
    FirstOrderTemporal().integrate(
        mesh, physics, MUSCL(), HLLSolver(), bcs, [], dt,
        source_terms=[ConstantSource(component=0, rate=RAIN)],
    )
    np.testing.assert_allclose(mesh.Q_array[1:-1, 0], 1.0 + RAIN * dt, rtol=0, atol=1e-15)
    np.testing.assert_allclose(mesh.Q_array[1:-1, 1], 0.0, atol=1e-15)


def test_rk2_rain_raises_surface_by_rate_dt():
    #A constant rate gives K1 = K2 = R, so the Heun average dt*(K1+K2)/2 is exact
    mesh = _uniform_mesh()
    physics = ShallowWater()
    bcs = _reflective_bcs(mesh)
    mesh.apply_external_boundary_conditions(bcs)
    dt = 0.01
    RK2().integrate(
        mesh, physics, MUSCL(), HLLSolver(), bcs, [], dt,
        source_terms=[ConstantSource(component=0, rate=RAIN)],
    )
    np.testing.assert_allclose(mesh.Q_array[1:-1, 0], 1.0 + RAIN * dt, rtol=0, atol=1e-15)
    np.testing.assert_allclose(mesh.Q_array[1:-1, 1], 0.0, atol=1e-15)


def test_rain_wets_a_dry_bed():
    mesh = Mesh1D(
        length=10.0,
        resolution=1.0,
        initial_conditions=lambda x: np.zeros((len(x), 2)),
    )
    physics = ShallowWater()
    bcs = _reflective_bcs(mesh)
    mesh.apply_external_boundary_conditions(bcs)
    dt = 0.01  #passed explicitly: dynamic_timestep is undefined on a fully dry domain
    FirstOrderTemporal().integrate(
        mesh, physics, MUSCL(), HLLSolver(), bcs, [], dt,
        source_terms=[ConstantSource(component=0, rate=RAIN)],
    )
    assert np.all(np.isfinite(mesh.Q_array))
    np.testing.assert_allclose(mesh.Q_array[1:-1, 0], RAIN * dt, rtol=0, atol=1e-15)
    np.testing.assert_allclose(mesh.Q_array[1:-1, 1], 0.0, atol=1e-15)


def test_clamped_sink_drains_thin_film_without_going_negative():
    h0 = 1e-4
    mesh = _uniform_mesh(eta=h0)
    physics = ShallowWater()
    bcs = _reflective_bcs(mesh)
    mesh.apply_external_boundary_conditions(bcs)
    dt = 0.01  #capacity * dt = 1e-2 m of demand against a 1e-4 m film
    sink = StateDependentSource(component=0, rate_fn=_clamped_infiltration(capacity=1.0))
    FirstOrderTemporal().integrate(mesh, physics, MUSCL(), HLLSolver(), bcs, [], dt, source_terms=[sink])
    h = mesh.Q_array[1:-1, 0] - mesh.zb[1:-1]
    assert np.all(h >= -1e-12)
    np.testing.assert_allclose(h, 0.0, atol=1e-12)


def test_unclamped_sink_removes_capacity_rate():
    mesh = _uniform_mesh(eta=1.0)
    physics = ShallowWater()
    bcs = _reflective_bcs(mesh)
    mesh.apply_external_boundary_conditions(bcs)
    dt = 0.01
    capacity = 1e-5
    sink = StateDependentSource(component=0, rate_fn=_clamped_infiltration(capacity))
    FirstOrderTemporal().integrate(mesh, physics, MUSCL(), HLLSolver(), bcs, [], dt, source_terms=[sink])
    np.testing.assert_allclose(mesh.Q_array[1:-1, 0], 1.0 - capacity * dt, rtol=0, atol=1e-15)


def test_rain_and_infiltration_combine_to_net_rate():
    mesh = _uniform_mesh(eta=1.0)
    physics = ShallowWater()
    bcs = _reflective_bcs(mesh)
    mesh.apply_external_boundary_conditions(bcs)
    dt = 0.01
    rain_rate = 2e-5
    capacity = 1e-5
    terms = [
        ConstantSource(component=0, rate=rain_rate),
        StateDependentSource(component=0, rate_fn=_clamped_infiltration(capacity)),
    ]
    FirstOrderTemporal().integrate(mesh, physics, MUSCL(), HLLSolver(), bcs, [], dt, source_terms=terms)
    np.testing.assert_allclose(mesh.Q_array[1:-1, 0], 1.0 + (rain_rate - capacity) * dt, rtol=0, atol=1e-15)


def test_term_order_through_integrator_is_identical():
    physics = ShallowWater()
    rain = ConstantSource(component=0, rate=RAIN)
    sink = StateDependentSource(component=0, rate_fn=_clamped_infiltration(1e-5))
    results = []
    for terms in ([rain, sink], [sink, rain]):
        mesh = _uniform_mesh()
        bcs = _reflective_bcs(mesh)
        mesh.apply_external_boundary_conditions(bcs)
        FirstOrderTemporal().integrate(mesh, physics, MUSCL(), HLLSolver(), bcs, [], 0.01, source_terms=terms)
        results.append(mesh.Q_array)
    np.testing.assert_array_equal(results[0], results[1])


def test_omitted_and_empty_source_terms_are_equivalent():
    #Regression: the default path must be bit-identical to passing an empty list
    physics = ShallowWater()
    results = []
    for source_terms in (None, []):
        mesh = Mesh1D(
            length=10.0,
            resolution=1.0,
            initial_conditions=lambda x: np.column_stack([
                np.where(x < 5.0, 2.0, 1.0),
                np.zeros_like(x),
            ]),
        )
        bcs = _reflective_bcs(mesh)
        mesh.apply_external_boundary_conditions(bcs)
        if source_terms is None:
            FirstOrderTemporal().integrate(mesh, physics, MUSCL(), HLLSolver(), bcs, [], 0.01)
        else:
            FirstOrderTemporal().integrate(mesh, physics, MUSCL(), HLLSolver(), bcs, [], 0.01, source_terms=source_terms)
        results.append(mesh.Q_array)
    np.testing.assert_array_equal(results[0], results[1])


def test_integrator_passes_mesh_time_to_sources():
    mesh = _uniform_mesh()
    mesh.t = 5.0
    physics = ShallowWater()
    bcs = _reflective_bcs(mesh)
    mesh.apply_external_boundary_conditions(bcs)
    dt = 0.01
    term = VariableSource(component=0, rate_t=lambda t: 0.001 * t)
    FirstOrderTemporal().integrate(mesh, physics, MUSCL(), HLLSolver(), bcs, [], dt, source_terms=[term])
    np.testing.assert_allclose(mesh.Q_array[1:-1, 0], 1.0 + (0.001 * 5.0) * dt, rtol=0, atol=1e-15)


def test_rk2_evaluates_state_dependent_source_at_predictor_state():
    #The second stage must hand terms the advected predictor, not the old state
    mesh = _uniform_mesh()
    physics = ShallowWater()
    bcs = _reflective_bcs(mesh)
    mesh.apply_external_boundary_conditions(bcs)
    dt = 0.01
    captured = []

    def spy(Q, m, t, dt):
        captured.append(Q[(*m.interior_slice, 0)].copy())
        return 0.0

    terms = [ConstantSource(component=0, rate=RAIN), StateDependentSource(component=0, rate_fn=spy)]
    RK2().integrate(mesh, physics, MUSCL(), HLLSolver(), bcs, [], dt, source_terms=terms)
    assert len(captured) == 2
    np.testing.assert_allclose(captured[0], 1.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(captured[1], 1.0 + RAIN * dt, rtol=0, atol=1e-15)


def test_first_order_2d_rain_raises_surface_uniformly():
    mesh = _uniform_mesh_2d()
    physics = ShallowWater()
    bcs = _reflective_bcs_2d(mesh)
    mesh.apply_external_boundary_conditions(bcs)
    dt = 0.01
    FirstOrderTemporal().integrate(
        mesh, physics, MUSCL(), HLLSolver(), bcs, [], dt,
        source_terms=[ConstantSource(component=0, rate=RAIN)],
    )
    np.testing.assert_allclose(mesh.Q_array[1:-1, 1:-1, 0], 1.0 + RAIN * dt, rtol=0, atol=1e-15)
    np.testing.assert_allclose(mesh.Q_array[1:-1, 1:-1, 1:], 0.0, atol=1e-15)


#---------------------------------------------------------------------------
#End-to-end: source terms through Simulation.run
#---------------------------------------------------------------------------

def test_simulation_forwards_source_terms():
    mesh = _uniform_mesh()
    physics = ShallowWater()
    bcs = _reflective_bcs(mesh)
    rate = 1e-3
    end_time = 0.5
    sim = Simulation(
        mesh, physics, MUSCL(), FirstOrderTemporal(), HLLSolver(), bcs,
        sources=[ConstantSource(component=0, rate=rate)],
    )
    sim.run(end_time=end_time)
    assert sim.t == pytest.approx(end_time)
    np.testing.assert_allclose(mesh.Q_array[1:-1, 0], 1.0 + rate * end_time, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mesh.Q_array[1:-1, 1], 0.0, atol=1e-12)


def test_simulation_without_source_terms_still_runs():
    #Regression: the constructor's new argument must default cleanly
    mesh = _uniform_mesh()
    physics = ShallowWater()
    bcs = _reflective_bcs(mesh)
    sim = Simulation(mesh, physics, MUSCL(), FirstOrderTemporal(), HLLSolver(), bcs)
    sim.run(end_time=0.5)
    np.testing.assert_allclose(mesh.Q_array[1:-1, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(mesh.Q_array[1:-1, 1], 0.0, atol=1e-12)