import numpy as np
import pytest

from swefvm.core.simulation import Simulation
from swefvm.core.mesh import Mesh1D
from swefvm.core.boundaries import ReflectiveBoundary, ClosedInterface
from swefvm.physics.shallow_water import ShallowWater
from swefvm.methods.spatial import MUSCL
from swefvm.methods.temporal import FirstOrderTemporal
from swefvm.methods.riemann_solvers import HLLSolver


def _make_sim(bcs=None):
    mesh = Mesh1D(
        length=10.0,
        resolution=1.0,
        initial_conditions=lambda x: np.column_stack([np.full_like(x, 1.0), np.zeros_like(x)]),
    )
    physics = ShallowWater()
    if bcs is None:
        bcs = [ReflectiveBoundary(0), ReflectiveBoundary(mesh.N + 1)]
    return Simulation(mesh, physics, MUSCL(), FirstOrderTemporal(), HLLSolver(), bcs)


def test_simulation_separates_external_and_internal_bcs():
    external = [ReflectiveBoundary(0), ReflectiveBoundary(11)]
    internal = [ClosedInterface(5)]
    sim = _make_sim(bcs=external + internal)
    assert len(sim.external_bcs) == 2
    assert len(sim.internal_bcs) == 1
    assert all(bc in external for bc in sim.external_bcs)
    assert all(bc in internal for bc in sim.internal_bcs)


def test_simulation_run_requires_an_end_condition():
    sim = _make_sim()
    with pytest.raises(ValueError):
        sim.run()


def test_simulation_run_rejects_both_end_conditions():
    sim = _make_sim()
    with pytest.raises(ValueError):
        sim.run(end_time=1.0, convergance_threshold=0.01)


def test_simulation_run_reaches_end_time():
    sim = _make_sim()
    sim.run(end_time=0.5)
    assert sim.t == pytest.approx(0.5, abs=1e-10)


def test_simulation_records_final_state_in_saved_times():
    sim = _make_sim()
    sim.run(end_time=0.5)
    assert sim.t in sim.saved_times
    assert sim.saved_times[sim.t].shape == sim.mesh.Q_array.shape


def test_simulation_records_requested_times():
    sim = _make_sim()
    sim.run(end_time=1.0, record_times=[0.25, 0.5])
    keys = sorted(sim.saved_times.keys())
    assert any(np.isclose(k, 0.25) for k in keys)
    assert any(np.isclose(k, 0.5) for k in keys)


def test_simulation_run_generator_yields_state_tuples():
    sim = _make_sim()
    frames = list(sim.run_generator(end_time=0.1))
    assert len(frames) >= 2
    t0, Q0 = frames[0]
    assert t0 == 0
    assert Q0.shape == sim.mesh.Q_array.shape