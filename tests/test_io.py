import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from swefvm.utils.io import write_1D_simulation_results
from swefvm.core.simulation import Simulation
from swefvm.core.mesh import Mesh1D
from swefvm.core.boundaries import ReflectiveBoundary
from swefvm.physics.shallow_water import ShallowWater1D
from swefvm.methods.spatial import MUSCL1D
from swefvm.methods.temporal import FirstOrderTemporal
from swefvm.methods.riemann_solvers import HLLSolver


def _completed_sim():
    mesh = Mesh1D(
        length=5.0,
        resolution=1.0,
        initial_conditions=lambda x: np.column_stack([np.full_like(x, 1.0), np.zeros_like(x)]),
    )
    physics = ShallowWater1D(dx=mesh.dx)
    bcs = [ReflectiveBoundary(0), ReflectiveBoundary(mesh.N + 1)]
    sim = Simulation(mesh, physics, MUSCL1D(), FirstOrderTemporal(), HLLSolver(), bcs)
    sim.run(end_time=0.2, record_times=[0.1])
    return sim


def test_write_1d_results_creates_file(tmp_path):
    sim = _completed_sim()
    output = tmp_path / "results.csv"
    write_1D_simulation_results(sim, output, name="test_run")
    assert output.exists()


def test_write_1d_results_metadata_lines(tmp_path):
    sim = _completed_sim()
    output = tmp_path / "results.csv"
    write_1D_simulation_results(sim, output, name="test_run")
    lines = output.read_text().splitlines()
    assert lines[0] == "Name,Length,Resolution"
    assert lines[1] == f"test_run,{sim.mesh.length},{sim.mesh.dx}"


def test_write_1d_results_dataframe_columns(tmp_path):
    sim = _completed_sim()
    output = tmp_path / "results.csv"
    write_1D_simulation_results(sim, output, name="test_run")
    df = pd.read_csv(output, skiprows=3)
    assert set(df.columns) == {"Time", "x", "eta", "q"}
    assert len(df) == sim.mesh.N * len(sim.saved_times)


def test_write_1d_results_creates_parent_dirs(tmp_path):
    sim = _completed_sim()
    output = tmp_path / "nested" / "dir" / "results.csv"
    write_1D_simulation_results(sim, output, name="test_run")
    assert output.exists()


def test_write_1d_results_rejects_non_mesh1d():
    class FakeMesh:
        pass

    class FakeSim:
        mesh = FakeMesh()
        saved_times = {}

    with pytest.raises(TypeError):
        write_1D_simulation_results(FakeSim(), Path("/tmp/x.csv"), name="x")
