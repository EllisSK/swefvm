import numpy as np

from pathlib import Path

from src.SotAMiH.core.simulation import Simulation
from src.SotAMiH.core.mesh import Mesh1D
from src.SotAMiH.physics.shallow_water import ShallowWater1D
from src.SotAMiH.methods.spatial.MUSCL import MUSCL1D
from src.SotAMiH.methods.temporal.range_kutta import RK2
from src.SotAMiH.methods.riemann_solvers import HLLSolver, OsherSolomonSolver
from src.SotAMiH.core.boundaries import ReflectiveBoundary, VariableConservedBoundary, TransmissiveBoundary, VariableDepthBoundary
from src.SotAMiH.utils.plot import Animation1D
from src.SotAMiH.utils.io import write_1D_simulation_results

def test_case_1():
    DOMAIN_LENGTH = 14000
    N_CELLS = 50
    RESOLUTION = DOMAIN_LENGTH / N_CELLS
    T_MAX = 5000

    def bed_fn(x):
        return (10) + (40 * x / DOMAIN_LENGTH) + (10 * np.sin(np.pi * ((4 * x / DOMAIN_LENGTH) - (1/2))))
    
    def initial_cond(x):
        return 65, 0
    
    mesh = Mesh1D(DOMAIN_LENGTH, RESOLUTION, initial_cond, bed_fn)
    physics = ShallowWater1D(mesh.dx)
    spatial = MUSCL1D()
    temporal = RK2()
    riemann = OsherSolomonSolver()

    bcs = {
        "left_boundary" : ReflectiveBoundary(),
        "right_boundary" : ReflectiveBoundary()
    }
    
    sim = Simulation(mesh, physics, spatial, temporal, riemann, bcs)

    sim.run(end_time=T_MAX)

    output_path = Path("./exports/testcase1.csv")
    write_1D_simulation_results(sim, output_path, "Test Case 1")

def test_case_2():
    DOMAIN_LENGTH = 14000
    N_CELLS = 50
    RESOLUTION = DOMAIN_LENGTH / N_CELLS
    T_MAX = 7552.13

    def bed_fn(x):
        return (10) + (40 * x / DOMAIN_LENGTH) + (10 * np.sin(np.pi * ((4 * x / DOMAIN_LENGTH) - (1/2))))
    
    def initial_cond(x):      
        return 60.5, 0
    
    def b_ht(t):
        return 64.5 - (4 * np.sin(np.pi * (((4 * t) / 86400) + (0.5))))

    def lb_qt(t):
        return (b_ht(0) - bed_fn(0)) * (((np.pi * (0-DOMAIN_LENGTH)) / (5400 * b_ht(t))) * np.cos(np.pi * (((4 * t) / 86400) + (0.5))))
    
    mesh = Mesh1D(DOMAIN_LENGTH, RESOLUTION, initial_cond, bed_fn)
    physics = ShallowWater1D(mesh.dx)
    spatial = MUSCL1D()
    temporal = RK2()
    riemann = OsherSolomonSolver()

    bcs = {
        "left_boundary" : VariableConservedBoundary(b_ht, lb_qt),
        "right_boundary" : ReflectiveBoundary()
    }
    
    sim = Simulation(mesh, physics, spatial, temporal, riemann, bcs)

    sim.run(end_time=T_MAX)

    output_path = Path("./exports/testcase2.csv")
    write_1D_simulation_results(sim, output_path, "Test Case 2")

def test_case_3a():
    DOMAIN_LENGTH = 50
    N_CELLS = 200
    RESOLUTION = DOMAIN_LENGTH / N_CELLS
    T_MAX = 5

    def bed_fn(x):
        return np.zeros_like(x)
    
    def initial_cond(x):
        Q = np.zeros((len(x), 2), dtype=float)
        Q[:, 0] = np.where(x < 25, 1.0, 0.1)
        return Q
    
    mesh = Mesh1D(DOMAIN_LENGTH, RESOLUTION, initial_cond, bed_fn)
    physics = ShallowWater1D(mesh.dx)
    spatial = MUSCL1D()
    temporal = RK2()
    riemann = HLLSolver()

    bcs = {
        "left_boundary" : TransmissiveBoundary(),
        "right_boundary" : TransmissiveBoundary()
    }
    
    sim = Simulation(mesh, physics, spatial, temporal, riemann, bcs)

    animator = Animation1D(sim, max_elevation=1.5, end_time=T_MAX)
    animator.show()

    sim.run(end_time=T_MAX)

    output_path = Path("./exports/testcase3a.csv")
    write_1D_simulation_results(sim, output_path, "Test Case 3a")

def test_case_3b():
    DOMAIN_LENGTH = 50
    N_CELLS = 200
    RESOLUTION = DOMAIN_LENGTH / N_CELLS
    T_MAX = 5

    def bed_fn(x):
        return np.zeros_like(x)
    
    def initial_cond(x):
        Q = np.zeros((len(x), 2), dtype=float)
        Q[:, 0] = np.where(x < 25, 1.0, 0.1)
        return Q
    
    mesh = Mesh1D(DOMAIN_LENGTH, RESOLUTION, initial_cond, bed_fn)
    physics = ShallowWater1D(mesh.dx)
    spatial = MUSCL1D()
    temporal = RK2()
    riemann = OsherSolomonSolver()

    bcs = {
        "left_boundary" : TransmissiveBoundary(),
        "right_boundary" : TransmissiveBoundary()
    }
    
    sim = Simulation(mesh, physics, spatial, temporal, riemann, bcs)

    animator = Animation1D(sim, max_elevation=1.5, end_time=T_MAX)
    animator.show()

    sim.run(end_time=T_MAX)

    output_path = Path("./exports/testcase3b.csv")
    write_1D_simulation_results(sim, output_path, "Test Case 3b")

def test_case_4():
    DOMAIN_LENGTH = 14000
    N_CELLS = 50
    RESOLUTION = DOMAIN_LENGTH / N_CELLS
    T_MAX = 32400
    TIMES_TO_RECORD = [10800, 32400]

    def bed_fn(x):
        zb = np.zeros_like(x)
        zb = np.where(abs(x-750) < 187.5, 8.0, 0.0)
        return zb
    
    def b_ht(t):
        return 20 - (4 * np.sin(np.pi * (((4 * t) / 86400) + (0.5))))
    
    def initial_cond(x):      
        return 16, 0
    
    mesh = Mesh1D(1500, 7.5, initial_cond, bed_fn)
    physics = ShallowWater1D(mesh.dx)
    spatial = MUSCL1D()
    temporal = RK2()
    riemann = OsherSolomonSolver()

    bcs = {
        "left_boundary" : VariableDepthBoundary(b_ht),
        "right_boundary" : ReflectiveBoundary()
    }
    
    sim = Simulation(mesh, physics, spatial, temporal, riemann, bcs)

    sim.run(end_time=T_MAX, record_times=TIMES_TO_RECORD)

    output_path = Path("./exports/testcase4.csv")
    write_1D_simulation_results(sim, output_path, "Test Case 4")

def main():
    test_case_1()
    print("Test Case 1 Completed.")

    test_case_2()
    print("Test Case 2 Completed.")

    test_case_3a()
    print("Test Case 3a Completed.")

    test_case_3b()
    print("Test Case 3b Completed.")

    test_case_4()
    print("Test Case 4 Completed.")


if __name__ == "__main__":
    main()
