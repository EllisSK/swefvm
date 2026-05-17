import numpy as np

from collections.abc import Sequence

from .mesh import Mesh
from .boundaries import BoundaryCondition, ExternalBoundary, InternalBoundary

from swefvm.methods.spatial import SpatialReconstruction
from swefvm.methods.temporal import TemporalIntegrator
from swefvm.methods.riemann_solvers import RiemannSolver
from swefvm.physics import Physics

class Simulation:
    def __init__(self, mesh: Mesh, physics: Physics, spatial: SpatialReconstruction, temporal: TemporalIntegrator, riemann: RiemannSolver, bcs: list[BoundaryCondition]):
        self.mesh = mesh
        self.physics = physics
        self.spatial = spatial
        self.temporal = temporal
        self.riemann = riemann
        self.bcs = bcs
        self.external_bcs = [bc for bc in bcs if isinstance(bc, ExternalBoundary)]
        self.internal_bcs = [bc for bc in bcs if isinstance(bc, InternalBoundary)]

        self.saved_times : dict = {}

    def run(self, end_time: int | float | None = None, convergance_threshold: int | float | None = None, record_times: Sequence[int | float] | None = None):
        if end_time and convergance_threshold:
            raise ValueError("Please provide only 1 end condition.")
        elif not end_time and not convergance_threshold:
            raise ValueError("Please provide 1 end condition.")
        
        self.t = 0
        self.max_change = 0
        
        record_time = False

        if not end_time:
            end_time = np.inf

        if not convergance_threshold:
            convergance_threshold = np.inf

        self.ttr = None
        if record_times:
            self.ttr = list(record_times)

        self.mesh.apply_external_boundary_conditions(self.external_bcs)

        while (self.t < end_time) and (self.max_change < convergance_threshold):
            dt = self.physics.dynamic_timestep(self.mesh.Q_array, self.mesh)

            #Flow control to determine correct timestep to capture desired times to record (ttr) or the final time
            if (self.t + dt) > end_time:
                dt = end_time - self.t
                record_time = True
            elif self.ttr:
                if (self.t + dt) > self.ttr[0]:
                    dt = self.ttr[0] - self.t
                    record_time = True
                    self.ttr.pop(0)
                else:
                    record_time = False
            else:
                record_time = False

            self.t += dt
            self.mesh.t = self.t

            self.temporal.integrate(self.mesh, self.physics, self.spatial, self.riemann, self.external_bcs, self.internal_bcs, dt)

            if record_time:
                current_Q = self.mesh.Q_array
                t = self.t
                self.saved_times[t] = current_Q

        current_Q = self.mesh.Q_array
        t = self.t
        self.saved_times[t] = current_Q

    def run_generator(self, end_time: int | float | None = None, convergance_threshold: int | float | None = None, record_times: Sequence[int | float] | None = None):
        #Run method adapted for live animation
        
        if end_time and convergance_threshold:
            raise ValueError("Please provide only 1 end condition.")
        elif not end_time and not convergance_threshold:
            raise ValueError("Please provide 1 end condition.")
        
        self.t = 0
        self.max_change = 0
        
        record_time = False

        if not end_time:
            end_time = np.inf

        if not convergance_threshold:
            convergance_threshold = np.inf

        self.ttr = None
        if record_times:
            self.ttr = list(record_times)

        self.mesh.apply_external_boundary_conditions(self.external_bcs)

        yield self.t, self.mesh.Q_array

        while (self.t < end_time) and (self.max_change < convergance_threshold):
            dt = self.physics.dynamic_timestep(self.mesh.Q_array, self.mesh)

            if (self.t + dt) > end_time:
                dt = end_time - self.t
                record_time = True
            elif self.ttr:
                if (self.t + dt) > self.ttr[0]:
                    dt = self.ttr[0] - self.t
                    record_time = True
                    self.ttr.pop(0)
                else:
                    record_time = False
            else:
                record_time = False

            self.t += dt
            self.mesh.t = self.t

            self.temporal.integrate(self.mesh, self.physics, self.spatial, self.riemann, self.external_bcs, self.internal_bcs, dt)

            if record_time:
                current_Q = self.mesh.Q_array
                t = self.t
                self.saved_times[t] = current_Q

            yield self.t, self.mesh.Q_array

        current_Q = self.mesh.Q_array
        t = self.t
        self.saved_times[t] = current_Q