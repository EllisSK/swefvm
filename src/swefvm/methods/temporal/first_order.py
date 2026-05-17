import numpy as np
from .base import TemporalIntegrator

class FirstOrderTemporal(TemporalIntegrator):
    def __init__(self):
        pass

    def integrate(self, mesh, physics, spatial, riemann, external_bcs, internal_bcs, dt):
        Q_n = mesh.Q_array

        Q_L, Q_R = spatial.reconstruct_conserved_variables(mesh)
        F = riemann.solve(Q_L, Q_R, physics, mesh.zb_interface)
        mesh.apply_internal_boundary_conditions(F, Q_L, Q_R, internal_bcs)
        flux_grad = (F[1:] - F[:-1]) / mesh.dx
        S = physics.source(Q_n, mesh.zb, mesh.mannings_n)
        
        Qn1 = np.zeros_like(Q_n)
        Qn1[1:-1] = -flux_grad + S[1:-1]

        mesh.Q_array = Q_n + (dt * Qn1)
        mesh.apply_external_boundary_conditions(external_bcs)