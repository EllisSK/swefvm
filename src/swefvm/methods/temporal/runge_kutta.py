import numpy as np

from .base import TemporalIntegrator

class RK2(TemporalIntegrator):
    def __init__(self):
        pass

    def integrate(self, mesh, physics, spatial, riemann, external_bcs, internal_bcs, dt):
        Q_n = mesh.Q_array

        Q_L, Q_R = spatial.reconstruct_conserved_variables(mesh)
        F_int = riemann.solve(Q_L, Q_R, physics, mesh.zb_interface)
        mesh.apply_internal_boundary_conditions(F_int, Q_L, Q_R, internal_bcs)
        flux_grad = (F_int[1:] - F_int[:-1]) / mesh.dx
        S = physics.source(Q_n, mesh, mesh.mannings_n)

        K1 = np.zeros_like(Q_n)
        K1[1:-1] = -flux_grad + S[1:-1]

        U_star = Q_n + (dt * K1)

        mesh.Q_array = U_star
        mesh.apply_external_boundary_conditions(external_bcs)

        Q_L_star, Q_R_star = spatial.reconstruct_conserved_variables(mesh)
        F_int_star = riemann.solve(Q_L_star, Q_R_star, physics, mesh.zb_interface)
        mesh.apply_internal_boundary_conditions(F_int_star, Q_L_star, Q_R_star, internal_bcs)
        flux_grad_star = (F_int_star[1:] - F_int_star[:-1]) / mesh.dx
        S_star = physics.source(U_star, mesh, mesh.mannings_n)

        K2 = np.zeros_like(Q_n)
        K2[1:-1] = -flux_grad_star + S_star[1:-1]

        mesh.Q_array = Q_n + 0.5 * dt * (K1 + K2)
        mesh.apply_external_boundary_conditions(external_bcs)

class RK3(TemporalIntegrator):
    def __init__(self):
        raise NotImplementedError()

    def integrate(self, mesh, physics, spatial, riemann, external_bcs, internal_bcs, dt):
        pass