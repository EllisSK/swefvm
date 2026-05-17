import numpy as np

from .base import TemporalIntegrator

class RK2(TemporalIntegrator):
    def __init__(self):
        pass

    def integrate(self, mesh, physics, spatial, riemann, external_bcs, internal_bcs, dt):
        Q_n = mesh.Q_array
        interior = mesh.interior_slice

        flux_div = np.zeros_like(Q_n[interior])

        for d in mesh.directions:
            Q_L, Q_R = spatial.reconstruct_conserved_variables(mesh, d)
            F_int = riemann.solve(Q_L, Q_R, physics, mesh.interface_bed(d), d + 1)
            mesh.apply_internal_boundary_conditions(F_int, Q_L, Q_R, internal_bcs, d)

            slicer = list(interior)
            slicer[d] = slice(None)
            flux_div += np.diff(F_int, axis=d)[tuple(slicer)] / mesh.spacing(d)

        S = physics.source(Q_n, mesh, mesh.mannings_n)

        K1 = np.zeros_like(Q_n)
        K1[interior] = -flux_div + S[interior]

        U_star = Q_n + (dt * K1)

        mesh.Q_array = U_star
        mesh.apply_external_boundary_conditions(external_bcs)

        flux_div_star = np.zeros_like(Q_n[interior])

        for d in mesh.directions:
            Q_L_star, Q_R_star = spatial.reconstruct_conserved_variables(mesh, d)
            F_int_star = riemann.solve(Q_L_star, Q_R_star, physics, mesh.interface_bed(d), d + 1)
            mesh.apply_internal_boundary_conditions(F_int_star, Q_L_star, Q_R_star, internal_bcs, d)

            slicer = list(interior)
            slicer[d] = slice(None)
            flux_div_star += np.diff(F_int_star, axis=d)[tuple(slicer)] / mesh.spacing(d)

        S_star = physics.source(U_star, mesh, mesh.mannings_n)

        K2 = np.zeros_like(Q_n)
        K2[interior] = -flux_div_star + S_star[interior]

        mesh.Q_array = Q_n + 0.5 * dt * (K1 + K2)
        mesh.apply_external_boundary_conditions(external_bcs)

class RK3(TemporalIntegrator):
    def __init__(self):
        raise NotImplementedError()

    def integrate(self, mesh, physics, spatial, riemann, external_bcs, internal_bcs, dt):
        pass