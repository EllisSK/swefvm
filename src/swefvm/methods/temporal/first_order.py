import numpy as np

from .base import TemporalIntegrator

class FirstOrderTemporal(TemporalIntegrator):
    def __init__(self):
        pass

    def integrate(self, mesh, physics, spatial, riemann, external_bcs, internal_bcs, dt):
        Q_n = mesh.Q_array
        interior = mesh.interior_slice

        flux_div = np.zeros_like(Q_n[interior])

        for d in mesh.directions:
            Q_L, Q_R = spatial.reconstruct_conserved_variables(mesh, d)
            F = riemann.solve(Q_L, Q_R, physics, mesh.interface_bed(d), d + 1)
            mesh.apply_internal_boundary_conditions(F, Q_L, Q_R, internal_bcs, d)

            slicer = list(interior)
            slicer[d] = slice(None)
            flux_div += np.diff(F, axis=d)[tuple(slicer)] / mesh.spacing(d)

        S = physics.source(Q_n, mesh, mesh.mannings_n)

        Qn1 = np.zeros_like(Q_n)
        Qn1[interior] = -flux_div + S[interior]

        mesh.Q_array = Q_n + (dt * Qn1)
        mesh.apply_external_boundary_conditions(external_bcs)