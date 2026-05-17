from .base import SpatialReconstruction, axis_slice

class FirstOrder(SpatialReconstruction):
    def reconstruct_conserved_variables(self, mesh, direction: int = 0):
        Q = mesh.Q_array
        Q_L = Q[axis_slice(Q.ndim, direction, None, -1)]
        Q_R = Q[axis_slice(Q.ndim, direction, 1, None)]

        return Q_L, Q_R