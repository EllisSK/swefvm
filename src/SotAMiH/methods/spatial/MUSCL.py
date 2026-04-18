import numpy as np

from .base import SpatialReconstruction

class MUSCL1D(SpatialReconstruction):
    def __init__(self):
        pass

    def _minmod(self, x, y):
        return 0.5 * (np.sign(x) + np.sign(y)) * np.minimum(np.abs(x), np.abs(y))

    def reconstruct_conserved_variables(self, mesh):
        grad_U = np.zeros_like(mesh.Q_array)
        
        #Calculate gradients with shifted arrays
        diff_bwd = (mesh.Q_array[1:-1] - mesh.Q_array[:-2]) / mesh.dx
        diff_fwd = (mesh.Q_array[2:] - mesh.Q_array[1:-1]) / mesh.dx
        
        grad_U[1:-1] = self._minmod(diff_bwd, diff_fwd)

        Q_L = mesh.Q_array[:-1] + (0.5 * mesh.dx * grad_U[:-1])
        Q_R = mesh.Q_array[1:]  - (0.5 * mesh.dx * grad_U[1:])
        
        return Q_L, Q_R