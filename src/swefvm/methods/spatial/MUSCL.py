import numpy as np

from .base import SpatialReconstruction, axis_slice

class MUSCL(SpatialReconstruction):
    def __init__(self):
        pass

    def _minmod(self, x, y):
        return 0.5 * (np.sign(x) + np.sign(y)) * np.minimum(np.abs(x), np.abs(y))

    def reconstruct_conserved_variables(self, mesh, direction: int = 0):
        Q = mesh.Q_array
        dx = mesh.spacing(direction)

        sl_lo = axis_slice(Q.ndim, direction, None, -2)
        sl_mid = axis_slice(Q.ndim, direction, 1, -1)
        sl_hi = axis_slice(Q.ndim, direction, 2, None)
        sl_L = axis_slice(Q.ndim, direction, None, -1)
        sl_R = axis_slice(Q.ndim, direction, 1, None)

        grad_U = np.zeros_like(Q)

        #Calculate gradients with shifted arrays
        diff_bwd = (Q[sl_mid] - Q[sl_lo]) / dx
        diff_fwd = (Q[sl_hi] - Q[sl_mid]) / dx

        grad_U[sl_mid] = self._minmod(diff_bwd, diff_fwd)

        Q_L = Q[sl_L] + (0.5 * dx * grad_U[sl_L])
        Q_R = Q[sl_R] - (0.5 * dx * grad_U[sl_R])

        return Q_L, Q_R