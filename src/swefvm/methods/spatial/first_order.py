from .base import SpatialReconstruction

class FirstOrder(SpatialReconstruction):
    def reconstruct_conserved_variables(self, mesh):

        Q_L = mesh.Q_array[:-1]
        Q_R = mesh.Q_array[1:]
        
        return Q_L, Q_R