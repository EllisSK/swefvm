from .base import SpatialReconstruction

class FirstOrder(SpatialReconstruction):
    def reconstruct(self, mesh):
        return mesh.Q_vector