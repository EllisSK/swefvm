from abc import ABC, abstractmethod

class SpatialReconstruction(ABC):
    @abstractmethod
    def reconstruct_conserved_variables(self, mesh) -> tuple:
        pass