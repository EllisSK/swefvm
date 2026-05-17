from abc import ABC, abstractmethod

class SpatialReconstruction(ABC):
    @abstractmethod
    def reconstruct_conserved_variables(self, mesh, direction: int = 0) -> tuple:
        pass

def axis_slice(ndim: int, axis: int, start, stop) -> tuple:
    sl = [slice(None)] * ndim
    sl[axis] = slice(start, stop)
    return tuple(sl)