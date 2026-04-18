from abc import ABC, abstractmethod

class Physics(ABC):
    @abstractmethod
    def dynamic_timestep(self, Q_array, zb) -> float:
        pass