from abc import ABC, abstractmethod

class Physics(ABC):
    @abstractmethod
    def flux(self, Q_array, zb, normal_idx: int = 1):
        pass

    @abstractmethod
    def source(self, Q_array, mesh, mannings_n):
        pass

    @abstractmethod
    def max_wave_speed(self, Q_array, zb):
        pass

    @abstractmethod
    def dynamic_timestep(self, Q_array, mesh) -> float:
        pass