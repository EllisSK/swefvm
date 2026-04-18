from abc import ABC, abstractmethod

class TemporalIntegrator(ABC):
    @abstractmethod
    def integrate(self, mesh, physics, spatial, riemann, bcs, dt):
        pass