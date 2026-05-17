from abc import ABC, abstractmethod

class TemporalIntegrator(ABC):
    @abstractmethod
    def integrate(self, mesh, physics, spatial, riemann, external_bcs, internal_bcs, dt):
        pass