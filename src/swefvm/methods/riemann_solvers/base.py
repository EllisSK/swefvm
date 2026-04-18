from abc import ABC, abstractmethod

class RiemannSolver(ABC):
    @abstractmethod
    def solve(self, Q_L, Q_R, physics, zb):
        pass