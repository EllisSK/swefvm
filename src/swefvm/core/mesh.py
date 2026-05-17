import numpy as np

from abc import ABC, abstractmethod
from typing import Callable
from .boundaries import ExternalBoundary, InternalBoundary

class Mesh(ABC):
    def __init__(self) -> None:
        self.Q_array : np.ndarray
        self.F_array : np.ndarray
        self.zb : np.ndarray
        self.mannings_n : float = 0.0
        self.t : float = 0.0

    @property
    @abstractmethod
    def directions(self) -> tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def interior_slice(self) -> tuple[slice, ...]:
        pass

    @abstractmethod
    def spacing(self, direction: int) -> float:
        pass

    @abstractmethod
    def interface_bed(self, direction: int) -> np.ndarray:
        pass

    @abstractmethod
    def _resolve_external(self, loc):
        pass

    @abstractmethod
    def _resolve_internal(self, loc, F_int: np.ndarray, Q_L: np.ndarray, Q_R: np.ndarray, direction: int):
        pass

    def apply_external_boundary_conditions(self, bcs: list[ExternalBoundary]):
        for bc in bcs:
            interior, ghost, normal_idx = self._resolve_external(bc.location)
            bc.apply(interior, ghost, normal_idx, t=self.t)

    def apply_internal_boundary_conditions(self, F_int: np.ndarray, Q_L: np.ndarray, Q_R: np.ndarray, bcs: list[InternalBoundary], direction: int = 0):
        for bc in bcs:
            f_view, ql_view, qr_view, zb, normal_idx = self._resolve_internal(bc.location, F_int, Q_L, Q_R, direction)
            bc.apply(f_view, ql_view, qr_view, zb, normal_idx, t=self.t)


class Mesh1D(Mesh):
    def __init__(self, length: float, resolution: float, initial_conditions: Callable, bed_function: Callable | None = None) -> None:
        super().__init__()
        self.length = length
        self.dx = resolution
        self.N = int(length / resolution)

        #Create 2D array that is as long as the domain + 2 ghost cells
        self.Q_array = np.zeros((self.N+2, 2))
        self.F_array = np.zeros((self.N+2, 2))

        self.x_vals = np.linspace(self.dx/2, self.length - (self.dx/2), self.N)
        self.Q_array[1:-1] = initial_conditions(self.x_vals)

        #Create bed elevations from bed function
        if bed_function:
            #Add elevations for ghost cells equal to the elevation of inner boundary cells
            x_vals = np.concatenate(([self.dx/2], self.x_vals, [self.length - (self.dx/2)]))
            self.zb = bed_function(x_vals)
            self.zb_interface = 0.5 * (self.zb[:-1] + self.zb[1:])
        else:
            #If no bed function, make all cells zb = 0
            self.zb = np.zeros((self.N+2,))
            self.zb_interface = np.zeros((self.N+1,))

    @property
    def directions(self) -> tuple[int, ...]:
        return (0,)

    @property
    def interior_slice(self) -> tuple[slice, ...]:
        return (slice(1, -1),)

    def spacing(self, direction: int) -> float:
        if direction != 0:
            raise ValueError(f"Mesh1D has no direction {direction}; valid directions are (0,)")
        return self.dx

    def interface_bed(self, direction: int) -> np.ndarray:
        if direction != 0:
            raise ValueError(f"Mesh1D has no direction {direction}; valid directions are (0,)")
        return self.zb_interface

    def _resolve_external(self, loc: int):
        if loc == 0:
            return self.Q_array[1], self.Q_array[0], 1
        elif loc == self.N + 1:
            return self.Q_array[-2], self.Q_array[-1], 1
        else:
            raise ValueError(f"Location {loc} is not a valid 1D ghost-cell index (expected 0 or {self.N + 1})")

    def _resolve_internal(self, loc: int, F_int: np.ndarray, Q_L: np.ndarray, Q_R: np.ndarray, direction: int):
        if direction != 0:
            raise ValueError(f"Mesh1D has no direction {direction}; valid directions are (0,)")
        if not (0 <= loc <= self.N):
            raise ValueError(f"Interface index {loc} out of range [0, {self.N}]")
        return F_int[loc], Q_L[loc], Q_R[loc], self.zb_interface[loc], 1