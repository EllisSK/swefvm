import numpy as np

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Callable
from .boundaries import BoundaryCondition, VariableBoundaryCondition

class Mesh(ABC):
    def __init__(self) -> None:
        self.Q_array : np.ndarray
        self.F_array : np.ndarray
        self.zb : np.ndarray
        self.zb_interface : np.ndarray
        self.mannings_n : float = 0.0
        self.t : float = 0.0
        self.x_vals : np.ndarray

    @abstractmethod
    def apply_boundary_conditions(self, boundary_conditions: Mapping[str, BoundaryCondition]):
        pass

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
            self.zb = np.zeros((self.N+2, 1))
            self.zb_interface = np.zeros((self.N+1, 1))

    def apply_boundary_conditions(self, boundary_conditions: Mapping[str, BoundaryCondition] | Mapping[str, VariableBoundaryCondition]):
        lb = boundary_conditions["left_boundary"]
        rb = boundary_conditions["right_boundary"]

        if isinstance(lb, VariableBoundaryCondition):
            lb.apply(
                interior_slice=self.Q_array[1], 
                ghost_slice=self.Q_array[0],
                t=self.t
            )
        elif isinstance(lb, BoundaryCondition):
            lb.apply(
                interior_slice=self.Q_array[1], 
                ghost_slice=self.Q_array[0], 
            )

        if isinstance(rb, VariableBoundaryCondition):
            rb.apply(
                interior_slice=self.Q_array[-2], 
                ghost_slice=self.Q_array[-1],
                t=self.t
            )
        elif isinstance(rb, BoundaryCondition):
            rb.apply(
                interior_slice=self.Q_array[-2], 
                ghost_slice=self.Q_array[-1], 
            )