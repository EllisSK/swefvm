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
    def apply_boundary_conditions(self, boundary_conditions: Mapping[int, BoundaryCondition]):
        pass

class Mesh1D(Mesh):
    def __init__(self, length: float, resolution: float, initial_conditions: Callable, boundary_conditions: Mapping[int, BoundaryCondition], bed_function: Callable | None = None) -> None:
        super().__init__()
        self.length = length
        self.dx = resolution
        self.N = int(length / resolution)
        self.bcs = boundary_conditions

        self._check_bcs(self.bcs)

        if len(self.bcs) > 2:
            N_ghost = 2 + (2 * (len(self.bcs) - 2))
        else:
            N_ghost = 2

        #Create 2D array that is as long as the domain + ghost cells
        self.Q_array = np.zeros((self.N+N_ghost, 2))
        self.F_array = np.zeros((self.N+N_ghost, 2))

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

    def _check_bcs(self, boundary_conditions: Mapping[int, BoundaryCondition] | Mapping[str, VariableBoundaryCondition]):
        if 0 not in boundary_conditions or -1 not in boundary_conditions:
                    raise ValueError("Missing perimeter boundary conditions!")

    def apply_boundary_conditions(self, boundary_conditions: Mapping[int, BoundaryCondition] | Mapping[str, VariableBoundaryCondition]):
        for index, bc in boundary_conditions.items():
            if index == 0:
                #Left boundary
                interior_slice = self.Q_array[1]
                ghost_slice = self.Q_array[0]
            elif index == -1:
                #Right boundary
                interior_slice = self.Q_array[-2]
                ghost_slice = self.Q_array[-1]  
            else:
                # Internal boundary
                pass

            if isinstance(bc, VariableBoundaryCondition):
                bc.apply(
                    interior_slice=self.Q_array[1], 
                    ghost_slice=self.Q_array[0],
                    t=self.t
                )
            elif isinstance(bc, BoundaryCondition):
                bc.apply(
                    interior_slice=self.Q_array[1], 
                    ghost_slice=self.Q_array[0], 
                )