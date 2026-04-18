import numpy as np

from abc import ABC, abstractmethod
from typing import Callable

class BoundaryCondition:
    @abstractmethod
    def apply(self, interior_slice: np.ndarray, ghost_slice: np.ndarray):
        pass

class ReflectiveBoundary(BoundaryCondition):
    def apply(self, interior_slice: np.ndarray, ghost_slice: np.ndarray, normal_idx: int = 1):
        #Copy the depth and mirror q
        ghost_slice[:] = interior_slice[:]
        ghost_slice[..., normal_idx] = -interior_slice[..., normal_idx]

class TransmissiveBoundary(BoundaryCondition):
    def apply(self, interior_slice: np.ndarray, ghost_slice: np.ndarray):
        #Copy the interior cell values to the ghost cell
        ghost_slice[:] = interior_slice[:]

class FixedFlowBoundary(TransmissiveBoundary):
    def __init__(self, target_q: float):
        self.target_q = target_q

    def apply(self, interior_slice: np.ndarray, ghost_slice: np.ndarray, normal_idx: int = 1):
        #Copy h and set a fixed q in the normal idx direction (future proofing for 2D)
        ghost_slice[:] = interior_slice[:]
        ghost_slice[..., normal_idx] = self.target_q

class FixedDepthBoundary(TransmissiveBoundary):
    def __init__(self, target_h: float):
        self.target_h = target_h

    def apply(self, interior_slice: np.ndarray, ghost_slice: np.ndarray):
        #Copy q and set a fixed depth
        ghost_slice[:] = interior_slice[:]
        ghost_slice[..., 0] = self.target_h

class VariableBoundaryCondition(BoundaryCondition):
    @abstractmethod
    def apply(self, interior_slice: np.ndarray, ghost_slice: np.ndarray, t: float):
        pass       

class VariableFlowBoundary(VariableBoundaryCondition):
    def __init__(self, q_t: Callable):
        self.q_t = q_t

    def apply(self, interior_slice: np.ndarray, ghost_slice: np.ndarray, t: float, normal_idx: int = 1):
        ghost_slice[:] = interior_slice[:]
        ghost_slice[..., normal_idx] = self.q_t(t)

class VariableDepthBoundary(VariableBoundaryCondition):
    def __init__(self, h_t: Callable):
        self.h_t = h_t

    def apply(self, interior_slice: np.ndarray, ghost_slice: np.ndarray, t: float):
        ghost_slice[:] = interior_slice[:]
        ghost_slice[..., 0] = self.h_t(t)

class VariableConservedBoundary(VariableBoundaryCondition):
    def __init__(self, h_t: Callable, q_t: Callable):
        #Store passed functions within class object
        self.h_t = h_t
        self.q_t = q_t

    def apply(self, interior_slice: np.ndarray, ghost_slice: np.ndarray, t: float, normal_idx: int = 1):
        ghost_slice[:] = interior_slice[:]

        #Apply Dirichlet boundary condition to eta and q, normal index implemented such that the method could potentially be applied to 2D problems (normal_idx = 2 in this case)
        ghost_slice[..., 0] = self.h_t(t)
        ghost_slice[..., normal_idx] = self.q_t(t)                