import numpy as np

from abc import ABC, abstractmethod
from typing import Callable

Location = int | tuple[int, int]

class BoundaryCondition:
    def __init__(self, location: Location):
        self.location = location

class ExternalBoundary(BoundaryCondition):
    @abstractmethod
    def apply(self, interior: np.ndarray, ghost: np.ndarray, normal_idx: int, t: float = 0.0):
        pass

class InternalBoundary(BoundaryCondition):
    @abstractmethod
    def apply(self, F_int_slice: np.ndarray, Q_L_slice: np.ndarray, Q_R_slice: np.ndarray, zb_interface: float, normal_idx: int, t: float = 0.0):
        pass

class ReflectiveBoundary(ExternalBoundary):
    def apply(self, interior: np.ndarray, ghost: np.ndarray, normal_idx: int, t: float = 0.0):
        ghost[:] = interior[:]
        ghost[..., normal_idx] = -interior[..., normal_idx]

class TransmissiveBoundary(ExternalBoundary):
    def apply(self, interior: np.ndarray, ghost: np.ndarray, normal_idx: int, t: float = 0.0):
        ghost[:] = interior[:]

class FixedFlowBoundary(ExternalBoundary):
    def __init__(self, location: Location, target_q: float):
        super().__init__(location)
        self.target_q = target_q

    def apply(self, interior: np.ndarray, ghost: np.ndarray, normal_idx: int, t: float = 0.0):
        ghost[:] = interior[:]
        ghost[..., normal_idx] = self.target_q

class FixedDepthBoundary(ExternalBoundary):
    def __init__(self, location: Location, target_h: float):
        super().__init__(location)
        self.target_h = target_h

    def apply(self, interior: np.ndarray, ghost: np.ndarray, normal_idx: int, t: float = 0.0):
        ghost[:] = interior[:]
        ghost[..., 0] = self.target_h

class VariableFlowBoundary(ExternalBoundary):
    def __init__(self, location: Location, q_t: Callable):
        super().__init__(location)
        self.q_t = q_t

    def apply(self, interior: np.ndarray, ghost: np.ndarray, normal_idx: int, t: float = 0.0):
        ghost[:] = interior[:]
        ghost[..., normal_idx] = self.q_t(t)

class VariableDepthBoundary(ExternalBoundary):
    def __init__(self, location: Location, h_t: Callable):
        super().__init__(location)
        self.h_t = h_t

    def apply(self, interior: np.ndarray, ghost: np.ndarray, normal_idx: int, t: float = 0.0):
        ghost[:] = interior[:]
        ghost[..., 0] = self.h_t(t)

class VariableConservedBoundary(ExternalBoundary):
    def __init__(self, location: Location, h_t: Callable, q_t: Callable):
        super().__init__(location)
        self.h_t = h_t
        self.q_t = q_t

    def apply(self, interior: np.ndarray, ghost: np.ndarray, normal_idx: int, t: float = 0.0):
        ghost[:] = interior[:]
        ghost[..., 0] = self.h_t(t)
        ghost[..., normal_idx] = self.q_t(t)

class FixedFluxInterface(InternalBoundary):
    def __init__(self, location: Location, target_flux):
        super().__init__(location)
        self.target_flux = np.asarray(target_flux, dtype=float)

    def apply(self, F_int_slice: np.ndarray, Q_L_slice: np.ndarray, Q_R_slice: np.ndarray, zb_interface: float, normal_idx: int, t: float = 0.0):
        F_int_slice[:] = self.target_flux

class VariableFluxInterface(InternalBoundary):
    def __init__(self, location: Location, flux_t: Callable):
        super().__init__(location)
        self.flux_t = flux_t

    def apply(self, F_int_slice: np.ndarray, Q_L_slice: np.ndarray, Q_R_slice: np.ndarray, zb_interface: float, normal_idx: int, t: float = 0.0):
        F_int_slice[:] = self.flux_t(t)

class ClosedInterface(InternalBoundary):
    def apply(self, F_int_slice: np.ndarray, Q_L_slice: np.ndarray, Q_R_slice: np.ndarray, zb_interface: float, normal_idx: int, t: float = 0.0):
        F_int_slice[:] = 0.0

class WeirInterface(InternalBoundary):
    def __init__(self, location: Location, crest_elevation: float, discharge_coefficient: float = 0.611, g: float = 9.81):
        super().__init__(location)
        self.z_crest = crest_elevation
        self.C_d = discharge_coefficient
        self.g = g

    def apply(self, F_int_slice: np.ndarray, Q_L_slice: np.ndarray, Q_R_slice: np.ndarray, zb_interface: float, normal_idx: int, t: float = 0.0):
        eta_L = Q_L_slice[0]
        eta_R = Q_R_slice[0]
        eta_up = max(eta_L, eta_R)
        sign = 1.0 if eta_L >= eta_R else -1.0
        H = max(eta_up - self.z_crest, 0.0)
        q = sign * self.C_d * (2.0 / 3.0) * np.sqrt(2.0 * self.g) * np.power(H, 1.5)
        F_int_slice[0] = q
        F_int_slice[normal_idx] = 0.5 * self.g * (np.power(eta_up, 2) - 2.0 * eta_up * zb_interface)

class SluiceGateInterface(InternalBoundary):
    def __init__(self, location: Location, gate_opening: float, discharge_coefficient: float = 0.6, g: float = 9.81):
        super().__init__(location)
        self.a = gate_opening
        self.C_d = discharge_coefficient
        self.g = g

    def apply(self, F_int_slice: np.ndarray, Q_L_slice: np.ndarray, Q_R_slice: np.ndarray, zb_interface: float, normal_idx: int, t: float = 0.0):
        eta_L = Q_L_slice[0]
        eta_R = Q_R_slice[0]
        eta_up = max(eta_L, eta_R)
        h_up = max(eta_up - zb_interface, 0.0)
        sign = 1.0 if eta_L >= eta_R else -1.0
        a_eff = min(self.a, h_up)
        q = sign * self.C_d * a_eff * np.sqrt(2.0 * self.g * h_up)
        F_int_slice[0] = q
        F_int_slice[normal_idx] = 0.5 * self.g * (np.power(eta_up, 2) - 2.0 * eta_up * zb_interface)

class VariableSluiceGateInterface(InternalBoundary):
    def __init__(self, location: Location, gate_opening_t: Callable, discharge_coefficient: float = 0.6, g: float = 9.81):
        super().__init__(location)
        self.a_t = gate_opening_t
        self.C_d = discharge_coefficient
        self.g = g

    def apply(self, F_int_slice: np.ndarray, Q_L_slice: np.ndarray, Q_R_slice: np.ndarray, zb_interface: float, normal_idx: int, t: float = 0.0):
        a = self.a_t(t)
        eta_L = Q_L_slice[0]
        eta_R = Q_R_slice[0]
        eta_up = max(eta_L, eta_R)
        h_up = max(eta_up - zb_interface, 0.0)
        sign = 1.0 if eta_L >= eta_R else -1.0
        a_eff = min(a, h_up)
        q = sign * self.C_d * a_eff * np.sqrt(2.0 * self.g * h_up)
        F_int_slice[0] = q
        F_int_slice[normal_idx] = 0.5 * self.g * (np.power(eta_up, 2) - 2.0 * eta_up * zb_interface)

class StageDischargeInterface(InternalBoundary):
    def __init__(self, location: Location, rating_curve: Callable, g: float = 9.81):
        super().__init__(location)
        self.rating_curve = rating_curve
        self.g = g

    def apply(self, F_int_slice: np.ndarray, Q_L_slice: np.ndarray, Q_R_slice: np.ndarray, zb_interface: float, normal_idx: int, t: float = 0.0):
        eta_L = Q_L_slice[0]
        eta_R = Q_R_slice[0]
        eta_up = max(eta_L, eta_R)
        h_up = max(eta_up - zb_interface, 0.0)
        sign = 1.0 if eta_L >= eta_R else -1.0
        q = sign * self.rating_curve(h_up)
        F_int_slice[0] = q
        F_int_slice[normal_idx] = 0.5 * self.g * (np.power(eta_up, 2) - 2.0 * eta_up * zb_interface)