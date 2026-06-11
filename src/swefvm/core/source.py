from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class SourceTerm(ABC):
    def __init__(self, component: int):
        self.component = component

    @abstractmethod
    def apply(
        self, S_array: np.ndarray, Q_array: np.ndarray, mesh, t: float, dt: float
    ) -> None: ...


class ConstantSource(SourceTerm):
    def __init__(self, component: int, rate: float):
        super().__init__(component)
        self.rate = rate

    def apply(self, S_array, Q_array, mesh, t, dt):
        S_array[(*mesh.interior_slice, self.component)] += self.rate


class VariableSource(SourceTerm):
    def __init__(self, component: int, rate_t: Callable[[float], float]):
        super().__init__(component)
        self.rate_t = rate_t

    def apply(self, S_array, Q_array, mesh, t, dt):
        S_array[(*mesh.interior_slice, self.component)] += self.rate_t(t)


class FieldSource(SourceTerm):
    def __init__(self, component: int, rate_fn: Callable[..., np.ndarray]):
        super().__init__(component)
        self.rate_fn = rate_fn

    def apply(self, S_array, Q_array, mesh, t, dt):
        S_array[(*mesh.interior_slice, self.component)] += self.rate_fn(mesh, t)


class StateDependentSource(SourceTerm):
    def __init__(self, component: int, rate_fn: Callable[..., np.ndarray]):
        super().__init__(component)
        self.rate_fn = rate_fn

    def apply(self, S_array, Q_array, mesh, t, dt):
        S_array[(*mesh.interior_slice, self.component)] += self.rate_fn(
            Q_array, mesh, t, dt
        )
