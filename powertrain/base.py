from __future__ import annotations

from abc import ABC, abstractmethod


class Powertrain(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_force(self, cmd: float, state, dt: float) -> float:
        raise NotImplementedError
