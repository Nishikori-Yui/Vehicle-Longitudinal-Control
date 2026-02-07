from __future__ import annotations

import numpy as np
from models.actuator import SecondOrderActuator
from powertrain.base import Powertrain


class ForcePowertrain(Powertrain):
    """
    纵向力输入的可插拔动力系统。
    负责执行器动态与限幅，不改变上层控制器接口。
    """
    def __init__(self, p):
        self.p = p
        self.act = SecondOrderActuator(p.wn, p.zeta)

    def reset(self) -> None:
        self.act.reset()

    def compute_force(self, cmd: float, state, dt: float) -> float:
        cmd = float(np.clip(cmd, self.p.F_min, self.p.F_max))
        fx = self.act.step(cmd, dt)
        return float(np.clip(fx, self.p.F_min, self.p.F_max))
