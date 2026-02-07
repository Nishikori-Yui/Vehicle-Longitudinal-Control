from __future__ import annotations

import numpy as np
from powertrain.base import Powertrain
from controllers.longitudinal.abs_controller import ABSController


class ThrottleBrakePowertrain(Powertrain):
    """
    简化油门/制动动力系统：
    - cmd 可以是 float (期望纵向力) 或 dict(throttle, brake)
    - throttle/brake 均为 0~1
    """
    def __init__(self, p, abs_enabled: bool = False):
        self.p = p
        self.abs_enabled = abs_enabled
        self.abs = ABSController(p) if abs_enabled else None
        self.f_drive = 0.0
        self.f_brake = 0.0

    def reset(self) -> None:
        self.f_drive = 0.0
        self.f_brake = 0.0
        if self.abs:
            self.abs.reset()

    def _force_to_tb(self, cmd: float):
        if cmd >= 0:
            throttle = min(cmd / self.p.F_drive_max, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(-cmd / self.p.F_brake_max, 1.0)
        return throttle, brake

    def compute_force(self, cmd, state, dt: float) -> float:
        _ = state
        if isinstance(cmd, dict):
            throttle = float(cmd.get("throttle", 0.0))
            brake = float(cmd.get("brake", 0.0))
        else:
            throttle, brake = self._force_to_tb(float(cmd))

        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        target_drive = throttle * self.p.F_drive_max
        target_brake = brake * self.p.F_brake_max

        # 一阶滞后
        self.f_drive += (dt / self.p.tau_drive) * (target_drive - self.f_drive)
        self.f_brake += (dt / self.p.tau_brake) * (target_brake - self.f_brake)

        brake_force = -self.f_brake
        if self.abs_enabled and self.abs and brake_force < 0.0:
            brake_force = self.abs.step(state.u, brake_force, dt)

        fx = self.f_drive + brake_force
        return float(np.clip(fx, self.p.F_min, self.p.F_max))
