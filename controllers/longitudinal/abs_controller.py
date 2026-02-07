import numpy as np

class ABSController:
    """
    ABS 防抱死制动控制器
    """
    def __init__(self, p):
        self.p = p
        self.r, self.J = p.r, p.J
        self.c1, self.c2, self.c3 = p.c1, p.c2, p.c3
        self.tau = p.tau_hyd
        self.omega = 0.0

    def burckhardt(self, slip):
        return self.c1 * (1 - np.exp(-self.c2 * slip)) - self.c3 * slip

    def reset(self):
        self.omega = 0.0

    def step(self, v, brake_force_cmd, dt):
        p = self.p
        slip = (v - self.omega * p.r) / max(v, self.omega * p.r, 1e-3)
        slip = np.clip(slip, -1, 1)
        mu   = self.burckhardt(slip)
        F_limit = mu * p.m * p.g

        # 目标滑移率，随制动力增大
        slip_target = 0.2 + 0.6 * min(abs(brake_force_cmd) / max(p.F_brake_max, 1e-3), 1.0)
        slip_target = np.clip(slip_target, 0.1, 0.8)
        omega_target = max(v * (1.0 - slip_target) / p.r, 0.0)
        self.omega += (dt / self.tau) * (omega_target - self.omega)

        brake_force = -min(abs(brake_force_cmd), F_limit)
        return brake_force
