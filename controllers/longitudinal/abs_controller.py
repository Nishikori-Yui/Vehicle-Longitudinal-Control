import numpy as np

class ABSController:
    """
    ABS 防抱死制动控制器
    """
    def __init__(self, p):
        self.r, self.J = p.r, p.J
        self.c1, self.c2, self.c3 = p.c1, p.c2, p.c3
        self.tau = p.tau_hyd
        self.omega = 0.0

    def burckhardt(self, slip):
        return self.c1 * (1 - np.exp(-self.c2 * slip)) - self.c3 * slip

    def reset(self):
        self.omega = 0.0

    def step(self, v, u_brake, dt, p):
        slip = (v - self.omega * p.r) / max(v, self.omega * p.r, 1e-3)
        slip = np.clip(slip, -1, 1)
        mu   = self.burckhardt(slip)
        F_des = mu * p.m * p.g * np.sign(u_brake)
        self.omega += (dt / self.tau) * (u_brake - self.omega)
        return F_des