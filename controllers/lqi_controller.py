import numpy as np
from scipy.linalg import solve_continuous_are
from models.actuator import SecondOrderActuator

class LQIController:
    """
    增广 LQI 控制器
    """
    def __init__(self, v0, p):
        a0 = 0.5 * p.rho * p.Cd * p.A * v0**2 + p.Cr * p.m * p.g
        b0 = p.rho * p.Cd * p.A * v0
        A_lin = np.array([[-b0 / p.m]])
        B_lin = np.array([[1 / p.m]])
        A_aug = np.block([[A_lin, np.zeros((1,1))], [-1, np.zeros((1,1))]])
        B_aug = np.vstack([B_lin, [0]])
        Q, R = np.diag([1000, 100]), np.array([[0.01]])
        P = solve_continuous_are(A_aug, B_aug, Q, R)
        self.K = np.linalg.inv(R) @ B_aug.T @ P
        self.x = np.zeros((2,1))
        self.act = SecondOrderActuator(p.wn, p.zeta)

    def reset(self):
        self.x.fill(0)
        self.act.reset()

    def step(self, v, v_ref, dt):
        e = v_ref - v
        self.x[0,0] = e
        self.x[1,0] += e * dt
        u = np.clip(-(self.K @ self.x).item(), -5000, 4000)
        return u, self.act.step(u, dt)