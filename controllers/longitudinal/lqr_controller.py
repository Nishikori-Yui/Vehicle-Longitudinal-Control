import numpy as np
from scipy.linalg import solve_continuous_are
from models.vehicle import longitudinal_resistance


class LQRController:
    """
    纵向 LQR 控制器（误差状态 + 前馈参考力）。
    输出为期望纵向力 Fx_cmd。
    """
    def __init__(self, q, r, p):
        self.q = float(q)
        self.r = float(r)
        self.p = p
        self._last_v_ref = None
        self._K = 0.0
        self._grade = 0.0

    def reset(self):
        self._last_v_ref = None
        self._K = 0.0

    def set_env(self, mu=None, grade=None):
        if grade is not None:
            self._grade = float(grade)

    def _compute_gain(self, v_ref: float) -> float:
        if self._last_v_ref == v_ref:
            return self._K
        A = -(self.p.rho * self.p.Cd * self.p.A / self.p.m) * v_ref
        B = 1.0 / self.p.m
        A_mat = np.array([[A]])
        B_mat = np.array([[B]])
        Q = np.array([[self.q]])
        R = np.array([[self.r]])
        P = solve_continuous_are(A_mat, B_mat, Q, R)
        K = np.linalg.inv(R) @ B_mat.T @ P
        self._K = float(K.item())
        self._last_v_ref = float(v_ref)
        return self._K

    def step(self, state, v_ref, dt):
        _ = dt
        e = state.u - v_ref
        K = self._compute_gain(v_ref)
        F_ff = longitudinal_resistance(v_ref, self.p, grade=self._grade)
        u = F_ff - K * e
        return float(np.clip(u, self.p.F_min, self.p.F_max))
