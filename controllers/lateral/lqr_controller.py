import numpy as np
from scipy.linalg import solve_continuous_are
from controllers.lateral.pure_pursuit import PurePursuitController


def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _ensure_path_geom(path):
    if "yaw" in path and "kappa" in path:
        return
    x = path["x"]
    y = path["y"]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    yaw = np.arctan2(dy, dx)
    denom = (dx * dx + dy * dy) ** 1.5
    denom = np.where(denom < 1e-6, 1e-6, denom)
    kappa = (dx * ddy - dy * ddx) / denom
    path["yaw"] = yaw
    path["kappa"] = kappa


class LQRLateralController:
    """
    LQR 横向控制器（动态自行车模型线性化）。
    状态: [e_y, e_psi, v, r]
    输出: 前轮转角 delta
    """
    def __init__(self, p, Q=None, R=1.0, delta_max: float = 0.6, v_switch_low: float = 3.0,
                 v_switch_high: float = 6.0):
        self.p = p
        self.Q = np.diag([4.0, 8.0, 1.0, 1.0]) if Q is None else np.array(Q)
        self.R = np.array([[R]])
        self._mu = p.mu0
        self._last_u = None
        self._K = None
        self.delta_max = float(delta_max)
        self.v_switch_low = float(v_switch_low)
        self.v_switch_high = float(v_switch_high)
        self._pp = PurePursuitController(p, Ld=5.0)

    def reset(self):
        self._last_u = None
        self._K = None
        self._pp.reset()

    def set_env(self, mu=None, grade=None):
        _ = grade
        if mu is not None:
            self._mu = float(mu)

    def _gain(self, u: float) -> np.ndarray:
        if self._last_u is not None and abs(self._last_u - u) < 0.2 and self._K is not None:
            return self._K
        u = max(u, 0.5)
        m = self.p.m
        Iz = self.p.Iz
        a = self.p.a
        b = self.p.b
        Cf = self.p.Caf * self._mu
        Cr = self.p.Car * self._mu

        A = np.array([
            [0.0, u, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -(2 * (Cf + Cr) / (m * u)), -(u + 2 * (a * Cf - b * Cr) / (m * u))],
            [0.0, 0.0, -(2 * (a * Cf - b * Cr) / (Iz * u)), -(2 * (a * a * Cf + b * b * Cr) / (Iz * u))],
        ])
        B = np.array([
            [0.0],
            [0.0],
            [2 * Cf / m],
            [2 * a * Cf / Iz],
        ])

        try:
            P = solve_continuous_are(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R) @ B.T @ P
        except Exception:
            # 低速或数值不稳定时退化为温和增益
            if self._K is not None:
                return self._K
            K = np.array([[0.2, 1.0, 0.0, 0.0]])
        self._K = K
        self._last_u = u
        return K

    def step(self, state, path, dt):
        _ = dt
        if state.u <= self.v_switch_low:
            return float(np.clip(self._pp.step(state, path, dt), -self.delta_max, self.delta_max))
        _ensure_path_geom(path)
        x_ref = path["x"]
        y_ref = path["y"]
        yaw_ref = path["yaw"]
        kappa_ref = path["kappa"]

        dx = state.x - x_ref
        dy = state.y - y_ref
        idx = int(np.argmin(dx * dx + dy * dy))

        psi_ref = yaw_ref[idx]
        kappa = kappa_ref[idx]

        # 有符号横向误差
        nx = -np.sin(psi_ref)
        ny = np.cos(psi_ref)
        e_y = dx[idx] * nx + dy[idx] * ny
        e_psi = _wrap_to_pi(state.psi - psi_ref)

        # 状态向量
        x = np.array([[e_y], [e_psi], [state.v], [state.r]])
        K = self._gain(state.u)

        # 前馈转向（几何曲率）
        delta_ff = np.arctan(self.p.wheelbase * kappa)
        delta_lqr = float((-K @ x).item() + delta_ff)
        if state.u >= self.v_switch_high:
            return float(np.clip(delta_lqr, -self.delta_max, self.delta_max))
        # 低速区间平滑过渡到纯追踪
        delta_pp = self._pp.step(state, path, dt)
        alpha = (state.u - self.v_switch_low) / max(self.v_switch_high - self.v_switch_low, 1e-3)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        delta = alpha * delta_lqr + (1.0 - alpha) * delta_pp
        return float(np.clip(delta, -self.delta_max, self.delta_max))
