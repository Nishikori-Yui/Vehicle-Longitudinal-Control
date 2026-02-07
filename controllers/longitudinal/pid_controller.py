import numpy as np

class PIDControllerAdaptive:
    """
    自适应 PID 控制器
    """
    def __init__(self, Kp0, Ki0, beta, p):
        self.Kp0, self.Ki0, self.beta = Kp0, Ki0, beta
        self.integrator = 0.0
        self.p = p

    def reset(self):
        self.integrator = 0.0

    def step(self, state, v_ref, dt):
        if not np.isfinite(state.u):
            return 0.0
        error = v_ref - state.u
        error = float(np.clip(error, -self.p.u_max, self.p.u_max))
        Kp = self.Kp0 * (1 + self.beta * abs(error))
        Ki = self.Ki0 * (1 + self.beta * abs(error))

        # 先计算未饱和控制量
        # 积分限幅，防止数值爆炸
        if Ki > 1e-6:
            int_limit = max(abs(self.p.F_max), abs(self.p.F_min)) / Ki
            self.integrator = float(np.clip(self.integrator, -int_limit, int_limit))

        u_unsat = Kp * error + Ki * self.integrator
        u_sat = np.clip(u_unsat, self.p.F_min, self.p.F_max)

        # 抗积分饱和：仅在不加剧饱和时积分
        if (u_unsat == u_sat) or (u_unsat > u_sat and error < 0) or (u_unsat < u_sat and error > 0):
            self.integrator += error * dt
            u_unsat = Kp * error + Ki * self.integrator
            u_sat = np.clip(u_unsat, self.p.F_min, self.p.F_max)

        return float(u_sat)
