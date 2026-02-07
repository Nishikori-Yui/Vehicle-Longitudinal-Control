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
        error = v_ref - state.u
        Kp = self.Kp0 * (1 + self.beta * abs(error))
        Ki = self.Ki0 * (1 + self.beta * abs(error))

        # 先计算未饱和控制量
        u_unsat = Kp * error + Ki * self.integrator
        u_sat = np.clip(u_unsat, self.p.F_min, self.p.F_max)

        # 抗积分饱和：仅在不加剧饱和时积分
        if (u_unsat == u_sat) or (u_unsat > u_sat and error < 0) or (u_unsat < u_sat and error > 0):
            self.integrator += error * dt
            u_unsat = Kp * error + Ki * self.integrator
            u_sat = np.clip(u_unsat, self.p.F_min, self.p.F_max)

        return float(u_sat)
