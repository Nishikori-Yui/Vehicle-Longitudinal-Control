import numpy as np
from models.actuator import SecondOrderActuator

class PIDControllerAdaptive:
    """
    自适应 PID 控制器
    """
    def __init__(self, Kp0, Ki0, beta, p):
        self.Kp0, self.Ki0, self.beta = Kp0, Ki0, beta
        self.integrator = 0.0
        self.act = SecondOrderActuator(p.wn, p.zeta)

    def reset(self):
        self.integrator = 0.0
        self.act.reset()

    def step(self, error, dt):
        Kp = self.Kp0 * (1 + self.beta * abs(error))
        Ki = self.Ki0 * (1 + self.beta * abs(error))
        self.integrator += error * dt
        u = np.clip(Kp * error + Ki * self.integrator, -5000, 4000)
        F_act = self.act.step(u, dt)
        return u, F_act