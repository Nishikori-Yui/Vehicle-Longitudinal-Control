import numpy as np

class SecondOrderActuator:
    """
    二阶执行机构模型
    """
    def __init__(self, wn, zeta):
        self.A = np.array([[0, 1], [-wn**2, -2*zeta*wn]])
        self.B = np.array([[0], [wn**2]])
        self.x = np.zeros((2,1))

    def reset(self):
        self.x.fill(0)

    def step(self, u, dt):
        self.x += (self.A @ self.x + self.B * u) * dt
        return self.x[0,0]