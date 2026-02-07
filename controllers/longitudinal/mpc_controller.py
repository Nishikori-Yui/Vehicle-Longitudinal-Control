import numpy as np
from scipy.optimize import minimize
from models.vehicle import longitudinal_resistance

class MPCControllerMultiObjective:
    """
    多目标 MPC 控制器
    """
    def __init__(self, N, q, r, e_energy, p):
        self.N = N
        self.q = q
        self.r = r
        self.e_energy = e_energy
        self.p = p                     # 保存车辆参数引用
        self.u_prev = 0.0
        self._grade = 0.0

    def reset(self):
        self.u_prev = 0.0

    def set_env(self, mu=None, grade=None):
        if grade is not None:
            self._grade = float(grade)

    def step(self, state, v_ref, dt):
        """
        执行一次 MPC 步长计算，返回最优纵向力命令
        v_curr: 当前速度
        v_ref: 目标速度
        dt:    时间步长
        """
        v_curr = float(state.u if np.isfinite(state.u) else 0.0)

        def obj(u):
            v_p = v_curr
            cost = 0.0
            u_last = self.u_prev
            for ui in u:
                # 阻力项计算
                d = longitudinal_resistance(v_p, self.p, grade=self._grade)
                # 预测速度（纵向动力学）
                v_p += dt * (ui - d) / self.p.m
                # 累计代价
                cost += (self.q * (v_p - v_ref) ** 2
                         + self.r * (ui - u_last) ** 2
                         + self.e_energy * ui ** 2)
                u_last = ui
            return cost

        # 优化初值与约束
        u_prev = float(self.u_prev if np.isfinite(self.u_prev) else 0.0)
        u0 = np.ones(self.N) * u_prev
        bounds = [(self.p.F_min, self.p.F_max) for _ in range(self.N)]
        res = minimize(obj, u0, method="L-BFGS-B", bounds=bounds, options={'maxiter': 100})

        u_cmd = float(res.x[0]) if res.success and np.isfinite(res.x[0]) else float(u0[0])
        self.u_prev = u_cmd
        return u_cmd
