import numpy as np
from scipy.optimize import minimize
from models.actuator import SecondOrderActuator
from models.vehicle import disturbance

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
        self.act = SecondOrderActuator(p.wn, p.zeta)
        self.u_prev = 0.0

    def reset(self):
        self.act.reset()
        self.u_prev = 0.0

    def step(self, v_curr, v_ref, dt):
        """
        执行一次 MPC 步长计算，返回最优输入和执行器输出
        v_curr: 当前速度
        v_ref: 目标速度
        dt:    时间步长
        """
        def obj(u):
            v_p = v_curr
            Fp = self.act.x[0,0]
            cost = 0.0
            for ui in u:
                # 执行器动力学预测
                Fp += (dt / self.p.tau) * (ui - Fp)
                # 干扰项计算
                d = disturbance(v_p, self.p)
                # 预测速度
                v_p += dt * (Fp - d) / self.p.m
                # 累计代价
                cost += ( self.q * (v_p - v_ref)**2
                          + self.r * (ui - self.u_prev)**2
                          + self.e_energy * ui**2 )
            return cost

        # 优化初值与约束
        u0 = np.ones(self.N) * self.u_prev
        bounds = [(-self.p.F_max, self.p.F_max) for _ in range(self.N)]
        res = minimize(obj, u0, bounds=bounds, options={'maxiter':100})

        u_cmd = res.x[0]
        self.u_prev = u_cmd
        # 真正执行一步
        u_opt = u_cmd
        F_act = self.act.step(u_opt, dt)
        return u_opt, F_act