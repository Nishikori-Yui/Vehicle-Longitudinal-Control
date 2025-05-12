class VehicleParams:
    def __init__(self):
        # 基础参数
        self.m = 1500        # 质量 (kg)
        self.g = 9.81        # 重力加速度 (m/s²)
        self.wheelbase = 2.7 # 轴距 (m)
        # 空气动力学
        self.Cd = 0.30
        self.A  = 2.2        # m²
        self.rho = 1.225     # kg/m³
        # 滚动阻力
        self.Cr = 0.015
        # 二阶执行器参数
        self.wn   = 5.65
        self.zeta = 0.707
        # ABS 参数
        self.r = 0.3         # m
        self.J = 0.35        # kg·m²
        self.c1, self.c2, self.c3 = 1.28, 23.99, 0.52
        self.tau_hyd = 0.5   # s
        # 牵引/制动力限制
        self.F_max =  4000   # N
        self.F_min = -5000   # N
        self.tau   = 0.5     # s