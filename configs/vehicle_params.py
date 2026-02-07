class VehicleParams:
    def __init__(self, preset=None):
        # 基础参数
        self.m = 1500        # 质量 (kg)
        self.g = 9.81        # 重力加速度 (m/s²)
        self.wheelbase = 2.7 # 轴距 (m)
        # 质心位置（相对前轴/后轴）
        self.a = 1.2         # 质心到前轴 (m)
        self.b = self.wheelbase - self.a  # 质心到后轴 (m)
        # 偏航转动惯量
        self.Iz = 2500.0     # kg·m²（中型轿车典型量级）
        # 空气动力学
        self.Cd = 0.30
        self.A  = 2.2        # m²
        self.rho = 1.225     # kg/m³
        # 滚动阻力
        self.Cr = 0.015
        # 轮胎侧偏刚度（线性模型）
        self.Caf = 80000.0   # N/rad
        self.Car = 80000.0   # N/rad
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
        # 油门/制动模型参数
        self.F_drive_max = 4000  # N
        self.F_brake_max = 5000  # N
        self.tau_drive = 0.3     # s
        self.tau_brake = 0.2     # s
        # 环境默认值
        self.mu0 = 0.9       # 干燥路面典型附着
        self.grade0 = 0.0    # 默认坡度 (tan(theta))
        # 数值安全限制
        self.u_max = 60.0    # 最大纵向速度 (m/s)
        self.v_max = 15.0    # 最大横向速度 (m/s)
        self.r_max = 2.0     # 最大偏航角速度 (rad/s)
        self.alpha_max = 0.35  # 最大侧偏角 (rad)

        if preset:
            self.apply_preset(preset)

    def apply_preset(self, name: str) -> None:
        if name == "mid_sedan":
            self.m = 1550
            self.wheelbase = 2.75
            self.a = 1.25
            self.b = self.wheelbase - self.a
            self.Iz = 2800.0
            self.Cd = 0.29
            self.A = 2.2
            self.Cr = 0.014
            self.Caf = 90000.0
            self.Car = 90000.0
            self.F_max = 4200
            self.F_min = -5200
            self.F_drive_max = 4200
            self.F_brake_max = 5200
        else:
            raise ValueError(f"Unknown preset: {name}")
