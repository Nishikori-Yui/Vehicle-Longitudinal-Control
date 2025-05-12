def disturbance(v, p):
    """
    计算空气阻力和滚动阻力
    v: 当前速度 (m/s)
    p: VehicleParams 实例
    """
    drag = 0.5 * p.rho * p.Cd * p.A * v**2
    roll = p.Cr * p.m * p.g
    return drag + roll