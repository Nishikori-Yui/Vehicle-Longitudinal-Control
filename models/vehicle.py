import numpy as np


def longitudinal_resistance(v, p, grade: float = 0.0) -> float:
    """
    计算空气阻力、滚动阻力与坡度阻力的合力。
    grade: 道路坡度 (tan(theta))
    """
    theta = np.arctan(grade)
    drag = 0.5 * p.rho * p.Cd * p.A * v**2
    roll = p.Cr * p.m * p.g * np.cos(theta)
    grade_force = p.m * p.g * np.sin(theta)
    return drag + roll + grade_force


def disturbance(v, p):
    """
    计算空气阻力和滚动阻力（兼容旧接口）。
    v: 当前速度 (m/s)
    p: VehicleParams 实例
    """
    return longitudinal_resistance(v, p, grade=0.0)
