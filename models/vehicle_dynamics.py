from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from models.vehicle import longitudinal_resistance


@dataclass
class VehicleState:
    x: float
    y: float
    psi: float
    u: float  # longitudinal velocity
    v: float  # lateral velocity
    r: float  # yaw rate


def step_dynamics(
    state: VehicleState,
    Fx: float,
    delta: float,
    p,
    mu: float,
    grade: float,
    dt: float,
) -> VehicleState:
    u = max(state.u, 0.0)
    v = state.v
    r = state.r

    u_eps = 0.5
    mu_eff = max(mu, 0.05)

    # 侧偏角
    alpha_f = delta - (v + p.a * r) / max(u, u_eps)
    alpha_r = -(v - p.b * r) / max(u, u_eps)
    alpha_f = float(np.clip(alpha_f, -p.alpha_max, p.alpha_max))
    alpha_r = float(np.clip(alpha_r, -p.alpha_max, p.alpha_max))

    # 线性轮胎侧向力
    Fyf = mu_eff * p.Caf * alpha_f
    Fyr = mu_eff * p.Car * alpha_r
    # 侧向力摩擦限幅（静态分配）
    fz_f = p.m * p.g * p.b / max(p.wheelbase, 1e-3)
    fz_r = p.m * p.g * p.a / max(p.wheelbase, 1e-3)
    Fyf = float(np.clip(Fyf, -mu_eff * fz_f, mu_eff * fz_f))
    Fyr = float(np.clip(Fyr, -mu_eff * fz_r, mu_eff * fz_r))

    # 纵向阻力
    Fres = longitudinal_resistance(u, p, grade)

    # 动力学方程
    u_dot = (Fx - Fres + p.m * v * r) / p.m
    v_dot = (Fyf + Fyr - p.m * u * r) / p.m
    r_dot = (p.a * Fyf - p.b * Fyr) / p.Iz

    x_dot = u * np.cos(state.psi) - v * np.sin(state.psi)
    y_dot = u * np.sin(state.psi) + v * np.cos(state.psi)
    psi_dot = r

    u_new = state.u + u_dot * dt
    v_new = state.v + v_dot * dt
    r_new = state.r + r_dot * dt
    u_new = float(np.clip(u_new, 0.0, p.u_max))
    v_new = float(np.clip(v_new, -p.v_max, p.v_max))
    r_new = float(np.clip(r_new, -p.r_max, p.r_max))

    return VehicleState(
        x=state.x + x_dot * dt,
        y=state.y + y_dot * dt,
        psi=state.psi + psi_dot * dt,
        u=u_new,
        v=v_new,
        r=r_new,
    )
