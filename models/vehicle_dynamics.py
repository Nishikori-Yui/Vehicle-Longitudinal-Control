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
    u = state.u
    v = state.v
    r = state.r

    u_eps = 0.1
    mu_eff = max(mu, 0.05)

    # 侧偏角
    alpha_f = delta - (v + p.a * r) / max(u, u_eps)
    alpha_r = -(v - p.b * r) / max(u, u_eps)

    # 线性轮胎侧向力
    Fyf = mu_eff * p.Caf * alpha_f
    Fyr = mu_eff * p.Car * alpha_r

    # 纵向阻力
    Fres = longitudinal_resistance(u, p, grade)

    # 动力学方程
    u_dot = (Fx - Fres + p.m * v * r) / p.m
    v_dot = (Fyf + Fyr - p.m * u * r) / p.m
    r_dot = (p.a * Fyf - p.b * Fyr) / p.Iz

    x_dot = u * np.cos(state.psi) - v * np.sin(state.psi)
    y_dot = u * np.sin(state.psi) + v * np.cos(state.psi)
    psi_dot = r

    return VehicleState(
        x=state.x + x_dot * dt,
        y=state.y + y_dot * dt,
        psi=state.psi + psi_dot * dt,
        u=state.u + u_dot * dt,
        v=state.v + v_dot * dt,
        r=state.r + r_dot * dt,
    )
