from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np
from models.vehicle_dynamics import VehicleState, step_dynamics


@dataclass
class SimulationResult:
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    u: np.ndarray
    v: np.ndarray
    r: np.ndarray
    fx_cmd: np.ndarray
    fx_act: np.ndarray
    delta: np.ndarray
    v_ref: np.ndarray
    mu: np.ndarray
    grade: np.ndarray
    lat_error: np.ndarray


def _lateral_error(state: VehicleState, path: Dict[str, np.ndarray]) -> float:
    dx = path["x"] - state.x
    dy = path["y"] - state.y
    idx = int(np.argmin(dx * dx + dy * dy))
    return float(np.hypot(dx[idx], dy[idx]))


def simulate(
    scenario,
    params,
    lon_controller,
    lat_controller,
    powertrain,
    dt: float,
    T_final: float,
) -> SimulationResult:
    time = np.arange(0.0, T_final + dt, dt)
    n = len(time)

    x = np.zeros(n)
    y = np.zeros(n)
    psi = np.zeros(n)
    u = np.zeros(n)
    v = np.zeros(n)
    r = np.zeros(n)

    fx_cmd = np.zeros(n)
    fx_act = np.zeros(n)
    delta = np.zeros(n)
    v_ref = np.zeros(n)
    mu_hist = np.zeros(n)
    grade_hist = np.zeros(n)
    lat_error = np.zeros(n)

    state = VehicleState(x=0.0, y=0.0, psi=0.0, u=0.0, v=0.0, r=0.0)

    if hasattr(lon_controller, "reset"):
        lon_controller.reset()
    if hasattr(lat_controller, "reset"):
        lat_controller.reset()
    if hasattr(powertrain, "reset"):
        powertrain.reset()

    # 初始状态记录
    x[0], y[0], psi[0] = state.x, state.y, state.psi
    u[0], v[0], r[0] = state.u, state.v, state.r
    lat_error[0] = _lateral_error(state, scenario.path)

    for i in range(n - 1):
        t = time[i]
        v_ref[i] = scenario.v_ref_profile(t)
        mu = scenario.mu_profile(t)
        grade = scenario.grade_profile(t)
        mu_hist[i] = mu
        grade_hist[i] = grade

        if hasattr(lon_controller, "set_env"):
            lon_controller.set_env(mu=mu, grade=grade)
        if hasattr(lat_controller, "set_env"):
            lat_controller.set_env(mu=mu, grade=grade)

        delta[i] = float(lat_controller.step(state, scenario.path, dt))
        fx_cmd[i] = float(lon_controller.step(state, v_ref[i], dt))

        # 轮胎附着限幅
        fx_limit = mu * params.m * params.g
        fx_cmd[i] = float(np.clip(fx_cmd[i], -fx_limit, fx_limit))

        fx_act[i] = float(powertrain.compute_force(fx_cmd[i], state, dt))

        # 状态更新
        state = step_dynamics(state, fx_act[i], delta[i], params, mu, grade, dt)

        x[i + 1], y[i + 1], psi[i + 1] = state.x, state.y, state.psi
        u[i + 1], v[i + 1], r[i + 1] = state.u, state.v, state.r
        lat_error[i + 1] = _lateral_error(state, scenario.path)

    # 补齐最后一个时刻的参考与环境
    v_ref[-1] = v_ref[-2]
    mu_hist[-1] = mu_hist[-2]
    grade_hist[-1] = grade_hist[-2]
    delta[-1] = delta[-2]
    fx_cmd[-1] = fx_cmd[-2]
    fx_act[-1] = fx_act[-2]

    return SimulationResult(
        time=time,
        x=x,
        y=y,
        psi=psi,
        u=u,
        v=v,
        r=r,
        fx_cmd=fx_cmd,
        fx_act=fx_act,
        delta=delta,
        v_ref=v_ref,
        mu=mu_hist,
        grade=grade_hist,
        lat_error=lat_error,
    )
