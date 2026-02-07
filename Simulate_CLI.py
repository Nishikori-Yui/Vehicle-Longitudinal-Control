import os
import numpy as np
import pandas as pd
from configs.vehicle_params import VehicleParams
from configs.scenarios import build_default_scenarios, load_scenarios
from controllers.longitudinal.pid_controller import PIDControllerAdaptive
from controllers.longitudinal.lqr_controller import LQRController
from controllers.longitudinal.mpc_controller import MPCControllerMultiObjective
from controllers.lateral.lqr_controller import LQRLateralController
from powertrain.force_powertrain import ForcePowertrain
from powertrain.throttle_brake import ThrottleBrakePowertrain
from sim.simulator import simulate
from utils.metrics import compute_metrics, compute_jerk_rms, compute_lateral_rms
from utils.plotting import (
    plot_speed_tracking, plot_normalized_signal, plot_error,
    plot_traction, plot_lateral_tracking, plot_metrics_bar,
    plot_heatmap, plot_radar
)


def main():
    """
    执行纵向与横向控制仿真，并生成性能指标与多种结果图。
    """
    # 仿真设置
    p = VehicleParams(preset="mid_sedan")
    dt = 0.2
    T_final = 40.0
    speeds = [10, 20, 30]

    scenarios = load_scenarios("configs/scenarios.json", default_mu=p.mu0, default_grade=p.grade0)
    if not scenarios:
        scenarios = build_default_scenarios(speeds, mu=p.mu0, grade=p.grade0)

    # 控制器配置
    controllers_info = [
        ("PID", PIDControllerAdaptive, {"Kp0": 6.0, "Ki0": 0.1, "beta": 0.5, "p": p}),
        ("LQR", LQRController, {"q": 1000.0, "r": 0.01, "p": p}),
        ("MPC", MPCControllerMultiObjective, {"N": 25, "q": 3.0, "r": 0.01, "e_energy": 1e-5, "p": p}),
    ]

    use_throttle_brake = True

    # 结果容器
    results = {name: [] for name, *_ in controllers_info}
    control_signals = {name: [] for name, *_ in controllers_info}
    error_results = {name: [] for name, *_ in controllers_info}
    control_forces = {name: [] for name, *_ in controllers_info}
    metrics_list = []

    lateral_sample = None

    # 纵横向联合仿真
    for scenario in scenarios:
        v_ref_const = scenario.v_ref_profile(0.0)
        for name, cls, kwargs in controllers_info:
            lon_ctrl = cls(**kwargs)
            lat_ctrl = LQRLateralController(p)
            powertrain = ThrottleBrakePowertrain(p) if use_throttle_brake else ForcePowertrain(p)

            result = simulate(scenario, p, lon_ctrl, lat_ctrl, powertrain, dt, T_final)

            results[name].append(result.u)
            control_signals[name].append(result.fx_cmd / (np.max(np.abs(result.fx_cmd)) + 1e-6))
            error_results[name].append(v_ref_const - result.u)
            control_forces[name].append(result.fx_act)

            mse, overshoot, energy = compute_metrics(result.u, result.fx_cmd, v_ref_const, dt)
            jerk = compute_jerk_rms(result.u, dt)
            lat_rms = compute_lateral_rms(result.lat_error)
            metrics_list.append({
                "Controller": name,
                "Speed (m/s)": v_ref_const,
                "MSE": mse,
                "Overshoot": overshoot,
                "Energy": energy,
                "Jerk": jerk,
                "LatRMS": lat_rms,
            })

            if lateral_sample is None:
                lateral_sample = (result.time, scenario.path, result.x, result.y, result.delta, result.lat_error)

    # 保存指标
    metrics_df = pd.DataFrame(metrics_list)
    os.makedirs("results", exist_ok=True)
    metrics_df.to_csv(os.path.join("results", "metrics_summary.csv"), index=False)

    # 图像输出目录
    fig_dir = "results/cli_figures"
    os.makedirs(fig_dir, exist_ok=True)

    time = np.arange(0, T_final + dt, dt)

    # 基本图表
    plot_speed_tracking(time, results, speeds, fig_dir)
    plot_normalized_signal(time, control_signals, speeds, fig_dir)
    plot_error(time, error_results, speeds, fig_dir)
    plot_traction(time, control_forces, speeds, fig_dir)

    # 横向跟踪
    if lateral_sample is not None:
        t_lat, path, x, y, delta_hist, lat_error = lateral_sample
        plot_lateral_tracking(t_lat, path["x"], path["y"], x, y, delta_hist, lat_error, fig_dir)

    # 高级图表
    plot_metrics_bar(metrics_df, fig_dir)
    plot_heatmap(metrics_df, fig_dir)
    for sp in speeds:
        plot_radar(metrics_df, sp, fig_dir)


if __name__ == "__main__":
    main()
