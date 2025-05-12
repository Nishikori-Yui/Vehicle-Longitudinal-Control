import os
import numpy as np
import pandas as pd
from configs.vehicle_params import VehicleParams
from models.vehicle import disturbance
from controllers.pid_controller import PIDControllerAdaptive
from controllers.lqi_controller import LQIController
from controllers.mpc_controller import MPCControllerMultiObjective
from utils.metrics import compute_metrics
from utils.plotting import (
    plot_speed_tracking, plot_normalized_signal, plot_error,
    plot_traction, plot_lateral_tracking, plot_metrics_bar,
    plot_heatmap, plot_radar
)
from lateral_control.pure_pursuit import pure_pursuit


def main():
    """
    执行纵向与横向控制仿真，并生成性能指标与多种结果图。
    """
    # 仿真设置
    p = VehicleParams()
    dt = 0.2
    T_final = 40.0
    time = np.arange(0, T_final + dt, dt)
    speeds = [10, 20, 30]

    # 控制器配置
    controllers_info = [
        ('PID', PIDControllerAdaptive, {'Kp0': 6.0, 'Ki0': 0.1, 'beta': 0.5, 'p': p}),
        ('LQI', LQIController,        {'v0': None, 'p': p}),
        ('MPC', MPCControllerMultiObjective, {'N': 25, 'q': 3.0, 'r': 0.01, 'e_energy': 1e-5, 'p': p})
    ]

    # 结果容器
    results = {name: [] for name, *_ in controllers_info}
    control_signals = {name: [] for name, *_ in controllers_info}
    error_results = {name: [] for name, *_ in controllers_info}
    control_forces = {name: [] for name, *_ in controllers_info}
    metrics_list = []

    # 纵向仿真
    for v_ref in speeds:
        for name, cls, kwargs in controllers_info:
            params = kwargs.copy()
            if name == 'LQI':
                params['v0'] = v_ref
            ctrl = cls(**params)
            ctrl.reset()

            v_hist, u_hist, e_hist, f_hist = [], [], [], []
            v_curr = 0.0
            for _ in time:
                error = v_ref - v_curr
                if name == 'PID':
                    u_cmd, f_act = ctrl.step(error, dt)
                else:
                    u_cmd, f_act = ctrl.step(v_curr, v_ref, dt)
                d = disturbance(v_curr, p)
                v_curr += (f_act - d) / p.m * dt

                v_hist.append(v_curr)
                u_hist.append(u_cmd)
                e_hist.append(error)
                f_hist.append(f_act)

            results[name].append(np.array(v_hist))
            control_signals[name].append(np.array(u_hist) / (np.max(np.abs(u_hist)) + 1e-6))
            error_results[name].append(np.array(e_hist))
            control_forces[name].append(np.array(f_hist))

            mse, overshoot, energy = compute_metrics(np.array(v_hist), np.array(u_hist), v_ref, dt)
            metrics_list.append({'Controller': name, 'Speed (m/s)': v_ref, 'MSE': mse, 'Overshoot': overshoot, 'Energy': energy})

    # 保存指标
    metrics_df = pd.DataFrame(metrics_list)
    os.makedirs('results', exist_ok=True)
    metrics_df.to_csv(os.path.join('results', 'metrics_summary.csv'), index=False)

    # 图像输出目录
    fig_dir = 'results/cli_figures'
    os.makedirs(fig_dir, exist_ok=True)

    # 基本图表
    plot_speed_tracking(time, results, speeds, fig_dir)
    plot_normalized_signal(time, control_signals, speeds, fig_dir)
    plot_error(time, error_results, speeds, fig_dir)
    plot_traction(time, control_forces, speeds, fig_dir)

    # 横向跟踪
    R, v_lat, Ld, T_lat = 50.0, 15.0, 5.0, 20.0
    t_lat, x_ref, y_ref, x, y, psi, delta_hist, lat_error = pure_pursuit(R, v_lat, Ld, dt, T_lat, p)
    plot_lateral_tracking(t_lat, x_ref, y_ref, x, y, delta_hist, lat_error, fig_dir)

    # 高级图表
    plot_metrics_bar(metrics_df, fig_dir)
    plot_heatmap(metrics_df, fig_dir)
    for sp in speeds:
        plot_radar(metrics_df, sp, fig_dir)


if __name__ == '__main__':
    main()