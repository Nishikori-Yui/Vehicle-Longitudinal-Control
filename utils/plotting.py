import os
import matplotlib.pyplot as plt
import numpy as np

def plot_speed_tracking(time, results, speeds, fig_dir):
    """
    绘制不同控制器在各目标速度下的速度跟踪曲线。

    参数：
    - time: 时间序列 (1D array)
    - results: dict, 键为控制器名，值为列表，每个元素为对应 speeds 中目标速度下的速度历史数组
    - speeds: 目标速度列表 (m/s)
    - fig_dir: 图像保存目录
    """
    os.makedirs(fig_dir, exist_ok=True)
    for idx, v_ref in enumerate(speeds):
        plt.figure()
        for name, v_hist_list in results.items():
            plt.plot(time, v_hist_list[idx], label=name)
        plt.plot(time, [v_ref] * len(time), 'k--', label='Reference')
        plt.title(f'Speed Tracking at {v_ref} m/s')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.legend()
        plt.grid(True)
        path = os.path.join(fig_dir, f'speed_tracking_{v_ref}ms.png')
        plt.savefig(path)
        plt.close()


def plot_normalized_signal(time, control_signals, speeds, fig_dir):
    """
    绘制归一化后的控制信号对比曲线。

    参数：
    - time: 时间序列 (1D array)
    - control_signals: dict, 键为控制器名，值为列表，每个元素为对应 speeds 中目标速度下的归一化控制信号数组
    - speeds: 目标速度列表 (m/s)
    - fig_dir: 图像保存目录
    """
    os.makedirs(fig_dir, exist_ok=True)
    for idx, v_ref in enumerate(speeds):
        plt.figure()
        for name, u_hist_list in control_signals.items():
            plt.plot(time, u_hist_list[idx], label=name)
        plt.title(f'Normalized Control Signal at {v_ref} m/s')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Signal')
        plt.legend()
        plt.grid(True)
        path = os.path.join(fig_dir, f'normalized_signal_{v_ref}ms.png')
        plt.savefig(path)
        plt.close()


def plot_error(time, error_results, speeds, fig_dir):
    """
    绘制速度跟踪误差曲线。

    参数：
    - time: 时间序列 (1D array)
    - error_results: dict, 键为控制器名，值为列表，每个元素为对应 speeds 中目标速度下的误差数组 (v_ref - v)
    - speeds: 目标速度列表
    - fig_dir: 图像保存目录
    """
    os.makedirs(fig_dir, exist_ok=True)
    for idx, v_ref in enumerate(speeds):
        plt.figure()
        for name, e_hist_list in error_results.items():
            plt.plot(time, e_hist_list[idx], label=name)
        plt.title(f'Speed Error at {v_ref} m/s')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (m/s)')
        plt.legend()
        plt.grid(True)
        path = os.path.join(fig_dir, f'speed_error_{v_ref}ms.png')
        plt.savefig(path)
        plt.close()


def plot_traction(time, control_forces, speeds, fig_dir):
    """
    绘制牵引力/制动力指令对比曲线。

    参数：
    - time: 时间序列 (1D array)
    - control_forces: dict, 键为控制器名，值为列表，每个元素为对应 speeds 中目标速度下的力命令数组
    - speeds: 目标速度列表
    - fig_dir: 图像保存目录
    """
    os.makedirs(fig_dir, exist_ok=True)
    for idx, v_ref in enumerate(speeds):
        plt.figure()
        for name, f_hist_list in control_forces.items():
            plt.plot(time, f_hist_list[idx], label=name)
        plt.title(f'Traction/Braking Command at {v_ref} m/s')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend()
        plt.grid(True)
        path = os.path.join(fig_dir, f'traction_command_{v_ref}ms.png')
        plt.savefig(path)
        plt.close()


def plot_lateral_tracking(t_lat, x_ref, y_ref, x, y, delta_hist, lat_error, fig_dir):
    """
    绘制横向跟踪结果：轨迹对比、转向角和横向误差。

    参数：
    - t_lat: 横向仿真时间序列
    - x_ref, y_ref: 参考轨迹坐标
    - x, y: 实际车辆轨迹坐标
    - delta_hist: 转向角历史序列
    - lat_error: 横向误差序列
    - fig_dir: 图像保存目录
    """
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure()
    plt.plot(x_ref, y_ref, 'k--', label='Reference Path')
    plt.plot(x, y, label='Actual Path')
    plt.title('Lateral Path Tracking')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'lateral_path_tracking.png'))
    plt.close()

    plt.figure()
    plt.plot(t_lat, delta_hist)
    plt.title('Steering Angle vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Angle (rad)')
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'steering_angle.png'))
    plt.close()

    plt.figure()
    plt.plot(t_lat, lat_error)
    plt.title('Lateral Error vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Lateral Error (m)')
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'lateral_error.png'))
    plt.close()

def plot_metrics_bar(metrics_df, fig_dir):
    """
    绘制性能指标柱状图，对比不同控制器在各速度下的 MSE、Overshoot 和 Energy。
    """
    os.makedirs(fig_dir, exist_ok=True)
    # Pivot 数据
    pivot_mse = metrics_df.pivot(index='Speed (m/s)', columns='Controller', values='MSE')
    pivot_os  = metrics_df.pivot(index='Speed (m/s)', columns='Controller', values='Overshoot')
    pivot_en  = metrics_df.pivot(index='Speed (m/s)', columns='Controller', values='Energy')
    # 绘制每个指标
    for name, pivot in [('mse', pivot_mse), ('overshoot', pivot_os), ('energy', pivot_en)]:
        pivot.plot(kind='bar')
        plt.title(name.upper() + ' Comparison')
        plt.xlabel('Speed (m/s)'); plt.ylabel(name)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'bar_{name}.png'))
        plt.close()


def plot_heatmap(metrics_df, fig_dir):
    """
    绘制热力图，展示 MSE 随速度与控制器变化的矩阵。
    """
    os.makedirs(fig_dir, exist_ok=True)
    pivot = metrics_df.pivot(index='Controller', columns='Speed (m/s)', values='MSE')
    data = pivot.values
    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect='auto')
    ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index))); ax.set_yticklabels(pivot.index)
    plt.title('MSE Heatmap')
    plt.xlabel('Speed (m/s)'); plt.ylabel('Controller')
    plt.colorbar(im)
    plt.savefig(os.path.join(fig_dir, 'heatmap_mse.png'))
    plt.close()


def plot_radar(metrics_df, speed, fig_dir):
    """
    绘制雷达图，比对各控制器在指定速度下的三个指标。
    speed: 要绘制的速度值
    """
    os.makedirs(fig_dir, exist_ok=True)
    df = metrics_df[metrics_df['Speed (m/s)'] == speed]
    labels = ['MSE','Overshoot','Energy']
    controllers = df['Controller'].tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(); ax = fig.add_subplot(111, polar=True)
    for _, row in df.iterrows():
        values = [row['MSE'], row['Overshoot'], row['Energy']]
        values += values[:1]
        ax.plot(angles, values, label=row['Controller'])
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    plt.legend(loc='best'); plt.title(f'Radar Plot @ {speed} m/s')
    plt.savefig(os.path.join(fig_dir, f'radar_{speed}ms.png'))
    plt.close()