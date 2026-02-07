import os
import matplotlib.pyplot as plt
import numpy as np

# 中文字体回退，避免中文标题/文件名渲染警告
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def _safe_filename(label: str) -> str:
    safe = label.replace(" ", "_")
    for ch in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]:
        safe = safe.replace(ch, "_")
    return safe


def _scenario_zh(label: str) -> str:
    if label.startswith("step_") and label.endswith("ms"):
        v = label[len("step_"):-len("ms")]
        return f"速度阶跃_{v}mps"
    if label == "start_stop":
        return "起步_加速_减速"
    if label.startswith("grade_step_"):
        return "坡度阶跃_" + label[len("grade_step_"):].replace("_to_", "到")
    if label.startswith("mu_step_"):
        return "附着阶跃_" + label[len("mu_step_"):].replace("_to_", "到")
    return label


def plot_speed_tracking(time, results, scenario_labels, fig_dir, ref_series_list=None):
    """
    绘制不同控制器在各场景下的速度跟踪曲线。

    参数：
    - time: 时间序列 (1D array)
    - results: dict, 键为控制器名，值为列表，每个元素为对应 speeds 中目标速度下的速度历史数组
    - scenario_labels: 场景标签列表
    - fig_dir: 图像保存目录
    - ref_series_list: 可选参考速度序列列表
    """
    os.makedirs(fig_dir, exist_ok=True)
    for idx, label in enumerate(scenario_labels):
        zh_label = _scenario_zh(label)
        plt.figure(figsize=(16, 8))
        for name, v_hist_list in results.items():
            plt.plot(time, v_hist_list[idx], label=name)
        if ref_series_list is not None:
            plt.plot(time, ref_series_list[idx], 'k--', label='Reference')
        plt.title(f'速度跟踪: {zh_label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.legend()
        plt.grid(True)
        path = os.path.join(fig_dir, f'速度跟踪_{_safe_filename(zh_label)}.png')
        plt.savefig(path)
        plt.close()


def plot_normalized_signal(time, control_signals, scenario_labels, fig_dir):
    """
    绘制归一化后的控制信号对比曲线。

    参数：
    - time: 时间序列 (1D array)
    - control_signals: dict, 键为控制器名，值为列表，每个元素为对应 speeds 中目标速度下的归一化控制信号数组
    - speeds: 目标速度列表 (m/s)
    - fig_dir: 图像保存目录
    """
    os.makedirs(fig_dir, exist_ok=True)
    for idx, label in enumerate(scenario_labels):
        zh_label = _scenario_zh(label)
        plt.figure(figsize=(16, 8))
        for name, u_hist_list in control_signals.items():
            plt.plot(time, u_hist_list[idx], label=name)
        plt.title(f'归一化控制信号: {zh_label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Signal')
        plt.legend()
        plt.grid(True)
        path = os.path.join(fig_dir, f'归一化控制信号_{_safe_filename(zh_label)}.png')
        plt.savefig(path)
        plt.close()


def plot_error(time, error_results, scenario_labels, fig_dir):
    """
    绘制速度跟踪误差曲线。

    参数：
    - time: 时间序列 (1D array)
    - error_results: dict, 键为控制器名，值为列表，每个元素为对应 speeds 中目标速度下的误差数组 (v_ref - v)
    - speeds: 目标速度列表
    - fig_dir: 图像保存目录
    """
    os.makedirs(fig_dir, exist_ok=True)
    for idx, label in enumerate(scenario_labels):
        zh_label = _scenario_zh(label)
        plt.figure(figsize=(16, 8))
        for name, e_hist_list in error_results.items():
            plt.plot(time, e_hist_list[idx], label=name)
        plt.title(f'速度误差: {zh_label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (m/s)')
        plt.legend()
        plt.grid(True)
        path = os.path.join(fig_dir, f'速度误差_{_safe_filename(zh_label)}.png')
        plt.savefig(path)
        plt.close()


def plot_traction(time, control_forces, scenario_labels, fig_dir):
    """
    绘制牵引力/制动力指令对比曲线。

    参数：
    - time: 时间序列 (1D array)
    - control_forces: dict, 键为控制器名，值为列表，每个元素为对应 speeds 中目标速度下的力命令数组
    - speeds: 目标速度列表
    - fig_dir: 图像保存目录
    """
    os.makedirs(fig_dir, exist_ok=True)
    for idx, label in enumerate(scenario_labels):
        zh_label = _scenario_zh(label)
        plt.figure(figsize=(16, 8))
        for name, f_hist_list in control_forces.items():
            plt.plot(time, f_hist_list[idx], label=name)
        plt.title(f'牵引/制动力: {zh_label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend()
        plt.grid(True)
        path = os.path.join(fig_dir, f'牵引制动力_{_safe_filename(zh_label)}.png')
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

    plt.figure(figsize=(16, 8))
    plt.plot(x_ref, y_ref, 'k--', label='Reference Path')
    plt.plot(x, y, label='Actual Path')
    plt.title('横向路径跟踪')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, '横向路径跟踪.png'))
    plt.close()

    plt.figure(figsize=(16, 8))
    plt.plot(t_lat, delta_hist)
    plt.title('转向角-时间')
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Angle (rad)')
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, '转向角.png'))
    plt.close()

    plt.figure(figsize=(16, 8))
    plt.plot(t_lat, lat_error)
    plt.title('横向误差-时间')
    plt.xlabel('Time (s)')
    plt.ylabel('Lateral Error (m)')
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, '横向误差.png'))
    plt.close()

def plot_metrics_bar(metrics_df, fig_dir):
    """
    绘制性能指标柱状图，对比不同控制器在各速度下的指标。
    """
    os.makedirs(fig_dir, exist_ok=True)
    idx_col = "Scenario" if "Scenario" in metrics_df.columns else "Speed (m/s)"
    metric_map = [
        ("MSE", "均方误差"),
        ("Overshoot", "超调量"),
        ("Energy", "能耗"),
        ("Jerk", "Jerk"),
        ("LatRMS", "横向误差RMS"),
        ("IAE", "IAE"),
        ("ITAE", "ITAE"),
        ("LatMax", "横向误差最大值"),
        ("EnergyPerDist", "单位距离能耗"),
    ]
    for col, name in metric_map:
        if col not in metrics_df.columns:
            continue
        pivot = metrics_df.pivot(index=idx_col, columns="Controller", values=col)
        pivot.plot(kind="bar", figsize=(16, 8))
        plt.title(name + " 对比")
        plt.xlabel(idx_col)
        plt.ylabel(name)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"指标_{_safe_filename(name)}.png"))
        plt.close()


def plot_heatmap(metrics_df, fig_dir):
    """
    绘制热力图，展示 MSE 随速度与控制器变化的矩阵。
    """
    os.makedirs(fig_dir, exist_ok=True)
    col_name = 'Scenario' if 'Scenario' in metrics_df.columns else 'Speed (m/s)'
    pivot = metrics_df.pivot(index='Controller', columns=col_name, values='MSE')
    data = pivot.values
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(data, aspect='auto')
    ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index))); ax.set_yticklabels(pivot.index)
    plt.title('MSE 热力图')
    plt.xlabel(col_name); plt.ylabel('Controller')
    plt.colorbar(im)
    plt.savefig(os.path.join(fig_dir, '热力图_MSE.png'))
    plt.close()


def plot_radar(metrics_df, scenario_label, fig_dir):
    """
    绘制雷达图，比对各控制器在指定速度下的三个指标。
    speed: 要绘制的速度值
    """
    os.makedirs(fig_dir, exist_ok=True)
    if 'Scenario' in metrics_df.columns:
        df = metrics_df[metrics_df['Scenario'] == scenario_label]
    else:
        df = metrics_df[metrics_df['Speed (m/s)'] == scenario_label]
    labels = ['MSE','Overshoot','Energy','Jerk','LatRMS']
    controllers = df['Controller'].tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, polar=True)
    for _, row in df.iterrows():
        values = [row['MSE'], row['Overshoot'], row['Energy'], row.get('Jerk', 0.0), row.get('LatRMS', 0.0)]
        values += values[:1]
        ax.plot(angles, values, label=row['Controller'])
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    zh_label = _scenario_zh(str(scenario_label))
    plt.legend(loc='best'); plt.title(f'雷达图: {zh_label}')
    plt.savefig(os.path.join(fig_dir, f'雷达图_{_safe_filename(zh_label)}.png'))
    plt.close()
