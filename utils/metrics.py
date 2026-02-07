import numpy as np

def compute_metrics(v_arr, u_arr, v_ref, dt):
    """
    计算 MSE、超调量及能量消耗
    """
    mse       = np.mean((v_arr - v_ref) ** 2)
    overshoot = np.max(v_arr - v_ref)
    energy    = np.sum(np.abs(u_arr)) * dt
    return mse, overshoot, energy


def compute_jerk_rms(v_arr, dt):
    """
    计算纵向加加速度（jerk）RMS
    """
    if len(v_arr) < 3:
        return 0.0
    acc = np.diff(v_arr) / dt
    jerk = np.diff(acc) / dt
    return float(np.sqrt(np.mean(jerk ** 2)))


def compute_lateral_rms(lat_error_arr):
    """
    计算横向误差 RMS
    """
    if len(lat_error_arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(lat_error_arr) ** 2)))
