import numpy as np

def compute_metrics(v_arr, u_arr, v_ref, dt):
    """
    计算 MSE、超调量及能量消耗
    """
    mse       = np.mean((v_arr - v_ref) ** 2)
    overshoot = np.max(v_arr - v_ref)
    energy    = np.sum(np.abs(u_arr)) * dt
    return mse, overshoot, energy