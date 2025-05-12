import numpy as np

def pure_pursuit(R, v_lat, Ld, dt, T_final, p):
    """
    纯旁路追踪算法生成参考轨迹与误差
    返回时间序列及参考/实际轨迹、横向误差、转向角
    """
    t_lat = np.arange(0, T_final + dt, dt)
    phi = v_lat * t_lat / R
    x_ref = R * np.sin(phi)
    y_ref = R * (1 - np.cos(phi))

    n = len(t_lat)
    x = np.zeros(n); y = np.zeros(n)
    psi = np.zeros(n); delta_hist = np.zeros(n)
    for k in range(n-1):
        idx = min(k + int(Ld/(v_lat*dt)), n-1)
        dx = x_ref[idx] - x[k]; dy = y_ref[idx] - y[k]
        alpha = np.arctan2(dy, dx) - psi[k]
        delta = np.arctan2(2 * p.wheelbase * np.sin(alpha), Ld)
        delta_hist[k] = delta
        x[k+1] = x[k] + v_lat * np.cos(psi[k]) * dt
        y[k+1] = y[k] + v_lat * np.sin(psi[k]) * dt
        psi[k+1] = psi[k] + v_lat / p.wheelbase * np.tan(delta) * dt

    lat_error = np.hypot(x - x_ref, y - y_ref)
    return t_lat, x_ref, y_ref, x, y, psi, delta_hist, lat_error