# === 1. 核心库加载与字体设置 ===
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize
import pandas as pd

# 确保图像保存目录存在
fig_dir = os.path.join(os.getcwd(), 'results/report')
os.makedirs(fig_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# === 2. 车辆参数类 ===
class VehicleParams:
    def __init__(self):
        # 基本参数
        self.m = 1500        # kg
        self.g = 9.81        # m/s²
        self.wheelbase = 2.7 # m
        # 空气动力学
        self.Cd = 0.30
        self.A  = 2.2        # m²
        self.rho = 1.225     # kg/m³
        # 滚动阻力
        self.Cr = 0.015
        # 执行器
        self.wn   = 5.65
        self.zeta = 0.707
        # ABS
        self.r = 0.3         # m
        self.J = 0.35        # kg·m²
        self.c1, self.c2, self.c3 = 1.28, 23.99, 0.52
        self.tau_hyd = 0.5   # s
        # 牵引／制动力限
        self.F_max =  4000   # N
        self.F_min = -5000   # N
        self.tau   = 0.5     # s (滞后)

# === 3. 工具函数：性能指标与阻力模型 ===
def compute_metrics(v_arr, u_arr, v_ref, dt):
    mse       = np.mean((v_arr - v_ref) ** 2)
    overshoot = np.max(v_arr - v_ref)
    energy    = np.sum(np.abs(u_arr)) * dt
    return mse, overshoot, energy

def disturbance(v, p: VehicleParams):
    drag = 0.5 * p.rho * p.Cd * p.A * v**2
    roll = p.Cr * p.m * p.g
    return drag + roll

# === 4. 二阶执行器模型 ===
class SecondOrderActuator:
    def __init__(self, wn, zeta):
        self.A = np.array([[0, 1], [-wn**2, -2*zeta*wn]])
        self.B = np.array([[0], [wn**2]])
        self.x = np.zeros((2, 1))
    def reset(self):
        self.x.fill(0)
    def step(self, u, dt):
        self.x += (self.A @ self.x + self.B * u) * dt
        return self.x[0, 0]

# === 5. ABS 控制器 ===
class ABSController:
    def __init__(self, p: VehicleParams):
        self.r, self.J = p.r, p.J
        self.c1, self.c2, self.c3 = p.c1, p.c2, p.c3
        self.tau = p.tau_hyd
        self.omega = 0.0
    def burckhardt(self, slip):
        return self.c1 * (1 - np.exp(-self.c2 * slip)) - self.c3 * slip
    def reset(self):
        self.omega = 0.0
    def step(self, v, u_brake, dt, p: VehicleParams):
        slip = (v - self.omega * p.r) / max(v, self.omega * p.r, 1e-3)
        slip = np.clip(slip, -1, 1)
        mu   = self.burckhardt(slip)
        F_des = mu * p.m * p.g * np.sign(u_brake)
        self.omega += (dt / self.tau) * (u_brake - self.omega)
        return F_des

# === 6. 自适应 PID 控制器 ===
class PIDControllerAdaptive:
    def __init__(self, Kp0, Ki0, beta, p: VehicleParams):
        self.Kp0, self.Ki0, self.beta = Kp0, Ki0, beta
        self.integrator = 0.0
        self.act = SecondOrderActuator(p.wn, p.zeta)
    def reset(self):
        self.integrator = 0.0
        self.act.reset()
    def step(self, error, dt):
        Kp = self.Kp0 * (1 + self.beta * abs(error))
        Ki = self.Ki0 * (1 + self.beta * abs(error))
        self.integrator += error * dt
        u = np.clip(Kp*error + Ki*self.integrator, -5000, 4000)
        F_act = self.act.step(u, dt)
        return u, F_act

# === 7. 增广 LQI 控制器 ===

'''
class LQIController:
    def __init__(self, v0, p: VehicleParams):
        a0    = 0.5 * p.rho * p.Cd * p.A * v0**2 + p.Cr * p.m * p.g
        b0    = p.rho * p.Cd * p.A * v0
        A_lin = np.array([[-b0 / p.m]])
        B_lin = np.array([[1.0 / p.m]])
        A_aug = np.block([
            [ A_lin,           np.zeros((1,1)) ],
            [   -1.0,          np.zeros((1,1)) ]
        ])
        B_aug = np.vstack([ B_lin, [0.0] ])

        e_max     = 0.5     # 最大速度误差 (m/s)
        I_max     = 5.0     # 最大误差积分 (m·s)
        u_pos_max = 4000.0  # 最大输出力 (N)
        Q11 = 1.0 / e_max**2
        Q22 = 1.0 / I_max**2
        R11 = 1.0 / u_pos_max**2

        Q = np.diag([Q11, Q22])
        R = np.array([[R11]])

        P      = solve_continuous_are(A_aug, B_aug, Q, R)
        self.K = np.linalg.inv(R) @ B_aug.T @ P  # shape (1,2)

        self.x       = np.zeros((2,1))              # [e; ∫e]
        self.u_min   = -5000.0
        self.u_max   = +4000.0
        self.act     = SecondOrderActuator(p.wn, p.zeta)

    def reset(self):
        self.x.fill(0.0)
        self.act.reset()

    def step(self, v, v_ref, dt):
        self.x[:] = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)

        e = v_ref - v
        self.x[0,0] = e
        self.x[1,0] += e * dt

        u_unsat = - (self.K @ self.x).item()
        if not np.isfinite(u_unsat):
            u_unsat = 0.0

        u = np.clip(u_unsat, self.u_min, self.u_max)
        if u != u_unsat:
            delta_I = e * dt
            if np.isfinite(delta_I):
                self.x[1,0] -= delta_I

        return u, self.act.step(u, dt)
        
---
完整版LQI控制器代码如下，由于客观限制，在实际执行中我们选择了简化版的LQI控制器。
'''
class LQIController:
    def __init__(self, v0, p: VehicleParams):
        a0 = 0.5 * p.rho * p.Cd * p.A * v0**2 + p.Cr * p.m * p.g
        b0 = p.rho * p.Cd * p.A * v0
        A_lin = np.array([[-b0 / p.m]])
        B_lin = np.array([[1 / p.m]])
        A_aug = np.block([[A_lin, np.zeros((1,1))], [-1, np.zeros((1,1))]])
        B_aug = np.vstack([B_lin, [0]])
        Q, R = np.diag([1000, 100]), np.array([[0.01]])
        P = solve_continuous_are(A_aug, B_aug, Q, R)
        self.K = np.linalg.inv(R) @ B_aug.T @ P
        self.x  = np.zeros((2,1))
        self.act = SecondOrderActuator(p.wn, p.zeta)
    def reset(self):
        self.x.fill(0)
        self.act.reset()
    def step(self, v, v_ref, dt):
        e = v_ref - v
        self.x[0,0] = e
        self.x[1,0] += e*dt
        u = np.clip(- (self.K @ self.x).item(), -5000, 4000)
        return u, self.act.step(u, dt)


# === 8. 多目标 MPC 控制器 ===
class MPCControllerMultiObjective:
    def __init__(self, N, q, r, e_energy, p: VehicleParams):
        self.N, self.q, self.r, self.e_energy = N, q, r, e_energy
        self.act = SecondOrderActuator(p.wn, p.zeta)
        self.u_prev = 0.0
    def reset(self):
        self.act.reset()
        self.u_prev = 0.0
    def step(self, v_curr, v_ref, dt):
        def obj(u):
            v_p, Fp = v_curr, self.act.x[0,0]
            cost = 0
            for ui in u:
                Fp += (dt/0.5)*(ui - Fp)
                d = disturbance(v_p, p)
                v_p += dt*(Fp - d)/p.m
                cost += ( self.q*(v_p - v_ref)**2
                        + self.r*(ui - self.u_prev)**2
                        + self.e_energy*ui**2 )
            return cost
        u0 = np.ones(self.N) * self.u_prev
        res = minimize(obj, u0, bounds=[(-5000, 4000)]*self.N, options={'maxiter':100})
        u_cmd = res.x[0]
        self.u_prev = u_cmd
        return u_cmd, self.act.step(u_cmd, dt)

# === 9. 主仿真循环 ===
p = VehicleParams()
dt, T_final = 0.2, 40
time = np.arange(0, T_final + dt, dt)
desired_speeds = [10, 20, 30]
results = {}
for v_t in desired_speeds:
    pid = PIDControllerAdaptive(300, 100, 0.1, p)
    lqi = LQIController(v_t, p)
    mpc = MPCControllerMultiObjective(40, 50, 0.005, 1e-6, p)
    abs_ctrl = ABSController(p)
    for c in (pid, lqi, mpc, abs_ctrl): c.reset()
    v_pid = np.zeros_like(time); v_lqi = v_pid.copy(); v_mpc = v_pid.copy()
    u_pid = v_pid.copy(); u_lqi = v_pid.copy(); u_mpc = v_pid.copy()
    for k in range(len(time)-1):
        e1, e2, e3 = v_t - v_pid[k], v_t - v_lqi[k], v_t - v_mpc[k]
        d1, d2, d3 = disturbance(v_pid[k],p), disturbance(v_lqi[k],p), disturbance(v_mpc[k],p)
        u1,_ = pid.step(e1, dt)
        u2,_ = lqi.step(v_lqi[k], v_t, dt)
        u3,_ = mpc.step(v_mpc[k], v_t, dt)
        F1 = pid.act.step(u1, dt) if u1>=0 else abs_ctrl.step(v_pid[k], u1, dt, p)
        F2 = lqi.act.step(u2, dt) if u2>=0 else abs_ctrl.step(v_lqi[k], u2, dt, p)
        F3 = mpc.act.step(u3, dt) if u3>=0 else abs_ctrl.step(v_mpc[k], u3, dt, p)
        v_pid[k+1] = v_pid[k] + dt*(F1 - d1)/p.m
        v_lqi[k+1] = v_lqi[k] + dt*(F2 - d2)/p.m
        v_mpc[k+1] = v_mpc[k] + dt*(F3 - d3)/p.m
        u_pid[k], u_lqi[k], u_mpc[k] = u1, u2, u3
    results[v_t] = {'v':(v_pid, v_lqi, v_mpc), 'u':(u_pid, u_lqi, u_mpc)}

# === 10. 绘图函数 ===
def plot_speed_tracking(time, results, speeds):
    fig, axes = plt.subplots(len(speeds), 1, figsize=(10, 3 * len(speeds)), sharex=True, constrained_layout=True)
    for ax, v_t in zip(axes, speeds):
        vp, vl, vm = results[v_t]['v']
        ax.plot(time, vp, label='PID', linewidth=1.5)
        ax.plot(time, vl, label='LQI', linewidth=1.5)
        ax.plot(time, vm, label='MPC', linewidth=1.5)
        ax.axhline(v_t, linestyle='--', color='k', label='目标')
        ax.set_ylabel('速度 (m/s)')
        ax.legend(frameon=False)
        ax.grid(True, linestyle=':')
    axes[-1].set_xlabel('时间 (s)')
    fig.suptitle('速度跟踪比较', fontsize=14)
    fig.savefig(os.path.join(fig_dir, 'speed_tracking.png'))
    plt.close(fig)

def plot_normalized_signal(time, results, speeds, p):
    fig, axes = plt.subplots(len(speeds), 1, figsize=(10, 3 * len(speeds)), sharex=True, constrained_layout=True)
    for ax, v_t in zip(axes, speeds):
        u1, u2, u3 = results[v_t]['u']
        u1n = np.where(u1 >= 0, u1 / p.F_max, u1 / (-p.F_min))
        u2n = np.where(u2 >= 0, u2 / p.F_max, u2 / (-p.F_min))
        u3n = np.where(u3 >= 0, u3 / p.F_max, u3 / (-p.F_min))
        ax.plot(time, u1n, label='PID')
        ax.plot(time, u2n, label='LQI')
        ax.plot(time, u3n, label='MPC')
        ax.set_ylabel('归一化信号')
        ax.legend(frameon=False)
        ax.grid(True, linestyle=':')
    axes[-1].set_xlabel('时间 (s)')
    fig.suptitle('油门/制动信号（归一化）', fontsize=14)
    fig.savefig(os.path.join(fig_dir, 'normalized_signal.png'))
    plt.close(fig)

def plot_error(time, results, speeds):
    fig, axes = plt.subplots(len(speeds), 1, figsize=(10, 3 * len(speeds)), sharex=True, constrained_layout=True)
    for ax, v_t in zip(axes, speeds):
        vp, vl, vm = results[v_t]['v']
        ax.plot(time, v_t - vp, label='PID')
        ax.plot(time, v_t - vl, label='LQI')
        ax.plot(time, v_t - vm, label='MPC')
        ax.set_ylabel('误差 (m/s)')
        ax.legend(frameon=False)
        ax.grid(True, linestyle=':')
    axes[-1].set_xlabel('时间 (s)')
    fig.suptitle('速度误差', fontsize=14)
    fig.savefig(os.path.join(fig_dir, 'error_tracking.png'))
    plt.close(fig)

def plot_traction(time, results, speeds):
    fig, axes = plt.subplots(len(speeds), 1, figsize=(10, 3 * len(speeds)), sharex=True, constrained_layout=True)
    for ax, v_t in zip(axes, speeds):
        u1, u2, u3 = results[v_t]['u']
        ax.plot(time, u1, label='PID')
        ax.plot(time, u2, label='LQI')
        ax.plot(time, u3, label='MPC')
        ax.set_ylabel('牵引力 (N)')
        ax.legend(frameon=False)
        ax.grid(True, linestyle=':')
    axes[-1].set_xlabel('时间 (s)')
    fig.suptitle('牵引力命令', fontsize=14)
    fig.savefig(os.path.join(fig_dir, 'traction_command.png'))
    plt.close(fig)

# === 11. 性能指标计算与 DataFrame ===
metrics = {'Speed':[], 'Controller':[], 'MSE':[], 'Overshoot':[], 'Energy':[]}
for v_t in desired_speeds:
    vp, vl, vm = results[v_t]['v']
    u1, u2, u3 = results[v_t]['u']
    for name, v_arr, u_arr in zip(['PID','LQI','MPC'], [vp,vl,vm], [u1,u2,u3]):
        mse, osr, ene = compute_metrics(v_arr, u_arr, v_t, dt)
        metrics['Speed'].append(v_t); metrics['Controller'].append(name)
        metrics['MSE'].append(mse); metrics['Overshoot'].append(osr); metrics['Energy'].append(ene)

df = pd.DataFrame(metrics)

# 横向模拟参数
R, v_lat, Ld = 50, 15, 5
t_lat = np.arange(0, T_final + dt, dt)
phi = v_lat * t_lat / R
x_ref = R * np.sin(phi)
y_ref = R * (1 - np.cos(phi))

n = len(t_lat)
x = np.zeros(n)
y = np.zeros(n)
psi = np.zeros(n)
delta_hist = np.zeros(n)
for k in range(n - 1):
    idx = min(k + int(Ld / (v_lat * dt)), n - 1)
    dx, dy = x_ref[idx] - x[k], y_ref[idx] - y[k]
    alpha = np.arctan2(dy, dx) - psi[k]
    delta = np.arctan2(2 * p.wheelbase * np.sin(alpha), Ld)
    delta_hist[k] = delta
    x[k + 1] = x[k] + v_lat * np.cos(psi[k]) * dt
    y[k + 1] = y[k] + v_lat * np.sin(psi[k]) * dt
    psi[k + 1] = psi[k] + v_lat / p.wheelbase * np.tan(delta) * dt

lat_error = np.hypot(x - x_ref, y - y_ref)

# 轨迹与误差图
fig, axd = plt.subplot_mosaic(
    [['traj', 'traj'],
     ['err',  'delta']],
    figsize=(10, 6),
    constrained_layout=True
)
# 轨迹
ax = axd['traj']
ax.plot(x_ref, y_ref, '--', label='参考')
ax.plot(x, y, label='实际')
ax.set_aspect('equal', 'box')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('纯旁路跟踪 — 轨迹')
ax.legend(frameon=False)
ax.grid(ls=':')
# 横向误差
ax = axd['err']
ax.plot(t_lat, lat_error)
ax.set_xlim(left=0)
ax.set_ylabel('横向误差 (m)')
ax.set_title('横向误差')
ax.grid(ls=':')
# 转向角
ax = axd['delta']
ax.plot(t_lat, delta_hist)
ax.set_xlim(left=0)
ax.set_ylabel('转向角 δ (rad)')
ax.set_title('转向角命令')
ax.grid(ls=':')
for key in ['err', 'delta']:
    axd[key].set_xlabel('时间 (s)')
fig.savefig(os.path.join(fig_dir, 'lateral_tracking.png'))
plt.close(fig)

# === 12. 结果绘制调用 ===
plot_speed_tracking(time, results, desired_speeds)
plot_normalized_signal(time, results, desired_speeds, p)
plot_error(time, results, desired_speeds)
plot_traction(time, results, desired_speeds)

# 性能柱状图
fig_bar, ax_bar = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)
for ax, metric, ylabel, title in zip(
        ax_bar,
        ['MSE', 'Overshoot', 'Energy'],
        ['MSE', '超调量 (m/s)', '能量消耗'],
        ['各控制器 MSE 比较', '各控制器 超调量 比较', '各控制器 能量消耗 比较']):
    df.pivot(index='Speed', columns='Controller', values=metric) \
      .plot(kind='bar', ax=ax, grid=True, legend=(metric == 'MSE'))
    ax.set_xlabel('速度 (m/s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if metric != 'MSE' and ax.get_legend() is not None:
        ax.get_legend().remove()
fig_bar.suptitle('三种控制器性能指标柱状图', fontsize=14)
fig_bar.savefig(os.path.join(fig_dir, 'performance_bar.png'))
plt.close(fig_bar)

# === 13. 雷达图与热力图 ===
metrics_list = ['MSE', 'Overshoot', 'Energy']
controls  = ['PID', 'LQI', 'MPC']
fig = plt.figure(figsize=(12, 7))
gs  = fig.add_gridspec(2, 3, height_ratios=[3, 2], hspace=0.15, wspace=0.25)
angles = np.linspace(0, 2*np.pi, len(metrics_list), endpoint=False).tolist() + [0]
for col, v_speed in enumerate(desired_speeds):
    ax = fig.add_subplot(gs[0, col], projection='polar')
    sub = df.set_index(['Speed', 'Controller']).loc[v_speed]
    vals = np.array([sub.loc[c, metrics_list].values for c in controls])
    mins, maxs = vals.min(0), vals.max(0)
    norm = (vals - mins) / np.where(maxs-mins == 0, 1, maxs-mins)
    for idx, c in enumerate(controls):
        data = np.r_[norm[idx], norm[idx, 0]]
        ax.plot(angles, data, label=c, linewidth=1.2)
        ax.fill(angles, data, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_list)
    ax.set_ylim(0, 1)
    ax.set_title(f'{v_speed} m/s', pad=8, fontsize=10)
fig.legend(controls, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False)

cmap = 'viridis'
for col, metric in enumerate(metrics_list):
    ax   = fig.add_subplot(gs[1, col])
    data = df.pivot(index='Speed', columns='Controller', values=metric)
    im   = ax.imshow(data, cmap=cmap, origin='lower', aspect='auto')
    ax.set_xticks(range(len(data.columns))); ax.set_xticklabels(data.columns)
    ax.set_yticks(range(len(data.index)));   ax.set_yticklabels(data.index)
    ax.set_title(metric, fontsize=10)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f'{data.iloc[i,j]:.2g}', ha='center', va='center', color='white', fontsize=8)
    if col == 0:
        ax.set_ylabel('速度 (m/s)')
    fig.colorbar(im, ax=ax, shrink=0.65, pad=0.015)
fig.suptitle('三种控制器性能雷达图与热力图', fontsize=14, y=0.99)
fig.savefig(os.path.join(fig_dir, 'radar_heatmap.png'))
plt.close(fig)
