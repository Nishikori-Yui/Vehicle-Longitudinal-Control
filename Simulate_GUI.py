import tkinter as tk
from tkinter import ttk, messagebox
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from configs.vehicle_params import VehicleParams
from models.vehicle import disturbance
from controllers.pid_controller import PIDControllerAdaptive
from controllers.lqi_controller import LQIController
from controllers.mpc_controller import MPCControllerMultiObjective
from controllers.abs_controller import ABSController
from utils.metrics import compute_metrics
from utils.plotting import (
    plot_speed_tracking, plot_normalized_signal,
    plot_error, plot_traction, plot_lateral_tracking,
    plot_metrics_bar, plot_heatmap, plot_radar
)
from lateral_control.pure_pursuit import pure_pursuit

class GuiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('车辆控制仿真')
        self.params = VehicleParams()
        self.image_paths = []
        self.image_index = 0
        self.categories = {}
        self._create_widgets()
        # 隐藏图片查看区，运行仿真后显示
        self.img_frame.grid_remove()

    def _create_widgets(self):
        # 状态标签
        self.status_label = ttk.Label(self, text='')
        self.status_label.grid(row=3, column=0, columnspan=2, pady=5, sticky='w')

        # 选择控制器
        ctrl_frame = ttk.LabelFrame(self, text='选择控制器')
        ctrl_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nw')
        self.ctrl_vars = {}
        for i, name in enumerate(['PID', 'LQI', 'MPC']):
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(ctrl_frame, text=name, variable=var, command=self._toggle_ctrl_params)
            chk.grid(row=i, column=0, sticky='w')
            self.ctrl_vars[name] = var

        # 控制器参数
        self.ctrl_param_frame = ttk.LabelFrame(self, text='控制器参数')
        self.ctrl_param_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nw')
        # PID 参数
        self.pid_frame = ttk.LabelFrame(self.ctrl_param_frame, text='PID')
        self.pid_frame.grid(row=0, column=0, padx=2, pady=2)
        ttk.Label(self.pid_frame, text='Kp0').grid(row=0, column=0)
        self.pid_kp = tk.DoubleVar(value=6.0)
        ttk.Entry(self.pid_frame, textvariable=self.pid_kp, width=8).grid(row=0, column=1)
        ttk.Label(self.pid_frame, text='Ki0').grid(row=1, column=0)
        self.pid_ki = tk.DoubleVar(value=0.1)
        ttk.Entry(self.pid_frame, textvariable=self.pid_ki, width=8).grid(row=1, column=1)
        ttk.Label(self.pid_frame, text='beta').grid(row=2, column=0)
        self.pid_beta = tk.DoubleVar(value=0.5)
        ttk.Entry(self.pid_frame, textvariable=self.pid_beta, width=8).grid(row=2, column=1)
        # LQI 参数
        self.lqi_frame = ttk.LabelFrame(self.ctrl_param_frame, text='LQI')
        self.lqi_frame.grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(self.lqi_frame, text='初始速度 v0').grid(row=0, column=0)
        self.lqi_v0 = tk.DoubleVar(value=10.0)
        ttk.Entry(self.lqi_frame, textvariable=self.lqi_v0, width=8).grid(row=0, column=1)
        # MPC 参数
        self.mpc_frame = ttk.LabelFrame(self.ctrl_param_frame, text='MPC')
        self.mpc_frame.grid(row=0, column=2, padx=2, pady=2)
        ttk.Label(self.mpc_frame, text='预测步长 N').grid(row=0, column=0)
        self.mpc_N = tk.IntVar(value=25)
        ttk.Entry(self.mpc_frame, textvariable=self.mpc_N, width=8).grid(row=0, column=1)
        ttk.Label(self.mpc_frame, text='q').grid(row=1, column=0)
        self.mpc_q = tk.DoubleVar(value=3.0)
        ttk.Entry(self.mpc_frame, textvariable=self.mpc_q, width=8).grid(row=1, column=1)
        ttk.Label(self.mpc_frame, text='r').grid(row=2, column=0)
        self.mpc_r = tk.DoubleVar(value=0.01)
        ttk.Entry(self.mpc_frame, textvariable=self.mpc_r, width=8).grid(row=2, column=1)
        ttk.Label(self.mpc_frame, text='能量权重').grid(row=3, column=0)
        self.mpc_e = tk.DoubleVar(value=1e-5)
        ttk.Entry(self.mpc_frame, textvariable=self.mpc_e, width=8).grid(row=3, column=1)

        # 车辆参数两列
        veh_frame = ttk.LabelFrame(self, text='车辆参数')
        veh_frame.grid(row=1, column=0, padx=5, pady=5, sticky='nw')
        param1 = [('质量 kg', 'm'), ('重力 g', 'g'), ('轴距 m', 'wheelbase'), ('Cd', 'Cd')]
        param2 = [('A m²', 'A'), ('rho', 'rho'), ('Cr', 'Cr'), ('wn', 'wn')]
        self.veh_vars = {}
        for i, (lbl, attr) in enumerate(param1):
            ttk.Label(veh_frame, text=lbl).grid(row=i, column=0, sticky='e')
            var = tk.DoubleVar(value=getattr(self.params, attr))
            ttk.Entry(veh_frame, textvariable=var, width=8).grid(row=i, column=1)
            self.veh_vars[attr] = var
        for i, (lbl, attr) in enumerate(param2):
            ttk.Label(veh_frame, text=lbl).grid(row=i, column=2, sticky='e')
            var = tk.DoubleVar(value=getattr(self.params, attr))
            ttk.Entry(veh_frame, textvariable=var, width=8).grid(row=i, column=3)
            self.veh_vars[attr] = var
        others = [('zeta', 'zeta'), ('r', 'r'), ('J', 'J'), ('c1', 'c1'), ('c2', 'c2'), ('c3', 'c3'),
                  ('tau_hyd', 'tau_hyd'), ('F_max', 'F_max'), ('F_min', 'F_min'), ('tau', 'tau')]
        for i, (lbl, attr) in enumerate(others):
            row = 4 + i // 2
            col = (i % 2) * 2
            ttk.Label(veh_frame, text=lbl).grid(row=row, column=col, sticky='e')
            var = tk.DoubleVar(value=getattr(self.params, attr))
            ttk.Entry(veh_frame, textvariable=var, width=8).grid(row=row, column=col + 1)
            self.veh_vars[attr] = var
        # self.abs_var = tk.BooleanVar(value=False)
        # ttk.Checkbutton(veh_frame, text='启用ABS', variable=self.abs_var).grid(row=row + 1, column=0, columnspan=2,
        #                                                                       sticky='w')

        # 选择绘图
        plot_frame = ttk.LabelFrame(self, text='选择绘图')
        plot_frame.grid(row=1, column=1, padx=5, pady=5, sticky='nw')
        plots = [('速度曲线', 'speed'), ('归一化信号', 'normalized'), ('误差曲线', 'error'), ('牵引力曲线', 'traction'),
                 ('横向跟踪', 'lateral'), ('柱状图', 'bar'), ('热力图', 'heatmap'), ('雷达图', 'radar')]
        self.plot_vars = {}
        for i, (lbl, key) in enumerate(plots):
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(plot_frame, text=lbl, variable=var).grid(row=i, column=0, sticky='w')
            self.plot_vars[key] = var

        # 运行按钮
        ttk.Button(self, text='运行仿真', command=self.run_sim).grid(row=2, column=0, columnspan=2, pady=10)

        # 图片查看区
        self.img_frame = ttk.LabelFrame(self, text='图片查看')
        self.img_frame.grid(row=0, column=2, rowspan=3, padx=5, pady=5)

        # 一级分类下拉
        ttk.Label(self.img_frame, text='图表类型:').pack(anchor='w')
        self.category_var = tk.StringVar()
        self.category_box = ttk.Combobox(self.img_frame, textvariable=self.category_var, state='readonly')
        self.category_box.pack(fill='x', padx=5)
        self.category_box.bind('<<ComboboxSelected>>', self._on_category_change)

        # 二级分类下拉
        ttk.Label(self.img_frame, text='子类型:').pack(anchor='w', pady=(5, 0))
        self.subcategory_var = tk.StringVar()
        self.subcategory_box = ttk.Combobox(self.img_frame, textvariable=self.subcategory_var, state='readonly')
        self.subcategory_box.pack(fill='x', padx=5)
        self.subcategory_box.bind('<<ComboboxSelected>>', self._on_subcategory_change)

        # 图片显示与导航按钮
        self.image_label = ttk.Label(self.img_frame)
        self.image_label.pack(pady=5)
        nav_frame = ttk.Frame(self.img_frame)
        nav_frame.pack(pady=5)
        ttk.Button(nav_frame, text='上一张', command=self._prev_image).grid(row=0, column=0, padx=2)
        ttk.Button(nav_frame, text='下一张', command=self._next_image).grid(row=0, column=1, padx=2)

    def _toggle_ctrl_params(self):
        self.pid_frame.grid() if self.ctrl_vars['PID'].get() else self.pid_frame.grid_remove()
        self.lqi_frame.grid() if self.ctrl_vars['LQI'].get() else self.lqi_frame.grid_remove()
        self.mpc_frame.grid() if self.ctrl_vars['MPC'].get() else self.mpc_frame.grid_remove()

    def _on_category_change(self, event=None):
        """一级分类改变，更新二级分类下拉"""
        cat = self.category_var.get()
        subcats = sorted(self.categories.get(cat, {}).keys())
        self.subcategory_box.config(values=subcats)
        if subcats:
            self.subcategory_var.set(subcats[0])
            self._on_subcategory_change()

    def _on_subcategory_change(self, event=None):
        """二级分类改变，加载对应图片列表"""
        cat = self.category_var.get()
        sub = self.subcategory_var.get()
        self.image_paths = self.categories.get(cat, {}).get(sub, [])
        self.image_index = 0
        self._show_image()

    def _show_image(self):
        """显示当前索引图片"""
        if not self.image_paths:
            self.image_label.config(image='')
            self.status_label.config(text='无可显示图片')
            return
        path = self.image_paths[self.image_index]
        img = Image.open(path).resize((400, 300), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)
        self.status_label.config(text=f'显示: {os.path.basename(path)}')

    def _prev_image(self):
        if not self.image_paths:
            return
        self.image_index = (self.image_index - 1) % len(self.image_paths)
        self._show_image()

    def _next_image(self):
        if not self.image_paths:
            return
        self.image_index = (self.image_index + 1) % len(self.image_paths)
        self._show_image()

    def _change_category(self):
        cat = self.category_var.get()
        self.image_paths = self.categories.get(cat, [])
        self.image_index = 0
        self._show_image()

    def run_sim(self):
        # 反馈开始仿真
        self.status_label.config(text='仿真进行中，请稍候...')
        self.update_idletasks()

        # 更新车辆参数
        for attr, var in self.veh_vars.items(): setattr(self.params, attr, var.get())
        # enable_abs = self.abs_var.get()
        # 组装控制器
        controllers = []
        if self.ctrl_vars['PID'].get(): controllers.append(('PID', PIDControllerAdaptive,
                                                            {'Kp0': self.pid_kp.get(), 'Ki0': self.pid_ki.get(),
                                                             'beta': self.pid_beta.get(), 'p': self.params}))
        if self.ctrl_vars['LQI'].get(): controllers.append(
            ('LQI', LQIController, {'v0': self.lqi_v0.get(), 'p': self.params}))
        if self.ctrl_vars['MPC'].get(): controllers.append(('MPC', MPCControllerMultiObjective,
                                                            {'N': self.mpc_N.get(), 'q': self.mpc_q.get(),
                                                             'r': self.mpc_r.get(), 'e_energy': self.mpc_e.get(),
                                                             'p': self.params}))
        # 仿真设置
        dt, T_final = 0.2, 40.0;
        time = np.arange(0, T_final + dt, dt);
        speeds = [10, 20, 30]
        results, sigs, mets = {n: [] for n, _, __ in controllers}, {k: {n: [] for n, _, __ in controllers} for k in
                                                                    ['normalized', 'speed', 'error', 'traction']}, []
        for v_ref in speeds:
            for name, cls, kw in controllers:
                ctrl = cls(**kw);
                ctrl.reset();
                v_curr = 0.0;
                hv, hu, he, hf = [], [], [], []
                for _ in time:
                    e = v_ref - v_curr
                    u, f = (ctrl.step(e, dt) if name == 'PID' else ctrl.step(v_curr, v_ref, dt))
                    # if enable_abs: f = ABSController(self.params).step(v_curr, u, dt)
                    v_curr += (f - disturbance(v_curr, self.params)) / self.params.m * dt
                    hv.append(v_curr);
                    hu.append(u);
                    he.append(e);
                    hf.append(f)
                results[name].append(np.array(hv))
                sigs['normalized'][name].append(np.array(hu) / (np.max(np.abs(hu)) + 1e-6))
                sigs['speed'][name].append(np.array(hv))
                sigs['error'][name].append(np.array(he))
                sigs['traction'][name].append(np.array(hf))
                mse, osht, energy = compute_metrics(np.array(hv), np.array(hu), v_ref, dt)
                mets.append({'Controller': name, 'Speed (m/s)': v_ref, 'MSE': mse, 'Overshoot': osht, 'Energy': energy})
        # 保存与绘图
        df = pd.DataFrame(mets);
        os.makedirs('results', exist_ok=True);
        df.to_csv('results/metrics_summary.csv', index=False)
        out = 'results/gui_figures';
        if self.plot_vars['speed'].get(): plot_speed_tracking(time, results, speeds, out)
        if self.plot_vars['normalized'].get(): plot_normalized_signal(time, sigs['normalized'], speeds, out)
        if self.plot_vars['error'].get(): plot_error(time, sigs['error'], speeds, out)
        if self.plot_vars['traction'].get(): plot_traction(time, sigs['traction'], speeds, out)
        if self.plot_vars['lateral'].get():
            tl,xr,yr,x,y,psi,dh,le=pure_pursuit(50.0,15.0,5.0,0.2,20.0,self.params)
            plot_lateral_tracking(tl,xr,yr,x,y,dh,le,out)
        if self.plot_vars['bar'].get(): plot_metrics_bar(df,out)
        if self.plot_vars['heatmap'].get(): plot_heatmap(df,out)
        if self.plot_vars['radar'].get():
            for sp in speeds: plot_radar(df,sp,out)
        os.makedirs(out, exist_ok=True)
        mapping = {
            'speed_tracking': '速度跟踪',
            'normalized_signal': '归一化信号',
            'speed_error': '速度误差',
            'traction_command': '牵引力命令',
            'lateral_error': '横向误差',
            'lateral_path_tracking': '横向路径跟踪',
            'steering_angle': '转向角',
            'bar_mse': '均方误差柱状图',
            'bar_overshoot': '最大超调柱状图',
            'bar_energy': '能耗柱状图',
            'heatmap_mse': '热力图(MSE)',
            'radar': '雷达图'
        }
        self.categories.clear()
        for fname in sorted(os.listdir(out)):
            if not fname.endswith('.png'): continue
            name = fname[:-4]
            for key, zh in mapping.items():
                if name.startswith(key):
                    if key in ['steering_angle', 'bar_mse', 'bar_overshoot', 'bar_energy', 'heatmap_mse']:
                        sub = '所有'
                    else:
                        parts = name.split('_')
                        sub = parts[1].replace('ms', ' m/s') if len(parts) > 1 else '所有'
                    self.categories.setdefault(zh, {}).setdefault(sub, []).append(os.path.join(out, fname))
                    break

        # 更新一级分类并触发事件
        cats = list(self.categories.keys())
        self.category_box.config(values=cats)
        if cats:
            self.category_var.set(cats[0])
            self._on_category_change()

        # 反馈完成
        self.status_label.config(text='仿真及绘图已完成')
        messagebox.showinfo('完成', '仿真及绘图已输出至: ' + os.path.abspath(out))
        self.img_frame.grid()

if __name__ == '__main__':
    GuiApp().mainloop()
