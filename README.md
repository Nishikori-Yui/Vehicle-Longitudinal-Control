# 车辆纵横向联合控制 Vehicle Longitudinal and Lateral Control

## 一、项目概述 (Project Overview)
本项目基于纯 Python 实现了一套车辆纵横向联合控制仿真框架，集成多种经典及前沿控制算法，包含自适应 PID、LQR、多目标 MPC 及 ABS 防抱死制动。目标在于通过数值仿真与可视化分析，评估不同控制策略在各种工况下的性能差异，并为未来的纵横向一体化和硬件在环验证奠定基础。

## 二、核心功能 (Key Features)
1. **多种控制器**  
   - 自适应 PID（PIDControllerAdaptive）  
   - LQR（LQRController）  
   - 多目标 MPC（MPCControllerMultiObjective）  
   - ABS 防抱死制动（ABSController）  

2. **精确车辆与执行器模型**  
   - 动态自行车模型（纵横向耦合）  
   - 空气阻力 + 滚动阻力 + 坡度阻力模型  
   - 二阶执行器动力学模型 + 简化油门/制动模型  

3. **全面性能评估**  
   - 速度跟踪误差 (MSE)  
   - 最大超调量 (Overshoot)  
   - 能量消耗 (Energy Consumption)  
   - 纵向舒适性（Jerk RMS）  
   - 横向跟踪误差（RMS）  

4. **可视化支持**  
   - Matplotlib 绘制速度曲线、误差曲线、雷达图、热力图等  

5. **模块化与可扩展**  
   - 清晰的目录与模块划分  
   - 支持参数化配置与批量仿真  

## 三、目录结构 (Directory Structure)
```text
Vehicle-Longitudinal-Control/
├─ configs/                        # 参数与场景配置
│   ├─ vehicle_params.py           # 车辆物理与执行器参数
│   ├─ scenarios.py                # 工况函数与道路路径
│   └─ scenarios.json              # 工况配置（默认）
│
├─ models/                         # 车辆与执行器模型
│   ├─ vehicle.py                  # 纵向阻力模型
│   ├─ vehicle_dynamics.py         # 动态自行车模型
│   └─ actuator.py                 # 二阶执行器模型
│
├─ controllers/                    # 控制器实现
│   ├─ longitudinal/               # 纵向控制器
│   │  ├─ pid_controller.py
│   │  ├─ lqr_controller.py
│   │  ├─ mpc_controller.py
│   │  └─ abs_controller.py
│   └─ lateral/                    # 横向控制器
│      ├─ pure_pursuit.py          # 纯追踪算法（历史）
│      └─ lqr_controller.py        # LQR 横向控制
│
├─ powertrain/                     # 可插拔动力系统
│   ├─ base.py
│   ├─ force_powertrain.py
│   └─ throttle_brake.py
│
├─ sim/                            # 仿真引擎
│   └─ simulator.py
│
├─ legacy/                         # 历史版本归档
│   └─ Report_Public.pdf           # 重构前报告
│
├─ utils/                          # 工具函数
│   ├─ metrics.py                  # MSE、超调、能耗计算
│   └─ plotting.py                 # 结果可视化函数
│
├─ Simulate_CLI.py                 # CLI 仿真入口
├─ vlc_report.py                   # 旧版报告绘图脚本（历史）
├─ requirements.txt                # Python 依赖
├─ LICENSE                         # 开源许可证
└─ README.md                       # 项目说明文档
````

## 四、安装与依赖 (Installation)

1. 克隆仓库：

   ```bash
   git clone https://github.com/Nishikori-Yui/Vehicle-Longitudinal-Control.git
   cd Vehicle-Longitudinal-Control
   ```
2. 创建并激活虚拟环境（以 `venv` 为例）：

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

## 五、快速开始 (Quick Start)

1. **CLI 命令行仿真**

   ```bash
   python Simulate_CLI.py
   ```

   脚本将依次对预设目标速度工况运行所有控制器，并在 `results/cli_runs/<timestamp>/` 目录输出图表与指标报告。

2.  **旧版报告图片复现（历史）**
   
      ```bash
      python vlc_report.py
      ```
   
      脚本将依次对依据论文要求输出所需图表，并在 `results/report/` 目录输出图表与指标报告。
      `legacy/Report_Public.pdf` 对应本次重构之前的版本。


## 六、参数配置 (Configuration)

* `configs/vehicle_params.py`：集中管理车辆质量、空气阻力系数、滚动阻力系数、执行器自然频率／阻尼比、最大驱动力／制动力、ABS 参数等参数变量；
* `configs/scenarios.py`：集中管理坡度、附着系数、目标速度等工况函数与道路路径构造。
* `configs/scenarios.json`：默认工况配置，可直接修改实现批量工况测试。
* 可在代码中使用 `VehicleParams(preset="mid_sedan")` 载入中型燃油轿车参数集。
* `Simulate_CLI.py` 中提供 `use_abs` 开关，用于验证 ABS 对制动通道的影响。
* 可通过脚本修改参数实现权重和限幅参数的动态调整。

## 七、结果可视化 (Visualization)

* 速度跟踪曲线
* 误差与牵引力命令对比
* MSE、超调、能耗雷达图与热力图
* 支持导出为 PNG、PDF 格式
