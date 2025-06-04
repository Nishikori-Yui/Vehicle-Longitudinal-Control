# 车辆纵向动力学模型控制 Vehicle Longitudinal Control

## 一、项目概述 (Project Overview)
本项目基于纯 Python 实现了一套车辆纵向控制仿真框架，集成多种经典及前沿控制算法，包含自适应 PID、增广 LQI、多目标 MPC 及 ABS 防抱死制动。目标在于通过数值仿真与可视化分析，评估不同控制策略在各种工况下的性能差异，并为未来的纵横向一体化和硬件在环验证奠定基础。

## 二、核心功能 (Key Features)
1. **多种控制器**  
   - 自适应 PID（PIDControllerAdaptive）  
   - 增广 LQI（LQIController）  
   - 多目标 MPC（MPCControllerMultiObjective）  
   - ABS 防抱死制动（ABSController）  

2. **精确车辆与执行器模型**  
   - 空气阻力 + 滚动阻力模型  
   - 二阶执行器动力学模型  

3. **全面性能评估**  
   - 速度跟踪误差 (MSE)  
   - 最大超调量 (Overshoot)  
   - 能量消耗 (Energy Consumption)  

4. **可视化与 GUI 支持**  
   - Matplotlib 绘制速度曲线、误差曲线、雷达图、热力图等  
   - 基于 Tkinter 的交互式图形界面  

5. **模块化与可扩展**  
   - 清晰的目录与模块划分  
   - 支持参数化配置与批量仿真  

## 三、目录结构 (Directory Structure)
```text
Vehicle-Longitudinal-Control/
├─ configs/                        # 参数与场景配置
│   └─ vehicle_params.py           # 车辆物理与执行器参数
│
├─ models/                         # 车辆与执行器模型
│   ├─ vehicle.py                  # 空气阻力 & 滚动阻力模型
│   └─ actuator.py                 # 二阶执行器模型
│
├─ controllers/                    # 各类控制器实现
│   ├─ pid_controller.py
│   ├─ lqi_controller.py
│   ├─ mpc_controller.py
│   └─ abs_controller.py
│
├─ lateral_control/                # 横向控制示例
│   └─ pure_pursuit.py             # 纯追踪算法
│
├─ utils/                          # 工具函数
│   ├─ metrics.py                  # MSE、超调、能耗计算
│   └─ plotting.py                 # 结果可视化函数
│
├─ Simulate_CLI.py                 # CLI 仿真入口
├─ Simulate_GUI_old-fashioned.py   # Tkinter GUI （旧版，已废弃）
├─ Simulate_GUI.py                 # GUI 仿真入口
├─ vlc_report.py                   # 报告撰写使用
├─ Report_Public.pdf               # 分析报告
├─ requirements.txt                # Python 依赖
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

   脚本将依次对预设目标速度工况运行所有控制器，并在 `results/cli_figures/` 目录输出图表与指标报告。

2. **GUI 交互式仿真**

   ```bash
   python Simulate_GUI.py
   ```

   在图形界面中选择控制算法、配置参数，仿真结束后自动展示结果,并在 `results/gui_figures/` 目录输出图表。
3.  **报告论文图片复现**
   
      ```bash
      python vlc_report.py
      ```
   
      脚本将依次对依据论文要求输出所需图表，并在 `results/report/` 目录输出图表与指标报告。


## 六、参数配置 (Configuration)

* `configs/vehicle_params.py`：集中管理车辆质量、空气阻力系数、滚动阻力系数、执行器自然频率／阻尼比、最大驱动力／制动力、ABS 参数等参数变量；
* 可在 GUI 中修改参数实现权重和限幅参数的动态调整。

## 七、结果可视化 (Visualization)

* 速度跟踪曲线
* 误差与牵引力命令对比
* MSE、超调、能耗雷达图与热力图
* 支持导出为 PNG、PDF 格式

