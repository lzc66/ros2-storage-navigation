# ROS 2 Autonomous Storage Navigation System

基于 ROS 2 Humble + Gazebo Classic 的自主仓储分拣 AMR 系统。

## Architecture

```
├── src/project/                          # 核心认知与感知节点
│   ├── scripts/
│   │   ├── brain_node.py                 # 优先级任务调度 + Nav2 抢占 + 视觉抓取
│   │   ├── vision_node.py                # YOLOv8 + HSV 双轨检测 + RGB-D 深度投影
│   │   ├── item_spawner.py               # 3D 投放引擎 (Z_CLEARANCE 防穿模)
│   │   ├── task_emitter.py               # FetchTask Action 客户端
│   │   ├── e2e_pick_place_test.py        # 端到端物理审计
│   │   ├── stress_test_preemption.py     # 抢占压力测试
│   │   └── sim_manager.py                # 统一生命周期管理
│   ├── launch/system_bringup.launch.py   # 一键启动 (Gazebo + Nav2 + Brain)
│   ├── action/FetchTask.action           # 4D 任务协议 (x, y, z, yaw)
│   └── params/nav2_params.yaml           # Nav2 参数
│
├── src/linorobot2_description/           # Mecanum 底盘 URDF
│   └── urdf/robots/mecanum.urdf.xacro    # 升降平台 + 真空吸盘 + RGB-D 相机
│
├── src/aws-robomaker-small-warehouse-world/  # AWS 工业仓储地图
│   └── worlds/no_roof_small_warehouse/       # 无屋顶版 (优化视野)
│
├── src/linorobot2_gazebo/                # Gazebo 仿真配置
│   ├── config/controller_manager.yaml    # ros2_control 控制器
│   └── launch/gazebo_classic.launch.py   # Gazebo Classic 启动
│
└── src/linorobot2_navigation/            # Nav2 + SLAM 配置
```

## Quick Start

```bash
# Build
cd ros2_ws
colcon build --symlink-install --packages-select project \
  aws_robomaker_small_warehouse_world linorobot2_description \
  linorobot2_gazebo linorobot2_navigation

# Launch (AWS warehouse + mecanum AMR + Nav2 + Brain)
source install/setup.bash
export DISPLAY=:1
python3 src/project/scripts/sim_manager.py start

# Send fetch task
ros2 run project task_emitter -p 2 --x 2.0 --y 1.0 --z 0.5 --yaw 0.0 \
  --drop-x -1.0 --drop-y -1.0

# E2E test
ros2 run project e2e_pick_place_test.py
```

## Key Features

- **Priority-Queue Task Engine**: preemptive Nav2 goal management
- **Standoff Kinematics**: 0.5m observation point for camera FOV
- **Active Search**: slow rotation (0.15 rad/s) for visual target acquisition
- **4D Task Protocol**: (x, y, z, yaw) with drop-off coordinates
- **Vacuum Gripper**: suction-based pick-and-place pipeline
- **YOLOv8 + HSV Fallback**: dual-track vision with depth projection
- **TF2 Spatial Projection**: camera_optical_frame → map

## Dependencies

- ROS 2 Humble
- Gazebo Classic 11
- Nav2
- Ultralytics YOLOv8
- PyTorch + CUDA

## Environment Notes

- Designed for `ROS_DOMAIN_ID=30` with Fast-DDS Discovery Server (127.0.0.1:11811)
- Headless rendering via Mesa llvmpipe (GPU: `LIBGL_ALWAYS_SOFTWARE=1`)
- System Python 3.10 required (Conda hijack mitigation: `PATH=/usr/bin:$PATH`)
