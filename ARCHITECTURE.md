# ROS 2 Autonomous Storage Navigation System — Architecture Design

**Version**: 1.0  
**ROS Distro**: Humble  
**Simulator**: Gazebo Classic 11  
**Hardware**: NVIDIA RTX 3080 Ti (CUDA 13.0) / AutoDL Cloud Container  

---

## 1. System Overview

A fully autonomous warehousing AMR (Autonomous Mobile Robot) capable of:

1. **Semantic object detection** (YOLOv8 + HSV dual-track vision with RGB-D depth projection)
2. **Priority-queue task scheduling** with safe Nav2 goal preemption
3. **Full pick-and-place physical pipeline**: navigate → approach → vacuum grip → retreat → transport → drop
4. **Cross-platform deployment**: AutoDL cloud ↔ WSL2 local, with DDS discovery server bridge

```
┌─────────────────────────────────────────────────────────────────┐
│                        /fetch_task (Action)                     │
│  task_emitter ──→ brain_node ──→ Nav2 ──→ planar_move ──→ odom │
│                       │                                        │
│                       ├──→ lift_controller (Z-axis prismatic)   │
│                       ├──→ /gripper/switch (vacuum suction)     │
│                       └──→ /target_object ←── vision_node       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Modules

### 2.1 Cognitive Center (`brain_node.py`)

**Priority-Queue Task Engine with Safe Nav2 Preemption**

| Feature | Implementation |
|---|---|
| Task protocol | `FetchTask.action`: 4D pose (x, y, z, yaw) + drop coordinates |
| Queue | `queue.PriorityQueue[(priority, seq, task)]` — lower value = higher urgency |
| Preemption | `cancel_goal_async()` → `_cancel_done_cb()` → `_flush_preempt()` async chain |
| Standoff kinematics | `standoff_point(tx, ty, tyaw, 0.5)` — observe target from 0.5m |
| Action lifecycle | flag-based polling (`_task_done`) → `goal_handle.succeed()` / `goal_handle.abort()` |
| Executor | `MultiThreadedExecutor` with `ReentrantCallbackGroup` |

**State Machine Flow**:

```
IDLE → STANDOFF_NAV → LIFT → ACTIVE_SEARCH → APPROACH → GRIP → RETREAT
  → DROP_NAV → DROP_OFF → IDLE
```

**Key design decisions**:

- `asyncio.Event` replaced with plain `bool` flag polling — rclpy Humble does not run a standard asyncio event loop
- Nav2 defensive callbacks: all non-SUCCEEDED statuses (CANCELED, ABORTED) reset `_nav_busy` and call `on_done(False)` to prevent deadlock
- Standoff projection avoids "zero-distance blind spot" where the camera cannot see the target

### 2.2 Perception Module (`vision_node.py`)

**Dual-Track Detection Architecture**

| Track | Method | Strengths | Weaknesses |
|---|---|---|---|
| **Primary** | YOLOv8n (TensorRT FP16, 1.4ms/inference) | Texture-rich objects (COCO classes) | Misses solid-color geometry |
| **Fallback** | OpenCV HSV filter → findContours → moments centroid | Solid-color boxes/barrels | Lighting-dependent |
| **Mock** | Fixed-position ground truth injection | Hardware-independent E2E testing | No real perception |

**Spatial Pipeline** (when camera is available):

```
RGB + Depth (message_filters sync) → YOLO/HSV bbox centroid (u,v)
  → Depth at (u,v) → Pinhole projection (Xc,Yc,Zc)
  → TF2: camera_depth_link → map
  → PointStamped on /target_object
```

**REP-103 Optical Frame**: `camera_depth_joint` has `rpy="-1.570796 0 -1.570796"` to align Z-forward with the camera lens axis.

### 2.3 Navigation Stack (Nav2)

```
map_server (AWS warehouse map, 5cm res) → /map (TRANSIENT_LOCAL QoS)
amcl (Monte Carlo Localization) → map→odom TF
planner_server → global_costmap
controller_server → /cmd_vel
behavior_server, bt_navigator, waypoint_follower, velocity_smoother
lifecycle_manager_localization (manages map_server + amcl)
lifecycle_manager_navigation (manages planners + controllers)
```

**Critical fix**: `bond_timeout=10.0` and startup ordering (`map_server` before `lifecycle_manager`) ensure reliable lifecycle activation through DDS discovery server.

### 2.4 Physics & Hardware (Gazebo Classic)

| Component | Plugin | Role |
|---|---|---|
| **Chassis** | `libgazebo_ros_planar_move.so` | Omnidirectional force on base_link |
| **Lift** | `libgazebo_ros_joint_pose_trajectory.so` | Prismatic joint Z-axis (0.0–0.8m) |
| **Gripper** | `gazebo_ros_vacuum_gripper` | Suction cup on lift_link front face |
| **Sensors** | `camera` + `depth` (Gazebo Classic) | RGB-D for vision pipeline |

**URDF → SDF Conversion**: `xacro → gz sdf -p → regex strip gz:namespace` enables Gazebo Classic compatibility for Jazzy-era linorobot2 URDF. Factory spawn via `spawn_entity.py -file <sdf>` ensures model plugins activate correctly.

---

## 3. Key Technical Challenges Resolved

### 3.1 Concurrency & Race Conditions

**Problem**: Calling `cancel_goal_async()` then immediately dispatching a new Nav2 goal caused race conditions — Nav2 rejected the new goal because the old one hadn't finished canceling.

**Solution**: Three-method async chain:
```
_preempt_current() → cancel_goal_async().add_done_callback(_cancel_done_cb)
_cancel_done_cb()  → wait for cancel future → _flush_preempt()
_flush_preempt()   → _nav_busy=False → _execute_task(new_prio)
```
All non-SUCCEEDED Nav2 result statuses (CANCELED, ABORTED) properly reset `_nav_busy` and call `on_done(False)` to unblock the pipeline.

### 3.2 Gazebo Plugin Silent Failure

**Problem**: Model plugins (planar_move, camera, laser) embedded in URDF `<gazebo>` blocks were not activated when spawned via `-topic robot_description`. ROS topics for odometry and sensors never appeared.

**Root cause**: `gz sdf -p` (URDF→SDF conversion) produced `gz:` namespace attributes (`gz:expressed_in`) that Gazebo Classic's SDF parser rejected with `Invalid XML`.

**Solution**: Post-process SDF with regex to strip `gz:` prefixed attributes. Factory-spawn the clean SDF via `-file` instead of topic-based spawn.

### 3.3 DDS Discovery Isolation

**Problem**: `ros2 topic list` returned only 2 topics (`/rosout`, `/parameter_events`) despite 28 nodes running. Terminal commands couldn't see topics published by Gazebo model plugins.

**Root cause**: `sim_manager.py` started `fast-discovery-server` with `stdout=subprocess.DEVNULL`, hiding the "Server is running" message and potentially misconfiguring the server. Additionally, `_ros2()` subprocess calls didn't inherit `ROS_DOMAIN_ID`/`ROS_DISCOVERY_SERVER` from the parent.

**Solution**: Remove DEVNULL redirect, pass `env=env` to `subprocess.Popen`, set `env['ROS_DISCOVERY_SERVER']` and `env['ROS_DOMAIN_ID']` before discovery server and all child processes.

### 3.4 Action Lifecycle: rclpy vs asyncio

**Problem**: `asyncio.Event()` used in Action Server's `execute_callback` produced `RuntimeWarning: coroutine 'Event.wait' was never awaited`. rclpy Humble does not run a standard Python asyncio event loop.

**Solution**: Replace `asyncio.Event` with a plain `bool` flag (`task['_task_done']`). The `_execute_fetch` coroutine polls this flag with `time.sleep(0.1)` — safe in `MultiThreadedExecutor` where the sleep blocks only the current thread, not the entire executor. `_succeed_task`/`_fail_task` set the flag and propagate result via `task['_result_ok']`/`task['_result_msg']`.

### 3.5 Camera Rendering in Headless Cloud

**Problem**: Gazebo Classic camera sensors require OGRE rendering engine, which requires an X11 display. In headless cloud containers, `gzserver` reported "Can't open display" and "Rendering will be disabled".

**Solution (partial)**: Xvfb (`:99`) provides a virtual X display, eliminating gzserver errors. However, OGRE FBO (Framebuffer Object) off-screen rendering still requires GPU OpenGL, which is unavailable in CUDA-only container GPU passthrough (missing `libnvidia-gpucomp`, `nvidia_drv.so`). **Workaround**: Mock vision node publishes ground-truth positions for E2E pipeline validation.

---

## 4. Directory Structure

```
ros2_ws/
├── src/project/                       # Core cognitive + perception nodes
│   ├── scripts/
│   │   ├── brain_node.py              # Priority-queue cognitive engine
│   │   ├── vision_node.py             # YOLO + HSV + mock ground-truth
│   │   ├── item_spawner.py            # 3D spawning with Z_CLEARANCE
│   │   ├── task_emitter.py            # FetchTask CLI client
│   │   ├── e2e_pick_place_test.py     # Physics-level E2E audit
│   │   ├── stress_test_preemption.py  # Preemption stress test
│   │   ├── calibration_check.py       # Spatial calibration probe
│   │   └── sim_manager.py             # Unified lifecycle + DDS + Xvfb
│   ├── launch/system_bringup.launch.py
│   ├── action/FetchTask.action
│   └── params/nav2_params.yaml
│
├── src/linorobot2_description/        # Mecanum AMR URDF
│   └── urdf/
│       ├── robots/mecanum.urdf.xacro  # Main assembly
│       ├── mech/lift.urdf.xacro       # Prismatic Z-axis lift
│       ├── mech/gripper.urdf.xacro    # Vacuum suction gripper
│       ├── sensors/                   # Camera, laser, IMU (Classic-adapted)
│       └── controllers/
│           └── gazebo_classic_drive.urdf.xacro  # planar_move plugin
│
├── src/linorobot2_gazebo/             # Gazebo launch + worlds
│   ├── launch/gazebo_classic.launch.py
│   └── config/controller_manager.yaml
│
├── src/linorobot2_navigation/         # Nav2 + SLAM configs
│
└── src/aws-robomaker-small-warehouse-world/  # AWS warehouse assets
    ├── worlds/no_roof_small_warehouse/
    ├── maps/005/                      # 5cm-resolution map
    └── models/                        # 14 warehouse fixture models
```

---

## 5. Build & Run

```bash
# Build
cd ros2_ws
colcon build --symlink-install --packages-select \
  project aws_robomaker_small_warehouse_world \
  linorobot2_description linorobot2_gazebo linorobot2_navigation

# Launch (one command)
source install/setup.bash
python3 src/project/scripts/sim_manager.py start

# E2E test (separate terminal)
source install/setup.bash
export ROS_DOMAIN_ID=30
export ROS_DISCOVERY_SERVER=127.0.0.1:11811
ros2 run project e2e_pick_place_test.py
```

## 6. Key Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `DISPLAY` | `:99` | Xvfb virtual framebuffer |
| `ROS_DOMAIN_ID` | `30` | DDS isolation domain |
| `ROS_DISCOVERY_SERVER` | `127.0.0.1:11811` | Fast-DDS Super Client mode |
| `QT_X11_NO_MITSHM` | `1` | Container GPU shared-memory workaround |
| `LINOROBOT2_BASE` | `mecanum` | Robot model selection |

## 7. Known Limitations

| Issue | Impact | Mitigation |
|---|---|---|
| No GPU OpenGL (missing `libnvidia-gpucomp`) | Camera/laser sensors inactive | Mock vision for E2E; GPU container for production |
| Gazebo Classic end-of-life Jan 2025 | No upstream fixes | SDF conversion workaround via `gz sdf -p` |
| DDS discovery server required | Extra process (`fast-discovery-server`) | Managed by `sim_manager.py` with graceful shutdown |
| AWS world models use `file://models/` URIs | Mesh loading fails without `model://` | Batch `sed` replacement in 14 SDF files |
