#!/usr/bin/env python3
"""Unified CLI for ROS 2 simulation lifecycle management."""
import argparse
import os
import signal
import subprocess
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(_SCRIPT_DIR)
WS_DIR = os.path.dirname(os.path.dirname(PKG_DIR))
SETUP_BASH = os.path.join(WS_DIR, 'install', 'setup.bash')


_LAUNCH_PROC = None
_DISCOVERY_PROC = None
_XVFB_PROC = None


def _ros2(cmd: str, check=True, timeout=None, capture=False):
    """Wrap a ROS 2 command with environment sourcing + DDS env inheritance."""
    full = f"source {SETUP_BASH} && {cmd}"
    print(f'[EXEC] {cmd[:120]}...' if len(cmd) > 120 else f'[EXEC] {cmd}')
    # Inherit parent env (ROS_DOMAIN_ID, ROS_DISCOVERY_SERVER, etc.)
    sub_env = os.environ.copy()
    sub_env.setdefault('ROS_DOMAIN_ID', '30')
    sub_env.setdefault('ROS_DISCOVERY_SERVER', '127.0.0.1:11811')
    try:
        return subprocess.run(full, shell=True, executable='/bin/bash', check=check,
                              timeout=timeout, capture_output=capture, text=True,
                              env=sub_env)
    except subprocess.CalledProcessError as e:
        print(f'[ERROR] Command failed (exit {e.returncode})')
        if capture:
            print(f'  stderr: {e.stderr[:300]}')
        if check:
            raise


# ===================== build =====================
def cmd_build(_args):
    print('[INFO] Building project...')
    _ros2('colcon build --packages-select project --symlink-install',
          timeout=120)
    print('[INFO] Build complete.')


# ===================== start =====================
def cmd_start(_args):
    global _LAUNCH_PROC, _XVFB_PROC
    print('[INFO] Launching system_bringup (GUI)...')

    # --- Xvfb virtual display (headless GPU containers) ---
    print('[INFO] Starting Xvfb on :99 (1024x768x24)...')
    _XVFB_PROC = subprocess.Popen(
        ['Xvfb', ':99', '-screen', '0', '1024x768x24', '-ac', '+extension', 'GLX'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    if _XVFB_PROC.poll() is not None:
        print('[ERROR] Xvfb failed to start. Camera sensors will be disabled.')

    env = os.environ.copy()
    env['DISPLAY'] = ':99'          # virtual framebuffer
    env['QT_X11_NO_MITSHM'] = '1'  # avoid MIT-SHM errors in containers
    env['PATH'] = '/usr/bin:' + env.get('PATH', '')
    # Mesa software rendering optimization for headless GPU containers
    env.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
    env.setdefault('GALLIUM_DRIVER', 'llvmpipe')
    env.setdefault('LP_NUM_THREADS', '4')
    env.setdefault('vblank_mode', '0')
    env.setdefault('mesa_glthread', 'true')
    # CUDA compute (available even when GLX is broken)
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    # Cross-platform model paths (AutoDL + WSL compatible)
    _aws_wh_models = os.path.join(WS_DIR, 'src', 'aws-robomaker-small-warehouse-world', 'models')
    _lino_models   = os.path.join(WS_DIR, 'src', 'linorobot2_gazebo', 'models')
    env['GAZEBO_MODEL_PATH'] = (
        os.path.join(PKG_DIR, 'models') + ':' +
        os.path.join(os.path.expanduser('~'), '.gazebo', 'models') + ':' +
        '/opt/ros/humble/share/turtlebot3_gazebo/models' + ':' +
        _aws_wh_models + ':' +
        _lino_models
    )
    # DDS Discovery Server — start before ROS 2 launch
    global _DISCOVERY_PROC
    print('[INFO] Starting Fast-DDS Discovery Server (127.0.0.1:11811)...')
    _DISCOVERY_PROC = subprocess.Popen(
        ['fast-discovery-server', '-i', '0', '-l', '127.0.0.1', '-p', '11811'],
        env=env, stdout=None, stderr=None,
    )
    time.sleep(2.0)
    env['ROS_DISCOVERY_SERVER'] = '127.0.0.1:11811'
    env['ROS_DOMAIN_ID'] = '30'
    launch_cmd = (
        f'source {SETUP_BASH} && '
        f'ros2 launch project system_bringup.launch.py'
    )

    _LAUNCH_PROC = subprocess.Popen(
        ['bash', '-c', launch_cmd],
        env=env,
        preexec_fn=os.setsid,
    )
    print(f'[INFO] Launch PID={_LAUNCH_PROC.pid} (group)')

    # Auto-init after 40s
    print('[INFO] Waiting 40s for system init...')
    for remaining in range(40, 0, -5):
        time.sleep(5)
        print(f'  ... {remaining}s')

    # Send initialpose via /initialpose topic
    print('[INFO] Sending /initialpose...')
    try:
        _ros2(
            'ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped '
            "'{header: {frame_id: \"map\"}, "
            "pose: {pose: {position: {x: -2.0, y: -0.5, z: 0.0}, "
            "orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, "
            "covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, "
            "0.0, 0.25, 0.0, 0.0, 0.0, 0.0, "
            "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
            "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
            "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
            "0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891909122467]}}'",
            timeout=10
        )
        print('[INFO] /initialpose sent.')
    except Exception as e:
        print(f'[WARNING] init pose failed: {e}')

    print('[INFO] System running. Press Ctrl+C to stop.')
    try:
        while _LAUNCH_PROC.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n[INFO] Caught SIGINT, shutting down launch group...')
        if _XVFB_PROC is not None and _XVFB_PROC.poll() is None:
            _XVFB_PROC.terminate()
            _XVFB_PROC.wait(timeout=5)
            print('[INFO] Xvfb terminated.')
        if _DISCOVERY_PROC is not None and _DISCOVERY_PROC.poll() is None:
            _DISCOVERY_PROC.terminate()
            _DISCOVERY_PROC.wait(timeout=5)
            print('[INFO] Discovery server terminated.')
        if _LAUNCH_PROC is not None and _LAUNCH_PROC.poll() is None:
            os.killpg(os.getpgid(_LAUNCH_PROC.pid), signal.SIGINT)
            try:
                _LAUNCH_PROC.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(_LAUNCH_PROC.pid), signal.SIGKILL)
            print('[INFO] Launch group terminated.')
    print('[INFO] Start command done.')


# ===================== spawn =====================
def cmd_spawn(args):
    target = args.target
    if target == 'red':
        print('[INFO] Spawning RED box at (1.5, 0.0, 0.5)...')
        _ros2(
            'ros2 service call /spawn_item project/srv/SpawnItem '
            '"{target_type: \'box\', x: 1.5, y: 0.0, z: 0.5}"',
            timeout=10
        )
        print('[INFO] Red box spawned.')
    elif target == 'blue':
        print('[INFO] Spawning BLUE barrel at (0.0, 1.5, 0.5)...')
        _ros2(
            'ros2 service call /spawn_item project/srv/SpawnItem '
            '"{target_type: \'barrel\', x: 0.0, y: 1.5, z: 0.5}"',
            timeout=10
        )
        print('[INFO] Blue barrel spawned.')
    elif target == 'clean':
        name = args.name if args.name else 'red_box'
        print(f'[INFO] Deleting entity [{name}]...')
        _ros2(
            f'ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity '
            f'"{{name: \'{name}\'}}"',
            timeout=10
        )
        print(f'[INFO] Entity [{name}] deleted.')
    else:
        print(f'[ERROR] Unknown target: {target}. Use red | blue | clean')


# ===================== nuke =====================
def cmd_nuke(_args):
    print('[INFO] NUKING all simulation processes...')
    global _DISCOVERY_PROC, _XVFB_PROC
    if _XVFB_PROC is not None and _XVFB_PROC.poll() is None:
        _XVFB_PROC.terminate()
        _XVFB_PROC.wait(timeout=5)
        print('[INFO] Xvfb terminated.')
        _XVFB_PROC = None
    if _DISCOVERY_PROC is not None and _DISCOVERY_PROC.poll() is None:
        _DISCOVERY_PROC.terminate()
        _DISCOVERY_PROC.wait(timeout=5)
        print('[INFO] Discovery server terminated.')
        _DISCOVERY_PROC = None
    for cmd in [
        'killall -9 gzserver gzclient 2>/dev/null',
        'pkill -9 -f "ros2 launch"',
        'pkill -9 -f "python3.*_node"',
    ]:
        print(f'[EXEC] {cmd}')
        subprocess.run(cmd, shell=True)
    time.sleep(2)
    print('[INFO] Nuke complete. All zombie processes eliminated.')


# ===================== CLI =====================
def main():
    parser = argparse.ArgumentParser(
        description='Simulation Manager - Unified CLI for ROS 2 lifecycle'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('build', help='colcon build --packages-select project')
    sub.add_parser('start', help='Launch system_bringup + auto init-pose')
    spawn_p = sub.add_parser('spawn', help='Spawn red box / blue barrel / clean entity')
    spawn_p.add_argument('target', choices=['red', 'blue', 'clean'],
                         help='Target type to spawn or delete')
    spawn_p.add_argument('--name', default='red_box',
                         help='Entity name for clean (default: red_box)')
    sub.add_parser('nuke', help='Kill all Gazebo and ROS 2 processes')

    args = parser.parse_args()

    cmds = {
        'build': cmd_build,
        'start': cmd_start,
        'spawn': cmd_spawn,
        'nuke': cmd_nuke,
    }
    try:
        cmds[args.command](args)
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted.')
        if args.command == 'start' and _LAUNCH_PROC:
            os.killpg(os.getpgid(_LAUNCH_PROC.pid), signal.SIGINT)


if __name__ == '__main__':
    main()
