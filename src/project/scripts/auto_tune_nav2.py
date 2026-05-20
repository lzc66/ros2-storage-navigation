#!/usr/bin/env python3
"""
Auto-tune Nav2 params: 5 iterations. Tunes max_vel_x [0.20, 0.60], inflation [0.10, 0.20].

Fix v4: map_server YAML origin already aligns world==map coords.
  - INIT_POSE = SPAWN directly (no origin offset)
  - Goal sent in world coords (= map coords)
  - cmd_vel checks linear.x|y AND angular.z
"""
import os, re, sys, time, signal, subprocess, math, random
from dataclasses import dataclass

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(_SCRIPT_DIR)
WS_DIR = os.path.dirname(os.path.dirname(PKG_DIR))
NAV2_PARAMS = os.path.join(PKG_DIR, 'params', 'nav2_params.yaml')

MAX_VEL_X_MIN, MAX_VEL_X_MAX = 0.20, 0.60
INFLATION_MIN, INFLATION_MAX = 0.23, 0.40

MAX_TIME = 90.0
CMD_VEL_TIMEOUT = 12.0
GOAL_TOLERANCE = 0.40
COLLISION_CHECK_INTERVAL = 0.5

SAFE_GOALS = [
    (1.5, -0.5),   # straight east, simple
    (1.0, 0.5),    # slight diagonal
    (1.5, 0.5),    # diagonal through cylinder gap
    (1.5, 1.5),    # diagonal crossing y=0 and y=1.1 cylinder rows
    (0.0, 1.8),    # north through center column gap
    (2.0, 0.0),    # far east past cylinder column
    (-1.5, 1.5),   # northwest diagonal
    (0.5, -1.8),   # southeast
]

SPAWN_X, SPAWN_Y = -2.0, -0.5
INIT_POSE_X = SPAWN_X   # map==world, no offset
INIT_POSE_Y = SPAWN_Y

CYLINDERS = [(x, y) for x in [-1.1, 0.0, 1.1] for y in [-1.1, 0.0, 1.1]]
CYLINDER_RADIUS = 0.15
HEXAGONS = [(3.5, 0.0), (1.8, 2.7), (1.8, -2.7), (-1.8, 2.7), (-1.8, -2.7)]
HEXAGON_RADIUS = 0.45
BOUND_X_MIN, BOUND_X_MAX = -2.3, 2.3
BOUND_Y_MIN, BOUND_Y_MAX = -2.3, 2.3
MIN_OBSTACLE_DIST = 0.5


@dataclass
class IterResult:
    iteration: int
    max_vel_x: float
    inflation_radius: float
    goal_x: float
    goal_y: float
    score: float
    elapsed_time: float
    reached_goal: bool
    plan_failure: bool
    stuck_failure: bool
    final_x: float
    final_y: float


def is_goal_safe(gx, gy):
    if not (BOUND_X_MIN <= gx <= BOUND_X_MAX and BOUND_Y_MIN <= gy <= BOUND_Y_MAX):
        return False
    for cx, cy in CYLINDERS:
        if math.sqrt((gx - cx)**2 + (gy - cy)**2) < CYLINDER_RADIUS + MIN_OBSTACLE_DIST:
            return False
    for hx, hy in HEXAGONS:
        if math.sqrt((gx - hx)**2 + (gy - hy)**2) < HEXAGON_RADIUS + MIN_OBSTACLE_DIST:
            return False
    return True


def pick_goal():
    """Randomly select a verified safe goal. Requires actual obstacle avoidance."""
    valid = [g for g in SAFE_GOALS if is_goal_safe(g[0], g[1])]
    if valid:
        return random.choice(valid)
    return SAFE_GOALS[0]


def modify_params(max_vel_x, inflation_radius):
    with open(NAV2_PARAMS, 'r') as f:
        content = f.read()
    content = re.sub(r'(FollowPath:\n(?:.*\n)*?\s{4,8}desired_linear_vel:\s*)[0-9.]+',
                     f'\\g<1>{max_vel_x}', content, count=1)
    content = re.sub(r'(FollowPath:\n(?:.*\n)*?\s{4,8}max_linear_vel:\s*)[0-9.]+',
                     f'\\g<1>{max_vel_x}', content, count=1)
    content = re.sub(r'(velocity_smoother:\n(?:.*\n)*?\s{4,8}max_velocity:\s*\[)[0-9.]+,',
                     f'\\g<1>{max_vel_x},', content, count=1)
    content = re.sub(r'(local_costmap:\n(?:.*\n)*?\s{6,10}inflation_radius:\s*)[0-9.]+',
                     f'\\g<1>{inflation_radius}', content, count=1)
    content = re.sub(r'(global_costmap:\n(?:.*\n)*?\s{6,10}inflation_radius:\s*)[0-9.]+',
                     f'\\g<1>{inflation_radius}', content, count=1)
    with open(NAV2_PARAMS, 'w') as f:
        f.write(content)
    return True


def kill_all():
    subprocess.run(['killall', '-9', 'gzserver', 'gzclient', 'ruby'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'navigation.launch'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'gzserver'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'gzclient'], capture_output=True)
    time.sleep(3)


def _get_amcl_pose():
    try:
        result = subprocess.run([
            'ros2', 'topic', 'echo', '/amcl_pose', '--once', '--field', 'pose.pose.position'
        ], capture_output=True, text=True, timeout=3, env=os.environ)
        match = re.search(r'x:\s*([\d.-]+)\s*\n\s*y:\s*([\d.-]+)', result.stdout)
        if match:
            return float(match.group(1)), float(match.group(2))
    except Exception:
        pass
    return None, None


def _get_cmd_vel_active():
    """Check if ANY velocity component is non-zero (handles spin recovery)."""
    try:
        result = subprocess.run([
            'ros2', 'topic', 'echo', '/cmd_vel', '--once'
        ], capture_output=True, text=True, timeout=3, env=os.environ)
        # Parse all velocity components
        lx = re.search(r'linear:\s*\n\s*x:\s*(-?[\d.]+)', result.stdout)
        ly = re.search(r'linear:\s*\n\s*x:\s*-?[\d.]+\s*\n\s*y:\s*(-?[\d.]+)', result.stdout)
        az = re.search(r'angular:\s*\n\s*x:\s*-?[\d.]+\s*\n\s*y:\s*-?[\d.]+\s*\n\s*z:\s*(-?[\d.]+)', result.stdout)
        vx = abs(float(lx.group(1))) if lx else 0.0
        vy = abs(float(ly.group(1))) if ly else 0.0
        vz = abs(float(az.group(1))) if az else 0.0
        return vx > 0.001 or vy > 0.001 or vz > 0.001
    except Exception:
        pass
    return False


def _send_initial_pose():
    msg = (
        '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: "map"}, '
        f'pose: {{pose: {{position: {{x: {INIT_POSE_X}, y: {INIT_POSE_Y}, z: 0.0}}, '
        'orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, '
        'covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, '
        '0.0, 0.25, 0.0, 0.0, 0.0, 0.0, '
        '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '
        '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '
        '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '
        '0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891909122467]}}'
    )
    try:
        subprocess.run(['ros2', 'topic', 'pub', '--once', '/initialpose',
                        'geometry_msgs/msg/PoseWithCovarianceStamped', msg],
                       timeout=5, capture_output=True, env=os.environ)
        print(f"[INIT] Initial pose at ({INIT_POSE_X}, {INIT_POSE_Y})")
        return True
    except Exception as e:
        print(f"[INIT] Initial pose failed: {e}")
        return False


def _verify_amcl(timeout=10.0):
    print("[AMCL] Verifying convergence...")
    start = time.time()
    while time.time() - start < timeout:
        pose = _get_amcl_pose()
        if pose[0] is not None:
            dist = math.sqrt((pose[0] - INIT_POSE_X)**2 + (pose[1] - INIT_POSE_Y)**2)
            if dist < 2.0:
                print(f"[AMCL] Converged: ({pose[0]:.2f}, {pose[1]:.2f}) offset={dist:.2f}m")
                return True
        time.sleep(1.0)
    print("[AMCL] WARNING: Not converged")
    return False


def _send_nav_goal(gx, gy, timeout_sec):
    """Send goal at (gx,gy) and monitor arrival via amcl_pose."""
    result = {
        'reached': False, 'elapsed': timeout_sec,
        'final_x': 0.0, 'final_y': 0.0,
        'plan_failure': False, 'stuck_failure': False
    }
    start_time = time.time()

    goal_cmd = [
        'ros2', 'action', 'send_goal', '/navigate_to_pose',
        'nav2_msgs/action/NavigateToPose',
        f'{{pose: {{header: {{frame_id: "map"}}, pose: {{position: {{x: {gx}, y: {gy}, z: 0.0}}, '
        f'orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}}}}}'
    ]
    proc = subprocess.Popen(goal_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, env=os.environ)

    cmd_vel_dead_time = 0.0
    initial_amcl = None
    first_cmd_vel = True

    while time.time() - start_time < timeout_sec:
        # cmd_vel: checks linear.x|y AND angular.z
        has_cmd_vel = _get_cmd_vel_active()
        if has_cmd_vel:
            cmd_vel_dead_time = 0.0
            if first_cmd_vel:
                print("  [OK] cmd_vel active")
                first_cmd_vel = False
        else:
            cmd_vel_dead_time += COLLISION_CHECK_INTERVAL
            if cmd_vel_dead_time > CMD_VEL_TIMEOUT and time.time() - start_time > 15.0:
                print(f"  [FAIL] No cmd_vel for {cmd_vel_dead_time:.0f}s")
                result['plan_failure'] = True
                break

        # amcl_pose: arrival + stuck detection
        amcl = _get_amcl_pose()
        if amcl[0] is not None:
            result['final_x'] = amcl[0]
            result['final_y'] = amcl[1]

            if initial_amcl is None:
                initial_amcl = amcl

            dist = math.sqrt((amcl[0] - gx)**2 + (amcl[1] - gy)**2)
            if dist < GOAL_TOLERANCE:
                result['reached'] = True
                result['elapsed'] = time.time() - start_time
                print(f"  [ARRIVED] ({amcl[0]:.2f}, {amcl[1]:.2f}) dist={dist:.3f}m")
                break

            total = math.sqrt((amcl[0] - initial_amcl[0])**2 + (amcl[1] - initial_amcl[1])**2)
            if time.time() - start_time > 30.0 and total < 0.3 and has_cmd_vel:
                print(f"  [STUCK] moved {total:.2f}m in {time.time()-start_time:.0f}s")
                result['stuck_failure'] = True
                break

        if proc.poll() is not None:
            break
        time.sleep(COLLISION_CHECK_INTERVAL)

    if not (result['reached'] or result['plan_failure'] or result['stuck_failure']):
        result['elapsed'] = timeout_sec

    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=2)
        except subprocess.TimeoutExpired: proc.kill()
    return result


def run_iteration(iter_num, max_vel_x, inflation_radius) -> IterResult:
    modify_params(max_vel_x, inflation_radius)
    gx, gy = pick_goal()

    print(f"\n{'='*60}")
    print(f"[Iter {iter_num}] v={max_vel_x:.2f} infl={inflation_radius:.2f}  goal=({gx:.1f}, {gy:.1f})")
    print(f"{'='*60}")

    print("[BUILD] Building...")
    subprocess.run(['colcon', 'build', '--packages-select', 'project', '--symlink-install'],
                   cwd=WS_DIR, capture_output=True, text=True, timeout=120)

    print("[LAUNCH] Starting simulation...")
    env = os.environ.copy()
    env['GAZEBO_MODEL_PATH'] = (
        f"{PKG_DIR}/models:{os.path.expanduser('~/.gazebo/models')}:"
        "/opt/ros/humble/share/turtlebot3_gazebo/models"
    )
    proc = subprocess.Popen(
        ['ros2', 'launch', 'project', 'navigation.launch.py'],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )

    print("[INIT] Waiting 25s...")
    time.sleep(25)
    _send_initial_pose()
    time.sleep(2)
    _verify_amcl(timeout=8.0)

    print(f"[GOAL] Navigating to ({gx:.1f}, {gy:.1f})...")
    result = _send_nav_goal(gx, gy, MAX_TIME)

    reached = result['reached']
    plan_failure = result['plan_failure']
    stuck_failure = result['stuck_failure']
    elapsed = result['elapsed']
    fx, fy = result['final_x'], result['final_y']

    if stuck_failure:
        score = 0.0; tag = "STUCK"
    elif plan_failure:
        score = 0.0; tag = "PLAN FAIL"
    elif reached:
        score = (MAX_TIME - elapsed) * 10.0; tag = f"REACHED {elapsed:.1f}s"
    else:
        score = 10.0
        dist = math.sqrt((fx - gx)**2 + (fy - gy)**2)
        tag = f"TIMEOUT dist={dist:.2f}m"

    print(f"[RESULT] {tag}  Score: {score:.1f}")

    print("[CLEANUP] Killing...")
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait(timeout=5)
    kill_all()

    return IterResult(iteration=iter_num, max_vel_x=max_vel_x, inflation_radius=inflation_radius,
                      goal_x=gx, goal_y=gy,
                      score=score, elapsed_time=elapsed, reached_goal=reached,
                      plan_failure=plan_failure, stuck_failure=stuck_failure,
                      final_x=fx, final_y=fy)


def heuristic_next_params(iter_num, all_results):
    if iter_num == 1:
        return 0.30, 0.25
    best = max(all_results, key=lambda r: r.score)
    last = all_results[-1]
    candidates = []
    if last.stuck_failure or last.plan_failure:
        candidates += [(max(0.20, last.max_vel_x-0.06), min(INFLATION_MAX, last.inflation_radius+0.03)),
                       (max(0.20, last.max_vel_x-0.03), min(INFLATION_MAX, last.inflation_radius+0.02))]
    elif last.reached_goal:
        candidates += [(min(MAX_VEL_X_MAX, last.max_vel_x+0.08), last.inflation_radius),
                       (min(MAX_VEL_X_MAX, last.max_vel_x+0.05), max(INFLATION_MIN, last.inflation_radius-0.03))]
    else:
        candidates += [(min(MAX_VEL_X_MAX, last.max_vel_x+0.06), last.inflation_radius),
                       (last.max_vel_x, max(INFLATION_MIN, last.inflation_radius-0.03)),
                       (min(MAX_VEL_X_MAX, last.max_vel_x+0.04), max(INFLATION_MIN, last.inflation_radius-0.02)),
                       (last.max_vel_x, min(INFLATION_MAX, last.inflation_radius+0.03))]
    candidates.append((best.max_vel_x+0.03 if best.max_vel_x<0.50 else best.max_vel_x-0.03,
                       best.inflation_radius-0.02 if best.inflation_radius>0.12 else best.inflation_radius+0.02))
    valid = [(v,r) for v,r in candidates if MAX_VEL_X_MIN<=v<=MAX_VEL_X_MAX and INFLATION_MIN<=r<=INFLATION_MAX]
    used = {(r.max_vel_x, r.inflation_radius) for r in all_results}
    for v,r in valid:
        if (v,r) not in used: return round(v,2), round(r,2)
    while True:
        v=round(random.uniform(MAX_VEL_X_MIN, MAX_VEL_X_MAX),2)
        r=round(random.uniform(INFLATION_MIN, INFLATION_MAX),2)
        if (v,r) not in used: return v,r


def git_commit(iter_num, max_vel_x, radius, score):
    try:
        subprocess.run(['git','add','-f',NAV2_PARAMS], cwd=PKG_DIR, capture_output=True)
        msg = f"[Auto-Tune] Iter {iter_num}: max_vel={max_vel_x:.2f}, radius={radius:.2f} | Score: {score:.1f}"
        subprocess.run(['git','commit','-m',msg], cwd=PKG_DIR, capture_output=True)
        print(f"[GIT] {msg}")
    except Exception as e:
        print(f"[GIT] Fail: {e}")


def main():
    print("="*60)
    print("Nav2 Auto-Tuning [v4: map==world coords]")
    print(f"  Spawn=({SPAWN_X},{SPAWN_Y})  Goals={SAFE_GOALS}")
    print(f"  infl=[{INFLATION_MIN},{INFLATION_MAX}]  tol={GOAL_TOLERANCE}m")
    print("="*60)
    subprocess.run([sys.executable, os.path.join(PKG_DIR,'scripts','generate_map.py')],
                   cwd=PKG_DIR, capture_output=True)
    kill_all()
    all_results = []
    for iteration in range(1,6):
        v, r = heuristic_next_params(iteration, all_results)
        result = run_iteration(iteration, v, r)
        all_results.append(result)
        git_commit(iteration, v, r, result.score)
        print(f"  => v={v:.2f} r={r:.2f} score={result.score:.1f} "
              f"reached={result.reached_goal} pf={result.plan_failure} st={result.stuck_failure}")
    best = max(all_results, key=lambda r: r.score)
    modify_params(best.max_vel_x, best.inflation_radius)
    print(f"\nBEST: v={best.max_vel_x:.2f} r={best.inflation_radius:.2f} score={best.score:.1f}")
    print(f"\n{'Iter':<6} {'v':<7} {'r':<7} {'Goal':<14} {'Score':<9} {'Time':<7} {'Reached':<8} {'Stuck':<7}")
    print("-"*72)
    for r in all_results:
        goal_str = f"({r.goal_x:.1f},{r.goal_y:.1f})"
        print(f"{r.iteration:<6} {r.max_vel_x:<7.2f} {r.inflation_radius:<7.2f} "
              f"{goal_str:<14} {r.score:<9.1f} {r.elapsed_time:<7.1f} "
              f"{str(r.reached_goal):<8} {str(r.stuck_failure):<7}")


if __name__ == '__main__':
    main()
