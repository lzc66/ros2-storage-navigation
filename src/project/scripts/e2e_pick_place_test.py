#!/usr/bin/python3
"""
E2E Pick-and-Place Audit: physics-level verification of Campaign 5.

Observes /gazebo/model_states to track the test box trajectory,
validating that the suction gripper lifts, transports, and releases.

Usage:
  ros2 run project e2e_pick_place_test.py

Environment:
  ROS_DOMAIN_ID=30  ROS_DISCOVERY_SERVER=127.0.0.1:11811
"""
import math
import sys
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnEntity
from project.action import FetchTask
from project.srv import SpawnItem

# ================================================================
# Test parameters
# ================================================================
BOX_NAME         = 'e2e_test_box'
SPAWN_X, SPAWN_Y, SPAWN_Z = 1.5, 0.0, 0.5
TARGET_YAW       = 0.0
DROP_X, DROP_Y   = -1.0, -1.0
DROP_YAW         = 0.0
GRASP_RISE_MIN   = 0.08   # box must rise at least this much from initial Z
DELIVERY_EPS     = 0.5    # tolerance radius around drop point
GROUND_Z_MAX     = 0.2    # box must fall below this to prove released
MAX_WALL_TIME    = 120.0  # overall timeout (seconds)


class E2ETestNode(Node):
    def __init__(self):
        super().__init__('e2e_pick_place_test')
        self._cbg = ReentrantCallbackGroup()
        self._lock = threading.Lock()

        # Trajectory log
        self._trajectory = []          # [(x, y, z, t), ...]
        self._grasp_detected = False
        self._grasp_rise = 0.0         # max rise from initial Z
        self._initial_z = None         # first recorded Z
        self._final_pos = None
        self._start_time = time.time()

        # Model states subscription (omniscient audit)
        self._states_sub = self.create_subscription(
            ModelStates, '/model_states', self._states_cb, 10,
            callback_group=self._cbg,
        )

        # Spawn service client
        self._spawn_cli = self.create_client(
            SpawnItem, '/spawn_item', callback_group=self._cbg,
        )

        # Action client for FetchTask
        self._task_cli = ActionClient(
            self, FetchTask, 'fetch_task', callback_group=self._cbg,
        )

        self._phase = 'INIT'
        self.get_logger().info('E2E Audit node ready. Starting test sequence...')

    # ============================================================
    # Model states callback — record trajectory
    # ============================================================
    def _states_cb(self, msg):
        try:
            idx = msg.name.index(BOX_NAME)
        except ValueError:
            return  # not spawned yet

        pose = msg.pose[idx]
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        t = time.time() - self._start_time

        with self._lock:
            self._trajectory.append((x, y, z, t))
            self._final_pos = (x, y, z)

            # Track initial Z (first sighting after spawn)
            if self._initial_z is None:
                self._initial_z = z

            # Detect grasp: Z rises significantly above initial (post-fall) level
            rise = z - (self._initial_z or 0.0)
            if rise > self._grasp_rise:
                self._grasp_rise = rise
            if not self._grasp_detected and rise > GRASP_RISE_MIN:
                self._grasp_detected = True
                self._grasp_time = t

    # ============================================================
    # Phase 1: Spawn test box
    # ============================================================
    def _spawn_box(self):
        if not self._spawn_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/spawn_item not available')
            return False
        req = SpawnItem.Request()
        req.target_type = 'box'
        req.x = SPAWN_X; req.y = SPAWN_Y; req.z = SPAWN_Z
        future = self._spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() and future.result().success:
            self.get_logger().info(
                f'[E2E AUDIT] Phase 1 (Spawn): PASS | '
                f'Box at ({SPAWN_X:.1f}, {SPAWN_Y:.1f}, {SPAWN_Z:.1f})'
            )
            return True
        self.get_logger().error('[E2E AUDIT] Phase 1 (Spawn): FAIL')
        return False

    # ============================================================
    # Phase 2+3: Send FetchTask and monitor
    # ============================================================
    def _send_task(self):
        if not self._task_cli.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('/fetch_task not available')
            return False
        goal = FetchTask.Goal()
        goal.priority = 2
        goal.target_x = SPAWN_X
        goal.target_y = SPAWN_Y
        goal.target_z = SPAWN_Z
        goal.target_yaw = TARGET_YAW
        goal.drop_x = DROP_X
        goal.drop_y = DROP_Y
        goal.drop_yaw = DROP_YAW

        self.get_logger().info(
            f'[E2E AUDIT] Phase 2: Dispatching task → '
            f'pick({SPAWN_X},{SPAWN_Y},{SPAWN_Z}) '
            f'drop({DROP_X},{DROP_Y})'
        )
        self._task_start_time = time.time()
        send_future = self._task_cli.send_goal_async(goal)
        send_future.add_done_callback(self._goal_response_cb)
        return True

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle or not handle.accepted:
            self.get_logger().error('[E2E AUDIT] Task rejected')
            return
        self.get_logger().info('[E2E AUDIT] Task accepted. Monitoring physics...')
        # Fallback: run assertions after MAX_WALL_TIME regardless of task callback
        self._fallback_timer = self.create_timer(
            MAX_WALL_TIME, self._run_assertions, callback_group=self._cbg,
        )
        handle.get_result_async().add_done_callback(self._task_done_cb)

    def _task_done_cb(self, future):
        """Task completed — run final assertions."""
        result = future.result()
        task_ok = result.result.success if result else False
        self.get_logger().info(
            f'[E2E AUDIT] Task reported: success={task_ok} '
            f'msg={result.result.message if result else "N/A"}'
        )
        # Run assertions after a brief settle period
        self._settle_timer = self.create_timer(
            2.0, self._run_assertions, callback_group=self._cbg,
        )

    # ============================================================
    # Assertions
    # ============================================================
    def _run_assertions(self):
        if self._settle_timer:
            self._settle_timer.cancel()

        with self._lock:
            traj = list(self._trajectory)
            final_pos = self._final_pos
            grasp_detected = self._grasp_detected

        all_pass = True

        # Assertion 1: Grasp — box elevated above shelf
        if grasp_detected:
            peak_z = max(p[2] for p in traj) if traj else 0.0
            self.get_logger().info(
                f'[E2E AUDIT] Phase 2 (Grasp): PASS | '
                f'Box rose {self._grasp_rise:.2f}m (peak Z={peak_z:.2f})'
            )
        else:
            peak_z = max(p[2] for p in traj) if traj else 0.0
            self.get_logger().error(
                f'[E2E AUDIT] Phase 2 (Grasp): FAIL | '
                f'Max rise={self._grasp_rise:.3f}m < {GRASP_RISE_MIN}m '
                f'(peak={peak_z:.2f}, initial={self._initial_z})'
            )
            all_pass = False

        # Assertion 2: Delivery — box near drop point
        if final_pos and len(traj) > 2:
            fx, fy, fz = final_pos
            dist = math.sqrt((fx - DROP_X)**2 + (fy - DROP_Y)**2)
            if dist <= DELIVERY_EPS:
                self.get_logger().info(
                    f'[E2E AUDIT] Phase 3 (Drop): PASS | '
                    f'Box resting at ({fx:.2f}, {fy:.2f}, {fz:.2f})'
                )
            else:
                self.get_logger().error(
                    f'[E2E AUDIT] Phase 3 (Drop): FAIL | '
                    f'Box at ({fx:.2f},{fy:.2f},{fz:.2f}) '
                    f'dist={dist:.2f} > {DELIVERY_EPS}'
                )
                all_pass = False
        else:
            self.get_logger().error(
                f'[E2E AUDIT] Phase 3 (Drop): FAIL | '
                f'Trajectory records={len(traj)}, final_pos_set={final_pos is not None}'
            )
            all_pass = False

        # Assertion 3: Released — box on/near ground
        if final_pos:
            _, _, fz = final_pos
            if fz <= GROUND_Z_MAX:
                self.get_logger().info(
                    f'[E2E AUDIT] Phase 3 (Release): PASS | '
                    f'Z={fz:.2f} <= {GROUND_Z_MAX} (grounded)'
                )
            else:
                self.get_logger().error(
                    f'[E2E AUDIT] Phase 3 (Release): FAIL | '
                    f'Z={fz:.2f} > {GROUND_Z_MAX} (still elevated)'
                )
                all_pass = False

        # Final verdict
        verdict = 'SYSTEM FULLY OPERATIONAL' if all_pass else 'SYSTEM HAS DEFECTS'
        self.get_logger().info(
            f'[E2E AUDIT] Final Result: {verdict}'
        )

        # Exit with appropriate code after log flush
        self._exit_timer = self.create_timer(
            1.0, lambda: sys.exit(0 if all_pass else 1),
            callback_group=self._cbg,
        )


def main():
    rclpy.init()
    node = E2ETestNode()

    # Phase 1: spawn
    time.sleep(2)  # brief settle for ROS graph
    if not node._spawn_box():
        node.get_logger().error('[E2E AUDIT] Aborting — spawn failed')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    # Phase 2: send task
    time.sleep(1)
    if not node._send_task():
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    # Spin until assertions run or timeout
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
