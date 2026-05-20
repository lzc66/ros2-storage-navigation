#!/usr/bin/python3
"""
E2E Pick-and-Place Audit: physics-level verification via GetEntityState service.

Polls /gazebo/get_entity_state at 2 Hz to track the test box trajectory,
avoiding DDS topic discovery issues with /model_states.

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

from gazebo_msgs.srv import GetEntityState
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
DELIVERY_EPS     = 0.8    # tolerance radius around drop point
GROUND_Z_MAX     = 0.2    # box must fall below this to prove released
MAX_WALL_TIME    = 120.0  # overall timeout (seconds)
POLL_PERIOD      = 0.5    # 2 Hz entity state polling


class E2ETestNode(Node):
    def __init__(self):
        super().__init__('e2e_pick_place_test')
        self._cbg = ReentrantCallbackGroup()
        self._lock = threading.Lock()

        # Trajectory from service polling
        self._trajectory = []          # [(x, y, z, t), ...]
        self._grasp_detected = False
        self._grasp_rise = 0.0
        self._initial_z = None
        self._final_pos = None
        self._start_time = time.time()
        self._task_done_time = None     # when brain_node reported completion

        # GetEntityState service client (Gazebo)
        self._entity_cli = self.create_client(
            GetEntityState, '/gazebo/get_entity_state', callback_group=self._cbg,
        )

        # Spawn service client
        self._spawn_cli = self.create_client(
            SpawnItem, '/spawn_item', callback_group=self._cbg,
        )

        # Action client for FetchTask
        self._task_cli = ActionClient(
            self, FetchTask, 'fetch_task', callback_group=self._cbg,
        )

        # Entity state polling timer (starts after spawn)
        self._poll_timer = None

        self.get_logger().info('E2E Audit node ready. Starting test sequence...')

    # ============================================================
    # Entity state polling — replaces /model_states subscription
    # ============================================================
    def _start_polling(self):
        """Begin 2 Hz polling of /gazebo/get_entity_state for the test box."""
        if not self._entity_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/gazebo/get_entity_state not available')
            return
        self.get_logger().info('Entity state polling started (2 Hz)')
        self._poll_timer = self.create_timer(
            POLL_PERIOD, self._poll_entity, callback_group=self._cbg,
        )

    def _poll_entity(self):
        """Query Gazebo for the test box pose."""
        req = GetEntityState.Request()
        req.name = BOX_NAME
        req.reference_frame = 'world'
        future = self._entity_cli.call_async(req)
        future.add_done_callback(self._entity_response_cb)

    def _entity_response_cb(self, future):
        try:
            resp = future.result()
            if not resp.success:
                return  # box not spawned yet or removed
            pose = resp.state.pose
            x, y, z = pose.position.x, pose.position.y, pose.position.z
        except Exception:
            return

        t = time.time() - self._start_time

        with self._lock:
            self._trajectory.append((x, y, z, t))
            self._final_pos = (x, y, z)

            if self._initial_z is None:
                self._initial_z = z

            rise = z - (self._initial_z or 0.0)
            if rise > self._grasp_rise:
                self._grasp_rise = rise
            if not self._grasp_detected and rise > GRASP_RISE_MIN:
                self._grasp_detected = True

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
            f'[E2E AUDIT] Phase 2: Dispatching task -> '
            f'pick({SPAWN_X},{SPAWN_Y},{SPAWN_Z}) '
            f'drop({DROP_X},{DROP_Y})'
        )
        send_future = self._task_cli.send_goal_async(goal)
        send_future.add_done_callback(self._goal_response_cb)
        return True

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle or not handle.accepted:
            self.get_logger().error('[E2E AUDIT] Task rejected')
            return
        self.get_logger().info('[E2E AUDIT] Task accepted. Polling entity state...')
        # Fallback: run assertions after MAX_WALL_TIME
        self._fallback_timer = self.create_timer(
            MAX_WALL_TIME, self._run_assertions, callback_group=self._cbg,
        )
        handle.get_result_async().add_done_callback(self._task_done_cb)

    def _task_done_cb(self, future):
        """Task reported by brain_node — run assertions after settle."""
        self._task_done_time = time.time()
        result = future.result()
        task_ok = result.result.success if result else False
        self.get_logger().info(
            f'[E2E AUDIT] Task reported: success={task_ok} '
            f'msg={result.result.message if result else "N/A"}'
        )
        self._settle_timer = self.create_timer(
            2.0, self._run_assertions, callback_group=self._cbg,
        )

    # ============================================================
    # Assertions
    # ============================================================
    def _run_assertions(self):
        if hasattr(self, '_settle_timer') and self._settle_timer:
            self._settle_timer.cancel()
        if hasattr(self, '_fallback_timer') and self._fallback_timer:
            self._fallback_timer.cancel()
        if self._poll_timer:
            self._poll_timer.cancel()

        with self._lock:
            traj = list(self._trajectory)
            final_pos = self._final_pos
            grasp_detected = self._grasp_detected
            grasp_rise = self._grasp_rise
            initial_z = self._initial_z

        all_pass = True

        # Assertion 1: Grasp
        if grasp_detected:
            peak_z = max(p[2] for p in traj) if traj else 0.0
            self.get_logger().info(
                f'[E2E AUDIT] Phase 2 (Grasp): PASS | '
                f'Box rose {grasp_rise:.2f}m (peak Z={peak_z:.2f})'
            )
        else:
            peak_z = max(p[2] for p in traj) if traj else 0.0
            self.get_logger().error(
                f'[E2E AUDIT] Phase 2 (Grasp): FAIL | '
                f'Max rise={grasp_rise:.3f}m < {GRASP_RISE_MIN}m '
                f'(records={len(traj)}, peak={peak_z:.2f}, initial={initial_z})'
            )
            all_pass = False

        # Assertion 2: Delivery
        if final_pos and len(traj) > 2:
            fx, fy, fz = final_pos
            dist = math.sqrt((fx - DROP_X)**2 + (fy - DROP_Y)**2)
            if dist <= DELIVERY_EPS:
                self.get_logger().info(
                    f'[E2E AUDIT] Phase 3 (Drop): PASS | '
                    f'Box at ({fx:.2f}, {fy:.2f}, {fz:.2f}) dist={dist:.2f}'
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
                f'Records={len(traj)}, final_pos_set={final_pos is not None}'
            )
            all_pass = False

        # Assertion 3: Release
        if final_pos:
            _, _, fz = final_pos
            if fz <= GROUND_Z_MAX:
                self.get_logger().info(
                    f'[E2E AUDIT] Phase 3 (Release): PASS | '
                    f'Z={fz:.2f} <= {GROUND_Z_MAX}'
                )
            else:
                self.get_logger().error(
                    f'[E2E AUDIT] Phase 3 (Release): FAIL | '
                    f'Z={fz:.2f} > {GROUND_Z_MAX} (still elevated)'
                )
                all_pass = False

        verdict = 'SYSTEM FULLY OPERATIONAL' if all_pass else 'SYSTEM HAS DEFECTS'
        self.get_logger().info(f'[E2E AUDIT] Final Result: {verdict}')
        self._exit_timer = self.create_timer(
            1.0, lambda: sys.exit(0 if all_pass else 1),
            callback_group=self._cbg,
        )


def main():
    rclpy.init()
    node = E2ETestNode()

    # Phase 1: spawn + start polling
    time.sleep(2)
    if not node._spawn_box():
        node.get_logger().error('[E2E AUDIT] Aborting — spawn failed')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    node._start_polling()

    # Phase 2: send task
    time.sleep(1)
    if not node._send_task():
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

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
