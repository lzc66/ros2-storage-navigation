#!/usr/bin/python3
"""
Campaign 5: Vacuum Gripper + Full Pick-and-Place Pipeline.

Architecture:
  - Action Server: FetchTask (priority, x/y/z/yaw, drop_x/y/yaw)
  - PriorityQueue with safe Nav2 preemption
  - Nav2 → Lift → Active Search → Approach → Grip → Retreat → Drop-off
  - Async gripper service client (/gripper/switch)
"""
import math
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Twist
from std_msgs.msg import Float64MultiArray, Header
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from project.action import FetchTask

# ================================================================
# Constants
# ================================================================
VISION_CONFIRM_EPS   = 0.3     # m — vision match threshold
LIFT_WAIT            = 3.0     # s — settle time after lift move
POLL_PERIOD          = 0.5     # s — queue poll interval

# Campaign 4: Active search
SEARCH_TIMEOUT       = 10.0    # s — max active search duration
SEARCH_ANGULAR_SPEED = 0.15   # rad/s — safe slow rotation (≤0.2)
SEARCH_POLL_PERIOD   = 0.1     # s — search loop frequency
HIGH_LIFT_Z          = 0.4     # m — above this, enforce speed limit

# Campaign 5+6: Standoff kinematics
STANDOFF_DIST        = 0.5     # m — observe target from this distance
APPROACH_SPEED       = 0.1     # m/s — forward creep to cross standoff gap
APPROACH_DURATION    = 5.5     # s — enough to cover STANDOFF_DIST + margin
RETREAT_SPEED        = -0.1    # m/s — pull back after gripping
RETREAT_DURATION     = 1.0     # s — retreat duration
DROP_LIFT_Z          = 0.05    # m — near-ground release height (anti-bounce)
BACKAWAY_SPEED       = -0.15   # m/s — exit drop zone after release
BACKAWAY_DURATION    = 1.5     # s — back away duration
DROPOFF_WAIT         = 2.0     # s — settle time after lift lowers for drop

# Drop zones (world coords)


def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def yaw_to_quat(yaw):
    """Convert yaw angle (rad) to quaternion (z, w only)."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def standoff_point(tx, ty, tyaw, dist=STANDOFF_DIST):
    """Compute an observation point `dist` metres behind the target,
    along the reverse of target_yaw, so the camera faces the target."""
    ox = tx - dist * math.cos(tyaw)
    oy = ty - dist * math.sin(tyaw)
    return ox, oy


def _make_lift_cmd(z_target):
    """Build a JointTrajectory message for the prismatic lift joint."""
    traj = JointTrajectory()
    traj.joint_names = ['lift_joint']
    point = JointTrajectoryPoint()
    point.positions = [float(z_target)]
    point.time_from_start.sec = 0
    point.time_from_start.nanosec = 500_000_000  # 0.5s
    traj.points = [point]
    return traj


# ================================================================
# Brain Node
# ================================================================
class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')

        # Reentrant callback group for concurrent action handling
        self._cbg = ReentrantCallbackGroup()

        # -- Nav2 action client --
        self._nav_client = ActionClient(
            self, NavigateToPose, '/navigate_to_pose',
            callback_group=self._cbg,
        )
        self._goal_handle = None
        self._nav_busy = False

        # -- Priority task queue: (priority, seq, goal_dict) --
        self._task_queue = queue.PriorityQueue()
        self._seq = 0

        # -- Active task state --
        self._active_task = None   # dict with priority, x, y, z, handle
        self._preempting = False

        # -- Robot state --
        self._rx = 0.0; self._ry = 0.0; self._ryaw = 0.0

        # -- Vision state --
        self._vision_x = 0.0; self._vision_y = 0.0; self._vision_z = 0.0
        self._vision_seen = False
        self._last_vision_t = 0.0

        # -- Subscriptions --
        self.create_subscription(
            PointStamped, '/target_object', self._vision_cb, 10,
            callback_group=self._cbg,
        )
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_cb, 10,
            callback_group=self._cbg,
        )

        # -- Publishers --
        self._lift_pub = self.create_publisher(
            JointTrajectory, '/lift_controller/commands', 10,
        )
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # -- Gripper service client (Campaign 5) --
        self._gripper_cli = self.create_client(
            SetBool, '/gripper/switch', callback_group=self._cbg,
        )

        # -- Action Server (Campaign 3 entry point) --
        self._action_server = ActionServer(
            self, FetchTask, 'fetch_task',
            execute_callback=self._execute_fetch,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cbg,
        )

        # -- Queue processor timer --
        self._poll_timer = self.create_timer(
            POLL_PERIOD, self._poll_queue, callback_group=self._cbg,
        )

        self.get_logger().info(
            'Brain Campaign 4 ready. /fetch_task (4D: x,y,z,yaw) + Active Search'
        )

    # ============================================================
    # Callbacks
    # ============================================================
    def _amcl_cb(self, msg):
        self._rx = msg.pose.pose.position.x
        self._ry = msg.pose.pose.position.y
        self._ryaw = quat_to_yaw(msg.pose.pose.orientation)

    def _vision_cb(self, msg):
        self._vision_x = msg.point.x
        self._vision_y = msg.point.y
        self._vision_z = msg.point.z   # type code (1.0=red, 2.0=blue)
        self._vision_seen = True
        self._last_vision_t = time.time()

    # ============================================================
    # Action Server: goal / cancel callbacks
    # ============================================================
    def _goal_callback(self, goal_request):
        self.get_logger().info(
            f'[GOAL] Received task P{goal_request.priority} '
            f'→ ({goal_request.target_x:.1f},{goal_request.target_y:.1f},'
            f'{goal_request.target_z:.1f})'
        )
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().info('[CANCEL] Task cancellation requested')
        return CancelResponse.ACCEPT

    async def _execute_fetch(self, goal_handle: ServerGoalHandle):
        """Enqueue task, wait for completion (async)."""
        request = goal_handle.request
        self._seq += 1

        task = {
            'priority': request.priority,
            'x': request.target_x,
            'y': request.target_y,
            'z': request.target_z,
            'yaw': request.target_yaw,
            'drop_x': request.drop_x,
            'drop_y': request.drop_y,
            'drop_yaw': request.drop_yaw,
            'seq': self._seq,
            'handle': goal_handle,
        }
        self._task_queue.put((request.priority, self._seq, task))
        self.get_logger().info(
            f'[ENQUEUE] P{request.priority} seq={self._seq} '
            f'→ ({request.target_x:.1f},{request.target_y:.1f},'
            f'{request.target_z:.1f})  queue_size={self._task_queue.qsize()}'
        )

        # Polling-based wait: check _task_done flag set by _succeed_task / _fail_task
        task['_task_done'] = False
        task['_result_ok'] = False
        task['_result_msg'] = ''
        result = FetchTask.Result()
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                result.success = False
                result.message = 'Cancelled by client'
                goal_handle.canceled()
                return result
            # Non-blocking poll: _succeed_task/_fail_task flip _task_done to True
            if task.get('_task_done'):
                break
            time.sleep(0.1)  # MultiThreadedExecutor — safe to sleep here
        result.success = task.get('_result_ok', False)
        result.message = task.get('_result_msg', 'Unknown')
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    # ============================================================
    # Queue processor (timer callback)
    # ============================================================
    def _poll_queue(self):
        if self._nav_busy:
            return
        if self._preempting:
            # Waiting for cancel confirmation — do not dispatch
            return

        if self._task_queue.empty():
            # No tasks queued; clear active task
            if self._active_task is None:
                return

        # Peek at highest-priority task
        if self._task_queue.empty():
            return
        prio, seq, task = self._task_queue.get()

        # If we have an active task and the new one has higher priority
        if self._active_task is not None:
            active_prio = self._active_task.get('priority', 99)
            if prio < active_prio:
                self.get_logger().info(
                    f'[PREEMPT] Higher priority task P{prio} arrived. '
                    f'Canceling current P{active_prio}...'
                )
                self._preempt_current(active_prio, prio, seq, task)
                return
            else:
                # Re-enqueue
                self._task_queue.put((prio, seq, task))
                return

        self._execute_task(prio, seq, task)

    def _preempt_current(self, old_prio, new_prio, new_seq, new_task):
        """Initiate async cancel of active Nav2 goal.  Will NOT dispatch
        the new task until the cancel is confirmed by Nav2."""
        self._preempting = True

        # Re-queue interrupted task
        if self._active_task:
            self._active_task['restored'] = True
            self._task_queue.put(
                (old_prio, self._active_task.get('seq', 0), self._active_task)
            )
        self._active_task = None

        # Store the preempting task for deferred dispatch
        self._pending_preempt = (new_prio, new_seq, new_task)

        if self._goal_handle is not None:
            self.get_logger().info(
                '[PREEMPT] Sent cancel request to Nav2, waiting for confirmation...'
            )
            cancel_future = self._goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self._cancel_done_cb)
        else:
            # No active goal — dispatch immediately
            self.get_logger().info(
                '[PREEMPT] No active Nav2 goal — dispatching immediately'
            )
            self._flush_preempt()

    def _cancel_done_cb(self, future):
        """Called when Nav2 has processed the cancel request."""
        try:
            result = future.result()
            self.get_logger().info(
                f'[PREEMPT] Nav2 confirmed cancellation. '
                f'Releasing lock and executing P{self._pending_preempt[0]} task.'
            )
        except Exception as e:
            self.get_logger().warn(
                f'[PREEMPT] Cancel future resolved with error: {e}. '
                'Proceeding anyway.'
            )

        # The old goal's _nav_result callback may also fire (status=CANCELED).
        # That handler sets _goal_handle=None — safe to call after cancel_done.
        self._flush_preempt()

    def _flush_preempt(self):
        """Release preemption lock and dispatch the pending high-priority task."""
        if self._pending_preempt is None:
            self.get_logger().error(
                '[PREEMPT] _flush_preempt called but no pending task!'
            )
            self._preempting = False
            return

        new_prio, new_seq, new_task = self._pending_preempt
        self._pending_preempt = None
        self._nav_busy = False
        self._preempting = False
        self._goal_handle = None

        # Dispatch the high-priority task directly
        self._execute_task(new_prio, new_seq, new_task)

    # ============================================================
    # Task execution pipeline
    # ============================================================
    def _execute_task(self, prio, seq, task):
        """Pipeline: STANDOFF_NAV → LIFT → SEARCH → APPROACH → GRIP → RETREAT → DROP."""
        self._active_task = task
        handle = task.get('handle')
        restored = task.get('restored', False)

        # Campaign 6: compute standoff observation point
        tyaw = task.get('yaw', 0.0)
        tx, ty = task['x'], task['y']
        ox, oy = standoff_point(tx, ty, tyaw)

        tag = 'RESTORED' if restored else 'NEW'
        self.get_logger().info(
            f'[EXEC {tag}] P{prio} seq={seq} '
            f'pick=({tx:.1f},{ty:.1f},{task["z"]:.1f}) yaw={tyaw:.2f} '
            f'standoff=({ox:.1f},{oy:.1f})'
        )

        if handle:
            feedback = FetchTask.Feedback()
            feedback.current_state = 'NAVIGATING'
            handle.publish_feedback(feedback)

        self._nav_busy = True
        self._send_nav_goal(
            ox, oy, tyaw,
            on_done=lambda ok: self._on_nav_done(ok, prio, seq, task),
        )

    # ============================================================
    # Nav2: send goal + result handler
    # ============================================================
    def _send_nav_goal(self, gx, gy, gyaw, on_done=None):
        self._nav_client.wait_for_server()
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = gx
        goal.pose.pose.position.y = gy
        _, _, qz, qw = yaw_to_quat(gyaw)
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        send_future = self._nav_client.send_goal_async(goal)
        send_future.add_done_callback(
            lambda f: self._nav_goal_response(f, on_done)
        )

    def _nav_goal_response(self, future, on_done):
        self._goal_handle = future.result()
        if self._goal_handle is None:
            self.get_logger().error(
                '[NAV] Goal rejected by server! State machine broken.'
            )
            self._nav_busy = False
            self._active_task = None
            # Do not call on_done — it would chain into lift/vision for a dead goal
            return
        self._goal_handle.get_result_async().add_done_callback(
            lambda f: self._nav_result(f, on_done)
        )

    def _nav_result(self, future, on_done):
        result = future.result()
        status = result.status if result else -1
        self._goal_handle = None

        if status == 4:  # SUCCEEDED
            self.get_logger().info('[NAV] Goal reached')
            if on_done:
                on_done(True)
        elif status == 6:  # CANCELED
            self.get_logger().error('[NAV] Goal canceled — aborting task pipeline')
            self._nav_busy = False
            self._active_task = None
            if on_done and not self._preempting:
                on_done(False)
        else:  # ABORTED (5), UNKNOWN (0), or other failures
            self.get_logger().error(f'[NAV] Failed (status={status}) — aborting pipeline')
            self._nav_busy = False
            self._active_task = None
            if on_done and not self._preempting:
                on_done(False)

    def _on_nav_done(self, ok, prio, seq, task):
        """Navigation complete → lift to target Z."""
        if self._preempting:
            return  # preemption in flight — ignore old task completion
        if not ok:
            self._nav_busy = False
            self._active_task = None
            self._fail_task(task, 'Nav2 failed')
            return

        handle = task.get('handle')
        if handle:
            fb = FetchTask.Feedback()
            fb.current_state = 'LIFTING'
            handle.publish_feedback(fb)

        # Move lift to target Z (JointTrajectory for Classic plugin)
        z_target = task['z']
        self.get_logger().info(f'[LIFT] Moving to Z={z_target:.2f}m')
        self._lift_pub.publish(_make_lift_cmd(z_target))

        # Wait for lift to settle, then confirm vision
        self._lift_timer = self.create_timer(
            LIFT_WAIT,
            lambda: self._on_lift_done(prio, seq, task),
            callback_group=self._cbg,
        )

    def _on_lift_done(self, prio, seq, task):
        """Lift settled → vision confirmation."""
        if self._lift_timer:
            self._lift_timer.cancel()
            self._lift_timer = None

        handle = task.get('handle')
        if handle:
            fb = FetchTask.Feedback()
            fb.current_state = 'CONFIRMING'
            handle.publish_feedback(fb)

        # Active search loop (faster poll for responsive rotation control)
        self._vision_start_time = time.time()
        self._vision_task = task
        self._vision_prio = prio
        self._vision_seq = seq
        self._vision_timer = self.create_timer(
            SEARCH_POLL_PERIOD, self._vision_poll, callback_group=self._cbg,
        )

    def _vision_poll(self):
        """Active search: check vision, if miss → slow rotate to scan FOV."""
        elapsed = time.time() - self._vision_start_time
        task = self._vision_task
        tx, ty = task['x'], task['y']

        # If we see the target within error threshold → success
        if self._vision_seen:
            err = math.sqrt(
                (self._vision_x - tx) ** 2 + (self._vision_y - ty) ** 2
            )
            if err < VISION_CONFIRM_EPS:
                self.get_logger().info(f'[CONFIRM] Target acquired! err={err:.2f}m')
                self._vision_timer.cancel()
                self._stop_rotation()
                # Campaign 5: approach → grip → retreat → drop-off
                self._start_approach(task)
                return

        # If timeout → fail
        if elapsed > SEARCH_TIMEOUT:
            self.get_logger().warn(f'[SEARCH] Timeout ({SEARCH_TIMEOUT:.1f}s)')
            self._vision_timer.cancel()
            self._stop_rotation()
            self._fail_task(task, 'Active search timeout — target not found')
            return

        # Active search: publish slow rotation so camera scans the FOV
        twist = Twist()
        twist.angular.z = SEARCH_ANGULAR_SPEED
        self._cmd_pub.publish(twist)
        if not hasattr(self, '_search_log_n'):
            self._search_log_n = 0
        self._search_log_n += 1
        if self._search_log_n % 20 == 0:  # every 2s
            self.get_logger().info(
                f'[SEARCH] Scanning... elapsed={elapsed:.1f}s '
                f'angular={SEARCH_ANGULAR_SPEED}rad/s'
            )

    def _stop_rotation(self):
        """Send zero Twist to halt search rotation."""
        stop = Twist()
        stop.angular.z = 0.0
        stop.linear.x = 0.0
        self._cmd_pub.publish(stop)
        self.get_logger().info('[SEARCH] Rotation stopped.')

    # ============================================================
    # Task completion
    # ============================================================
    def _succeed_task(self, task):
        self._nav_busy = False
        self._active_task = None
        task['_result_ok'] = True
        task['_result_msg'] = 'Pick-and-place complete'
        task['_task_done'] = True  # unblock _execute_fetch polling loop
        handle = task.get('handle')
        if handle and handle.is_active:
            fb = FetchTask.Feedback()
            fb.current_state = 'DONE'
            handle.publish_feedback(fb)
        self.get_logger().info('[TASK] SUCCESS — full pick-and-place complete')

    def _fail_task(self, task, reason):
        self._nav_busy = False
        self._active_task = None
        task['_result_ok'] = False
        task['_result_msg'] = reason
        task['_task_done'] = True  # unblock _execute_fetch polling loop
        self.get_logger().error(f'[TASK] FAILED: {reason}')

    # ============================================================
    # Campaign 5: Grasping pipeline
    # ============================================================
    def _start_approach(self, task):
        """Creep forward to make suction cup contact with target."""
        self.get_logger().info('[APPROACH] Forward creep to contact target...')
        twist = Twist()
        twist.linear.x = APPROACH_SPEED
        self._cmd_pub.publish(twist)
        self._approach_timer = self.create_timer(
            APPROACH_DURATION,
            lambda: self._on_approach_done(task),
            callback_group=self._cbg,
        )

    def _on_approach_done(self, task):
        """Approach complete — brake and activate suction."""
        self._approach_timer.cancel()
        self._stop_rotation()  # zero Twist
        self._start_grip(task)

    def _start_grip(self, task):
        """Activate vacuum gripper via /gripper/switch."""
        self.get_logger().info('[GRIP] Activating suction...')
        if not self._gripper_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('[GRIP] /gripper/switch unavailable')
            self._fail_task(task, 'Gripper service not available')
            return
        req = SetBool.Request()
        req.data = True
        future = self._gripper_cli.call_async(req)
        future.add_done_callback(lambda f: self._on_grip_done(f, task))

    def _on_grip_done(self, future, task):
        """Suction engaged — retreat to pull object out."""
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info('[GRIP] Suction ON — object attached')
                self._start_retreat(task)
            else:
                self.get_logger().error(f'[GRIP] Suction failed: {resp.message}')
                self._fail_task(task, 'Grip failed')
        except Exception as e:
            self.get_logger().error(f'[GRIP] Service call error: {e}')
            self._fail_task(task, 'Grip service error')

    def _start_retreat(self, task):
        """Pull object out of shelf by backing up."""
        self.get_logger().info('[RETREAT] Pulling object out...')
        twist = Twist()
        twist.linear.x = RETREAT_SPEED
        self._cmd_pub.publish(twist)
        self._retreat_timer = self.create_timer(
            RETREAT_DURATION,
            lambda: self._on_retreat_done(task),
            callback_group=self._cbg,
        )

    def _on_retreat_done(self, task):
        """Retreat complete — navigate to drop zone."""
        self._retreat_timer.cancel()
        self._stop_rotation()
        self._start_drop_nav(task)

    def _start_drop_nav(self, task):
        """Navigate to drop coordinates with cargo, using standoff."""
        handle = task.get('handle')
        if handle:
            fb = FetchTask.Feedback()
            fb.current_state = 'DROPPING'
            handle.publish_feedback(fb)

        gx = task.get('drop_x', task.get('x', 0.0))
        gy = task.get('drop_y', task.get('y', 0.0))
        gyaw = task.get('drop_yaw', 0.0)
        # Standoff for drop approach
        dox, doy = standoff_point(gx, gy, gyaw)
        self.get_logger().info(
            f'[DROP-NAV] drop=({gx:.1f},{gy:.1f}) standoff=({dox:.1f},{doy:.1f})'
        )
        self._send_nav_goal(dox, doy, gyaw, on_done=lambda ok: (
            self._on_drop_nav_done(ok, task)
        ))

    def _on_drop_nav_done(self, ok, task):
        """Arrived at drop zone — lower lift + release + back away."""
        if not ok:
            self._fail_task(task, 'Drop nav failed')
            return
        self._start_drop_off(task)

    def _start_drop_off(self, task):
        """Lower lift to near-ground, release object, back away."""
        self.get_logger().info(f'[DROP-OFF] Lowering lift to Z={DROP_LIFT_Z}m...')
        self._lift_pub.publish(_make_lift_cmd(DROP_LIFT_Z))

        self._dropoff_timer = self.create_timer(
            DROPOFF_WAIT,
            lambda: self._on_drop_off_release(task),
            callback_group=self._cbg,
        )

    def _on_drop_off_release(self, task):
        """Lift settled — release suction + back away."""
        if self._dropoff_timer:
            self._dropoff_timer.cancel()
            self._dropoff_timer = None

        self.get_logger().info('[DROP-OFF] Releasing suction...')
        if self._gripper_cli.wait_for_service(timeout_sec=2.0):
            req = SetBool.Request()
            req.data = False
            future = self._gripper_cli.call_async(req)
            future.add_done_callback(lambda f: self._on_release_done(f, task))
        else:
            self.get_logger().error('[DROP-OFF] Gripper unavailable for release')
            self._finish_drop_off(task)

    def _on_release_done(self, future, task):
        """Suction off — back away from cargo."""
        try:
            resp = future.result()
            self.get_logger().info(
                f'[DROP-OFF] Suction OFF: {resp.message if resp.success else "FAILED"}'
            )
        except Exception:
            pass
        self._finish_drop_off(task)

    def _finish_drop_off(self, task):
        """Back away and complete the task."""
        self.get_logger().info('[DROP-OFF] Backing away...')
        twist = Twist()
        twist.linear.x = BACKAWAY_SPEED
        self._cmd_pub.publish(twist)
        self._backaway_timer = self.create_timer(
            BACKAWAY_DURATION,
            lambda: self._on_drop_off_done(task),
            callback_group=self._cbg,
        )

    def _on_drop_off_done(self, task):
        """Full pick-and-place complete."""
        if self._backaway_timer:
            self._backaway_timer.cancel()
            self._backaway_timer = None
        self._stop_rotation()
        self._succeed_task(task)


# ============================================================
# Entrypoint
# ============================================================
def main():
    rclpy.init()
    node = BrainNode()
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
