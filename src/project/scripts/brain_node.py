#!/usr/bin/python3
"""
Campaign 3: Priority-Queue Cognitive Engine with Safe Nav2 Preemption.

Architecture:
  - Action Server: accepts FetchTask (priority, target_x/y/z)
  - PriorityQueue: (priority, seq, goal) — lower priority value = higher urgency
  - Preemption: cancel_goal_async → wait CANCELED → send new goal
  - Orchestration: Nav2 move → lift to Z → vision confirm (PointStamped)
  - Async: MultiThreadedExecutor, no time.sleep in callbacks
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
from std_msgs.msg import Float64MultiArray

from project.action import FetchTask

# ================================================================
# Constants
# ================================================================
APPROACH_THRESHOLD = 0.5     # m — considered "arrived" at Nav2 goal
VISION_CONFIRM_EPS  = 0.3    # m — vision match threshold for grab success
VISION_TIMEOUT      = 8.0    # s — max wait for vision after arriving
LIFT_WAIT           = 3.0    # s — settle time after lift move
POLL_PERIOD         = 0.5    # s — queue poll interval
PREEMPT_WAIT        = 3.0    # s — max wait for cancel to resolve

# Drop zones (world coords)
RED_ZONE  = (-2.0, -0.5)
BLUE_ZONE = (-2.0, 1.5)


def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


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
            Float64MultiArray, '/lift_controller/commands', 10,
        )
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

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
            'Brain Campaign 3 ready. Action Server: /fetch_task'
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
            'seq': self._seq,
            'handle': goal_handle,
        }
        self._task_queue.put((request.priority, self._seq, task))
        self.get_logger().info(
            f'[ENQUEUE] P{request.priority} seq={self._seq} '
            f'→ ({request.target_x:.1f},{request.target_y:.1f},'
            f'{request.target_z:.1f})  queue_size={self._task_queue.qsize()}'
        )

        # Wait for task completion (non-blocking)
        result = FetchTask.Result()
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                result.success = False
                result.message = 'Cancelled'
                goal_handle.canceled()
                return result
            if goal_handle.is_active and not goal_handle.is_executing:
                # Task completed externally
                break
            await self._sleep_async(0.5)

        result.success = True
        result.message = 'Fetch completed'
        goal_handle.succeed()
        return result

    async def _sleep_async(self, seconds):
        """Non-blocking sleep for async contexts."""
        start = time.time()
        while time.time() - start < seconds:
            await self._yield()
    async def _yield(self):
        pass  # rclpy executor yields naturally

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
        """Stateful pipeline: NAVIGATE → LIFT → VISION_CONFIRM."""
        self._active_task = task
        handle = task.get('handle')
        restored = task.get('restored', False)

        tag = 'RESTORED' if restored else 'NEW'
        self.get_logger().info(
            f'[EXEC {tag}] P{prio} seq={seq} '
            f'→ ({task["x"]:.1f},{task["y"]:.1f},{task["z"]:.1f})'
        )

        if handle:
            feedback = FetchTask.Feedback()
            feedback.current_state = 'NAVIGATING'
            handle.publish_feedback(feedback)

        self._nav_busy = True
        self._send_nav_goal(task['x'], task['y'], on_done=lambda ok: (
            self._on_nav_done(ok, prio, seq, task)
        ))

    # ============================================================
    # Nav2: send goal + result handler
    # ============================================================
    def _send_nav_goal(self, gx, gy, on_done=None):
        self._nav_client.wait_for_server()
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = gx
        goal.pose.pose.position.y = gy
        goal.pose.pose.orientation.w = 1.0

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
            self.get_logger().info(
                '[NAV] Previous goal was canceled (preemption confirmed)'
            )
            # Do NOT call on_done — the preempt path is handling dispatch
        else:
            self.get_logger().warn(f'[NAV] Failed (status={status})')
            if not self._preempting:
                if on_done:
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

        # Move lift to target Z
        z_target = task['z']
        self.get_logger().info(f'[LIFT] Moving to Z={z_target:.2f}m')
        msg = Float64MultiArray()
        msg.data = [float(z_target)]
        self._lift_pub.publish(msg)

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

        # Vision confirmation loop (poll via timer)
        self._vision_start_time = time.time()
        self._vision_task = task
        self._vision_prio = prio
        self._vision_seq = seq
        self._vision_timer = self.create_timer(
            0.5, self._vision_poll, callback_group=self._cbg,
        )

    def _vision_poll(self):
        """Check if vision sees the target within error threshold."""
        elapsed = time.time() - self._vision_start_time
        task = self._vision_task
        tx, ty, tz = task['x'], task['y'], task['z']

        if self._vision_seen:
            err = math.sqrt(
                (self._vision_x - tx) ** 2 +
                (self._vision_y - ty) ** 2
                # Z comparison skipped (type_code in point.z, not world Z)
            )
            dist = math.sqrt(
                (self._vision_x - self._rx) ** 2 +
                (self._vision_y - self._ry) ** 2
            )
            if dist < VISION_CONFIRM_EPS:
                self.get_logger().info(
                    f'[CONFIRM] Vision match! err={err:.2f}m dist={dist:.2f}m'
                )
                self._vision_timer.cancel()
                self._succeed_task(task)
                return

        if elapsed > VISION_TIMEOUT:
            self.get_logger().warn(
                f'[CONFIRM] Vision timeout ({elapsed:.1f}s)'
            )
            self._vision_timer.cancel()
            self._fail_task(task, 'Vision confirmation timeout')

    # ============================================================
    # Task completion
    # ============================================================
    def _succeed_task(self, task):
        self._nav_busy = False
        self._active_task = None
        self.get_logger().info('[TASK] SUCCESS')

    def _fail_task(self, task, reason):
        self._nav_busy = False
        self._active_task = None
        self.get_logger().error(f'[TASK] FAILED: {reason}')


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
