#!/usr/bin/python3
"""
Preemption stress test: verifies brain_node cancels L2 → executes L1 correctly.

Workflow:
  1. Send L2 task (distant target)
  2. Wait 2.0s for robot to begin moving
  3. Send L1 task (close target)
  4. Monitor brain_node feedback — assert switch to L1 within 5s
"""
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from project.action import FetchTask


class StressTester(Node):
    def __init__(self):
        super().__init__('stress_tester')
        self._client = ActionClient(self, FetchTask, 'fetch_task')
        self._current_state = None
        self._l1_active = False
        self._l2_active = False
        self._start_time = None
        self._passed = False

    def run(self):
        self.get_logger().info('=== Preemption Stress Test ===')
        if not self._client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('/fetch_task server not available')
            sys.exit(1)

        # Step 1: send Level 2 task (distant)
        self.get_logger().info('[STEP 1] Sending L2 task → (5.0, 3.0, 0.8)')
        self._send_task(2, 5.0, 3.0, 0.8, self._l2_response_cb, self._l2_feedback_cb)

        # Step 2: after 2.0s, send Level 1 task (preempt)
        self._l2_timer = self.create_timer(2.0, self._step2_send_l1)

    def _step2_send_l1(self):
        self._l2_timer.cancel()
        self.get_logger().info('[STEP 2] Sending L1 task → (1.0, 0.0, 1.2)')
        self._start_time = time.time()
        self._send_task(1, 1.0, 0.0, 1.2, self._l1_response_cb, self._l1_feedback_cb)

    def _send_task(self, prio, x, y, z, response_cb, feedback_cb):
        goal = FetchTask.Goal()
        goal.priority = prio
        goal.target_x = x
        goal.target_y = y
        goal.target_z = z
        future = self._client.send_goal_async(goal, feedback_callback=feedback_cb)
        future.add_done_callback(response_cb)

    def _l2_feedback_cb(self, fb):
        state = fb.feedback.current_state
        if state != self._current_state:
            self._current_state = state
            self.get_logger().info(f'[L2 FEEDBACK] state={state}')
            if state == 'NAVIGATING':
                self._l2_active = True

    def _l2_response_cb(self, future):
        handle = future.result()
        if handle and handle.accepted:
            self.get_logger().info('[L2] Goal accepted')
        else:
            self.get_logger().error('[L2] Goal rejected')

    def _l1_feedback_cb(self, fb):
        state = fb.feedback.current_state
        self.get_logger().info(f'[L1 FEEDBACK] state={state}')
        if state == 'NAVIGATING':
            elapsed = time.time() - self._start_time
            self.get_logger().info(
                f'[RESULT] L1 task ACTIVE after {elapsed:.1f}s — PREEMPTION WORKS'
            )
            if elapsed < 5.0:
                self._passed = True
                self.get_logger().info('[PASS] Preemption completed within 5s threshold')
            else:
                self.get_logger().error('[FAIL] Preemption took >5s')
            self._shutdown()

    def _l1_response_cb(self, future):
        handle = future.result()
        if handle and handle.accepted:
            self.get_logger().info('[L1] Goal accepted')
        else:
            self.get_logger().error('[L1] Goal rejected — preemption may have failed')

    def _shutdown(self):
        if self._passed:
            self.get_logger().info('=== STRESS TEST PASSED ===')
        else:
            self.get_logger().error('=== STRESS TEST FAILED ===')
        # Allow 1s for log flush
        self.create_timer(1.0, lambda: sys.exit(0 if self._passed else 1))


def main():
    rclpy.init()
    tester = StressTester()
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
