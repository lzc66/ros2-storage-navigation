#!/usr/bin/python3
"""
Task Emitter: send FetchTask actions to brain_node via CLI.

Usage:
  ros2 run project task_emitter --priority 1 --x 2.0 --y 1.0 --z 1.2
  ros2 run project task_emitter -p 2 -x 0.0 -y -1.0 -z 0.8
"""
import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from project.action import FetchTask


class TaskEmitter(Node):
    def __init__(self, priority, x, y, z, yaw, wait):
        super().__init__('task_emitter')
        self._client = ActionClient(self, FetchTask, 'fetch_task')
        self._priority = priority
        self._x = x; self._y = y; self._z = z; self._yaw = yaw
        self._wait = wait
        self._done = False

    def send(self):
        self.get_logger().info(
            f'Sending task P{self._priority} → '
            f'({self._x:.1f},{self._y:.1f},{self._z:.1f}) yaw={self._yaw:.2f}'
        )
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server /fetch_task not available')
            self._done = True
            return

        goal = FetchTask.Goal()
        goal.priority = self._priority
        goal.target_x = self._x
        goal.target_y = self._y
        goal.target_z = self._z
        goal.target_yaw = self._yaw

        send_future = self._client.send_goal_async(
            goal, feedback_callback=self._feedback_cb,
        )
        send_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            self._done = True
            return
        self.get_logger().info('Goal accepted. Waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, fb):
        self.get_logger().info(
            f'[FEEDBACK] {fb.feedback.current_state} '
            f'dist={fb.feedback.distance_to_target:.2f}'
        )

    def _result_cb(self, future):
        result = future.result()
        self.get_logger().info(
            f'[RESULT] success={result.result.success} '
            f'msg={result.result.message}'
        )
        if self._wait:
            self._done = True


def main():
    rclpy.init()
    parser = argparse.ArgumentParser(description='FetchTask emitter')
    parser.add_argument('-p', '--priority', type=int, default=2,
                        help='1=URGENT, 2=NORMAL (default: 2)')
    parser.add_argument('--x', type=float, default=2.0,
                        help='target X (world)')
    parser.add_argument('--y', type=float, default=1.0,
                        help='target Y (world)')
    parser.add_argument('--z', type=float, default=0.8,
                        help='target Z (lift height)')
    parser.add_argument('--yaw', type=float, default=0.0,
                        help='target yaw in rad (0=+X, 1.57=+Y, 3.14=-X)')
    parser.add_argument('--wait', action='store_true',
                        help='wait for task completion before exit')
    args = parser.parse_args()

    node = TaskEmitter(args.priority, args.x, args.y, args.z, args.yaw, args.wait)
    node.send()

    if args.wait:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)
    else:
        rclpy.spin_once(node, timeout_sec=1.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
