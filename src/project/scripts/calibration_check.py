#!/usr/bin/python3
"""
Calibration probe: compare vision_node output against ground-truth spawn position.

Subscribes to /target_object (PointStamped, map frame).
Computes Euclidean error against a configurable ground-truth target.
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped


class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_check')

        self.declare_parameter('truth_x', 1.0)
        self.declare_parameter('truth_y', 0.0)
        self.declare_parameter('truth_z', 1.25)
        self._tx = self.get_parameter('truth_x').value
        self._ty = self.get_parameter('truth_y').value
        self._tz = self.get_parameter('truth_z').value

        self.sub = self.create_subscription(
            PointStamped, '/target_object', self.cb, 10,
        )
        self.get_logger().info(
            f'Calibration probe active. Truth=({self._tx:.2f},{self._ty:.2f},{self._tz:.2f})'
        )

    def cb(self, msg):
        vx = msg.point.x
        vy = msg.point.y
        vz = msg.point.z
        err = math.sqrt(
            (vx - self._tx) ** 2 +
            (vy - self._ty) ** 2 +
            (vz - self._tz) ** 2
        )
        self.get_logger().info(
            f'[CALIBRATION] Vision: ({vx:.2f}, {vy:.2f}, {vz:.2f}) | '
            f'Truth: ({self._tx:.2f}, {self._ty:.2f}, {self._tz:.2f}) | '
            f'Error: {err:.3f} meters'
        )


def main():
    rclpy.init()
    node = CalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
