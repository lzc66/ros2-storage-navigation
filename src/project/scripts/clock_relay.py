#!/usr/bin/python3
"""Clock QoS bridge: subscribes /clock (BEST_EFFORT), republishes (RELIABLE)."""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rosgraph_msgs.msg import Clock

# Gazebo Classic: BEST_EFFORT, VOLATILE
BE_CLOCK = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
# Nav2 Humble: RELIABLE, VOLATILE (default system clock QoS)
RL_CLOCK = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


class ClockRelay(Node):
    def __init__(self):
        super().__init__('clock_relay')
        self._sub = self.create_subscription(Clock, '/clock', self._cb, BE_CLOCK)
        self._pub = self.create_publisher(Clock, '/clock', RL_CLOCK)
        self.get_logger().info('Clock relay active: /clock BE→RL')

    def _cb(self, msg):
        self._pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(ClockRelay())


if __name__ == '__main__':
    main()
