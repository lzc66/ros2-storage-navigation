#!/usr/bin/python3
"""
Semantic 3D Vision: YOLOv8 + RGB-D depth + TF2 spatial projection.

Pipeline:
  1. message_filters aligns /camera/color/image_raw + /camera/depth/image_raw
  2. YOLOv8n infers bounding boxes (COCO baseline)
  3. Depth read at bbox centre pixel (u, v)
  4. Pinhole projection: pixel → camera_optical_frame (Xc, Yc, Zc)
  5. TF2 transform: camera_depth_link → map (absolute world coords)
  6. Publishes geometry_msgs/PointStamped on /target_object (Xw, Yw, Zw)
"""
import os
import time
import numpy as np
import cv2
import torch

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge
import message_filters

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped as PointStampedMsg

# Paths
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_MODEL_PATH = os.path.join(_SCRIPT_DIR, 'yolov8n.pt')

# YOLO classes used for sorting objects (COCO indices)
TARGET_CLASSES = {
    39: 'bottle',
    41: 'cup',
    44: 'bowl',
    45: 'bowl',
    73: 'book',
    75: 'vase',
    77: 'scissors',
    84: 'book',  # 'book' in some model versions
}
CONF_THRESH = 0.25
IOU_THRESH = 0.45

# Depth: ignore 0 (no data) and NaN
MIN_DEPTH = 0.1   # meters
MAX_DEPTH = 15.0  # meters

# HSV colour ranges for target classification
RED_RANGES = [([0, 50, 50], [10, 255, 255]), ([170, 50, 50], [180, 255, 255])]
BLUE_RANGES = [([100, 80, 80], [130, 255, 255])]


def _classify_color_hsv(bgr_crop):
    """Return 'red', 'blue', or 'none' via HSV mask area vote."""
    if bgr_crop is None or bgr_crop.size == 0:
        return 'none'
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return 'none'

    red_px = 0
    for low, high in RED_RANGES:
        red_px += cv2.countNonZero(cv2.inRange(hsv, np.array(low), np.array(high)))
    blue_px = 0
    for low, high in BLUE_RANGES:
        blue_px += cv2.countNonZero(cv2.inRange(hsv, np.array(low), np.array(high)))

    red_r = red_px / total
    blue_r = blue_px / total
    if red_r > 0.05 and red_r > blue_r:
        return 'red'
    if blue_r > 0.05 and blue_r > red_r:
        return 'blue'
    return 'none'


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()

        # -- TF2 --
        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5))
        self._tf_listener = tf2_ros.TransformListener(
            self._tf_buffer, self, spin_thread=True
        )

        # -- YOLO model (PyTorch, FP16 on GPU) --
        self._load_model()

        # -- Camera intrinsics (populated from /camera/color/camera_info) --
        self._fx = self._fy = 0.0
        self._cx = self._cy = 0.0
        self._cam_info_ready = False

        # -- Subscriptions --
        self._cam_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self._cam_info_cb, 10
        )

        # -- message_filters: approximate time synchronizer for RGB + Depth --
        self._rgb_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw'
        )
        self._depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw'
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=10,
            slop=0.1,  # 100ms max time difference
        )
        self._sync.registerCallback(self._synced_callback)

        # -- Publisher: world-frame 3D position (PointStamped, brain_node-compatible) --
        self._pub = self.create_publisher(PointStamped, '/target_object', 10)

        self.get_logger().info(
            f'3D Vision ready. YOLOv8n model={_MODEL_PATH}, '
            f'optical frame=camera_depth_link→map'
        )

    # ================================================================
    # Model loading
    # ================================================================
    def _load_model(self):
        from ultralytics import YOLO
        if os.path.exists(_MODEL_PATH):
            self._model = YOLO(_MODEL_PATH, task='detect')
            self.get_logger().info(f'Loaded YOLOv8n from {_MODEL_PATH}')
        else:
            # Download from ultralytics hub
            self.get_logger().info('Downloading YOLOv8n from ultralytics hub...')
            self._model = YOLO('yolov8n.pt', task='detect')

        # Move to GPU FP16 for maximum throughput
        try:
            self._model.model = self._model.model.to('cuda').half()
        except Exception:
            self._model.model = self._model.model.to('cuda')
            self.get_logger().warn('FP16 not available, using FP32')

        # Warmup with a small dummy image on GPU
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self._model(dummy, imgsz=640, half=True, verbose=False)
        self.get_logger().info('YOLOv8n GPU warmup complete')

    # ================================================================
    # Camera info callback
    # ================================================================
    def _cam_info_cb(self, msg):
        if not self._cam_info_ready:
            self._fx = msg.k[0]
            self._fy = msg.k[4]
            self._cx = msg.k[2]
            self._cy = msg.k[5]
            self._cam_info_ready = True
            self.get_logger().info(
                f'Camera intrinsics: fx={self._fx:.1f} fy={self._fy:.1f} '
                f'cx={self._cx:.1f} cy={self._cy:.1f}'
            )
            # Unsubscribe after first message
            self.destroy_subscription(self._cam_info_sub)

    # ================================================================
    # Synchronised RGB + Depth callback
    # ================================================================
    def _synced_callback(self, rgb_msg, depth_msg):
        if not self._cam_info_ready:
            return  # no intrinsics yet

        t0 = time.perf_counter()

        # --- Decode images ---
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f'Image decode failed: {e}')
            return

        if depth is None or depth.size == 0:
            return

        # Depth: use msg.encoding for unambiguous unit conversion
        enc = depth_msg.encoding
        if enc in ('32FC1', '32FC2'):
            depth_m = depth.astype(np.float32)             # metres
        elif enc in ('16UC1', '16UC2'):
            depth_m = depth.astype(np.float32) / 1000.0   # millimetres → metres
        elif enc == 'mono16':
            # Ambiguous: Gazebo Classic rgbd_camera often uses mono16 in mm
            depth_m = depth.astype(np.float32) / 1000.0
        elif enc == 'mono8':
            depth_m = depth.astype(np.float32) / 255.0    # normalised
        else:
            self.get_logger().error(f'Unknown depth encoding: {enc}')
            return

        h_img, w_img = bgr.shape[:2]

        # --- YOLO inference (GPU FP16) ---
        t1 = time.perf_counter()
        results = self._model(
            bgr, imgsz=640, half=True, verbose=False,
            conf=CONF_THRESH, iou=IOU_THRESH, max_det=10,
        )
        t2 = time.perf_counter()

        if results is None or len(results[0].boxes) == 0:
            return  # no detections

        # --- Select highest-confidence target within our class set ---
        boxes = results[0].boxes
        best_conf = 0.0
        best_xyxy = None
        best_cls = None

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id in TARGET_CLASSES:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_xyxy = box.xyxy[0].cpu().numpy()
                    best_cls = cls_id

        if best_xyxy is None:
            return  # no target-class objects

        # --- Bounding box centre pixel ---
        x1, y1, x2, y2 = best_xyxy
        u = int((x1 + x2) / 2.0)
        v = int((y1 + y2) / 2.0)

        # Clamp to image bounds
        u = max(0, min(u, w_img - 1))
        v = max(0, min(v, h_img - 1))

        # --- Read depth at centre pixel ---
        d = depth_m[v, u]
        if np.isnan(d) or d < MIN_DEPTH or d > MAX_DEPTH:
            # Try median depth in a 5×5 window around the centre
            r = 5
            y0, y1c = max(0, v - r), min(h_img, v + r + 1)
            x0, x1c = max(0, u - r), min(w_img, u + r + 1)
            patch = depth_m[y0:y1c, x0:x1c]
            patch = patch[(patch > MIN_DEPTH) & (patch < MAX_DEPTH)]
            if patch.size > 0:
                d = float(np.median(patch))
            else:
                return  # no valid depth

        # --- Pinhole camera projection: pixel → camera_optical_frame ---
        x_c = (u - self._cx) * d / self._fx
        y_c = (v - self._cy) * d / self._fy
        z_c = d

        # --- TF2: camera_depth_link → map ---
        stamp = rgb_msg.header.stamp
        target_time = rclpy.time.Time(
            seconds=stamp.sec, nanoseconds=stamp.nanosec
        )
        try:
            transform = self._tf_buffer.lookup_transform(
                'map',                    # target frame
                'camera_depth_link',      # source frame (optical)
                target_time,
                timeout=Duration(seconds=0.5),
            )
        except tf2_ros.TransformException as e:
            # Retry with latest available
            try:
                transform = self._tf_buffer.lookup_transform(
                    'map', 'camera_depth_link',
                    rclpy.time.Time(),  # latest
                    timeout=Duration(seconds=0.2),
                )
            except tf2_ros.TransformException as e2:
                self.get_logger().debug(f'TF lookup failed: {e2}')
                return

        # Apply transform to (Xc, Yc, Zc)
        pt_cam = PointStampedMsg()
        pt_cam.header.frame_id = 'camera_depth_link'
        pt_cam.header.stamp = stamp
        pt_cam.point.x = x_c
        pt_cam.point.y = y_c
        pt_cam.point.z = z_c

        try:
            pt_map = tf2_geometry_msgs.do_transform_point(pt_cam, transform)
        except Exception as e:
            self.get_logger().error(f'TF2 transform point failed: {e}')
            return

        # --- Publish 3D world position with type code ---
        # HSV color classification on bbox crop to determine type
        crop = bgr[int(y1):int(y2), int(x1):int(x2)]
        color = _classify_color_hsv(crop)
        type_code = 1.0 if color == 'red' else (2.0 if color == 'blue' else 0.0)

        out_ps = PointStamped()
        out_ps.header.stamp = stamp
        out_ps.header.frame_id = 'map'
        out_ps.point.x = pt_map.point.x
        out_ps.point.y = pt_map.point.y
        out_ps.point.z = float(type_code)  # 1.0=red, 2.0=blue
        self._pub.publish(out_ps)

        t3 = time.perf_counter()

        # --- Performance log ---
        if not hasattr(self, '_perf_n'):
            self._perf_n = 0
        self._perf_n += 1
        if self._perf_n % 30 == 0:
            infer_ms = (t2 - t1) * 1000
            total_ms = (t3 - t0) * 1000
            cls_name = TARGET_CLASSES.get(best_cls, f'cls_{best_cls}')
            self.get_logger().info(
                f'[PERF] YOLO:{infer_ms:.1f}ms Total:{total_ms:.1f}ms | '
                f'{cls_name} (u={u},v={v}) d={d:.2f}m → '
                f'({pt_map.point.x:.2f},{pt_map.point.y:.2f},{pt_map.point.z:.2f}) [map]'
            )


def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
