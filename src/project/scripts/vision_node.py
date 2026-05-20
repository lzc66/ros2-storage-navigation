#!/usr/bin/python3
"""
YOLOv8n TensorRT vision: GPU semantic detection + HSV color classification.

Pipeline:
  1. CvBridge BGR8 -> GPU FP16 tensor (640x640 NCHW)
  2. TensorRT engine inference (FP16)
  3. GPU-side NMS + bbox decode
  4. HSV color vote on bbox crops -> red (z=1.0) / blue (z=2.0)
  5. Publish geometry_msgs/Point on /target_object (brain_node compatible)
"""
import os
import time
import numpy as np
import cv2
import torch
import tensorrt as trt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge


_MODEL_DIR = os.path.dirname(os.path.realpath(__file__))
_ENGINE_PATH = os.path.join(_MODEL_DIR, 'yolov8n.engine')

CONF_THRESH = 0.25
IOU_THRESH  = 0.45
MAX_DET     = 10
IMGSZ       = 640

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]

# HSV color classification masks (same as original vision_node)
RED_RANGES  = [([0, 50, 50], [10, 255, 255]), ([170, 50, 50], [180, 255, 255])]
BLUE_RANGES = [([100, 80, 80], [130, 255, 255])]


def _classify_color_hsv(bgr_crop):
    """Return 'red', 'blue', or 'none' via HSV mask area vote."""
    if bgr_crop.size == 0:
        return 'none'
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]

    red_px = 0
    for lower, upper in RED_RANGES:
        red_px += cv2.countNonZero(cv2.inRange(hsv, np.array(lower), np.array(upper)))
    blue_px = 0
    for lower, upper in BLUE_RANGES:
        blue_px += cv2.countNonZero(cv2.inRange(hsv, np.array(lower), np.array(upper)))

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
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.pub = self.create_publisher(Point, '/target_object', 10)

        self._load_engine()
        self._warmup()
        self.get_logger().info(
            f'TensorRT vision ready. Engine={_ENGINE_PATH} '
            f'FP16 input={IMGSZ}x{IMGSZ}'
        )

    # ====================================================================
    # TensorRT engine lifecycle
    # ====================================================================
    def _load_engine(self):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(_ENGINE_PATH, 'rb') as f:
            # Strip ultralytics metadata header (4-byte length + JSON)
            import struct
            meta_len = struct.unpack('<I', f.read(4))[0]
            f.seek(meta_len, 1)  # skip metadata JSON
            engine_data = f.read()
        self._engine = runtime.deserialize_cuda_engine(engine_data)
        self._ctx = self._engine.create_execution_context()
        stream = torch.cuda.Stream()
        self._stream = stream

        # Input binding
        self._in_name = self._engine.get_tensor_name(0)
        in_shape = tuple(self._engine.get_tensor_shape(self._in_name))
        # Output binding
        self._out_name = self._engine.get_tensor_name(1)
        out_shape = tuple(self._engine.get_tensor_shape(self._out_name))

        self.get_logger().info(
            f'Engine bindings: {self._in_name} {in_shape} -> '
            f'{self._out_name} {out_shape}'
        )

        # Pre-allocate GPU buffers
        self._in_buf  = torch.empty(in_shape,  dtype=torch.float16, device='cuda')
        self._out_buf = torch.empty(out_shape, dtype=torch.float16, device='cuda')

    def _warmup(self):
        """Single dry-run to initialise CUDA kernels."""
        dummy = torch.randn(1, 3, IMGSZ, IMGSZ, dtype=torch.float16, device='cuda')
        with torch.cuda.stream(self._stream):
            self._ctx.set_tensor_address(self._in_name, dummy.data_ptr())
            self._ctx.set_tensor_address(self._out_name, self._out_buf.data_ptr())
            self._ctx.execute_async_v3(self._stream.cuda_stream)
            self._stream.synchronize()
        self.get_logger().info('TensorRT warmup complete.')

    # ====================================================================
    # Preprocessing: BGR8 ndarray -> GPU FP16 NCHW tensor
    # ====================================================================
    def _preprocess(self, bgr):
        """Resize, BGR->RGB, normalize, CHW, FP16, upload to GPU."""
        # Letterbox resize (preserve aspect ratio, pad to 640x640)
        h0, w0 = bgr.shape[:2]
        scale = IMGSZ / max(h0, w0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Pad to square
        dw = IMGSZ - new_w
        dh = IMGSZ - new_h
        top, left = dh // 2, dw // 2
        bottom, right = dh - top, dw - left
        letterbox = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        # BGR -> RGB, HWC -> CHW, normalize [0,1], FP16
        rgb = letterbox[..., ::-1].copy()  # BGR to RGB (.copy avoids neg stride)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device='cuda', dtype=torch.float16) / 255.0
        return tensor, (h0, w0, top, left, scale)

    # ====================================================================
    # Inference
    # ====================================================================
    def _infer(self, tensor):
        with torch.cuda.stream(self._stream):
            self._ctx.set_tensor_address(self._in_name, tensor.data_ptr())
            self._ctx.set_tensor_address(self._out_name, self._out_buf.data_ptr())
            self._ctx.execute_async_v3(self._stream.cuda_stream)
            self._stream.synchronize()
        return self._out_buf.clone()

    # ====================================================================
    # Postprocessing: GPU decode + NMS -> list of (xyxy, conf, cls_id)
    # ====================================================================
    @torch.no_grad()
    def _postprocess(self, output, meta):
        h0, w0, top, left, scale = meta

        # output shape: (1, 84, 8400)
        preds = output.squeeze(0).float()  # (84, 8400)

        # Split: first 4 = bbox (cx,cy,w,h), rest = class logits
        bbox_raw = preds[:4]   # (4, 8400)
        cls_raw  = preds[4:]   # (80, 8400)

        # Decode bbox: ultralytics export uses DFL-decode internally;
        # output bbox values are in [0,1] xywh relative to 640x640.
        # Scale to 640x640 pixel coords then map back to original image.
        cx = bbox_raw[0] * IMGSZ   # (8400,)
        cy = bbox_raw[1] * IMGSZ
        w  = bbox_raw[2] * IMGSZ
        h  = bbox_raw[3] * IMGSZ

        # Convert to xyxy
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Remove letterbox padding effect
        x1 = (x1 - left) / scale
        y1 = (y1 - top)  / scale
        x2 = (x2 - left) / scale
        y2 = (y2 - top)  / scale

        # Clip to original image bounds
        w0_t = torch.tensor(w0, device=x1.device, dtype=torch.float32)
        h0_t = torch.tensor(h0, device=x1.device, dtype=torch.float32)
        x1 = x1.clamp(0, w0_t)
        y1 = y1.clamp(0, h0_t)
        x2 = x2.clamp(0, w0_t)
        y2 = y2.clamp(0, h0_t)

        # Class confidence
        cls_conf, cls_id = cls_raw.sigmoid().max(dim=0)  # (8400,), (8400,)

        # Filter by confidence
        keep = cls_conf > CONF_THRESH
        if not keep.any():
            return []

        x1 = x1[keep]; y1 = y1[keep]; x2 = x2[keep]; y2 = y2[keep]
        cls_conf = cls_conf[keep]; cls_id = cls_id[keep]

        # Filter out tiny boxes
        area = (x2 - x1) * (y2 - y1)
        keep_area = area > 16
        if not keep_area.any():
            return []
        x1 = x1[keep_area]; y1 = y1[keep_area]; x2 = x2[keep_area]; y2 = y2[keep_area]
        cls_conf = cls_conf[keep_area]; cls_id = cls_id[keep_area]

        # GPU NMS via torchvision
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        nms_idx = torchvision_ops_nms(boxes, cls_conf, IOU_THRESH)
        nms_idx = nms_idx[:MAX_DET]

        detections = []
        for i in nms_idx:
            detections.append((
                (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                float(cls_conf[i]),
                int(cls_id[i]),
            ))
        return detections

    # ====================================================================
    # Callback
    # ====================================================================
    def callback(self, msg):
        t0 = time.perf_counter()

        # --- Decode ROS image ---
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge decode: {e}')
            return

        h, w = frame.shape[:2]
        total_px = h * w
        half_w = w / 2.0

        # --- Preprocess ---
        tensor, meta = self._preprocess(frame)

        # --- TensorRT inference ---
        t1 = time.perf_counter()
        output = self._infer(tensor)
        t2 = time.perf_counter()

        # --- Postprocess ---
        detections = self._postprocess(output, meta)

        # --- Color classification + best target selection ---
        best_type  = 0.0
        best_area  = 0.0
        best_cx    = 0.0

        for (x1, y1, x2, y2), conf, cls_id in detections:
            crop = frame[y1:y2, x1:x2]
            color = _classify_color_hsv(crop)

            # Map color to brain_node type code
            if color == 'red':
                z_type = 1.0
            elif color == 'blue':
                z_type = 2.0
            else:
                # Skip uncolored objects (person, floor, etc.)
                continue

            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / total_px
            if area_ratio > best_area:
                best_area = area_ratio
                best_type = z_type
                best_cx = (x1 + x2) / 2.0

        t3 = time.perf_counter()

        # --- Publish (brain_node-compatible format) ---
        if best_type > 0.0:
            error_x_norm = (best_cx - half_w) / half_w
            area_ratio   = best_area / total_px

            msg_out = Point()
            msg_out.x = error_x_norm
            msg_out.y = area_ratio
            msg_out.z = best_type
            self.pub.publish(msg_out)

        # Performance log (every 30 frames to reduce log spam)
        if not hasattr(self, '_perf_counter'):
            self._perf_counter = 0
        self._perf_counter += 1
        if self._perf_counter % 30 == 0:
            pre_ms  = (t1 - t0) * 1000
            infer_ms = (t2 - t1) * 1000
            post_ms = (t3 - t2) * 1000
            total_ms = (t3 - t0) * 1000
            self.get_logger().info(
                f'[PERF] Pre:{pre_ms:.1f}ms | TRT:{infer_ms:.1f}ms | '
                f'Post:{post_ms:.1f}ms | Total:{total_ms:.1f}ms | '
                f'Dets:{len(detections)}'
            )


# Lazy import to avoid torchvision import overhead at module load
from torchvision.ops import nms as torchvision_ops_nms


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
