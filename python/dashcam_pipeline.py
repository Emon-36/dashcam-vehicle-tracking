#!/usr/bin/env python3
"""
Multi-Threaded Dashboard-Cam Suite (Python / YOLOv8)
=====================================================
- Colorful per-class detection boxes (car/bus/truck), filled label strips.
- Distance estimate FUSES two independent cues:
    1) ground-plane geometry (camera height + focal length)
    2) known-object-size geometry (average real vehicle width)
  so it isn't dependent on a single arbitrary fudge factor.
- Colorful, dense optical-flow visualization (HSV direction/magnitude wash
  + magnitude-graded, direction-colored arrows).
- True 4-stage pipeline, each stage its own thread with its own bounded
  queue.Queue (thread-safe by construction -- no manual locks needed):
      capture -> tracking(inference) -> optical-flow -> writer
  OpenCV/numpy release the GIL during their C-level calls, so these threads
  genuinely run concurrently across cores, not just interleaved.

Usage:
    python3 dashcam_pipeline.py <model.pt|model.onnx> <video_path> [options]

Requires (in a clean venv/conda env, Python 3.10-3.12 recommended):
    pip install ultralytics opencv-python numpy
"""

import argparse
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
    # This CPU doesn't support NNPACK's required instruction set, so torch
    # falls back automatically anyway -- disabling it outright avoids the
    # repeated "Could not initialize NNPACK! Unsupported hardware" warning
    # spam on every forward pass, and skips the tiny wasted init-attempt cost.
    torch.backends.nnpack.enabled = False
except ImportError:
    pass

# ============================================================================
# Camera / distance model
# ----------------------------------------------------------------------------
# Focal length is computed at RUNTIME from the actual video resolution, not
# hardcoded, so it stays correct regardless of the input clip's size. A
# generic dashcam horizontal-FOV assumption is used (~90-110 deg is typical
# for forward-facing dashcams; 100 deg is a reasonable default). If you know
# your exact camera's HFOV from its spec sheet, or have done a real
# checkerboard calibration (cv2.calibrateCamera), pass --hfov / edit
# CAM_MOUNT_HEIGHT_M for better accuracy.
# ============================================================================
DASHCAM_HFOV_DEG_DEFAULT = 100.0
CAM_MOUNT_HEIGHT_M_DEFAULT = 1.3  # typical windshield mount height, average sedan

# Average real-world vehicle widths (meters) -- the "known object size" cue.
VEHICLE_WIDTH_M = {
    "car": 1.80,
    "bus": 2.55,
    "truck": 2.50,
}
VEHICLE_CLASSES = set(VEHICLE_WIDTH_M.keys())
# Standard 0-indexed COCO-80 ids ultralytics uses internally: car=2, bus=5, truck=7
VEHICLE_CLASS_IDS = [2, 5, 7]

CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.25
MAX_MISSED_FRAMES = 8

# ============================================================================
# Optical-flow visualization params
# ============================================================================
ARROW_GRID_STEP = 8            # px spacing between arrows (denser grid -> more arrows)
MOTION_THRESHOLD = 0.9          # px/frame minimum motion to draw an arrow (lower -> more arrows)
FLOW_HSV_BLEND_ALPHA = 0.30     # strength of the full-frame color-flow wash
ARROW_THICKNESS = 2
ARROW_LENGTH_MULTIPLIER = 2.4   # longer arrows (was effectively 1.6)

MAX_QUEUE_SIZE = 4

# Class colors (BGR) -- distinct, saturated per class.
CLASS_COLOR = {
    "car": (70, 200, 60),     # green
    "bus": (0, 140, 255),     # orange
    "truck": (50, 50, 230),   # red
}
DEFAULT_COLOR = (15, 196, 241)  # yellow fallback


# ============================================================================
# Camera intrinsics (computed once from actual frame size)
# ============================================================================
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    cam_height_m: float


def init_camera_intrinsics(frame_w: int, frame_h: int, hfov_deg: float, cam_height_m: float) -> CameraIntrinsics:
    half_fov_rad = math.radians(hfov_deg / 2.0)
    fx = (frame_w / 2.0) / math.tan(half_fov_rad)
    fy = fx  # assumes square pixels, true for essentially all consumer dashcams
    cx = frame_w / 2.0
    cy = frame_h / 2.0  # assumes a level-mounted dashcam (horizon at image center)
    intr = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, cam_height_m=cam_height_m)
    print(f"[calib] frame={frame_w}x{frame_h} HFOV={hfov_deg}deg -> "
          f"fx=fy={fx:.1f}px, camHeight={cam_height_m}m")
    return intr


def compute_distance(v_bottom_px: float, bbox_width_px: float, class_name: str, intr: CameraIntrinsics) -> float:
    """
    Fuses ground-plane distance (flat-road + camera-height assumption) with
    known-object-width distance (average real vehicle width). Averaging the
    two is more robust than either alone -- no arbitrary fudge multiplier
    needed once both cues are individually correct.

    NOTE: near the horizon, pixel_disp (v_bottom_px - cy) shrinks toward
    zero, so ground_dist = height*fy/pixel_disp explodes and becomes
    wildly noisy from a 1-2px detection jitter -- this is what causes far
    cars to flicker between e.g. 60m and 180m frame-to-frame (and the
    resulting fake 400+ km/h speed spikes). Requiring a larger minimum
    pixel_disp before trusting the ground-plane cue, and falling back to
    the much more stable width-based cue near the horizon, fixes that.
    """
    real_width = VEHICLE_WIDTH_M.get(class_name, VEHICLE_WIDTH_M["car"])
    width_dist = (real_width * intr.fx) / bbox_width_px if bbox_width_px > 1.0 else -1.0

    pixel_disp = v_bottom_px - intr.cy
    MIN_PIXEL_DISP = 8.0  # was 1.5 -- too small a denominator made distant/near-horizon cars unstable
    ground_dist = (intr.cam_height_m * intr.fy) / pixel_disp if pixel_disp > MIN_PIXEL_DISP else -1.0

    if ground_dist > 0 and width_dist > 0:
        dist = 0.5 * (ground_dist + width_dist)
    elif width_dist > 0:
        dist = width_dist
    elif ground_dist > 0:
        dist = ground_dist
    else:
        dist = 150.0  # both cues degenerate -> generic "far away" placeholder

    return min(dist, 200.0)  # clamp: beyond this, dashcam-scale detections are unreliable anyway


# ============================================================================
# Simple 1D (distance, velocity) Kalman filter per track -- smooths distance
# and derives a stable speed estimate instead of using raw frame-to-frame
# differences (which are noisy).
# ============================================================================
PROCESS_NOISE = 3.0
MEASUREMENT_NOISE = 0.6


@dataclass
class Track:
    track_id: int
    class_name: str
    last_box: Tuple[float, float, float, float]  # x1,y1,x2,y2
    last_distance: float
    last_timestamp: float
    missed_frames: int = 0
    smoothed_speed: float = 0.0
    kf_d: float = 0.0
    kf_v: float = 0.0
    P00: float = 1000.0
    P01: float = 0.0
    P10: float = 0.0
    P11: float = 1000.0
    kf_initialized: bool = False
    reject_streak: int = 0  # consecutive frames where a measurement was rejected as an outlier

    def kalman_predict_update(self, measured_distance: float, dt: float):
        if not self.kf_initialized:
            self.kf_d = measured_distance
            self.kf_v = 0.0
            self.P00, self.P01, self.P10, self.P11 = 1000.0, 0.0, 0.0, 1000.0
            self.kf_initialized = measured_distance > 0.0
            return

        d_pred = self.kf_d + self.kf_v * dt
        v_pred = self.kf_v
        P00_pred = self.P00 + dt * self.P10 + dt * self.P01 + dt * dt * self.P11
        P01_pred = self.P01 + dt * self.P11
        P10_pred = self.P10 + dt * self.P11
        P11_pred = self.P11
        dt3 = (dt ** 3) / 3.0
        dt2 = (dt ** 2) / 2.0
        P00_pred += PROCESS_NOISE * dt3
        P01_pred += PROCESS_NOISE * dt2
        P10_pred += PROCESS_NOISE * dt2
        P11_pred += PROCESS_NOISE * dt

        # Outlier rejection: a real highway vehicle's distance can't jump by
        # more than ~55 m/s (~200 km/h) relative to us in one frame. If the
        # raw measurement implies more than that, it's near-horizon pixel
        # jitter, not real motion -- treat this frame as a missed measurement
        # (predict-only) instead of letting it corrupt the velocity estimate.
        #
        # ESCAPE HATCH: rejecting every frame forever is worse than the
        # occasional bad frame -- if the filter's own prediction has drifted
        # away from reality (e.g. a bad early measurement locked in a wrong
        # velocity), EVERY future real measurement looks like an "outlier"
        # relative to that bad prediction, freezing the track's displayed
        # distance/speed permanently. After a few consecutive rejections we
        # force-accept the next measurement to resync with reality.
        MAX_PLAUSIBLE_CLOSING_SPEED_MPS = 55.0
        MAX_CONSECUTIVE_REJECTS = 4
        if measured_distance > 0.0 and dt > 0.0:
            implied_speed = abs(measured_distance - d_pred) / dt
            if implied_speed > MAX_PLAUSIBLE_CLOSING_SPEED_MPS and self.reject_streak < MAX_CONSECUTIVE_REJECTS:
                self.reject_streak += 1
                measured_distance = -1.0  # discard this frame's measurement
            else:
                self.reject_streak = 0  # accepted (either plausible, or forced resync)

        if measured_distance > 0.0:
            y = measured_distance - d_pred
            S = P00_pred + MEASUREMENT_NOISE
            K0 = P00_pred / S
            K1 = P10_pred / S
            self.kf_d = d_pred + K0 * y
            self.kf_v = v_pred + K1 * y
            self.P00 = (1 - K0) * P00_pred
            self.P01 = (1 - K0) * P01_pred
            self.P10 = P10_pred - K1 * P00_pred
            self.P11 = P11_pred - K1 * P01_pred
        else:
            self.kf_d, self.kf_v = d_pred, v_pred
            self.P00, self.P01, self.P10, self.P11 = P00_pred, P01_pred, P10_pred, P11_pred

        self.kf_d = max(self.kf_d, 1.0)  # never let predict-only drift push distance to/below zero


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class VehicleTracker:
    def __init__(self, intr: CameraIntrinsics):
        self.tracks = []
        self._next_id = 0
        self.intr = intr

    def update(self, detections, timestamp: float):
        matched = [False] * len(detections)

        for track in self.tracks:
            best_idx, best_iou = -1, IOU_THRESHOLD
            for i, det in enumerate(detections):
                if matched[i]:
                    continue
                score = iou(track.last_box, det["box"])
                if score > best_iou:
                    best_iou, best_idx = score, i

            if best_idx != -1:
                det = detections[best_idx]
                bw = det["box"][2] - det["box"][0]
                new_dist = compute_distance(det["box"][3], bw, det["class_name"], self.intr)
                dt = timestamp - track.last_timestamp
                if dt > 1e-3:
                    track.kalman_predict_update(new_dist, dt)
                    track.smoothed_speed = max(-200.0, min((-track.kf_v) * 3.6, 200.0))
                if track.kf_initialized:
                    track.last_distance = track.kf_d
                elif new_dist > 0:
                    track.last_distance = new_dist
                track.class_name = det["class_name"]
                track.last_box = det["box"]
                track.last_timestamp = timestamp
                track.missed_frames = 0
                matched[best_idx] = True
            else:
                track.missed_frames += 1

        self.tracks = [t for t in self.tracks if t.missed_frames <= MAX_MISSED_FRAMES]

        for i, det in enumerate(detections):
            if matched[i]:
                continue
            bw = det["box"][2] - det["box"][0]
            dist = compute_distance(det["box"][3], bw, det["class_name"], self.intr)
            t = Track(
                track_id=self._next_id, class_name=det["class_name"],
                last_box=det["box"], last_distance=dist, last_timestamp=timestamp,
            )
            self._next_id += 1
            t.kf_d, t.kf_v = dist, 0.0
            t.kf_initialized = dist > 0
            self.tracks.append(t)


# ============================================================================
# Drawing
# ============================================================================
def draw_overlay(frame: np.ndarray, track: Track):
    color = CLASS_COLOR.get(track.class_name, DEFAULT_COLOR)
    x1, y1, x2, y2 = [int(round(v)) for v in track.last_box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)

    label = f"{track.class_name.capitalize()}  {track.last_distance:.1f}m  {abs(track.smoothed_speed):.1f}km/h" \
        if track.last_distance > 0 else f"{track.class_name.capitalize()}  --"

    font, scale, thickness = cv2.FONT_HERSHEY_DUPLEX, 0.85, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)

    bg_tl = (x1, max(0, y1 - th - 18))
    bg_br = (x1 + tw + 18, y1)
    h, w = frame.shape[:2]
    rx1, ry1 = max(0, bg_tl[0]), max(0, bg_tl[1])
    rx2, ry2 = min(w, bg_br[0]), min(h, bg_br[1])
    if rx2 > rx1 and ry2 > ry1:
        roi = frame[ry1:ry2, rx1:rx2]
        overlay = np.full_like(roi, color, dtype=np.uint8)
        cv2.addWeighted(overlay, 0.75, roi, 0.25, 0, dst=roi)
    cv2.rectangle(frame, bg_tl, bg_br, color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x1 + 9, y1 - 9), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def apply_colorful_optical_flow(frame: np.ndarray, flow: np.ndarray):
    """
    1) Full-frame HSV color wash: hue = motion direction, value = magnitude.
    2) Dense grid of thick arrows, individually colored by DIRECTION (full
       rainbow hue) and brightened by local MAGNITUDE -- gives varied,
       "interactive" colors instead of a flat single-axis gradient.
    """
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    max_mag = max(float(mag.max()), 1e-3)

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)          # OpenCV hue range 0-179
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.addWeighted(frame, 1.0 - FLOW_HSV_BLEND_ALPHA, flow_bgr, FLOW_HSV_BLEND_ALPHA, 0, dst=frame)

    h, w = flow.shape[:2]
    for y in range(0, h, ARROW_GRID_STEP):
        for x in range(0, w, ARROW_GRID_STEP):
            dx, dy = flow[y, x]
            m = math.hypot(dx, dy)
            if m < MOTION_THRESHOLD:
                continue
            start = (x, y)
            end = (int(round(x + dx * ARROW_LENGTH_MULTIPLIER)), int(round(y + dy * ARROW_LENGTH_MULTIPLIER)))
            angle_deg = math.degrees(math.atan2(dy, dx)) % 360.0
            value = min(0.35 + 0.65 * (m / max_mag), 1.0)
            color = _hsv_to_bgr(angle_deg / 60.0, 1.0, value)
            cv2.arrowedLine(frame, start, end, color, ARROW_THICKNESS, cv2.LINE_AA, tipLength=0.35)


def _hsv_to_bgr(h: float, s: float, v: float) -> Tuple[int, int, int]:
    i = int(h) % 6
    f = h - int(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    u = v * (1 - s * (1 - f))
    r, g, b = [(v, u, p), (q, v, p), (p, v, u), (p, q, v), (u, p, v), (v, p, q)][i]
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


# ============================================================================
# Pipeline stages (threads)
# ============================================================================
SENTINEL = None  # signals "no more items" through a queue


def capture_worker(source, track_in_q: "queue.Queue", flow_in_q: "queue.Queue"):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[capture] Could not open source: {source}", file=sys.stderr)
        track_in_q.put(SENTINEL)
        flow_in_q.put(SENTINEL)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:  # NaN-safe check
        fps = 25.0
    dt, ts, idx = 1.0 / fps, 0.0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        track_in_q.put({"frame": frame.copy(), "index": idx, "timestamp": ts})
        flow_in_q.put({"frame": frame.copy(), "index": idx, "timestamp": ts})
        idx += 1
        ts += dt
        if idx % 50 == 0:
            print(f"[capture] read {idx} frames...")

    cap.release()
    track_in_q.put(SENTINEL)
    flow_in_q.put(SENTINEL)
    print(f"[capture] finished, total frames: {idx}")


def tracking_worker(model_path: str, track_in_q: "queue.Queue", track_out_q: "queue.Queue", intr: CameraIntrinsics):
    model = YOLO(model_path)
    tracker = VehicleTracker(intr)

    while True:
        item = track_in_q.get()
        if item is SENTINEL:
            break
        frame, ts = item["frame"], item["timestamp"]

        results = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                                 classes=VEHICLE_CLASS_IDS, verbose=False)[0]
        detections = []
        names = results.names
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            detections.append({"box": (x1, y1, x2, y2), "class_name": names[cls_id]})

        tracker.update(detections, ts)
        for t in tracker.tracks:
            if t.missed_frames == 0:
                draw_overlay(frame, t)

        item["frame"] = frame
        track_out_q.put(item)

    track_out_q.put(SENTINEL)
    print("[tracking] finished.")


def flow_worker(flow_in_q: "queue.Queue", flow_out_q: "queue.Queue"):
    gray_prev = None
    while True:
        item = flow_in_q.get()
        if item is SENTINEL:
            break
        frame = item["frame"]
        gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray_prev is not None:
            flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_next, None, 0.5, 1, 5, 2, 5, 1.1, 0)
            apply_colorful_optical_flow(frame, flow)
        gray_prev = gray_next
        item["frame"] = frame
        flow_out_q.put(item)

    flow_out_q.put(SENTINEL)
    print("[flow] finished.")


def writer_worker(track_out_q: "queue.Queue", flow_out_q: "queue.Queue",
                   width: int, height: int, fps: float, frames_dir: str):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    track_writer = cv2.VideoWriter("output_tracking_calibrated.avi", fourcc, fps, (width, height))
    flow_writer = cv2.VideoWriter("output_optical_flow_arrows.avi", fourcc, fps, (width, height))

    track_open, flow_open = True, True
    while track_open or flow_open:
        if track_open:
            try:
                item = track_out_q.get(timeout=0.01)
                if item is SENTINEL:
                    track_open = False
                else:
                    track_writer.write(item["frame"])
                    cv2.imwrite(os.path.join(frames_dir, f"frame_{item['index']:04d}.jpg"), item["frame"])
            except queue.Empty:
                pass
        if flow_open:
            try:
                item = flow_out_q.get(timeout=0.01)
                if item is SENTINEL:
                    flow_open = False
                else:
                    flow_writer.write(item["frame"])
            except queue.Empty:
                pass

    track_writer.release()
    flow_writer.release()
    print("[writer] finished, both videos closed.")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-threaded dashcam detection+tracking+optical-flow pipeline")
    parser.add_argument("model", help="Path to YOLO model (.pt or .onnx)")
    parser.add_argument("video", help="Path to input video file, or camera index e.g. 0")
    parser.add_argument("--hfov", type=float, default=DASHCAM_HFOV_DEG_DEFAULT,
                         help="Dashcam horizontal FOV in degrees (default: %(default)s)")
    parser.add_argument("--cam-height", type=float, default=CAM_MOUNT_HEIGHT_M_DEFAULT,
                         help="Camera mount height in meters (default: %(default)s)")
    parser.add_argument("--frames-dir", default="combined_output_frames",
                         help="Directory to save tracking-annotated frame jpgs")
    args = parser.parse_args()

    source = int(args.video) if args.video.isdigit() else args.video

    probe = cv2.VideoCapture(source)
    if not probe.isOpened():
        print(f"Could not open input to read properties: {source}", file=sys.stderr)
        sys.exit(1)
    frame_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = probe.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 25.0
    probe.release()

    intr = init_camera_intrinsics(frame_w, frame_h, args.hfov, args.cam_height)
    os.makedirs(args.frames_dir, exist_ok=True)

    track_in_q: "queue.Queue" = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    flow_in_q: "queue.Queue" = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    track_out_q: "queue.Queue" = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    flow_out_q: "queue.Queue" = queue.Queue(maxsize=MAX_QUEUE_SIZE)

    print(">> 4-stage pipeline starting: capture | tracking(inference) | optical-flow | writer")

    threads = [
        threading.Thread(target=capture_worker, args=(source, track_in_q, flow_in_q), daemon=True),
        threading.Thread(target=tracking_worker, args=(args.model, track_in_q, track_out_q, intr), daemon=True),
        threading.Thread(target=flow_worker, args=(flow_in_q, flow_out_q), daemon=True),
        threading.Thread(target=writer_worker, args=(track_out_q, flow_out_q, frame_w, frame_h, fps, args.frames_dir)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n[Success] Pipeline completed: output_tracking_calibrated.avi, "
          f"output_optical_flow_arrows.avi, and frames in {args.frames_dir}/")


if __name__ == "__main__":
    main()
