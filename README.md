# Dashcam Vision Pipeline

Real-time vehicle detection, multi-object tracking, distance/speed estimation,
and dense optical-flow visualization from dashcam footage — implemented as a
multithreaded pipeline in **two versions**:

| | Detector | Language | Status |
|---|---|---|---|
| [`python/`](./python) | YOLOv8 (ultralytics) | Python | Actively developed |
| [`cpp/`](./cpp) | SSD MobileNetV2 (OpenCV DNN) | C++17 | Stable, legacy |

Both versions share the same core design:

- **Detection** filtered to vehicles (car / bus / truck), colorful per-class overlay
- **Tracking**: IOU-based multi-object tracker with a 1D (distance, velocity) Kalman filter per track
- **Distance estimation** fuses two independent geometric cues instead of one fragile assumption:
  1. Ground-plane geometry (camera mount height + focal length)
  2. Known-object-size geometry (average real-world vehicle width)

  Includes near-horizon and outlier guards so a single noisy detection can't
  spike into an implausible reported speed.
- **Dense optical flow**: Farneback flow rendered as a full-frame colorful
  motion field (direction → hue, magnitude → brightness) plus direction-colored arrows
- **Multithreaded 4-stage pipeline**: `capture → tracking(inference) → optical-flow → writer`,
  each stage decoupled through its own queue so no single slow stage stalls the others

---

## Python version (YOLOv8) — recommended

### Setup

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **CPU-only machines**: `ultralytics` pulls in `torch` by default, which may
> grab a large CUDA build even without a GPU. Install the CPU-only wheel
> first to avoid that:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### Run

```bash
python3 dashcam_pipeline.py yolov8n.pt /path/to/video.mp4
```

`yolov8n.pt` is downloaded automatically by `ultralytics` on first run if not
already present. A custom-trained `.pt` or an exported `.onnx` also works —
just pass its path.

Optional flags:

```bash
python3 dashcam_pipeline.py yolov8n.pt video.mp4 --hfov 110 --cam-height 1.4
```

| Flag | Description | Default |
|---|---|---|
| `--hfov` | Dashcam horizontal field of view (degrees) | `100.0` |
| `--cam-height` | Camera mount height above ground (meters) | `1.3` |
| `--frames-dir` | Directory to save annotated tracking frames | `combined_output_frames` |

### Outputs

- `output_tracking_calibrated.avi` — detection + tracking overlay
- `output_optical_flow_arrows.avi` — colorful optical-flow visualization
- `combined_output_frames/` — per-frame tracking-annotated JPGs

---

## C++ version (SSD MobileNetV2)

### Setup (Ubuntu/Debian)

The setup script installs system dependencies, downloads the SSD
MobileNetV2 COCO model, and builds the binary with CMake:

```bash
cd cpp
chmod +x setup_dnn.sh
./setup_dnn.sh
```

This creates `cpp/dashcam_dnn_project/` containing:
- `models/frozen_inference_graph.pb` and `models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt`
- `build/dashcam_dnn` — the compiled binary

### Run

```bash
./dashcam_dnn_project/build/dashcam_dnn \
    dashcam_dnn_project/models/frozen_inference_graph.pb \
    dashcam_dnn_project/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt \
    /path/to/video.mp4
```

Live camera (index 0) instead of a video file:

```bash
./dashcam_dnn_project/build/dashcam_dnn \
    dashcam_dnn_project/models/frozen_inference_graph.pb \
    dashcam_dnn_project/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt \
    0
```

### Manual build (without the setup script)

```bash
sudo apt-get install -y build-essential cmake pkg-config libopencv-dev wget
g++ dashcam_dnn.cpp -o dashcam_dnn -O2 -std=c++17 \
    $(pkg-config --cflags --libs opencv4) -lpthread
```

### Outputs

Same as the Python version: `output_tracking_calibrated.avi`,
`output_optical_flow_arrows.avi`, `combined_output_frames/`.

---

## Notes on calibration

Focal length is derived at runtime from the actual video resolution and an
assumed horizontal FOV (generic dashcams are typically 90–110°) — in both
versions. For best accuracy:

- Pass your camera's real spec-sheet HFOV (`--hfov` in Python; edit
  `DASHCAM_HFOV_DEG` in the C++ source), or
- Calibrate properly with `cv2.calibrateCamera` / `cv::calibrateCamera` and a
  checkerboard, and substitute the resulting focal length directly.

## Which version should I use?

- **Python**: better detector (YOLOv8 vs. a 2018-era SSD model), easier to
  swap in a custom-trained model, simpler to modify.
- **C++**: lower runtime overhead once built, no Python/torch environment
  needed on the deployment machine, no NNPACK/GIL considerations.

## License

MIT 
