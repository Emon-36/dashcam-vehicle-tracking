#!/bin/bash
# ============================================================================
# setup_dnn.sh — Environment setup for the C++ SSD MobileNetV2 version
#
# Covers: system deps -> SSD MobileNetV2 model download (.pb + .pbtxt)
#         -> build -> run command
# ============================================================================
set -e

PROJECT_ROOT="$(pwd)/dashcam_dnn_project"
MODEL_DIR="$PROJECT_ROOT/models"
BUILD_DIR="$PROJECT_ROOT/build"

mkdir -p "$PROJECT_ROOT" "$MODEL_DIR" "$PROJECT_ROOT/src"

# ----------------------------------------------------------------------------
# 1. System dependencies
# ----------------------------------------------------------------------------
echo "==> Installing system dependencies"
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake pkg-config \
    libopencv-dev \
    wget

# ----------------------------------------------------------------------------
# 2. Download SSD MobileNetV2 COCO model files for OpenCV DNN
#    - frozen_inference_graph.pb  : TF frozen graph weights
#    - .pbtxt                     : OpenCV's text graph description
#      (OpenCV needs this second file to parse a TF frozen graph correctly;
#      it's a well-known, pre-generated file maintained in opencv_extra)
# ----------------------------------------------------------------------------
echo "==> Downloading SSD MobileNetV2 COCO model"
cd "$MODEL_DIR"

if [ ! -f frozen_inference_graph.pb ]; then
    wget -O ssd_mobilenet_v2_coco_2018_03_29.tar.gz \
        "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
    tar -xzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    cp ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb .
fi

if [ ! -f ssd_mobilenet_v2_coco_2018_03_29.pbtxt ]; then
    wget -O ssd_mobilenet_v2_coco_2018_03_29.pbtxt \
        "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
fi

echo "==> Model files ready:"
ls -la "$MODEL_DIR"

# ----------------------------------------------------------------------------
# 3. Copy source + write CMakeLists
# ----------------------------------------------------------------------------
cp dashcam_dnn.cpp "$PROJECT_ROOT/src/" 2>/dev/null || \
    echo "NOTE: copy dashcam_dnn.cpp into $PROJECT_ROOT/src/ manually if not found in current dir"

cat > "$PROJECT_ROOT/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(dashcam_dnn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)

add_executable(dashcam_dnn src/dashcam_dnn.cpp)
target_include_directories(dashcam_dnn PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(dashcam_dnn PRIVATE ${OpenCV_LIBS} pthread)
EOF

# ----------------------------------------------------------------------------
# 4. Build
# ----------------------------------------------------------------------------
echo "==> Building"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ..
cmake --build . -j"$(nproc)"

echo ""
echo "============================================================"
echo " Setup complete."
echo " Binary: $BUILD_DIR/dashcam_dnn"
echo ""
echo " Run with a video file:"
echo "   $BUILD_DIR/dashcam_dnn $MODEL_DIR/frozen_inference_graph.pb $MODEL_DIR/ssd_mobilenet_v2_coco_2018_03_29.pbtxt /path/to/video.mp4"
echo ""
echo " Run with a live camera (index 0):"
echo "   $BUILD_DIR/dashcam_dnn $MODEL_DIR/frozen_inference_graph.pb $MODEL_DIR/ssd_mobilenet_v2_coco_2018_03_29.pbtxt 0"
echo "============================================================"
