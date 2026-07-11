// ============================================================================
// Multi-Threaded Dashboard-Cam Suite v2
// - Colorful per-class detection boxes (thick, filled label strips)
// - Colorful dense optical-flow visualization (HSV blend + magnitude-graded arrows)
// - True 4-stage pipeline: Capture -> Inference(track) -> OpticalFlow -> Writer
//   Each stage is its own std::thread with its OWN queue (own mutex+cv), so no
//   thread ever waits while holding a lock another thread needs (no deadlock),
//   and no queue's producer/consumer pair blocks another queue's pair.
// ============================================================================

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <sys/stat.h>

// ----------------------------------------------------------------------------
// Camera / distance model
//
// Runtime-computed, not baked-in: focal length is derived from the ACTUAL
// frame size + a general-purpose dashcam horizontal FOV assumption (most
// forward-facing dashcams sit in the ~90-110 deg range; 100 deg is a
// reasonable default). Replace DASHCAM_HFOV_DEG / CAM_MOUNT_HEIGHT_M with
// your camera's real spec-sheet values, or a proper cv::calibrateCamera
// checkerboard calibration, for best accuracy.
// ----------------------------------------------------------------------------
static const float DASHCAM_HFOV_DEG   = 100.0f;
static const float CAM_MOUNT_HEIGHT_M = 1.3f;    // typical windshield mount height, average sedan

// Average real-world vehicle widths (meters) -- the "known object size" cue,
// fused with ground-plane geometry in computeDistance() below.
static const float CAR_WIDTH_M   = 1.80f;
static const float BUS_WIDTH_M   = 2.55f;
static const float TRUCK_WIDTH_M = 2.50f;

// Populated once in main() from the actual video's resolution.
float FX = 0.0f, FY = 0.0f, CX = 0.0f, CY = 0.0f;

void initCameraIntrinsics(int frameWidth, int frameHeight) {
    float halfFovRad = (DASHCAM_HFOV_DEG * 0.5f) * static_cast<float>(CV_PI) / 180.0f;
    FX = (frameWidth / 2.0f) / std::tan(halfFovRad);
    FY = FX; // assumes square pixels, true for essentially all consumer dashcams
    CX = frameWidth / 2.0f;
    CY = frameHeight / 2.0f; // assumes a level-mounted dashcam (horizon at image center)
    std::cout << "[calib] frame=" << frameWidth << "x" << frameHeight
              << " HFOV=" << DASHCAM_HFOV_DEG << "deg -> fx=fy=" << FX
              << "px, camHeight=" << CAM_MOUNT_HEIGHT_M << "m" << std::endl;
}

static const float CONF_THRESHOLD    = 0.45f;
static const float IOU_THRESHOLD     = 0.25f;
static const int   MAX_MISSED_FRAMES = 8;
static const std::vector<int> VEHICLE_CLASS_IDS = {3, 6, 8};

// ----------------------------------------------------------------------------
// Optical flow visualization params
// ----------------------------------------------------------------------------
static const int    ARROW_GRID_STEP = 22;
static const double MOTION_THRESHOLD = 2.0;
static const float  FLOW_HSV_BLEND_ALPHA = 0.35f; // how strong the color-flow overlay is
static const int    ARROW_THICKNESS = 3;

static const size_t MAX_QUEUE_SIZE = 4;

struct FrameData {
    cv::Mat frame;
    int index = 0;
    double timestamp = 0.0;
};

// ----------------------------------------------------------------------------
// Generic bounded, thread-safe queue. Each instance owns its OWN mutex and
// its OWN pair of condition variables. Because every queue is independent,
// no thread ever needs to hold two different queues' locks at once while
// waiting -- this is what removes the deadlock class of bug entirely.
// ----------------------------------------------------------------------------
class FrameQueue {
public:
    explicit FrameQueue(size_t maxSize) : maxSize_(maxSize) {}

    // Blocking push: waits until there's room.
    void push(FrameData data) {
        std::unique_lock<std::mutex> lock(mtx_);
        cvNotFull_.wait(lock, [&] { return queue_.size() < maxSize_; });
        queue_.push(std::move(data));
        lock.unlock();
        cvNotEmpty_.notify_one();
    }

    // Blocking pop: waits until an item is available OR producerDone is set
    // and the queue is empty (returns false in that case -> stream ended).
    bool pop(FrameData& out, std::atomic<bool>& producerDone) {
        std::unique_lock<std::mutex> lock(mtx_);
        cvNotEmpty_.wait(lock, [&] { return !queue_.empty() || producerDone.load(); });
        if (queue_.empty()) return false;
        out = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        cvNotFull_.notify_one();
        return true;
    }

    // Non-blocking-ish pop with timeout, used by the single writer thread so
    // it can service TWO queues (tracking-out and flow-out) without ever
    // permanently blocking on just one of them.
    bool tryPopFor(FrameData& out, std::atomic<bool>& producerDone, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (!cvNotEmpty_.wait_for(lock, timeout, [&] { return !queue_.empty() || producerDone.load(); }))
            return false; // timed out, stream still open, nothing ready yet
        if (queue_.empty()) return false; // producer finished and drained
        out = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        cvNotFull_.notify_one();
        return true;
    }

    bool finishedAndEmpty(std::atomic<bool>& producerDone) {
        std::lock_guard<std::mutex> lock(mtx_);
        return producerDone.load() && queue_.empty();
    }

private:
    std::queue<FrameData> queue_;
    std::mutex mtx_;
    std::condition_variable cvNotFull_;
    std::condition_variable cvNotEmpty_;
    size_t maxSize_;
};

// Pipeline queues
FrameQueue trackInQueue(MAX_QUEUE_SIZE);   // capture -> tracking(inference)
FrameQueue flowInQueue(MAX_QUEUE_SIZE);    // capture -> optical flow
FrameQueue trackOutQueue(MAX_QUEUE_SIZE);  // tracking -> writer
FrameQueue flowOutQueue(MAX_QUEUE_SIZE);   // flow -> writer

std::atomic<bool> captureDone(false);   // capture thread finished reading input
std::atomic<bool> trackingDone(false);  // tracking thread finished producing output
std::atomic<bool> flowDone(false);      // flow thread finished producing output

// ----------------------------------------------------------------------------
// Class colors (BGR, since OpenCV) -- distinct, saturated colors per class so
// the overlay reads as "colorful" rather than flat black/white.
// ----------------------------------------------------------------------------
cv::Scalar classColor(int classId) {
    switch (classId) {
        case 3: return cv::Scalar(70, 200, 60);    // Car    -> green
        case 6: return cv::Scalar(0, 140, 255);    // Bus    -> orange
        case 8: return cv::Scalar(50, 50, 230);    // Truck  -> red
        default: return cv::Scalar(15, 196, 241);  // fallback -> yellow
    }
}

std::string classIdToName(int classId) {
    switch (classId) {
        case 3: return "Car";
        case 6: return "Bus";
        case 8: return "Truck";
        default: return "Vehicle";
    }
}

struct BBox { float x_min, y_min, x_max, y_max; };
struct Detection { BBox box; int classId; float score; };

struct Track {
    int id; int classId; BBox lastBox; float lastDistance; double lastTimestamp;
    float rawSpeed; float smoothedSpeed; int missedFrames; bool hasSpeed;
    float kf_d = 0.0f; float kf_v = 0.0f;
    float P00 = 1000.0f, P01 = 0.0f, P10 = 0.0f, P11 = 1000.0f;
    bool kfInitialized = false;
};

float avgVehicleWidthM(int classId) {
    switch (classId) {
        case 6: return BUS_WIDTH_M;
        case 8: return TRUCK_WIDTH_M;
        default: return CAR_WIDTH_M; // car and any unrecognized class fall back to sedan width
    }
}

// Fuses ground-plane geometry (flat-road + camera-height assumption) with
// known-object-size geometry (average real vehicle width) -- more robust
// than either cue alone, and avoids the arbitrary fudge multiplier a single
// uncalibrated cue would otherwise need. Guards against the near-horizon
// instability where a tiny pixel_disp denominator makes ground-plane
// distance explode from a 1-2px detection jitter.
float computeDistance(float v_bottom_px, float bboxWidthPx, int classId) {
    float widthDist = -1.0f;
    if (bboxWidthPx > 1.0f) {
        widthDist = (avgVehicleWidthM(classId) * FX) / bboxWidthPx;
    }

    float pixelDisplacement = v_bottom_px - CY;
    const float MIN_PIXEL_DISP = 8.0f; // below this, ground-plane cue is too noisy to trust
    float groundDist = -1.0f;
    if (pixelDisplacement > MIN_PIXEL_DISP) {
        groundDist = (CAM_MOUNT_HEIGHT_M * FY) / pixelDisplacement;
    }

    float dist;
    if (groundDist > 0.0f && widthDist > 0.0f) dist = 0.5f * (groundDist + widthDist);
    else if (widthDist > 0.0f) dist = widthDist;
    else if (groundDist > 0.0f) dist = groundDist;
    else dist = 150.0f; // both cues degenerate -> generic "far away" placeholder

    return std::min(dist, 200.0f); // clamp: beyond this, dashcam-scale detections are unreliable anyway
}

static const float PROCESS_NOISE     = 3.0f;
static const float MEASUREMENT_NOISE = 0.6f;

void kalmanPredictUpdate(Track& track, float measuredDistance, double dt) {
    if (!track.kfInitialized) {
        track.kf_d = measuredDistance; track.kf_v = 0.0f;
        track.P00 = 1000.0f; track.P01 = 0.0f; track.P10 = 0.0f; track.P11 = 1000.0f;
        track.kfInitialized = (measuredDistance > 0.0f);
        return;
    }
    float d_pred = track.kf_d + track.kf_v * (float)dt;
    float v_pred = track.kf_v;
    float P00_pred = track.P00 + (float)dt * track.P10 + (float)dt * track.P01 + (float)dt * (float)dt * track.P11;
    float P01_pred = track.P01 + (float)dt * track.P11;
    float P10_pred = track.P10 + (float)dt * track.P11;
    float P11_pred = track.P11;
    float dt3 = (float)(dt * dt * dt) / 3.0f; float dt2 = (float)(dt * dt) / 2.0f;
    P00_pred += PROCESS_NOISE * dt3; P01_pred += PROCESS_NOISE * dt2; P10_pred += PROCESS_NOISE * dt2; P11_pred += PROCESS_NOISE * (float)dt;

    if (measuredDistance > 0.0f) {
        float y = measuredDistance - d_pred; float S = P00_pred + MEASUREMENT_NOISE;
        float K0 = P00_pred / S; float K1 = P10_pred / S;
        track.kf_d = d_pred + K0 * y; track.kf_v = v_pred + K1 * y;
        track.P00 = (1.0f - K0) * P00_pred; track.P01 = (1.0f - K0) * P01_pred;
        track.P10 = P10_pred - K1 * P00_pred; track.P11 = P11_pred - K1 * P01_pred;
    } else {
        track.kf_d = d_pred; track.kf_v = v_pred;
        track.P00 = P00_pred; track.P01 = P01_pred; track.P10 = P10_pred; track.P11 = P11_pred;
    }
}

float computeIOU(const BBox& a, const BBox& b) {
    float x1 = std::max(a.x_min, b.x_min); float y1 = std::max(a.y_min, b.y_min);
    float x2 = std::min(a.x_max, b.x_max); float y2 = std::min(a.y_max, b.y_max);
    float interArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float areaA = std::max(0.0f, a.x_max - a.x_min) * std::max(0.0f, a.y_max - a.y_min);
    float areaB = std::max(0.0f, b.x_max - b.x_min) * std::max(0.0f, b.y_max - b.y_min);
    float unionArea = areaA + areaB - interArea;
    if (unionArea <= 0.0f) return 0.0f;
    return interArea / unionArea;
}

class VehicleTracker {
public:
    std::vector<Track> tracks;
    void update(std::vector<Detection>& detections, double timestamp) {
        std::vector<bool> matched(detections.size(), false);
        for (auto& track : tracks) {
            int bestMatch = -1; float bestIOU = IOU_THRESHOLD;
            for (size_t i = 0; i < detections.size(); ++i) {
                if (matched[i]) continue;
                float iou = computeIOU(track.lastBox, detections[i].box);
                if (iou > bestIOU) { bestIOU = iou; bestMatch = static_cast<int>(i); }
            }
            if (bestMatch != -1) {
                float matchedWidth = detections[bestMatch].box.x_max - detections[bestMatch].box.x_min;
                float newDist = computeDistance(detections[bestMatch].box.y_max, matchedWidth, detections[bestMatch].classId);
                double dt = timestamp - track.lastTimestamp;
                if (dt > 1e-3) {
                    kalmanPredictUpdate(track, newDist, dt);
                    float speed_kmh = (-track.kf_v) * 3.6f;
                    track.smoothedSpeed = speed_kmh;
                    track.rawSpeed = (newDist > 0.0f && track.lastDistance > 0.0f) ? ((track.lastDistance - newDist) / (float)dt) * 3.6f : track.rawSpeed;
                    track.hasSpeed = true;
                }
                if (track.kfInitialized) track.lastDistance = track.kf_d;
                else if (newDist > 0.0f) track.lastDistance = newDist;
                track.classId = detections[bestMatch].classId; track.lastBox = detections[bestMatch].box;
                track.lastTimestamp = timestamp; track.missedFrames = 0; matched[bestMatch] = true;
            } else { track.missedFrames++; }
        }
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](const Track& t) { return t.missedFrames > MAX_MISSED_FRAMES; }), tracks.end());
        static int nextID = 0;
        for (size_t i = 0; i < detections.size(); ++i) {
            if (matched[i]) continue;
            Track t; t.id = nextID++; t.classId = detections[i].classId; t.lastBox = detections[i].box;
            float newWidth = detections[i].box.x_max - detections[i].box.x_min;
            t.lastDistance = computeDistance(detections[i].box.y_max, newWidth, detections[i].classId); t.lastTimestamp = timestamp;
            t.rawSpeed = 0.0f; t.smoothedSpeed = 0.0f; t.missedFrames = 0; t.hasSpeed = false;
            t.kf_d = t.lastDistance; t.kf_v = 0.0f; t.P00 = 1000.0f; t.P01 = 0.0f; t.P10 = 0.0f; t.P11 = 1000.0f;
            t.kfInitialized = (t.lastDistance > 0.0f); tracks.push_back(t);
        }
    }
};

class VehicleDetector {
public:
    bool load(const std::string& pbPath, const std::string& pbtxtPath) {
        net_ = cv::dnn::readNetFromTensorflow(pbPath, pbtxtPath);
        if (net_.empty()) return false;
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        return true;
    }
    std::vector<Detection> infer(const cv::Mat& frame) {
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(0, 0, 0), true, false);
        net_.setInput(blob); cv::Mat output = net_.forward();
        cv::Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());
        int frameW = frame.cols; int frameH = frame.rows;
        std::vector<Detection> results;
        for (int i = 0; i < detectionMat.rows; ++i) {
            float score = detectionMat.at<float>(i, 2);
            if (score < CONF_THRESHOLD) continue;
            int classId = static_cast<int>(detectionMat.at<float>(i, 1));
            if (std::find(VEHICLE_CLASS_IDS.begin(), VEHICLE_CLASS_IDS.end(), classId) == VEHICLE_CLASS_IDS.end()) continue;
            float left = detectionMat.at<float>(i, 3) * frameW; float top = detectionMat.at<float>(i, 4) * frameH;
            float right = detectionMat.at<float>(i, 5) * frameW; float bottom = detectionMat.at<float>(i, 6) * frameH;
            Detection d; d.classId = classId; d.score = score; d.box = { left, top, right, bottom };
            results.push_back(d);
        }
        return results;
    }
private:
    cv::dnn::Net net_;
};

// ----------------------------------------------------------------------------
// Colorful overlay: thick, class-colored box + filled label strip (alpha
// blended so it isn't a flat opaque block) + white bold text on top.
// ----------------------------------------------------------------------------
void drawOverlay(cv::Mat& frame, const Track& t) {
    cv::Scalar color = classColor(t.classId);
    cv::Point tl(cvRound(t.lastBox.x_min), cvRound(t.lastBox.y_min));
    cv::Point br(cvRound(t.lastBox.x_max), cvRound(t.lastBox.y_max));

    cv::rectangle(frame, tl, br, color, 4, cv::LINE_AA);

    std::string name = classIdToName(t.classId);
    char text[160];
    if (t.lastDistance > 0.0f) {
        snprintf(text, sizeof(text), "%s  %.1fm  %.1fkm/h", name.c_str(), t.lastDistance, std::abs(t.smoothedSpeed));
    } else {
        snprintf(text, sizeof(text), "%s  --", name.c_str());
    }

    double fontScale = 0.85; int thickness = 2; int fontFace = cv::FONT_HERSHEY_DUPLEX;
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

    cv::Point bgTopLeft(tl.x, std::max(0, tl.y - textSize.height - 18));
    cv::Point bgBottomRight(tl.x + textSize.width + 18, tl.y);

    // Semi-transparent filled label strip in the class color (colorful, not
    // a flat white/black bar) so it reads clearly against any background.
    cv::Mat roi = frame(cv::Rect(bgTopLeft, bgBottomRight) & cv::Rect(0, 0, frame.cols, frame.rows));
    if (!roi.empty()) {
        cv::Mat colorLayer(roi.size(), roi.type(), color);
        cv::addWeighted(colorLayer, 0.75, roi, 0.25, 0, roi);
    }
    cv::rectangle(frame, bgTopLeft, bgBottomRight, color, 2, cv::LINE_AA);

    cv::putText(frame, text, cv::Point(tl.x + 9, tl.y - 9), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}

// ----------------------------------------------------------------------------
// Colorful dense-flow visualization:
//  1) Convert flow (dx,dy) -> HSV (hue=direction, value=magnitude), blend
//     over the frame so overall motion is visible as color at a glance.
//  2) Draw thick arrows on a sparse grid, colored by local magnitude on a
//     blue -> green -> red gradient, so strong motion pops out.
// ----------------------------------------------------------------------------
cv::Scalar magnitudeToColor(float mag, float maxMag) {
    float t = maxMag > 1e-3f ? std::min(mag / maxMag, 1.0f) : 0.0f;
    if (t < 0.5f) {
        float k = t / 0.5f; // blue -> green
        return cv::Scalar(255.0f * (1.0f - k), 255.0f * k, 0.0f);
    } else {
        float k = (t - 0.5f) / 0.5f; // green -> red
        return cv::Scalar(0.0f, 255.0f * (1.0f - k), 255.0f * k);
    }
}

void applyColorfulOpticalFlow(cv::Mat& displayFrame, const cv::Mat& flow) {
    cv::Mat flowParts[2];
    cv::split(flow, flowParts);
    cv::Mat magnitude, angle;
    cv::cartToPolar(flowParts[0], flowParts[1], magnitude, angle, true);

    double maxMagD = 0.0;
    cv::minMaxLoc(magnitude, nullptr, &maxMagD);
    float maxMag = static_cast<float>(std::max(maxMagD, 1e-3));

    // --- (1) full-frame HSV color wash for overall motion feel ---
    cv::Mat magNorm, angleNorm, sat, hsvMerged, flowColorBGR;
    magnitude.convertTo(magNorm, CV_8U, 255.0 / maxMag);
    angle.convertTo(angleNorm, CV_8U, 255.0 / 360.0);
    sat = cv::Mat(angle.size(), CV_8U, cv::Scalar(255));
    std::vector<cv::Mat> hsvCh = { angleNorm, sat, magNorm };
    cv::merge(hsvCh, hsvMerged);
    cv::cvtColor(hsvMerged, flowColorBGR, cv::COLOR_HSV2BGR);
    cv::addWeighted(displayFrame, 1.0f - FLOW_HSV_BLEND_ALPHA, flowColorBGR, FLOW_HSV_BLEND_ALPHA, 0.0, displayFrame);

    // --- (2) thick, magnitude-colored arrows on a sparse grid ---
    for (int y = 0; y < flow.rows; y += ARROW_GRID_STEP) {
        for (int x = 0; x < flow.cols; x += ARROW_GRID_STEP) {
            const cv::Vec2f& f = flow.at<cv::Vec2f>(y, x);
            float dx = f[0], dy = f[1];
            float mag = std::sqrt(dx * dx + dy * dy);
            if (mag < MOTION_THRESHOLD) continue;
            cv::Point start(x, y);
            cv::Point end(cvRound(x + dx * 1.6f), cvRound(y + dy * 1.6f));
            cv::Scalar color = magnitudeToColor(mag, maxMag);
            cv::arrowedLine(displayFrame, start, end, color, ARROW_THICKNESS, cv::LINE_AA, 0, 0.35);
        }
    }
}

// ---------------------------------------------------------------------------
// STAGE 1: CAPTURE -- the only thread that touches the input source.
// Pushes each frame (as two independent deep copies) into trackInQueue and
// flowInQueue for the two downstream compute stages.
// ---------------------------------------------------------------------------
void captureWorker(const std::string& source) {
    cv::VideoCapture cap(source);
    if (!cap.isOpened()) {
        std::cerr << "[capture] Could not open source: " << source << std::endl;
        captureDone = true;
        return;
    }
    double inputFps = cap.get(cv::CAP_PROP_FPS);
    if (inputFps <= 0.0 || std::isnan(inputFps)) inputFps = 25.0;
    double dt = 1.0 / inputFps;
    double ts = 0.0;
    int idx = 0;

    cv::Mat frame;
    while (cap.read(frame)) {
        FrameData dTrack, dFlow;
        frame.copyTo(dTrack.frame);   // independent deep copy per queue --
        frame.copyTo(dFlow.frame);    // avoids any cross-thread pixel-buffer race
        dTrack.index = dFlow.index = idx++;
        dTrack.timestamp = dFlow.timestamp = ts;
        ts += dt;

        trackInQueue.push(std::move(dTrack));
        flowInQueue.push(std::move(dFlow));

        if (idx % 50 == 0) std::cout << "[capture] read " << idx << " frames..." << std::endl;
    }
    cap.release();
    captureDone = true; // signals both trackInQueue and flowInQueue consumers
    std::cout << "[capture] finished, total frames: " << idx << std::endl;
}

// ---------------------------------------------------------------------------
// STAGE 2: INFERENCE / TRACKING -- pure compute, no disk I/O here at all.
// Pops from trackInQueue, runs detector+tracker, draws colorful overlay,
// pushes the annotated frame to trackOutQueue for the writer thread.
// ---------------------------------------------------------------------------
void trackingWorker(std::string pbPath, std::string pbtxtPath) {
    VehicleDetector detector;
    if (!detector.load(pbPath, pbtxtPath)) {
        std::cerr << "[tracking] Failed to load model." << std::endl;
        trackingDone = true;
        return;
    }
    VehicleTracker tracker;

    FrameData data;
    while (trackInQueue.pop(data, captureDone)) {
        std::vector<Detection> detections = detector.infer(data.frame);
        tracker.update(detections, data.timestamp);
        for (const auto& t : tracker.tracks) {
            if (t.missedFrames == 0) drawOverlay(data.frame, t);
        }
        trackOutQueue.push(std::move(data));
    }
    trackingDone = true;
    std::cout << "[tracking] finished." << std::endl;
}

// ---------------------------------------------------------------------------
// STAGE 3: OPTICAL FLOW -- pure compute, no disk I/O here at all.
// ---------------------------------------------------------------------------
void flowWorker() {
    cv::Mat grayPrev, grayNext, flow;

    FrameData data;
    while (flowInQueue.pop(data, captureDone)) {
        cv::cvtColor(data.frame, grayNext, cv::COLOR_BGR2GRAY);
        if (!grayPrev.empty()) {
            cv::calcOpticalFlowFarneback(grayPrev, grayNext, flow, 0.5, 1, 5, 2, 5, 1.1, 0);
            applyColorfulOpticalFlow(data.frame, flow);
        }
        grayNext.copyTo(grayPrev);
        flowOutQueue.push(std::move(data));
    }
    flowDone = true;
    std::cout << "[flow] finished." << std::endl;
}

// ---------------------------------------------------------------------------
// STAGE 4: WRITER -- the ONLY thread that touches output files. Services
// BOTH output queues (interleaved with short timeouts) so neither stream
// stalls behind the other, and writes both videos + tracking frame jpgs.
// ---------------------------------------------------------------------------
void writerWorker(int width, int height, double fps, const std::string& folderName) {
    cv::VideoWriter trackingWriter("output_tracking_calibrated.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height), true);
    cv::VideoWriter flowWriter("output_optical_flow_arrows.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height), true);

    bool trackOpen = true, flowOpen = true;
    const std::chrono::milliseconds pollTimeout(10);

    while (trackOpen || flowOpen) {
        if (trackOpen) {
            FrameData d;
            if (trackOutQueue.tryPopFor(d, trackingDone, pollTimeout)) {
                if (trackingWriter.isOpened()) trackingWriter.write(d.frame);
                char filename[256];
                snprintf(filename, sizeof(filename), "%s/frame_%04d.jpg", folderName.c_str(), d.index);
                cv::imwrite(filename, d.frame);
            } else if (trackOutQueue.finishedAndEmpty(trackingDone)) {
                trackOpen = false;
            }
        }
        if (flowOpen) {
            FrameData d;
            if (flowOutQueue.tryPopFor(d, flowDone, pollTimeout)) {
                if (flowWriter.isOpened()) flowWriter.write(d.frame);
            } else if (flowOutQueue.finishedAndEmpty(flowDone)) {
                flowOpen = false;
            }
        }
    }

    if (trackingWriter.isOpened()) trackingWriter.release();
    if (flowWriter.isOpened()) flowWriter.release();
    std::cout << "[writer] finished, both videos closed." << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model.pb> <graph.pbtxt> <video_path>" << std::endl;
        return 1;
    }
    std::string pbPath = argv[1]; std::string pbtxtPath = argv[2]; std::string source = argv[3];

    // Budget OpenCV's internal parallel-for pool at 2 threads: this is the
    // pool the DNN forward pass (heaviest op) draws on internally, so
    // inference effectively gets up to 2 cores' worth of internal work.
    // Our own pipeline stages (capture/tracking/flow/writer) are separate
    // std::threads on top of this and run truly concurrently now that I/O
    // is isolated in the writer thread and no queue blocks another.
    cv::setNumThreads(2);

    cv::VideoCapture probe(source);
    if (!probe.isOpened()) {
        std::cerr << "Could not open input to read properties: " << source << std::endl;
        return 1;
    }
    int frameWidth  = static_cast<int>(probe.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(probe.get(cv::CAP_PROP_FRAME_HEIGHT));
    double inputFps = probe.get(cv::CAP_PROP_FPS);
    if (inputFps <= 0.0 || std::isnan(inputFps)) inputFps = 25.0;
    probe.release();

    // Must run BEFORE any worker thread starts, since trackingWorker's
    // distance computation reads the global FX/FY/CX/CY set here.
    initCameraIntrinsics(frameWidth, frameHeight);

    std::string folderName = "combined_output_frames";
    mkdir(folderName.c_str(), 0777);

    std::cout << ">> 4-stage pipeline starting: capture | tracking(inference) | optical-flow | writer" << std::endl;

    std::thread tCapture(captureWorker, source);
    std::thread tTracking(trackingWorker, pbPath, pbtxtPath);
    std::thread tFlow(flowWorker);
    std::thread tWriter(writerWorker, frameWidth, frameHeight, inputFps, folderName);

    tCapture.join();
    tTracking.join();
    tFlow.join();
    tWriter.join();

    std::cout << "\n[Success] Pipeline completed: output_tracking_calibrated.avi, "
              << "output_optical_flow_arrows.avi, and frames in " << folderName << "/" << std::endl;
    return 0;
}
