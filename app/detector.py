from ultralytics import YOLO
import cv2
import numpy as np

import os

# Use fine-tuned model if available, otherwise fall back to default
_model_path = "models/best.pt" if os.path.exists("models/best.pt") else "models/yolov8n.pt"
model       = YOLO(_model_path)
print(f"✅ Loaded model: {_model_path}")

prev_gray       = None
_speed_history  = []
_count_history  = []
_MAX_HISTORY    = 10

SPEED_THRESHOLD      = 3.5
UNIFORMITY_THRESHOLD = 0.65
DENSITY_SPIKE_RATIO  = 0.30


def detect_people(frame):
    results      = model(frame, classes=[0], verbose=False)
    person_count = 0
    boxes        = []

    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, conf))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (0, 200, 100), 1, cv2.LINE_AA)

    return frame, person_count, boxes


def get_density_label(count):
    if count <= 5:
        return "Low"
    elif count <= 15:
        return "Medium"
    else:
        return "High"


def generate_heatmap(frame, boxes):
    h, w  = frame.shape[:2]
    heat  = np.zeros((h, w), dtype=np.float32)

    for (x1, y1, x2, y2, conf) in boxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        for dy in range(-60, 61):
            for dx in range(-60, 61):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    dist = np.sqrt(dx**2 + dy**2)
                    heat[ny, nx] += max(0, 1 - dist / 60)

    if heat.max() > 0:
        heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)

    heatmap_color = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)
    blended       = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
    return blended


def detect_flow_direction(frame):
    global prev_gray
    gray           = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    direction_label = "Analysing..."

    if prev_gray is not None:
        if prev_gray.shape != gray.shape:
            prev_gray = None
        else:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                h, w                   = frame.shape[:2]
                step                   = 30
                total_dx, total_dy, count = 0, 0, 0

                for y in range(0, h, step):
                    for x in range(0, w, step):
                        fx, fy = flow[y, x]
                        mag    = np.sqrt(fx**2 + fy**2)
                        if mag > 1.5:
                            ex = int(x + fx * 3)
                            ey = int(y + fy * 3)
                            ex = max(0, min(w - 1, ex))
                            ey = max(0, min(h - 1, ey))
                            cv2.arrowedLine(frame, (x, y), (ex, ey),
                                            (0, 220, 255), 1, tipLength=0.3)
                            total_dx += fx
                            total_dy += fy
                            count    += 1

                if count > 10:
                    avg_dx = total_dx / count
                    avg_dy = total_dy / count
                    angle  = np.degrees(np.arctan2(avg_dy, avg_dx))
                    if -45 <= angle < 45:
                        direction_label = "Moving RIGHT →"
                    elif 45 <= angle < 135:
                        direction_label = "Moving DOWN ↓"
                    elif angle >= 135 or angle < -135:
                        direction_label = "Moving LEFT ←"
                    else:
                        direction_label = "Moving UP ↑"
                    speed = np.sqrt((total_dx/count)**2 + (total_dy/count)**2)
                    if speed > 4:
                        direction_label = "⚠️ RAPID " + direction_label
                else:
                    direction_label = "Crowd stationary"
            except cv2.error:
                prev_gray = None
                return frame, "Analysing..."

    prev_gray = gray.copy()
    return frame, direction_label


def detect_stampede(frame, boxes, person_count):
    global prev_gray, _speed_history, _count_history

    gray               = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    signal_speed       = 0.0
    signal_uniformity  = 0.0
    signal_density     = 0.0
    avg_speed          = 0.0
    direction_label    = "—"

    if prev_gray is not None and prev_gray.shape == gray.shape:
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            h, w    = frame.shape[:2]
            step    = 20
            fx_vals = []
            fy_vals = []
            mags    = []

            for y in range(0, h, step):
                for x in range(0, w, step):
                    fx, fy = flow[y, x]
                    mag    = np.sqrt(fx**2 + fy**2)
                    if mag > 0.5:
                        fx_vals.append(fx / mag)
                        fy_vals.append(fy / mag)
                        mags.append(mag)

            if mags:
                avg_speed = float(np.mean(mags))
                _speed_history.append(avg_speed)
                if len(_speed_history) > _MAX_HISTORY:
                    _speed_history.pop(0)

                signal_speed = min(1.0, avg_speed / (SPEED_THRESHOLD * 2))

                if fx_vals:
                    ux = float(np.mean(fx_vals))
                    uy = float(np.mean(fy_vals))
                    signal_uniformity = float(np.sqrt(ux**2 + uy**2))
                    angle = np.degrees(np.arctan2(uy, ux))
                    if -45 <= angle < 45:
                        direction_label = "→ RIGHT"
                    elif 45 <= angle < 135:
                        direction_label = "↓ DOWN"
                    elif angle >= 135 or angle < -135:
                        direction_label = "← LEFT"
                    else:
                        direction_label = "↑ UP"
        except cv2.error:
            pass

    prev_gray = gray.copy()

    _count_history.append(person_count)
    if len(_count_history) > _MAX_HISTORY:
        _count_history.pop(0)

    if len(_count_history) >= 5:
        past_avg    = float(np.mean(_count_history[:-3]))
        current_avg = float(np.mean(_count_history[-3:]))
        if past_avg > 0:
            spike          = (current_avg - past_avg) / past_avg
            signal_density = max(0.0, min(1.0, spike / DENSITY_SPIKE_RATIO))

    stampede_score = (
        signal_speed      * 0.45 +
        signal_uniformity * 0.35 +
        signal_density    * 0.20
    )

    if stampede_score < 0.25:
        level   = "Normal"
        color   = (34, 197, 94)
        message = "Normal movement"
    elif stampede_score < 0.50:
        level   = "Warning"
        color   = (36, 191, 251)
        message = f"Abnormal movement — {direction_label}"
    elif stampede_score < 0.75:
        level   = "Danger"
        color   = (0, 140, 255)
        message = f"DANGER: Rapid movement — {direction_label}"
    else:
        level   = "STAMPEDE"
        color   = (0, 0, 220)
        message = f"STAMPEDE ALERT! — {direction_label}"

    if level != "Normal":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0),
                      (frame.shape[1], frame.shape[0]),
                      color, 6)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.rectangle(frame, (0, 0),
                      (frame.shape[1], 44), color, -1)
        cv2.putText(frame, message, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        bar_w = int(frame.shape[1] * stampede_score)
        cv2.rectangle(frame,
                      (0, frame.shape[0] - 8),
                      (bar_w, frame.shape[0]),
                      color, -1)

    return frame, {
        "level":      level,
        "score":      round(stampede_score * 100, 1),
        "speed":      round(avg_speed, 2),
        "uniformity": round(signal_uniformity * 100, 1),
        "density":    round(signal_density * 100, 1),
        "direction":  direction_label,
        "message":    message
    }