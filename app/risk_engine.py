import cv2
import numpy as np

GRID_COLS = 4
GRID_ROWS = 3

ZONE_THRESHOLDS = {
    "Safe":     (0,  3),
    "Moderate": (4,  7),
    "High":     (8,  12),
    "Critical": (13, float("inf"))
}

ZONE_COLORS = {
    "Safe":     (34,  197, 94),
    "Moderate": (36,  191, 251),
    "High":     (68,  113, 248),
    "Critical": (28,  28,  185),
}

ALPHA = 0.35


def get_zone_risk(count):
    for level, (lo, hi) in ZONE_THRESHOLDS.items():
        if lo <= count <= hi:
            return level
    return "Critical"


def calculate_grid_risk(frame, boxes):
    h, w      = frame.shape[:2]
    zone_w    = w // GRID_COLS
    zone_h    = h // GRID_ROWS
    zone_counts = [[0] * GRID_COLS for _ in range(GRID_ROWS)]

    for (x1, y1, x2, y2, conf) in boxes:
        cx  = (x1 + x2) // 2
        cy  = (y1 + y2) // 2
        col = min(cx // zone_w, GRID_COLS - 1)
        row = min(cy // zone_h, GRID_ROWS - 1)
        zone_counts[row][col] += 1

    overlay       = frame.copy()
    zone_data     = []
    risk_priority = ["Safe", "Moderate", "High", "Critical"]
    overall_risk  = "Safe"

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            count = zone_counts[row][col]
            risk  = get_zone_risk(count)
            color = ZONE_COLORS[risk]
            zx1   = col * zone_w
            zy1   = row * zone_h
            zx2   = zx1 + zone_w
            zy2   = zy1 + zone_h

            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), color, -1)
            cv2.rectangle(frame,   (zx1, zy1), (zx2, zy2), color,  1)

            label = f"{count}p | {risk}"
            font  = cv2.FONT_HERSHEY_SIMPLEX
            fs    = 0.42
            (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
            tx = zx1 + (zone_w - tw) // 2
            ty = zy1 + (zone_h + th) // 2
            cv2.rectangle(frame, (tx-4, ty-th-4), (tx+tw+4, ty+4), (0,0,0), -1)
            cv2.putText(frame, label, (tx, ty), font, fs, color, 1, cv2.LINE_AA)

            zone_data.append({"row": row, "col": col,
                               "count": count, "risk": risk})

            if risk_priority.index(risk) > risk_priority.index(overall_risk):
                overall_risk = risk

    cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)
    return frame, zone_data, overall_risk, zone_counts


def calculate_risk_score(person_count, density_label, frame_area):
    base  = {"Low": 10, "Medium": 40, "High": 75}.get(density_label, 10)
    score = min(100, base)
    if score <= 25:
        category, color = "Safe", "green"
    elif score <= 50:
        category, color = "Moderate", "orange"
    elif score <= 75:
        category, color = "High", "red"
    else:
        category, color = "Critical", "darkred"
    return score, category, color