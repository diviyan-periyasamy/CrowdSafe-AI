
import os
import io
import logging
from datetime import datetime

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from ultralytics import YOLO


# LOGGING SETUP (for Model Monitoring task)

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/api_requests.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

app = Flask(__name__)


# LOAD MODEL

MODEL_PATH = "models/best.pt" if os.path.exists("models/best.pt") else "models/yolov8n.pt"
model = YOLO(MODEL_PATH)
logging.info(f"Model loaded: {MODEL_PATH}")


@app.route("/health", methods=["GET"])
def health():
    """Used by Docker HEALTHCHECK and the CI/CD pipeline smoke test."""
    return jsonify({"status": "ok", "model": MODEL_PATH}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form field name 'file'."}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img_np = np.array(img)

    start = datetime.now()
    results = model(img_np, classes=[0], verbose=False) 
    latency_ms = (datetime.now() - start).total_seconds() * 1000

    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": round(conf, 3)})

    response = {
        "person_count": len(boxes),
        "boxes": boxes,
        "inference_time_ms": round(latency_ms, 2),
        "timestamp": datetime.now().isoformat(),
    }

    
    logging.info(
        f"prediction | person_count={len(boxes)} | latency_ms={round(latency_ms, 2)} "
        f"| avg_confidence={round(np.mean([b['confidence'] for b in boxes]), 3) if boxes else 0}"
    )

    return jsonify(response), 200


if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5000, debug=False)
