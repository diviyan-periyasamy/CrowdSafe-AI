

import os
import mlflow
import mlflow.pytorch
from ultralytics import YOLO


# 1. CONFIG 

DATA_YAML_PATH   = r"C:\Users\USER\Desktop\crowdsafe_ai_mlops\crowd-counting-1\data.yaml" 
BASE_MODEL       = "yolov8n.pt"
EPOCHS           = 2
IMG_SIZE         = 640
BATCH_SIZE       = 16
PATIENCE         = 10
MLFLOW_TRACKING_URI = "file:./mlruns"  
EXPERIMENT_NAME  = "CrowdSafe-AI-YOLOv8-Finetune"


# 2. MLFLOW SETUP

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="yolov8n_crowd_finetune"):

    
    # 3. LOG HYPERPARAMETERS
    
    mlflow.log_param("base_model", BASE_MODEL)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("imgsz", IMG_SIZE)
    mlflow.log_param("batch", BATCH_SIZE)
    mlflow.log_param("patience", PATIENCE)
    mlflow.log_param("data_yaml", DATA_YAML_PATH)
    mlflow.log_param("device", "GPU (Colab T4)" if os.path.exists("/content") else "CPU")


    # 4. DATA PREPROCESSING NOTE
    
    mlflow.log_param("preprocessing", "Roboflow: resize 640x640, auto-orient, split 70/20/10")

    
    # 5. TRAIN THE MODEL 
    
    # NOTE: full fine-tuning (50 epochs) already done on Colab GPU, producing
    # models/best.pt. We load that here and validate + log through MLflow
    # instead of re-running training on CPU (too slow for a quick demo).
    model = YOLO("models/best.pt")

    
    # 6. EVALUATE & LOG METRICS
    
    metrics = model.val(data=r"C:\Users\USER\Desktop\crowdsafe_ai_mlops\crowd-counting-1\data.yaml", imgsz=IMG_SIZE, device="cpu")
    mlflow.log_metric("precision", float(metrics.box.mp))
    mlflow.log_metric("recall", float(metrics.box.mr))
    mlflow.log_metric("mAP50", float(metrics.box.map50))
    mlflow.log_metric("mAP50-95", float(metrics.box.map))

    
    # 7. SAVE & VERSION THE MODEL
    
    mlflow.log_artifact("models/best.pt", artifact_path="model")

    print("Validation + MLflow logging complete.")
    print(f"MLflow run logged under experiment: {EXPERIMENT_NAME}")
    print("Run 'mlflow ui --port 5001' in this folder to view the dashboard")
