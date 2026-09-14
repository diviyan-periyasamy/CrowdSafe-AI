# CrowdSafe AI — Real-Time Crowd Monitoring & Stampede Detection
An end-to-end crowd safety monitoring system built with YOLOv8, integrating computer vision, crowd density analysis, and a full MLOps pipeline for production readiness.

Overview
CrowdSafe AI detects and analyzes crowd density in real time from video, webcam, or image input, flags high-risk zones using a grid-based scoring system, and raises stampede-risk alerts. The project also demonstrates a complete MLOps workflow — from experiment tracking to CI/CD — built entirely on a Windows CPU-only environment.

Features
Real-time people detection using a fine-tuned YOLOv8 Nano model
Gaussian crowd density heatmaps for visual density mapping
Farneback Dense Optical Flow for crowd movement analysis
4×3 grid-based zone risk scoring for localized stampede risk detection
Streamlit dashboard supporting video, webcam, DroidCam, and image inputs
CSV export and alert logging
MLOps Pipeline
MLflow — experiment tracking, model registry, staging/production versioning
DVC — dataset and model version control
GitHub Actions — CI/CD automation
Docker — containerized deployment
Flask REST API — model serving with /health and /predict endpoints
Drift detection — monitors model performance degradation over time
Tech Stack
Python · YOLOv8 · Streamlit · OpenCV · MLflow · DVC · Docker · Flask · GitHub Actions

Model Training
The YOLOv8 Nano model was fine-tuned on a custom crowd dataset sourced from Roboflow, trained via Google Colab (50 epochs, T4 GPU).

Project Structure
crowdsafe-ai/
├── app.py                  # Streamlit dashboard
├── src/
│   ├── detection.py        # YOLOv8 inference
│   ├── heatmap.py          # Gaussian density heatmaps
│   ├── optical_flow.py     # Farneback movement analysis
│   └── risk_scoring.py     # Zone-based risk grid
├── api/
│   └── app.py               # Flask REST API
├── mlflow/                  # MLflow tracking configs
├── .github/workflows/       # CI/CD pipeline
├── Dockerfile
├── requirements.txt
└── README.md
Running Locally
git clone https://github.com/DhulakshanKannan/crowdsafe-ai.git
cd crowdsafe-ai
pip install -r requirements.txt
streamlit run app.py
Running via Docker
docker build -t crowdsafe-ai .
docker run -p 8501:8501 crowdsafe-ai
Author
Dhulakshan Kannan BSc (Hons) Data Science, Coventry University | NIBM LinkedIn · GitHub

Developed as part of HND Machine Learning 2 coursework.
