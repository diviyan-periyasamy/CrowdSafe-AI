# CrowdSafe AI — Real-Time Crowd Monitoring & Stampede Detection

An end-to-end crowd safety monitoring system built with **YOLOv8**, integrating computer vision, crowd density analysis, and a complete **MLOps pipeline** for production readiness.

## Overview

CrowdSafe AI detects and analyzes crowd density in real time from **video, webcam, or image input**, flags high-risk zones using a grid-based scoring system, and raises potential stampede-risk alerts.

The project also demonstrates a complete MLOps workflow — from **experiment tracking and model versioning to CI/CD and deployment** — built and tested in a Windows CPU-only environment.

## Features

* Real-time people detection using a fine-tuned **YOLOv8 Nano** model
* **Gaussian crowd density heatmaps** for visual density mapping
* **Farneback Dense Optical Flow** for crowd movement analysis
* **4×3 grid-based zone risk scoring** for localized stampede-risk detection
* **Streamlit dashboard** supporting:

  * Video input
  * Webcam input
  * DroidCam input
  * Image input
* Real-time crowd analysis and risk visualization
* CSV export
* Alert logging

## MLOps Pipeline

* **MLflow** — experiment tracking, model registry, and staging/production versioning
* **DVC** — dataset and model version control
* **GitHub Actions** — CI/CD automation
* **Docker** — containerized deployment
* **Flask REST API** — model serving with `/health` and `/predict` endpoints
* **Drift Detection** — monitors potential model performance degradation over time

## Tech Stack

| Category            | Technologies                |
| ------------------- | --------------------------- |
| Programming         | Python                      |
| Computer Vision     | OpenCV                      |
| Object Detection    | YOLOv8                      |
| Dashboard           | Streamlit                   |
| API                 | Flask                       |
| Experiment Tracking | MLflow                      |
| Version Control     | DVC, Git, GitHub            |
| Containerization    | Docker                      |
| CI/CD               | GitHub Actions              |
| Training            | Google Colab, NVIDIA T4 GPU |

## Model Training

The **YOLOv8 Nano** model was fine-tuned on a custom crowd dataset sourced from **Roboflow**.

### Training Details

| Parameter         | Details              |
| ----------------- | -------------------- |
| Model             | YOLOv8 Nano          |
| Dataset           | Custom Crowd Dataset |
| Dataset Source    | Roboflow             |
| Training Platform | Google Colab         |
| GPU               | NVIDIA T4            |
| Epochs            | 50                   |
| Task              | Person Detection     |

## Crowd Analysis Pipeline

The system combines multiple computer vision techniques to analyze crowd safety.

```text
Input
  │
  ├── Video
  ├── Webcam
  ├── DroidCam
  └── Image
       │
       ▼
┌─────────────────────┐
│     YOLOv8 Nano     │
│   Person Detection  │
└──────────┬──────────┘
           │
           ├──────────────────┐
           │                  │
           ▼                  ▼
┌──────────────────┐  ┌────────────────────┐
│ Crowd Density    │  │ Dense Optical Flow │
│ Gaussian Heatmap │  │    (Farneback)     │
└────────┬─────────┘  └──────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │    4×3 Risk Grid     │
          │   Zone Risk Scoring  │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   Risk Assessment    │
          │  & Alert Generation  │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Streamlit Dashboard  │
          │  Visualization/Logs  │
          └──────────────────────┘
```

## Zone-Based Risk Scoring

The monitored area is divided into a **4×3 grid** containing 12 individual zones.

```text
┌─────┬─────┬─────┬─────┐
│ A1  │ A2  │ A3  │ A4  │
├─────┼─────┼─────┼─────┤
│ B1  │ B2  │ B3  │ B4  │
├─────┼─────┼─────┼─────┤
│ C1  │ C2  │ C3  │ C4  │
└─────┴─────┴─────┴─────┘
```

Each zone is analyzed using crowd-related measurements such as:

* Person density
* Crowd concentration
* Movement intensity
* Optical-flow information

The system assigns a risk score to each zone and highlights areas with elevated crowd-safety risk.

## Project Structure

```text
crowdsafe-ai/
│
├── app.py                       # Streamlit dashboard
│
├── src/
│   ├── detection.py             # YOLOv8 inference
│   ├── heatmap.py               # Gaussian density heatmaps
│   ├── optical_flow.py          # Farneback movement analysis
│   └── risk_scoring.py          # Zone-based risk scoring
│
├── api/
│   └── app.py                   # Flask REST API
│
├── mlflow/                      # MLflow tracking configurations
│
├── .github/
│   └── workflows/               # GitHub Actions CI/CD pipeline
│
├── Dockerfile                   # Docker configuration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Running via Docker

### Build the Docker Image

```bash
docker build -t crowdsafe-ai .
```

### Run the Container

```bash
docker run -p 8501:8501 crowdsafe-ai
```

The Streamlit dashboard can then be accessed through the exposed port.

## REST API

CrowdSafe AI includes a Flask REST API for model serving.

### Health Check

```http
GET /health
```

### Prediction

```http
POST /predict
```

The API can be used to integrate the crowd detection model with external applications or services.

## MLOps Workflow

```text
Dataset
   │
   ▼
DVC Version Control
   │
   ▼
Model Training
   │
   ▼
MLflow Experiment Tracking
   │
   ▼
Model Registry
   │
   ├── Staging
   │
   └── Production
   │
   ▼
Docker Container
   │
   ▼
Flask API / Streamlit
   │
   ▼
Monitoring & Drift Detection
   │
   ▼
GitHub Actions CI/CD
```

## Future Improvements

* Multi-object tracking using **ByteTrack** or **DeepSORT**
* Advanced crowd behavior classification
* Improved stampede-event detection
* Real-time notification system
* Cloud deployment
* Kubernetes-based deployment
* Automated model retraining
* Advanced model monitoring
* Multi-camera crowd monitoring
* Edge-device deployment
* Historical crowd analytics

## Disclaimer

CrowdSafe AI is an **academic and experimental project** developed to demonstrate computer vision, machine learning, and MLOps concepts.

The generated risk scores and alerts are algorithmic indicators and should **not be considered a replacement for professional security systems, emergency response procedures, or human supervision**.

## Author

**Diviyan Periyasmay**

BSc (Hons) Data Science, Coventry University | NIBM

Developed as part of **HND Machine Learning 2 coursework**.

---

⭐ If you find this project useful, consider giving the repository a star!
