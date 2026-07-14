---
title: AC Fault Detection & Diagnosis
emoji: ❄️
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: "4.36.1"
app_file: app.py
pinned: false
---

# AC Unit Fault Detection & Diagnosis (AC-FDD)

A real-time machine learning application for air conditioner telemetry analysis. This Gradio-based web app classifies the operating condition of split AC units and performs root-cause diagnosis using a hybrid pipeline of supervised and unsupervised models.

## 🧠 Model Architecture

The application relies on a two-stage inference pipeline to evaluate physical sensor readings:

1. **Fault Detection (Random Forest):** The `best_model.pkl` classifier evaluates engineered thermodynamic and electrical features to categorize the severity of the system's state into discrete conditions (`NORMAL`, `MAINTENANCE`, `TROUBLE`, `ABNORMAL`).
2. **Fault Diagnosis (Autoencoder):** If a fault is detected, the data is passed to `ac_autoencoder.keras`. This neural network reconstructs the telemetry and calculates feature-specific Z-scores based on reconstruction errors. The feature with the highest Z-score identifies the thermodynamic or electrical root cause of the anomaly, immune to natural parameter variance.

All inputs undergo a custom piecewise **Tiered Scaling** process to normalize data based on the specific AC unit's capacity
