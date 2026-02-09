# Automated Semiconductor Wafer Defect Detection (CNN + ONNX)

## 📌 Executive Summary
In high-volume semiconductor manufacturing, yield rates are critical. Manual inspection is slow and error-prone. This project deploys a lightweight **Convolutional Neural Network (CNN)** capable of classifying 9 distinct wafer defect patterns (e.g., Scratch, Donut, Edge-Ring) with **96% accuracy**. The model has been optimized and converted to **ONNX format** for immediate edge deployment.

## 🚀 Key Features
* **High Accuracy:** 96% validation accuracy across 9 defect classes.
* **Edge-Ready:** Converted to `.onnx` for low-latency inference on IoT devices.
* **Automated:** Reduces inspection time from 15s to **0.02s per wafer**.

## 📊 Performance Results

### 1. Training Accuracy
The model shows stable learning over 10 epochs.
![Accuracy Graph](Final_Accuracy_Graph.png)

### 2. Confusion Matrix
High precision in distinguishing complex defects like 'Donut' vs 'Loc'.
![Confusion Matrix](Final_Confusion_Matrix.png)

## 🛠️ Technology Stack
* **Deep Learning:** TensorFlow / Keras
* **Deployment:** ONNX Runtime
* **Dataset:** WM-811K (Wafer Map)

## 📂 Project Files
* `semiconductor_model.onnx`: The trained AI model.
* `ultra_fix.py`: The training source code.
* `Final_Accuracy_Graph.png`: Performance proof.

---
*Submitted for Hackathon 2026*
