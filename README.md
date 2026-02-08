# Edge-AI Defect Classification for Semiconductor Images

## 1. Problem Understanding
Semiconductor fabrication is a zero-error industry. This project builds a lightweight, Edge-AI system to detect 8 types of wafer defects (Shorts, Opens, Cracks, etc.) directly on NXP i.MX RT hardware, solving the latency issues of cloud inspection.

## 2. Approach
* **Data**: 8 classes (Clean, Other, Short, Open, Bridge, Malformed_Via, Scratch, Crack).
* **Model**: Custom lightweight CNN with Depthwise Separable Convolutions.
* **Deployment**: TensorFlow -> ONNX -> NXP eIQ.

## 3. How to Run
1. `pip install -r requirements.txt`
2. `python main.py`
3. Output: `semiconductor_defect_model.onnx`
