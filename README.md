# KavachX AI/ML Technical Assessment

Automated active safety surveillance edge-to-cloud intelligence system.

---

## Quick Start & Setup

### 1. Prerequisites
- Linux / macOS / Windows with **Python 3.10+** (Tested on Python 3.12 with CUDA GPU & CPU)

### 2. Environment Setup
```bash
git clone https://github.com/Kashyap-10/KavachX-Assesment.git
cd KavachX-Assesment

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Running Task 1: Plain-English Rule to Event JSON Pipeline

Converts natural language rules into open-vocabulary detection, tracks targets with ByteTrack inside polygon zones, measures dwell time, blurs face/head regions for privacy, and emits schema-compliant JSON events.

```bash
python task1/run_task1.py \
    --rule "Alert me if a person carrying a bag enters the loading bay and stays more than 10 seconds." \
    --video clips/loading_bay.mp4 \
    --zones zones.json \
    --output-dir task1/outputs
```

> **Note:** If `clips/loading_bay.mp4` is not present, a synthetic CCTV test clip is automatically generated so the script executes seamlessly.

**Outputs generated:**
- `task1/outputs/sample_event.json` (Validates against `event_schema.json`)
- `task1/outputs/evidence_*.jpg` (Privacy-masked frames with blurred face/head regions)

---

## Running Task 2: Model Training, Quantization & Edge Budget

### 1. Model Training (Google Colab or Local GPU)
- **Colab Notebook:** Open and run `task2/train_notebook.ipynb` on a free T4 GPU.
- **Local Training:**
```bash
python task2/train.py --data data/ppe/data.yaml --epochs 15 --imgsz 416 --device 0
```

### 2. Automated Quantization Pipeline (FP32 → FP16 → INT8)
Exports trained weights to ONNX and runs static Post-Training Quantization (PTQ) with calibration data:
```bash
python task2/quantize_pipeline.py \
    --weights task2/outputs/best.pt \
    --calibration-dir data/ppe/train/images \
    --output-dir task2/outputs
```

### 3. Edge Budget Simulation: Motion Gating
Simulates detector execution on surveillance video behind a frame-difference motion gate with periodic heartbeat passes:
```bash
python task2/motion_gate.py \
    --video clips/static_heavy.mp4 \
    --model task2/outputs/model_int8.onnx \
    --interval 30
```

### 4. Performance Benchmarking Across Variants
Evaluates model size (MB), CPU latency (ms), mAP@50, and recall:
```bash
python task2/benchmark.py
```

### 5. Validation Metrics & 5 Failure Cases Analysis
Evaluates baseline and generates detailed illustrations and explanations for 5 failure cases:
```bash
python task2/evaluate.py
```

**Outputs generated:**
- `task2/outputs/model_fp32.onnx`, `model_fp16.onnx`, `model_int8.onnx`
- `task2/outputs/benchmark_table.md` & `benchmark_table.csv`
- `task2/outputs/failure_cases_analysis.md` & failure case illustrations

---

## Running Task 3: Cascaded False-Positive Verifier

Second-stage zero-shot Vision-Language verifier using OpenCLIP with engineered prompt ensembles to filter false alarms (shadows, headlights, rain, insects, stray animals) from true intrusions.

### 1. Run Verifier Scoring
```bash
python task3/verifier.py \
    --alerts-dir alerts \
    --labels alerts/labels.csv \
    --output-dir task3/outputs
```

### 2. Evaluate Precision/Recall & Threshold Selection
Compares Edge Alone vs. Edge + Secondary Verifier across operating confidence thresholds:
```bash
python task3/evaluate.py
```

**Outputs generated:**
- `task3/outputs/verifier_scores.csv`
- `task3/outputs/verifier_evaluation.md` (Precision/recall sweep table & threshold rationale)

---

## Comprehensive Engineering Notes

Refer to **[NOTES.md](file:///home/monarch/Desktop/PRO/NOTES.md)** for:
1. CCTV data augmentation engineering rationale.
2. Quantization trade-offs & outlier mitigation technique.
3. Motion gating edge compute reduction analysis.
4. Economic analysis: Monthly API costs for 200 alerts/day.
5. Customer dashboard *"Flag as incorrect"* active learning feedback loop design.
6. Transparency & AI assistant disclosures.
7. Next-step improvements with one additional week.
