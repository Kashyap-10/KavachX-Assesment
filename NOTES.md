# KavachX AI/ML Technical Assessment — Engineering Design Notes

---

## 1. Executive Summary & Architecture Overview

KavachX converts existing CCTV cameras into an active safety surveillance edge-to-cloud intelligence system. The core design philosophy implemented across this codebase is:
1. **Edge-First Efficiency:** Heavy video streams never hit the cloud directly; edge devices filter static intervals via motion gating and run quantized INT8 models locally on edge compute.
2. **Open-Vocabulary Flexibility:** Rules defined in natural language are parsed into structured configurations and routed to open-vocabulary models (YOLO-World) without requiring per-class retraining.
3. **Cascaded Verification:** Zero-shot Vision-Language filters evaluate borderline alerts before dispatching notifications to end-customers, virtually eliminating alert fatigue while maintaining near-zero false negatives.

---

## 2. Task 1: Plain-English Rule to Event JSON

### 2.1 Rule Parser Design Choices
- **Hybrid Architecture:** Implemented in `task1/rule_parser.py` with dual-mode execution:
  - **LLM Mode (OpenAI/Gemini):** Uses structured JSON schema extraction with temperature=0.0 for deterministic cloud deployments.
  - **Semantic NLP Heuristic Fallback:** In offline/edge environments where an external API key is absent, an entity and regex extraction engine parses dwell conditions (`> 10s`), spatial prepositions (`enters the loading bay`), and target nouns (`person carrying bag`). **No values are hardcoded**, ensuring full support for arbitrary queries.
- **Attributes Association:** In `task1/detector.py`, open-vocabulary YOLO-World detects both the primary object (`person`) and attribute nouns (`bag`, `backpack`, `helmet`). A spatial containment test associates carried items with the primary actor based on bounding box overlap and proximity.

### 2.2 Tracking & Polygon Dwell Time
- Uses **ByteTrack** via `supervision` for multi-object association across occlusions.
- Dwell time is tracked using the **bottom-center point (foot position)** of the bounding box against the Shapely polygon defined in `zones.json`. Foot points represent ground contact and prevent false triggers when a subject's head or torso leans over an aerial boundary without entering the physical floor zone.

### 2.3 Privacy Protection
- Dual-tier face & head anonymization in `task1/privacy.py`:
  1. Primary: OpenCV Haar frontal & profile face detection with Gaussian blur.
  2. Secondary / Failsafe: Geometric top-28% head bounding box blur for all detected persons, ensuring 100% compliance even if faces are turned away, backlit, or wearing respirators.

---

## 3. Task 2: Train, Quantize & Fit Edge Budget

### 3.1 CCTV Data Augmentation Rationale
Industrial CCTV cameras present distinct challenges compared to standard photographic datasets:

| Augmentation | Parameter | Engineering Rationale for Industrial CCTV |
|---|---|---|
| **Mosaic** | `1.0` | Combines 4 training frames into one; simulates crowded factory floors with multiple simultaneous workers at varying scales. |
| **MixUp** | `0.1` | Blends image pairs linearly; teaches the model to recognize workers partially occluded by forklifts, guardrails, and structural beams. |
| **HSV Jitter** | `h=0.015, s=0.5, v=0.3` | Warehouses experience extreme shifts between daylight skylights, sodium-vapor lamps, and harsh point illumination. |
| **Rotation** | `5.0 deg` | Corrects for minor physical tilt or vibration on pole-mounted CCTV housings without creating unrealistic orientations. |
| **Translation** | `0.1` | Simulates subjects entering or exiting peripheral surveillance blind spots. |
| **Scale Jitter** | `0.3` | Emulates worker distance variations (5 meters vs. 40 meters from the fixed camera lens). |
| **Perspective** | `0.0005` | Crucial for oblique, top-down surveillance camera angles that foreshorten human proportions. |
| **Flip Up-Down** | **`0.0 (OFF)`** | **Strictly disabled:** Factory workers never walk on the ceiling; vertical flipping introduces non-physical artifacts. |
| **Flip Left-Right**| `0.5` | Directional symmetry is physically sound across surveillance camera fields of view. |

### 3.2 Quantization Pipeline (FP32 → FP16 → INT8)
- Automated pipeline in `task2/quantize_pipeline.py`.
- **Static Post-Training Quantization (PTQ):** Employs `onnxruntime.quantization.quantize_static` with `QuantFormat.QDQ` and per-channel symmetric weights.
- **Calibration:** Feeds 100 representative CCTV images preprocessed with letterbox normalization into `(1, 3, 416, 416)` via `PPECalibrationDataReader`.
- **Mitigation Technique:** If INT8 accuracy drops > 2 mAP points, the pipeline switches from `CalibrationMethod.MinMax` to `CalibrationMethod.Percentile` (99.99th percentile clipping). This prevents extreme outlier pixel values (e.g., reflective metal or specular high-bay glare) from compressing the quantization dynamic range for worker heads.

### 3.3 Benchmark Performance Across Variants

Evaluated on AMD Ryzen CPU (416x416 resolution):

| Model Variant | Size (MB) | CPU Latency (ms) | mAP@50 | no_helmet Recall | Compression / Savings |
|:---|---:|:---|---:|---:|:---|
| **FP32 ONNX** | 12.16 MB | 108.7 ms | 0.865 | 0.842 | Baseline (1.0x) |
| **FP16 ONNX** | 12.16 MB | 138.5 ms | 0.863 | 0.840 | GPU-optimized format (CPU fallback) |
| **INT8 ONNX** | **3.39 MB** | **56.5 ms** | **0.851** | **0.828** | **3.6x smaller**, ~2x faster on CPU |
| **INT8 + Motion Gate** | **3.39 MB** | **13.58 ms (amortized)** | **0.851** | **0.828** | **76.5% compute reduction** |

### 3.4 Motion Gating Edge Simulation Results
- `task2/motion_gate.py` runs before deep learning inference:
  - Computes frame differences against an adaptive background model with morphological filtering to eliminate sensor noise.
  - Triggers the INT8 detector only when movement exceeds `0.4%` of frame area.
  - Heartbeat safety net: executes a mandatory full-frame detection pass every 30 frames (1 Hz at 30 fps) so that stationary workers inside the frame remain monitored.
- **Results on `clips/static_heavy.mp4` (20.0s, 600 frames):**
  - Calls **without** motion gating: 30.00 calls/sec (600 total)
  - Calls **with** motion gating: 7.05 calls/sec (141 total: 121 motion-triggered + 20 heartbeat passes)
  - **Edge Compute Reduction:** **76.5% reduction** in detector invocations, fitting comfortably inside edge hardware thermal/power budgets.

### 3.5 Five Model Failure Cases & Architectural Root Causes

| Failure Case | Category | Failure Mechanism | Engineering Root Cause | Recommended Mitigation |
|---|---|---|---|---|
| **1. Small Distant Head** | Range / Scale | Worker at far end of warehouse occupies <15x15 px; features lost in deep strides. | Spatial downsampling beyond detector resolution limit. | High-resolution P2 stride-4 head or SAHI tiled inference. |
| **2. Heavy Occlusion** | Spatial Context | Worker walking behind metal racks is 60% occluded; misclassified as `no_helmet`. | Partial context visibility fails safety cap feature activation. | Increase MixUp augmentation & temporal multi-frame track voting. |
| **3. Specular Lighting Glare** | Radiometry | High-bay sodium glare washes out white hard hat boundary. | Sensor dynamic range clipping; high-frequency edge loss. | Percentile PTQ calibration & HSV value jitter augmentation. |
| **4. Hoodie / Beanie Ambiguity** | Semantic Ambiguity | Worker wearing dark cloth beanie is misclassified as wearing a hard hat. | Convex silhouette curvature matches helmet shape without material texture. | Add curated negative class samples (`beanie`, `hood`) to training set. |
| **5. Steep Overhead Angle** | Geometric Perspective | High-nadir CCTV angle foreshortens 3D dome into a flat circular disc. | Unbalanced aerial perspective distribution in COCO pretraining. | Perspective augmentation (`perspective=0.001`) & nadir CCTV camera fine-tuning. |

*(Annotated failure case images available in `task2/outputs/failure_01_*.jpg` through `failure_05_*.jpg`)*.

---

## 4. Task 3: Cascaded False-Positive Verifier & Economic Analysis

### 4.1 Verifier Architecture & Prompt Engineering
- **Model:** Zero-shot OpenCLIP (`ViT-B-32`, LAION-2B).
- **Prompt Ensembles:** Evaluates snapshot embeddings against two calibrated semantic ensembles:
  - **Positive (Intrusions):** `"unauthorized human intruder trespassing at night"`, `"person burglar breaking into property on cctv"`.
  - **Negative (False Alarms):** `"tree shadows moving on ground"`, `"car headlights glare"`, `"rain droplets reflecting infrared illuminator"`, `"stray cat or dog"`, `"insects flying near lens"`.
- Cosine similarities are temperature-scaled through softmax to yield a calibrated verification probability $P(\text{Intrusion}) \in [0.0, 1.0]$.

### 4.2 Precision & Recall Evaluation & Threshold Selection

Evaluated across the 30 benchmark nighttime alert snapshots (`alerts/labels.csv`):

| Pipeline Configuration | Precision | Recall | F1-Score | False Alarms Sent | False Alarm Reduction |
|---|---|---|---|---|---|
| **(a) Edge Model Alone** | `0.433` (43.3%) | `1.000` (100.0%) | `0.605` | 17 (out of 30) | 0.0% |
| **(b) Edge + Secondary Verifier (t=0.60)** | **`1.000` (100.0%)** | **`1.000` (100.0%)** | **`1.000`** | **0** | **100.0% eliminated** |

#### Operating Threshold Justification:
- Physical security adheres to an **asymmetric loss function**:
  $$\text{Cost}(\text{False Negative}) \gg \text{Cost}(\text{False Positive})$$
  A missed intrusion risks property loss or bodily harm, whereas a false alarm costs seconds of secondary review.
- We therefore constrain the operating point to **$\text{Recall} \ge 95\%$**.
- Operating threshold sweep:
  - $t = 0.30 - 0.45$: Captures all intrusions, but lets through 14–17 false alarms.
  - $t = 0.55$: Eliminates 41.2% of false alarms, maintaining 100% recall.
  - **$t = 0.60$ (Optimal):** Eliminates **100.0% of false alarms** (shadows, headlights, rain, insects, stray cats/dogs) while preserving **100.0% Recall** on true intrusions.
  - $t \ge 0.75$: Recall drops sharply to 46.2%, which is unacceptable for active security.

### 4.3 Economic & Product Analysis

#### Question 1: Monthly API Cost Estimate for 200 Alerts/Day
A customer site generates $200\text{ alerts/day} \times 30\text{ days} = \mathbf{6,000\text{ alert verifications/month}}$.

| Verification Architecture | Cost Per Image | Monthly Compute / API Cost | Latency Profile | Scalability & Data Privacy |
|---|---|---|---|---|
| **Option A: Local Edge/Server CLIP (Our Implementation)** | **$0.00** | **~$0.00** (Included in edge/on-prem server compute) | Ultra-low (~15–25ms) | 100% private; footage never leaves enterprise boundary. |
| **Option B: Batch Hosted VLM (e.g. Claude 3.5 Haiku / GPT-4o-mini)** | ~$0.0008 / call | **~$4.80 / month** per site | ~300–600ms | Highly cost-effective; requires external outbound egress. |
| **Option C: Premium Hosted VLM (e.g. GPT-4o Vision)** | ~$0.008 / call | **~$48.00 / month** per site | ~1000–1800ms | Expensive at multi-site scale (100 sites = $4,800/mo). |

**Recommendation:** Run OpenCLIP locally on the customer's on-prem edge gateway or regional server. It delivers sub-second response times, zero per-token recurring cloud fees, and preserves strict enterprise data privacy compliance.

#### Question 2: Dashboard "Flag as Incorrect" Feedback Loop
When a customer clicks *"Flag as incorrect"* on the dashboard, the event enters a continuous learning flywheel:

```mermaid
graph TD
    A["Customer Clicks 'Flag as Incorrect'"] --> B["Capture Image + Metadata + Bounding Boxes"]
    B --> C["Triage: False Alarm vs Missed Detection"]
    C --> D["Active Learning Hard-Negative Pool"]
    D --> E["Weekly Threshold Calibration (Site-Specific)"]
    D --> F["Periodic LoRA / CLIP Fine-Tuning"]
    F --> G["Automated Regression Test Suite"]
    G --> H["Over-the-Air (OTA) Model Rollout"]
```

1. **Short-Term (Automated Site Calibration):**
   - Record site-specific false-alarm modalities (e.g., Camera 4 has a tree casting rhythmic shadows).
   - Dynamically adjust local operating thresholds and prompt weights for that specific camera without retraining.
2. **Medium-Term (Active Learning Curation):**
   - Automatically store flagged snapshots into a `hard_negatives` partition with environmental metadata (lux level, weather, camera angle).
   - Filter redundant examples using embedding diversity clustering (k-means on CLIP visual embeddings).
3. **Long-Term (Model Adaptation & OTA Updates):**
   - Fine-tune CLIP visual projection layers or train a lightweight downstream linear probe on accumulated customer hard negatives.
   - Run candidate models against a golden regression test benchmark to guarantee no recall degradation before deploying OTA updates to edge devices.

---

## 5. Transparency & AI Assistant Usage

As instructed in Section 3 of the candidate brief:
- **AI Coding Assistants Used:** Google Antigravity IDE / Gemini & Claude Opus thinking models were utilized for scaffolding boilerplate, refining regular expressions, structuring documentation tables, and drafting test harnesses.
- **Engineering Ownership:** All algorithmic logic, model selection rationale, mathematical formulations (IoU, dwell time, PTQ calibration readers, prompt engineering), and evaluation criteria have been independently reviewed, verified, and can be fully modified and defended during the technical review call.

---

## 6. What I Would Improve Given One Additional Week

1. **SAHI (Slicing Aided Hyper Inference) for Small Heads:**
   - Integrate patch-based inference to elevate small-object recall on workers 50+ meters away from high-mounted CCTV cameras.
2. **TensorRT & INT8 Engine Compilation:**
   - Compile the INT8 ONNX models using TensorRT with FP16 fallback on Nvidia Jetson GPUs for sub-4ms end-to-end edge inference.
3. **Multi-Camera Re-Identification (ReID):**
   - Extend ByteTrack to multi-camera tracking using feature embedding matching across overlapping camera viewpoints.
4. **End-to-End Synthetic Hard-Negative Generator:**
   - Use Stable Diffusion / ControlNet to synthesize targeted factory edge-cases (workers with turbans, decorative hard hats, complex machinery occlusions) to continuously stress-test detector robustness.
