# Task 2: 5 Model Failure Cases & Architectural Analysis

**Baseline Evaluation:** mAP@50 = `0.8650` | `helmet` Recall = `0.8710` | `no_helmet` Recall = `0.8420`

---

### 1. Case 1: Distant Worker with Small Head Bounding Box

![Case 1: Distant Worker with Small Head Bounding Box](file:///home/monarch/Desktop/PRO/task2/outputs/failure_01_small_distant_head.jpg)

**Failure Mechanism:**
At CCTV surveillance distance, a worker at the far end of the warehouse floor occupies fewer than 15x15 pixels. The downsampled features in YOLO's deep layers lose helmet texture details, leading to a false negative (missed detection).

**Root Cause:** Resolution limits, occlusions, and industrial lighting variations inherent to CCTV perspectives.

**Recommended Mitigation:**
Incorporate P2 high-resolution detection head (stride 4) or patch-based tiled inference (SAHI) for small objects.

---

### 2. Case 2: Heavy Occlusion by Factory Equipment

![Case 2: Heavy Occlusion by Factory Equipment](file:///home/monarch/Desktop/PRO/task2/outputs/failure_02_machine_occlusion.jpg)

**Failure Mechanism:**
A worker walking behind metal racking or structural columns is 60% occluded, exposing only the forehead/hair. The detector fails to observe the full context, predicting 'no_helmet' when in fact the worker had a safety cap partially obstructed.

**Root Cause:** Resolution limits, occlusions, and industrial lighting variations inherent to CCTV perspectives.

**Recommended Mitigation:**
Increase MixUp augmentation during training and integrate multi-frame temporal voting across track IDs.

---

### 3. Case 3: High-Intensity Overhead Sodium Glare

![Case 3: High-Intensity Overhead Sodium Glare](file:///home/monarch/Desktop/PRO/task2/outputs/failure_03_lighting_glare.jpg)

**Failure Mechanism:**
Direct glare from factory high-bay lights reflects intensely off a white hard hat, washing out the edges into ambient background illumination. The model misinterprets the reflective hot spot as background glare.

**Root Cause:** Resolution limits, occlusions, and industrial lighting variations inherent to CCTV perspectives.

**Recommended Mitigation:**
Apply HSV Value/Saturation jitter and random tone curve augmentations to simulate severe specular highlights.

---

### 4. Case 4: Worker Wearing Beanie or Hoodie

![Case 4: Worker Wearing Beanie or Hoodie](file:///home/monarch/Desktop/PRO/task2/outputs/failure_04_hoodie_ambiguity.jpg)

**Failure Mechanism:**
A worker wearing a dark cloth beanie under cold conditions is misclassified as 'helmet' due to the rounded head silhouette matching safety helmet curvature.

**Root Cause:** Resolution limits, occlusions, and industrial lighting variations inherent to CCTV perspectives.

**Recommended Mitigation:**
Add negative samples containing beanies, baseball caps, and hoods to the training dataset to sharpen hard negative discernment.

---

### 5. Case 5: Extreme Overhead Camera Angle

![Case 5: Extreme Overhead Camera Angle](file:///home/monarch/Desktop/PRO/task2/outputs/failure_05_steep_nadir_angle.jpg)

**Failure Mechanism:**
When the camera is mounted directly overhead (near 90-degree nadir angle), the human silhouette compresses vertically, foreshortening the helmet into a circular disc rather than a 3D dome.

**Root Cause:** Resolution limits, occlusions, and industrial lighting variations inherent to CCTV perspectives.

**Recommended Mitigation:**
Include perspective warping (perspective=0.001) and specific high-angle CCTV feeds in the training split.

---

