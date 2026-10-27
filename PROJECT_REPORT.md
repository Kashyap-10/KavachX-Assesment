# KavachX AI/ML Project Report
Prepared 26 October 2026

## Implementation

Task 1 converts an English monitoring rule into a structured configuration, registers open-vocabulary targets with YOLO-World, tracks them using ByteTrack, measures qualifying dwell inside a polygon, masks evidence and emits schema-validated JSON events. The demonstration includes a bag rule, a car rule using the same weights, and a five-second control.

Task 2 fine-tunes YOLOv8n at 416 pixels for 15 epochs on 240 training / 60 validation images. The package includes the checkpoint, training logs and plots, validation outputs, five reviewed examples, FP32/FP16/INT8 ONNX exports, static calibration from 100 training images and CPU benchmark scripts.

Task 3 uses local CLIP ViT-B/32 with explicit positive/negative text prompt ensembles. A threshold is selected on 20 calibration images and evaluated on 30 separate test images. Outputs include every score, decision, threshold sweep and precision/recall comparison. Cost and feedback-workflow notes are included.

## Measured results

Task 1 emitted one bag candidate at video time 34.3 seconds after 10.1 seconds of qualifying dwell; needs_verification=true. The car control emitted one event at 3.1 seconds. The five-second control emitted zero. Independent pedestrian identity agreement covered 143 sampled observations, with mean box IoU 0.8557.

| PPE model | Size (MiB) | CPU latency (ms) | mAP@50 | helmet recall | no_helmet recall |
|---|---:|---:|---:|---:|---:|
| FP32 | 11.580 | 12.390 | 0.501414 | 0.656595 | 0.294118 |
| FP16 | 5.829 | 13.419 | 0.501651 | 0.655862 | 0.294118 |
| INT8 | 3.197 | 7.411 | 0.491185 | 0.660159 | 0.235294 |

PPE measurements use Apple M4 CPU, ONNX Runtime, batch 1, four threads, ten warmups and 100 timed runs. Latency covers model execution. INT8 is approximately 3.62 times smaller and 1.67 times faster than FP32 on this host; mAP differs by 1.023 percentage points.

| Snapshot pipeline | TP | FP | FN | TN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Forward all candidates | 15 | 15 | 0 | 0 | 50.0% | 100.0% |
| CLIP, threshold 0.3187942803 | 11 | 6 | 4 | 9 | 64.7% | 73.3% |

CLIP's warm CPU average is 18.79 ms per image, p95 19.48 ms, including image decoding/preprocessing/scoring. At 200 alerts/day over 30 days, the local setup performs 6,000 verifications with $0 hosted inference API charges. Hardware and operations are separate from API charges.

## Execution and verification

The project runs on Python 3.12. Install requirements.txt and follow README.md for exact commands. Tasks 1 and 3 use pretrained models; Task 2's checkpoint is included. The Colab launcher creates a separate managed Python 3.12 environment. A GPU is used for the supplied training workflow.

The recorded regression run passed 29 tests: 17 for Task 1, six for Task 2 and six for Task 3. Fresh Task 3 inference reproduced all 50 stored scores exactly in a separate local environment. Input hashes, recording splits, source annotations and packaged artifacts were checked.

## Demonstration sources and authorship

Task 1: MOTChallenge MOT17-04. Task 2: public Ultralytics Construction-PPE subset. Task 3: BIRDSAI real thermal images, with labels indicating human presence. These are attributed substitute demonstration assets. Model, source and dataset metadata are included in the package. The measured results refer to saved local runs.

OpenAI Codex assisted implementation, source discovery, debugging, verification and documentation. Candidate review and understanding remain part of the handover. The delivered package contains implementation and measured demonstrations; this report makes no blanket claim that every assessment requirement has been completed.
