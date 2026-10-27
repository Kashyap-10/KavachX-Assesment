# Task 3 measured verifier evaluation

Human presence proxy on real nighttime thermal images; not authorization or intent.

| Pipeline | TP | FP | FN | TN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| All candidate alerts sent | 15 | 15 | 0 | 0 | 0.500 | 1.000 |
| CLIP secondary verifier | 11 | 6 | 4 | 9 | 0.647 | 0.733 |

Threshold: **0.31879428**, selected on 20 calibration images, with empirical recall >= 95%. Test set: 30 images.
Test false-alarm reduction: 60.0%.

The score is not a calibrated probability. The threshold is optimal only under this finite calibration set and the declared selection rule; it is not a production optimum. Baseline recall of 1.0 means all labeled positives in these candidates are forwarded, not that an upstream edge model detects every intrusion.

The verifier holds 4 true-positive candidates in this test set. Held alerts must remain auditable; do not silently destroy them or infer intrusion intent from these proxy labels.
