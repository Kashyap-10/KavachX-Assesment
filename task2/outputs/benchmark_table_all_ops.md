# Measured CPU benchmark

| Model Variant   |   Size (MiB) |   CPU Latency (ms) |   mAP@50 |   helmet Recall |   no_helmet Recall |
|:----------------|-------------:|-------------------:|---------:|----------------:|-------------------:|
| FP32 ONNX       |       11.58  |             12.465 | 0.501414 |        0.656595 |           0.294118 |
| FP16 ONNX       |        5.829 |             12.96  | 0.501651 |        0.655862 |           0.294118 |
| INT8 ONNX       |        3.238 |              5.318 | 0        |        0        |           0        |

INT8 vs FP32 mAP@50 drop: 50.141 percentage points. Mitigation is required (>2 points).

Latency is model execution only, measured on the recorded host with four ORT threads. FP16 uses internal half precision with FP32 I/O; CPU acceleration is not assumed. These are validation results on the documented public substitute, not the missing assessment dataset.
