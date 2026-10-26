# Task 2 Benchmark Results

| Model Variant      |   Size (MB) | CPU Latency (ms)   |   mAP@50 |   no_helmet Recall |
|:-------------------|------------:|:-------------------|---------:|-------------------:|
| FP32 ONNX          |        12.2 | 28.4               |    0.865 |              0.842 |
| FP16 ONNX          |         6.1 | 25.1               |    0.863 |              0.84  |
| INT8 ONNX          |         3.3 | 11.2               |    0.851 |              0.828 |
| INT8 + Motion Gate |         3.3 | 3.61 (amortized)   |    0.851 |              0.828 |

### Observations:
- **INT8 Quantization:** Achieves ~3.7x model compression with less than 1.5 mAP degradation.
- **Motion Gating:** Drastically cuts detector executions on static surveillance footage by ~78%, yielding an effective amortized latency of under 4ms per frame on CPU.
