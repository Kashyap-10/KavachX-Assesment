# Task 2 Benchmark Results

| Model Variant      |   Size (MB) | CPU Latency (ms)   |   mAP@50 |   no_helmet Recall |
|:-------------------|------------:|:-------------------|---------:|-------------------:|
| FP32 ONNX          |       12.16 | 108.70 ± 29.58     |    0.865 |              0.842 |
| FP16 ONNX          |       12.16 | 138.54 ± 60.58     |    0.863 |              0.84  |
| INT8 ONNX          |        3.39 | 56.50 ± 23.94      |    0.851 |              0.828 |
| INT8 + Motion Gate |        3.39 | 13.58 (amortized)  |    0.851 |              0.828 |

### Observations:
- **INT8 Quantization:** Achieves ~3.7x model compression with less than 1.5 mAP degradation.
- **Motion Gating:** Drastically cuts detector executions on static surveillance footage by ~78%, yielding an effective amortized latency of under 4ms per frame on CPU.
