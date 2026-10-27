# KavachX: recovered training and measured model exports

The recovered Colab training completed 15 epochs. Its checkpoint loads locally and the exact 240 training / 60 validation images match the source manifest. No retraining was performed during recovery.

## Measured results

All comparisons below use the same 60 validation images, 416 x 416 square letterboxing and batch size 1. CPU timings are ONNX Runtime model execution only on an Apple M4 with four intra-op threads, ten warm-ups and 100 timed runs. Decode, preprocessing and NMS are excluded.

| Model | Size (MiB) | Mean CPU latency (ms) | mAP@50 | helmet recall | no_helmet recall |
|---|---:|---:|---:|---:|---:|
| FP32 ONNX | 11.580 | 12.390 | 0.501414 | 0.656595 | 0.294118 |
| FP16 ONNX | 5.829 | 13.419 | 0.501651 | 0.655862 | 0.294118 |
| INT8 ONNX | 3.197 | 7.411 | 0.491185 | 0.660159 | 0.235294 |

The final INT8 model is about 3.62 times smaller and 1.67 times faster than FP32 on this host. Its mAP@50 drops 1.023 percentage points. Its missing-helmet recall drops 5.882 points, a material safety-related trade-off even though the brief's mAP condition is met.

The local FP32 score differs from Colab's reported 0.480 because this comparison fixes batch=1 and square 416 preprocessing; the original run used batch=16 and the default rectangular validation policy. Do not present these as the same evaluation configuration.

## Quantization diagnosis and mitigation

The first static QDQ conversion quantized all supported operations and yielded zero mAP. A direct real-image probe showed all class confidence scores rounded to zero. Final decoding combines box coordinates, which can exceed 400, and confidence values in [0,1]; sharing an 8-bit scale can erase the much smaller scores. This is the likely numerical mechanism; the recorded output-score regression and full validation establish the failure without claiming an internal activation trace.

Percentile calibration alone still produced zero class scores on the diagnostic image; it was not fully validated. Restricting static PTQ to convolution layers, while keeping decoding and confidence operations in float, restored nonzero scores and yielded the measured final INT8 accuracy above. Calibration uses 100 real training images, selected with seed 42. FP16 has real internal half-precision weights and FP32 input/output; CPU speedup is not assumed.

The failed model variants, hashes and first all-ops benchmark are preserved as diagnostics. Use model_int8.onnx, not an _all_ops model.

## Evidence and limitations

- Raw benchmark timings, model hashes, per-class metrics and environment: task2/outputs/benchmark_measurements.json.
- Checkpoint validation: task2/outputs/baseline_metrics.json.
- Calibration provenance: task2/outputs/quantization.json.
- Five actual diagnostic failures, source hashes, boxes and annotated images: task2/outputs/failure_cases.json and failure_01_actual.jpg through failure_05_actual.jpg.
- Visual interpretation: task2/outputs/failure_cases_visual_review.md.
- Original Colab results.csv, logs, plots, best/last weights, training configuration and environment are retained.

This is an attributed public Construction-PPE substitute, not the missing KavachX dataset. Publisher: Dalvi, Singh, Bhingarde and Chalke (2025), Ultralytics. Publisher-declared license: AGPL-3.0. Source: https://docs.ultralytics.com/datasets/detect/construction-ppe/.

Visual inspection found non-construction scenes and questionable source annotations, including bare heads labeled helmet. Only 17 no_helmet boxes occur in validation. Report these scores as a pipeline demonstration on fixed source labels, not factory-CCTV or production acceptance. Do not correct labels after inspecting validation results and then silently compare against the original score.

## Reproduce locally

Use Python 3.12. Extract this package into a new folder, create a virtual environment, and install requirements.txt. The Colab notebook keeps Colab's supplied CUDA-enabled PyTorch; these requirements are for the tested local CPU workflow.

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Update the dataset's absolute root after extraction:

    python - <<'PY'
    from pathlib import Path
    import yaml
    path = Path("data/ppe/data.yaml")
    config = yaml.safe_load(path.read_text())
    config["path"] = str(path.parent.resolve())
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    PY

Run evaluation and exports in this order. Run benchmarks without concurrent training/evaluation workloads:

    python task2/evaluate.py
    python task2/quantize_pipeline.py
    python task2/benchmark.py
    python -m unittest discover -s tests -v

Exporting regenerates calibration/model hashes for the current environment. Visual-review notes describe the recovered checkpoint and fixed five images; review newly generated failures if retraining changes the selected cases.

## Checks completed

- ZIP integrity and manifest verification for all 300 images.
- Saved checkpoint loaded as helmet/no_helmet.
- Full CPU validation for the checkpoint and all three recommended ONNX variants.
- ONNX checker plus finite-output inference for FP32, FP16 and INT8.
- Failed all-ops INT8 output reproduced; corrected convolution-only output verified.
- Six focused regression tests passed.
- Python compilation and Git whitespace checks passed.

## Remaining assignment work

This recovery package does not complete the whole assessment. Motion-gating measurement on suitable footage remains outstanding. Tasks 1 and 3 are not included or represented as completed. The original assessment datasets and clips remain unavailable.

The original Colab script failed after training with a NumPy environment error. The revised notebook runs the script in a fresh Python process after installation, and JSON serialization now handles NumPy scalar metric values. The revised full Colab workflow has not been rerun on the remote GPU.

AI assistance was used to prepare the substitute data workflow, repair evaluation/export scripts, diagnose quantization, inspect failure examples and write this report. The candidate must review and understand the code and disclose the assistance accurately.
