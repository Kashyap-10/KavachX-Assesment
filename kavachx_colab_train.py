"""KavachX PPE training on Google Colab (T4).

Install first:
  %pip install -q ultralytics==8.3.235 numpy==2.2.6 opencv-python==4.12.0.88 onnx==1.19.1 onnxruntime==1.23.2

Then run this file with %run, or upload the accompanying notebook.
Uses real PUBLIC SUBSTITUTE data, not the missing KavachX dataset.
"""
import argparse
import hashlib
import json
import logging
import platform
import random
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

DATA_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
DATA_SHA256 = "bef8dcb599aa4e9d9f5e602cb6fa7143d3c84d7f6a0ff40463d7f2a4c2632ccc"


def prepare_dataset(archive, root):
    source = root / "source"
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if not (source / member.filename).resolve().is_relative_to(source.resolve()):
                raise ValueError("Unsafe archive path")
        z.extractall(source)
    dataset = root / "bundle" / "data" / "ppe"
    if any(dataset.glob("images/*/*")):
        raise FileExistsError("Dataset already prepared. Use a new --work-dir.")
    rng = random.Random(42)
    hashes, inventory = [], []
    counts = {}
    for split, required in [("train", 240), ("val", 60)]:
        candidates = sorted((source / "images" / split).glob("*"))
        rng.shuffle(candidates)
        accepted = 0
        instances = {"helmet": 0, "no_helmet": 0}
        for image in candidates:
            label = source / "labels" / split / (image.stem + ".txt")
            if not label.is_file():
                raise FileNotFoundError(label)
            remapped = []
            for line in label.read_text().splitlines():
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Invalid YOLO label in {label}")
                class_id = int(fields[0])
                coords = list(map(float, fields[1:]))
                if not all(0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
                    raise ValueError(f"Invalid bounding box in {label}")
                if class_id in (0, 7):
                    remapped.append(" ".join([str({0: 0, 7: 1}[class_id]), *fields[1:]]))
            if not remapped:
                continue
            pixels = cv2.imread(str(image))
            if pixels is None:
                raise ValueError(f"Unreadable image: {image}")
            gray = cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY), (32, 32))
            dct = cv2.dct(gray.astype(np.float32))[:8, :8].flatten()[1:]
            phash = dct > np.median(dct)
            if any(np.count_nonzero(phash != previous) <= 6 for previous in hashes):
                continue
            hashes.append(phash)
            image_out = dataset / "images" / split / image.name
            label_out = dataset / "labels" / split / label.name
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, image_out)
            label_out.write_text("\n".join(remapped) + "\n")
            for line in remapped:
                instances[["helmet", "no_helmet"][int(line.split()[0])]] += 1
            inventory.append({
                "split": split, "source": str(image.relative_to(source)),
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "labels": remapped,
            })
            accepted += 1
            if accepted == required:
                break
        if accepted != required or min(instances.values()) == 0:
            raise ValueError(f"Insufficient usable data in {split}: {accepted}, {instances}")
        counts[split] = {"images": accepted, "instances": instances}
    manifest = {
        "substitute": True, "source": DATA_URL, "source_sha256": DATA_SHA256,
        "documentation": "https://docs.ultralytics.com/datasets/detect/construction-ppe/",
        "license": "AGPL-3.0 as declared by Ultralytics",
        "attribution": "Construction-PPE: Dalvi, Singh, Bhingarde and Chalke (2025), Ultralytics",
        "seed": 42, "original_class_mapping": {"0": 0, "7": 1},
        "counts": counts, "images": inventory,
        "duplicate_filter": "63-bit perceptual hash; reject Hamming distance <= 6",
        "limitations": [
            "This is public construction imagery, not the missing factory CCTV dataset.",
            "Original train/val boundaries retained; source scene and worker IDs unavailable.",
            "Perceptual deduplication reduces leakage but does not establish scene independence.",
            "Only 17 no_helmet validation boxes in this fixed subset; recall is uncertain.",
        ],
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest, indent=2))
    yaml_path = dataset / "data.yaml"
    yaml_path.write_text(
        f"path: {dataset.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: helmet\n  1: no_helmet\n"
    )
    print(json.dumps(counts, indent=2), flush=True)
    return yaml_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", default="/content/kavachx_" +
                   datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--archive", help="Existing official zip; otherwise download it")
    p.add_argument("--prepare-only", action="store_true", help="Data audit without training")
    args = p.parse_args(argv)
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Training must use Colab CUDA. No silent CPU fallback.
    if not args.prepare_only:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU. In Colab select Runtime > Change runtime type > T4 GPU.")
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    archive = Path(args.archive).resolve() if args.archive else root / "construction-ppe.zip"
    if not archive.is_file():
        print("Downloading real Construction-PPE images (about 170 MiB)...", flush=True)
        urllib.request.urlretrieve(DATA_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"Source archive changed: {digest}. Verify the source before proceeding.")
    data_yaml = prepare_dataset(archive, root)
    if args.prepare_only:
        return data_yaml

    import torch
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    bundle = root / "bundle"
    output = bundle / "task2" / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(output / "training.log")
    LOGGER.addHandler(handler)
    try:
        settings = dict(
            data=str(data_yaml), epochs=15, imgsz=416, batch=16, device=0,
            workers=2, seed=42, deterministic=True, project=str(output),
            name="ppe_yolov8n", exist_ok=False, plots=True,
            mosaic=1.0, close_mosaic=5, mixup=0.1, hsv_h=0.015, hsv_s=0.5,
            hsv_v=0.3, degrees=5.0, translate=0.1, scale=0.3,
            perspective=0.0005, flipud=0.0, fliplr=0.5,
        )
        (output / "training_config.json").write_text(json.dumps(settings, indent=2))
        environment = {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "source_archive_sha256": digest,
        }
        (output / "environment.json").write_text(json.dumps(environment, indent=2))
        (output / "environment.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        )
        model = YOLO("yolov8n.pt")
        model.train(**settings)
        best_path = Path(model.trainer.best)
        if not best_path.is_file():
            raise RuntimeError("Training did not produce best.pt")
        shutil.copy2(best_path, output / "best.pt")
        best = YOLO(str(output / "best.pt"))
        metrics = best.val(data=str(data_yaml), split="val", imgsz=416,
                           device=0, batch=16, workers=2, plots=True,
                           project=str(output), name="baseline_validation", exist_ok=False)
        report = {
            "dataset": "Public substitute Construction-PPE 240/60 subset",
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "per_class": [{k: v.item() if isinstance(v, np.generic) else v for k, v in row.items()}
                          for row in metrics.summary(normalize=False, decimals=6)],
            "recall_definition": "Ultralytics per-class recall at its validation operating point",
            "qualification": "Validation estimates only; not KavachX-data or production performance.",
        }
        (output / "baseline_metrics.json").write_text(json.dumps(report, indent=2))
        print("\nMEASURED VALIDATION RESULTS\n" + json.dumps(report, indent=2), flush=True)
        # FP32 is the input to the later CPU quantization/benchmark pipeline.
        exported = best.export(format="onnx", imgsz=416, batch=1, half=False,
                               dynamic=False, simplify=False, opset=17, device="cpu")
        shutil.copy2(exported, output / "model_fp32.onnx")
        (bundle / "README_TRAINING.txt").write_text(
            "Real 15-epoch YOLOv8n run, 416px, on a public substitute dataset.\n"
            "Attribution: Construction-PPE, Dalvi et al. (2025), Ultralytics, AGPL-3.0.\n"
            "https://docs.ultralytics.com/datasets/detect/construction-ppe/\n"
            "See data/ppe/manifest.json for provenance, exact selected images and limitations.\n"
            "After extracting elsewhere, update data/ppe/data.yaml path to its new absolute root.\n"
            "Contains best.pt, last.pt, FP32 ONNX, raw results.csv, logs, plots, validation metrics, "
            "training settings, environment, and the exact 240/60 dataset.\n"
            "Not included: measured FP16/INT8 accuracy, CPU benchmarks, real failure-case selection, "
            "motion-gating or Task 1/3 results. Those must be completed separately.\n"
            "Do not claim these results came from the original missing KavachX assets.\n"
        )
    finally:
        LOGGER.removeHandler(handler)
        handler.close()

    # Archive only after every required training/evaluation/export step succeeded.
    destination = root / "kavachx_training_results"
    shutil.make_archive(str(destination), "zip", root_dir=bundle)
    archive_path = destination.with_suffix(".zip")
    print("\nSUCCESS. Download and send back:", archive_path, flush=True)
    return archive_path


if __name__ == "__main__":
    main()
