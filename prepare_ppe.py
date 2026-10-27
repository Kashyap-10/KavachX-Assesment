"""Build a reproducible two-class substitute from extracted Construction-PPE data."""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.glob("images/*/*")):
        raise FileExistsError(f"Dataset already exists: {output}")
    rng = random.Random(42)
    hashes, rows = [], []
    for split, count in [("train", 240), ("val", 60)]:
        candidates = sorted((source / "images" / split).glob("*"))
        rng.shuffle(candidates)
        accepted = 0
        for image in candidates:
            label = source / "labels" / split / (image.stem + ".txt")
            if not label.is_file():
                raise FileNotFoundError(label)
            labels = []
            for line in label.read_text().splitlines():
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Invalid YOLO label: {label}")
                cls = int(fields[0])
                coords = list(map(float, fields[1:]))
                if not all(0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
                    raise ValueError(f"Invalid YOLO box: {label}")
                if cls in (0, 7):
                    labels.append(" ".join([str({0: 0, 7: 1}[cls]), *fields[1:]]))
            if not labels:
                continue
            pixels = cv2.imread(str(image))
            if pixels is None:
                raise ValueError(f"Unreadable image: {image}")
            gray = cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY), (32, 32))
            dct = cv2.dct(gray.astype(np.float32))[:8, :8].flatten()[1:]
            phash = dct > np.median(dct)
            # Reject near duplicates within and across the selected splits.
            if any(np.count_nonzero(phash != prior) <= 6 for prior in hashes):
                continue
            hashes.append(phash)
            image_out = output / "images" / split / image.name
            label_out = output / "labels" / split / label.name
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, image_out)
            label_out.write_text("\n".join(labels) + "\n")
            rows.append(dict(split=split, source=str(image.relative_to(source)),
                             sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                             labels=labels))
            accepted += 1
            if accepted == count:
                break
        if accepted != count:
            raise ValueError(f"Need {count} usable {split} images; found {accepted}")
    # Ultralytics resolves path relative to its datasets setting, so use an explicit root.
    (output / "data.yaml").write_text(
        f"path: {output.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: helmet\n  1: no_helmet\n")
    manifest = dict(source="https://docs.ultralytics.com/datasets/detect/construction-ppe/",
                    license="AGPL-3.0 (as declared by the publisher)", seed=42,
                    substitute=True, original_class_mapping={"0": 0, "7": 1},
                    duplicate_filter="63-bit perceptual hash Hamming distance > 6",
                    limitation="Source scene/worker IDs unavailable; scene independence is not established.",
                    images=rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(rows)} real source images at {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="Extracted official archive")
    p.add_argument("--output", default="data/ppe")
    a = p.parse_args()
    prepare(a.source, a.output)
