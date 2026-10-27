"""Restore and verify the fixed public BIRDSAI subset; never synthesize missing data."""
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checked_asset(path, url, digest):
    if not path.exists():
        data = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Downloaded asset hash mismatch: {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Existing asset hash mismatch: {path}")


def prepare():
    manifest = json.loads((ROOT / "task3/demo_source/manifest.json").read_text())
    target = ROOT / "alerts/task3_demo"
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in manifest["samples"]:
        annotation = ROOT / sample["annotation_file"]
        checked_asset(annotation, sample["annotation_url"], sample["annotation_sha256"])
        frame_rows = []
        with annotation.open(newline="") as file:
            for raw in csv.reader(file):
                try:
                    values = list(map(float, raw[:10]))
                except ValueError:  # Publisher header, if present.
                    continue
                if values and int(values[0]) == sample["frame"]:
                    frame_rows.append(values)
        if not frame_rows or frame_rows != sample["rows"]:
            raise ValueError(f"Source frame annotations differ: {sample['filename']}")
        label = int(any(row[6] == 1 for row in frame_rows))
        if label != sample["label"] or (not label and not any(row[6] == 0 for row in frame_rows)):
            raise ValueError("Proxy label does not match original human/animal annotations")
        expected_partition = "train" if sample["split"] == "calibration" else "test"
        if sample["original_partition"] != expected_partition:
            raise ValueError("Publisher TrainReal/TestReal split was not preserved")
        checked_asset(target / sample["filename"], sample["url"], sample["sha256"])
        rows.append({**{key: sample[key] for key in
                     ("filename", "label", "split", "source_group", "sequence", "frame")},
                     "negative_kind": "animal_only" if label == 0 else "", "sha256": sample["sha256"]})
    with (target / "labels.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} real images and original frame annotations")


if __name__ == "__main__":
    prepare()
