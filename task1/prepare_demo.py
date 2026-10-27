"""Download the exact public research-demo clip, tracking labels and model weights."""
import hashlib
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "video": ("https://motchallenge.net/sequenceVideos/MOT17-04-DPM-raw.mp4",
              "ab0d2fc03bce5e112abe9a173de7a2d70b39875c0af3eca49d6160c6e317e495"),
    "labels": ("https://motchallenge.net/data/MOT17Labels.zip",
               "0aa79322e91583369f42f17c4d79a0b145380d8732487bba59272048dc82b2b9"),
    "model": ("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt",
              "9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792"),
}


def download(name, path):
    url, expected = ASSETS[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Existing {path} differs from the recorded source; it will not be overwritten")
        return path
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        candidate = Path(temp) / "download"
        print(f"Downloading {name}: {url}", flush=True)
        urllib.request.urlretrieve(url, candidate)
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Upstream {name} changed. Review it before using new bytes.")
        candidate.replace(path)
    return path


def prepare():
    video = download("video", ROOT / "clips/mot17-04.mp4")
    download("model", ROOT / "models/yolov8s-worldv2.pt")
    source = ROOT / "task1/demo_source"
    source.mkdir(parents=True, exist_ok=True)
    # Extract only the two named metadata files, never arbitrary archive paths.
    with tempfile.TemporaryDirectory() as temp:
        archive = download("labels", Path(temp) / "MOT17Labels.zip")
        with zipfile.ZipFile(archive) as z:
            for filename in ("gt/gt.txt", "seqinfo.ini"):
                data = z.read("train/MOT17-04-DPM/" + filename)
                destination = source / Path(filename).name
                if destination.exists() and destination.read_bytes() != data:
                    raise ValueError(f"Existing metadata differs: {destination}")
                destination.write_bytes(data)
    manifest = {
        "substitute": True, "sequence": "MOT17-04-DPM raw sequence preview",
        "publisher": "MOTChallenge, introduced in MOT16; see the official benchmark page",
        "source_page": "https://motchallenge.net/data/MOT17/",
        "license": "CC BY-NC-SA 3.0, as declared by MOTChallenge",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
        "use": "Private non-commercial research/evaluation demonstration",
        "assets": {name: {"url": url, "sha256": sha} for name, (url, sha) in ASSETS.items()},
        "video": {"filename": str(video.relative_to(ROOT)), "width": 960, "height": 540,
                  "fps": 30, "frames": 1050, "duration_s": 35},
        "transformations": "Video bytes unmodified. Evidence derivatives are blurred, mean-masked and annotated.",
        "ground_truth": {"filename": "task1/demo_source/gt.txt", "source_width": 1920,
                         "source_height": 1080, "coordinate_scale_to_video": .5,
                         "labels": "Pedestrian boxes/IDs only; no carrying or bag ground truth."},
        "zones": "task1/demo_zones.json: image-space walkway and parking polygons defined for this substitute.",
        "capture_time": "Unknown. Event ISO timestamps use an explicitly documented anchor.",
        "limitations": "This is a pedestrian street, not the original loading-bay footage.",
    }
    (source / "provenance.json").write_text(json.dumps(manifest, indent=2))
    print("Public substitute assets verified. Run Task 1 from the repository root.")


if __name__ == "__main__":
    prepare()
