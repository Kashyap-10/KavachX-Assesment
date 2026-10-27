"""Audit input provenance, saved decisions and optional fresh CLIP inference."""
import argparse
import hashlib
import json
from pathlib import Path

from evaluate import choose_threshold, metrics, read_scores
from prepare_demo import ROOT, prepare
from verifier import CascadedVerifier, load_samples


def verify(run_dir, rescore=False):
    prepare()
    run = Path(run_dir)
    image_dir = ROOT / "alerts/task3_demo"
    labels = image_dir / "labels.csv"
    inputs = load_samples(labels, image_dir)
    rows = read_scores(run / "verifier_scores.csv")
    config = json.loads((run / "scoring_config.json").read_text())
    config_digest = config.pop("configuration_sha256")
    if hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() != config_digest:
        raise ValueError("Scoring configuration fingerprint differs")
    if hashlib.sha256(labels.read_bytes()).hexdigest() != config["labels_sha256"]:
        raise ValueError("Labels differ from scoring run")
    if hashlib.sha256((ROOT / "task3/verifier.py").read_bytes()).hexdigest() != config["verifier_source_sha256"]:
        raise ValueError("Verifier implementation changed since scoring")
    if json.loads((ROOT / "task3/prompts.json").read_text()) != config["prompts"]:
        raise ValueError("Prompt configuration differs")
    if len(rows) != len(inputs) or any(any(row.get(k) != sample[k] for k in
        ("filename", "label", "split", "source_group", "sha256")) for row, sample in zip(rows, inputs)):
        raise ValueError("Score rows do not match the verified images and labels")
    calibration = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "test"]
    if len(calibration) != 20 or len(test) != 30:
        raise ValueError("Expected 20 calibration and 30 test images")
    policy = json.loads((run / "threshold.json").read_text())
    if hashlib.sha256((run / "verifier_scores.csv").read_bytes()).hexdigest() != policy["scores_sha256"]:
        raise ValueError("Scores differ from the selected policy")
    selected, _ = choose_threshold(calibration, policy["minimum_calibration_recall"])
    if selected["threshold"] != policy["threshold"]:
        raise ValueError("Threshold is not the calibration-only selection")
    report = json.loads((run / "evaluation.json").read_text())
    truth = [r["label"] for r in test]
    if metrics(truth, [True] * len(test)) != report["edge_alone"] or metrics(
        truth, [r["verifier_score"] >= policy["threshold"] for r in test]) != report["edge_plus_verifier"]:
        raise ValueError("Reported metrics differ from the score table")
    max_error = None
    if rescore:
        model = CascadedVerifier()
        if model.weights_sha256 != config["weights_sha256"]:
            raise ValueError("Model weights differ")
        errors = [abs(model.score_image(image_dir / row["filename"]) - row["verifier_score"]) for row in rows]
        max_error = max(errors)
        if max_error > 1e-5:
            raise ValueError(f"Fresh inference differs by {max_error}; inspect runtime versions")
    result = {"verified_images": len(rows), "test_images": len(test), "calibration_images": len(calibration),
              "fresh_inference": rescore, "maximum_score_error": max_error,
              "status": "Artifact integrity verified; this does not establish deployment accuracy"}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(ROOT / "task3/outputs/real_demo"))
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    verify(args.run_dir, args.rescore)
