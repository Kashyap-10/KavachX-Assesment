"""Meaningful boundaries for calibration, metrics and input integrity. No model download."""
import csv
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from task3.evaluate import choose_threshold, evaluate_verifier_metrics, metrics, read_scores
from task3.verifier import load_samples, pad_image


class Task3Tests(unittest.TestCase):
    def test_metrics_count_missed_positive_and_false_alarm(self):
        result = metrics([1, 1, 0, 0], [True, False, True, False])
        self.assertEqual([result[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["recall"], .5)
        self.assertIsNone(metrics([1, 0], [False, False])["precision"])

    def test_recall_floor_overrides_better_precision_and_boundary_is_inclusive(self):
        rows = [{"label": label, "verifier_score": score} for label, score in
                [(1, .2), (1, .8), (0, .3), (0, .1)]]
        selected, _ = choose_threshold(rows, .95)
        self.assertEqual(selected["threshold"], .2)
        self.assertEqual((selected["tp"], selected["fp"]), (2, 1))
        self.assertEqual(choose_threshold(rows, .5)[0]["threshold"], .8)

    def test_padding_preserves_edges_and_aspect_ratio(self):
        image = Image.new("RGB", (200, 100), "white")
        image.putpixel((0, 50), (255, 0, 0))
        padded = pad_image(image, 200)
        self.assertEqual(padded.getpixel((0, 100)), (255, 0, 0))
        self.assertEqual(padded.getpixel((100, 0)), (0, 0, 0))
        self.assertEqual(padded.getpixel((199, 100)), (255, 255, 255))

    def test_duplicates_traversal_hash_mismatch_and_recording_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2, 2), "white").save(root / "a.png")
            Image.new("RGB", (2, 2), "black").save(root / "b.png")
            base = [{"filename": "a.png", "label": "1", "split": "calibration", "source_group": "A"},
                    {"filename": "b.png", "label": "0", "split": "test", "source_group": "B"}]
            for change in [{"filename": "../escape.png"}, {"filename": "a.png"},
                           {"sha256": "wrong"}, {"source_group": "A"}]:
                with self.subTest(change=change):
                    rows = [base[0], {**base[1], **change}]
                    fields = sorted(set().union(*(row.keys() for row in rows)))
                    with (root / "labels.csv").open("w", newline="") as file:
                        writer = csv.DictWriter(file, fieldnames=fields)
                        writer.writeheader(); writer.writerows(rows)
                    with self.assertRaises(ValueError):
                        load_samples(root / "labels.csv", root)
            (root / "b.png").write_bytes((root / "a.png").read_bytes())
            with (root / "labels.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(base[0]))
                writer.writeheader(); writer.writerows(base)
            with self.assertRaisesRegex(ValueError, "Duplicate image content"):
                load_samples(root / "labels.csv", root)

    def test_score_table_rejects_nan_and_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            path.write_text("filename,label,split,source_group,verifier_score\na,1,test,A,nan\n")
            with self.assertRaisesRegex(ValueError, "finite"):
                read_scores(path)
            path.write_text("filename,label\na,1\n")
            with self.assertRaisesRegex(ValueError, "columns"):
                read_scores(path)

    def test_test_labels_cannot_change_selected_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(filename=str(i), label=label, split=split, source_group=split,
                         verifier_score=score) for i, (label, split, score) in enumerate([
                             (1, "calibration", .4), (0, "calibration", .3),
                             (1, "test", .8), (0, "test", .2)])]
            thresholds = []
            for iteration in range(2):
                with (root / "scores.csv").open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                with contextlib.redirect_stdout(io.StringIO()):
                    report = evaluate_verifier_metrics(root / "scores.csv", root / str(iteration))
                thresholds.append(report["threshold"])
                for row in rows:
                    if row["split"] == "test":
                        row["label"] = 1 - row["label"]
            self.assertEqual(thresholds, [.4, .4])


if __name__ == "__main__":
    unittest.main()
