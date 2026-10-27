"""Regression checks for the error analysis and calibration input boundary."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from task2.evaluate import compare_boxes
from task2.quantize_pipeline import PPECalibrationDataReader
from utils.video_utils import frame_to_tensor


class Task2Tests(unittest.TestCase):
    def test_duplicate_predictions_do_not_hide_false_positives(self):
        gt = [{"class": 0, "bbox": [0, 0, 20, 20]}]
        predictions = [dict(gt[0]), dict(gt[0])]
        errors = compare_boxes(gt, predictions)
        self.assertEqual([e["kind"] for e in errors], ["unmatched_prediction"])

    def test_wrong_class_is_reported_even_when_localization_is_correct(self):
        errors = compare_boxes([{"class": 1, "bbox": [0, 0, 20, 20]}],
                               [{"class": 0, "bbox": [0, 0, 20, 20]}])
        self.assertEqual([e["kind"] for e in errors], ["class_confusion"])

    def test_no_predictions_reports_a_missed_head(self):
        errors = compare_boxes([{"class": 1, "bbox": [0, 0, 20, 20]}], [])
        self.assertEqual([e["kind"] for e in errors], ["missed_ground_truth"])

    def test_letterbox_keeps_color_order_and_aspect_ratio(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:] = (0, 0, 255)
        tensor = frame_to_tensor(image)
        self.assertEqual(tensor.shape, (1, 3, 416, 416))
        np.testing.assert_allclose(tensor[0, :, 208, 208], [1, 0, 0])
        np.testing.assert_allclose(tensor[0, :, 0, 208], [114/255] * 3)

    def test_unreadable_calibration_image_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = PPECalibrationDataReader([Path(directory) / "missing.jpg"], "images")
            with self.assertRaises(ValueError):
                reader.get_next()

    def test_calibration_rewind_repeats_the_same_real_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "real.png"
            cv2.imwrite(str(path), np.full((30, 40, 3), 127, np.uint8))
            reader = PPECalibrationDataReader([path], "images")
            first = reader.get_next()["images"]
            self.assertIsNone(reader.get_next())
            reader.rewind()
            np.testing.assert_array_equal(first, reader.get_next()["images"])


if __name__ == "__main__":
    unittest.main()
