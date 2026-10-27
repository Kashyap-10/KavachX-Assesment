"""Task 1 regression tests; controlled fixtures never count as demo evidence."""
import base64
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import jsonschema
import numpy as np
from shapely.geometry import Polygon

from task1.detector import Detection, OpenVocabDetector
from task1.event_emitter import EventEmitter
from task1.privacy import PrivacyMasker
from task1.rule_parser import RuleParser
from task1.run_task1 import ROOT, run_pipeline
from task1.tracker import DwellTracker, TrackViolation
from utils.geometry import load_zones

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
ZONE = Polygon([(0,0), (100,0), (100,100), (0,100)])


def config(duration=1, comparison="more than", item="bag"):
    attribute = f" carrying a {item}" if item else ""
    return RuleParser().parse(
        f"Alert me if a person{attribute} enters the walkway and stays {comparison} {duration} seconds.")


def person(attributes=None, box=None):
    return Detection(box or [10,10,40,90], .9, "person",
                     attributes if attributes is not None else {"carrying": "bag"}, .8)


class ParserTests(unittest.TestCase):
    def test_original_example_extracts_all_fields(self):
        result = RuleParser().parse(
            "Alert me if a person carrying a bag enters the loading bay and stays more than 10 seconds.")
        self.assertEqual((result.target_class, result.attributes, result.zone,
                          result.dwell_threshold_s, result.comparison),
                         ("person", {"carrying": "bag"}, "loading_bay", 10, "gt"))

    def test_new_nouns_and_minute_units_are_not_defaulted(self):
        result = RuleParser().parse(
            "Notify me if a forklift enters the storage area and remains at least 2.5 minutes.")
        self.assertEqual((result.target_class, result.zone, result.dwell_threshold_s, result.comparison),
                         ("forklift", "storage_area", 150, "gte"))
        helmet = RuleParser().parse("Alert if a helmet enters the walkway and stays over 5 s.")
        self.assertEqual(helmet.target_class, "helmet")

    def test_wearing_and_multiword_item(self):
        result = RuleParser().parse(
            "Alert me if a worker wearing a hard hat enters the loading bay and stays at least 2 m.")
        self.assertEqual((result.target_class, result.attributes, result.dwell_threshold_s),
                         ("person", {"wearing": "hard hat"}, 120))

    def test_missing_fields_negation_and_compound_rules_fail(self):
        for text in ["Alert me about people", "Alert me if a person enters the walkway.",
                     "Alert me if a person without a helmet enters the walkway and stays over 2 seconds.",
                     "Alert me if a person carrying a bag and a toolbox enters the walkway and stays over 2 seconds.",
                     "Alert me if a person enters the walkway and stays more than 0 seconds."]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                RuleParser().parse(text)

    def test_threshold_and_comparison_change_rule_identity(self):
        self.assertNotEqual(config(1).rule_id, config(2).rule_id)
        self.assertNotEqual(config(1).rule_id, config(1, "at least").rule_id)


class AssociationTests(unittest.TestCase):
    def detector(self, relation, item, allowed):
        detector = OpenVocabDetector.__new__(OpenVocabDetector)
        detector.attributes = {relation: item}
        detector.item_labels = allowed
        return detector

    def test_backpack_cannot_satisfy_toolbox(self):
        detector = self.detector("carrying", "toolbox", ["toolbox"])
        targets = detector.associate([person({})], [Detection([20,40,30,60], .9, "backpack", {})])
        self.assertEqual(targets[0].attributes, {})

    def test_wearing_requires_an_item_near_the_head(self):
        detector = self.detector("wearing", "helmet", ["helmet", "hard hat"])
        self.assertEqual(detector.associate([person({})], [])[0].attributes, {})
        low = Detection([20,70,30,85], .9, "helmet", {})
        self.assertEqual(detector.associate([person({})], [low])[0].attributes, {})
        head = Detection([15,5,35,25], .7, "helmet", {})
        self.assertEqual(detector.associate([person({})], [head])[0].attributes, {"wearing":"helmet"})

    def test_one_bag_and_duplicate_prompts_do_not_qualify_two_people(self):
        detector = self.detector("carrying", "bag", ["bag", "backpack"])
        targets = [person({}, [0,0,50,100]), person({}, [25,0,75,100])]
        items = [Detection([25,30,35,50], .8, "bag", {}),
                 Detection([26,31,36,51], .7, "backpack", {})]
        result = detector.associate(targets, items)
        self.assertEqual(sum(bool(t.attributes) for t in result), 1)


class DwellTests(unittest.TestCase):
    def test_strict_threshold_and_exactly_one_event(self):
        tracker = DwellTracker(config(), ZONE, 4, START)
        events = []
        for i in range(9):
            current = tracker.update([person()], i/4)
            if i <= 4:
                self.assertEqual(current, [])
            events.extend(current)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].dwell_s, 1.25)
        self.assertEqual((datetime.fromisoformat(events[0].ts_end) -
                          datetime.fromisoformat(events[0].ts_start)).total_seconds(), 1.25)

    def test_at_least_includes_exact_boundary(self):
        tracker = DwellTracker(config(comparison="at least"), ZONE, 4, START)
        events = []
        for i in range(5):
            events.extend(tracker.update([person()], i/4))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].dwell_s, 1)

    def test_missing_attribute_preserves_id_but_does_not_add_dwell(self):
        tracker = DwellTracker(config(100), ZONE, 4, START)
        for i in range(3):
            tracker.update([person()], i/4)
        track_id = tracker.current[0]["track_id"]
        tracker.update([person({})], .75)
        tracker.update([person()], 1)
        self.assertEqual(tracker.current[0]["track_id"], track_id)
        self.assertEqual(tracker.current[0]["dwell_s"], .5)

    def test_occlusion_and_outside_time_do_not_count(self):
        tracker = DwellTracker(config(100), ZONE, 4, START)
        tracker.update([person()], 0)
        tracker.update([person()], .25)
        tracker.update([], .5)
        tracker.update([person()], .75)
        self.assertEqual(tracker.current[0]["dwell_s"], .25)
        tracker.update([person()], 1)
        self.assertEqual(tracker.current[0]["dwell_s"], .5)
        # A separate subject outside the zone can never violate the rule.
        outside = DwellTracker(config(.1), ZONE, 4, START)
        events = []
        for i in range(8):
            events.extend(outside.update([person(box=[110,10,140,90])], i/4))
        self.assertEqual(events, [])

    def test_nonmonotonic_time_is_rejected(self):
        tracker = DwellTracker(config(), ZONE, 4, START)
        tracker.update([], 1)
        with self.assertRaises(ValueError):
            tracker.update([], .9)


class EvidenceTests(unittest.TestCase):
    def violation(self, confidence=.5):
        return TrackViolation(1, "person", {"carrying":"bag"}, confidence, 1.25,
                              "walkway", START.isoformat(), "2026-01-01T00:00:01.250000+00:00")

    def test_schema_masked_jpeg_and_verification_flag(self):
        emitter = EventEmitter(ROOT / "event_schema.json")
        image = np.zeros((100,100,3), dtype=np.uint8)
        event = emitter.create_event(config(), self.violation(), image)
        self.assertTrue(event["needs_verification"])
        self.assertFalse(emitter.create_event(config(), self.violation(.9), image)["needs_verification"])
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(event["evidence_frame"]), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, image.shape)
        with self.assertRaises(ValueError):
            emitter.create_event(config(), self.violation(float("nan")), image)
        event["confidence"] = 2
        with self.assertRaises(jsonschema.ValidationError), tempfile.TemporaryDirectory() as temp:
            emitter.save_event(event, temp)

    def test_privacy_covers_nonqualifying_people_and_background(self):
        image = np.random.default_rng(42).integers(0,256,(120,160,3), dtype=np.uint8)
        original = image.copy()
        masked = PrivacyMasker().mask_evidence_frame(image, [[10,10,50,100], [80,10,130,100]])
        np.testing.assert_array_equal(image, original)
        self.assertLess(masked[15:30,20:40].std(), 3)
        self.assertLess(masked[15:30,90:115].std(), 3)
        # Also blur areas outside all detected boxes, protecting missed background faces.
        self.assertLess(masked[90:110,135:155].std(), original[90:110,135:155].std()/4)

    def test_invalid_polygon_and_unknown_zone_fail_without_a_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"zones.json"
            path.write_text(json.dumps({"walkway":[[0,0],[10,10],[0,10],[10,0]]}))
            with self.assertRaises(ValueError):
                load_zones(path)
            path.write_text(json.dumps({"other":[[0,0],[10,0],[10,10]]}))
            output = Path(temp)/"out"
            with self.assertRaisesRegex(ValueError, "Unknown zone"):
                run_pipeline(config().rule_text, Path(temp)/"missing.mp4", path, output)
            self.assertFalse(output.exists())

    def test_controlled_end_to_end_outputs_match_embedded_jpeg(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root/"fixture.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4, (64,64))
            image = np.random.default_rng(1).integers(0,256,(64,64,3), dtype=np.uint8)
            for _ in range(16):
                writer.write(image)
            writer.release()
            zone = root/"zones.json"
            zone.write_text(json.dumps({"walkway":[[0,0],[63,0],[63,63],[0,63]]}))
            weights = root/"fixture-model.txt"
            weights.write_text("Controlled test fixture, no neural inference")
            detector = SimpleNamespace(model=SimpleNamespace(ckpt_path=str(weights)),
                                       vocab=["person","bag"], confidence_threshold=.1,
                                       person_boxes=[[10,5,35,50]],
                                       detect=lambda frame: [person(box=[10,5,35,50])])
            output = root/"out"
            with patch("task1.run_task1.OpenVocabDetector", return_value=detector):
                manifest = run_pipeline(config().rule_text, video, zone, output,
                                        stride=1, video_start=START.isoformat())
            self.assertEqual(len(manifest["events"]), 1)
            self.assertEqual(manifest["review_frames"], 3)
            event = json.loads((output/"sample_event.json").read_text())
            evidence = output/manifest["events"][0]["evidence_file"]
            self.assertEqual(base64.b64decode(event["evidence_frame"]), evidence.read_bytes())
            self.assertEqual(len(list(output.glob("event_*.json"))), 1)
            with self.assertRaises(FileExistsError):
                run_pipeline(config().rule_text, video, zone, output, stride=1)


if __name__ == "__main__":
    unittest.main()
