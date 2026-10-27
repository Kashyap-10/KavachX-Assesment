"""Audit saved real-demo events against traces, schemas, source bytes and independent pedestrian labels."""
import argparse
import base64
import configparser
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import jsonschema
import numpy as np
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parents[1]


def box_iou(a, b):
    intersection = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0, min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / max(union, 1e-9)


def audit(run):
    manifest = json.loads((run/"run_manifest.json").read_text())
    trace = [json.loads(line) for line in (run/"trace.jsonl").read_text().splitlines()]
    video = ROOT/"clips"/manifest["video"]["filename"]
    assert hashlib.sha256(video.read_bytes()).hexdigest() == manifest["video"]["sha256"]
    model = ROOT/"models"/manifest["model"]["filename"]
    assert hashlib.sha256(model.read_bytes()).hexdigest() == manifest["model"]["sha256"]
    zones_path = ROOT/"task1/demo_zones.json"
    assert hashlib.sha256(zones_path.read_bytes()).hexdigest() == manifest["zones_sha256"]
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert (width,height,fps,total) == (960,540,30,1050)
    limit = min(total, manifest["source_frame_limit"] or total)
    assert [r["frame_index"] for r in trace] == list(range(0,limit,manifest["tracking"]["stride"]))
    assert len(trace) == manifest["processed_samples"]
    for row in trace:
        assert math.isclose(row["video_time_s"], row["frame_index"]/fps, abs_tol=1e-9)
    validator = jsonschema.Draft7Validator(json.loads((ROOT/"event_schema.json").read_text()),
                                          format_checker=jsonschema.FormatChecker())
    rule = manifest["rule"]
    polygon = Polygon(json.loads(zones_path.read_text())[rule["zone"]])
    anchor = datetime.fromisoformat(manifest["video_start_anchor"])
    records = manifest["events"]
    assert len({e["track_id"] for e in records}) == len(records), "Duplicate event for a tracked identity"
    assert len(list(run.glob("event_*.json"))) == len(records)
    event_results = []
    for record in records:
        event = json.loads((run/record["event_file"]).read_text())
        validator.validate(event)
        assert event["label"] == rule["target_class"] and event["attributes"] == rule["attributes"]
        assert event["rule_text"] == rule["rule_text"] and event["zone"] == rule["zone"]
        assert event["track_id"] == record["track_id"]
        image_bytes = base64.b64decode(event["evidence_frame"], validate=True)
        assert image_bytes == (run/record["evidence_file"]).read_bytes()
        image = cv2.imdecode(np.frombuffer(image_bytes,np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and image.shape[:2] == (height,width)
        observed, credit, first, previous, confidence, first_trigger = [], 0., None, None, 1., None
        for row in trace:
            track = next((t for t in row["tracks"] if t["track_id"] == event["track_id"]), None)
            if track is None:
                continue
            detection = next(d for d in row["detections"] if d["bbox"] == track["bbox"])
            x1,y1,x2,y2 = track["bbox"]
            inside = polygon.covers(Point((x1+x2)/2,y2))
            qualified = inside and all(detection["attributes"].get(k)==v for k,v in rule["attributes"].items())
            assert inside == track["inside"] and qualified == track["qualified"]
            now = row["video_time_s"]
            if previous and qualified and previous[1] and now-previous[0] <= 1.5/manifest["tracking"]["sample_fps"]:
                credit += now-previous[0]
            if qualified:
                first = now if first is None else first
                confidence = min(confidence,detection["confidence"],detection["attribute_confidence"])
                if rule["attributes"]:
                    assert detection["attribute_label"] in manifest["model"]["vocabulary"]
            assert math.isclose(credit,track["dwell_s"],abs_tol=1e-6)
            exceeds = (round(credit,6) > rule["dwell_threshold_s"] if rule["comparison"]=="gt"
                       else round(credit,6) >= rule["dwell_threshold_s"])
            if qualified and exceeds and first_trigger is None:
                first_trigger = now
            observed.append((row,track))
            previous = now,qualified
            if math.isclose(now,record["video_time_s"],abs_tol=1e-9):
                assert math.isclose(credit,event["dwell_s"],abs_tol=1e-6)
                assert math.isclose(confidence,event["confidence"],abs_tol=1e-7)
                assert event["needs_verification"] == (event["confidence"] < .65)
                assert datetime.fromisoformat(event["ts_start"]) == anchor+timedelta(seconds=first)
                assert datetime.fromisoformat(event["ts_end"]) == anchor+timedelta(seconds=now)
        assert first_trigger == record["video_time_s"]
        event_results.append({"track_id": event["track_id"],"video_time_s":record["video_time_s"],
                              "dwell_s":event["dwell_s"],"confidence":event["confidence"],
                              "needs_verification":event["needs_verification"]})
    for sample in manifest["review_samples"]:
        assert sample["frame_index"] in [r["frame_index"] for r in trace]
        image = cv2.imread(str(run/sample["filename"]))
        assert image is not None and image.shape[:2] == (height,width)
    assert len(manifest["review_samples"]) == manifest["review_frames"]
    return manifest, trace, {"samples":len(trace),"events":event_results,"review_frames":manifest["review_frames"]}


def main(runs_root):
    root = Path(runs_root)
    results = {}
    manifests, traces = {}, {}
    for name in ("mot17_bag_demo","car_noun_smoke","short_clip_negative"):
        manifests[name],traces[name],results[name] = audit(root/name)
    bag,car,negative = (manifests[n] for n in ("mot17_bag_demo","car_noun_smoke","short_clip_negative"))
    assert bag["rule"]["target_class"]=="person" and bag["rule"]["attributes"]=={"carrying":"bag"}
    assert bag["rule"]["dwell_threshold_s"]==10 and len(bag["events"])==1
    assert bag["review_frames"]==3 and bag["complete_video"]
    assert car["rule"]["target_class"]=="car" and len(car["events"])==1
    assert len(negative["events"])==0 and negative["source_frame_limit"]==150
    assert bag["model"]["sha256"] == car["model"]["sha256"] == negative["model"]["sha256"]
    gt = defaultdict(list)
    config = configparser.ConfigParser()
    config.read(ROOT/"task1/demo_source/seqinfo.ini")
    sx = bag["video"]["width"]/config.getint("Sequence","imWidth")
    sy = bag["video"]["height"]/config.getint("Sequence","imHeight")
    with (ROOT/"task1/demo_source/gt.txt").open() as file:
        for values in csv.reader(file):
            frame,identity,x,y,w,h,mark,cls,visibility = map(float,values)
            if int(mark)==1 and int(cls)==1:
                gt[int(frame)].append((int(identity),[x*sx,y*sy,(x+w)*sx,(y+h)*sy]))
    target_id = bag["events"][0]["track_id"]
    matches = []
    for row in traces["mot17_bag_demo"]:
        track = next((t for t in row["tracks"] if t["track_id"]==target_id),None)
        if track:
            overlap,identity = max((box_iou(track["bbox"],bbox),identity)
                                   for identity,bbox in gt[row["frame_index"]+1])
            assert overlap >= .3, "No sufficiently overlapping independent pedestrian label"
            matches.append({"frame_index":row["frame_index"],"ground_truth_id":identity,
                            "iou":overlap,"qualified":track["qualified"]})
    assert matches and len({m["ground_truth_id"] for m in matches})==1, "Alert track changes pedestrian identity"
    results["independent_pedestrian_identity"] = {
        "track_id":target_id,"ground_truth_id":matches[0]["ground_truth_id"],
        "sample_count":len(matches),"mean_iou":float(np.mean([m["iou"] for m in matches])),
        "minimum_iou":min(m["iou"] for m in matches),"matches":matches,
        "scope":"Only the alerting pedestrian identity; no bag/carrying labels or global IDF1 claim."}
    destination = root/"task1_verification.json"
    destination.write_text(json.dumps({"passed":True,"results":results},indent=2))
    print(f"PASS: source/model hashes, schemas, JPEG bytes, dwell arithmetic, timing, duplicate prevention, "
          f"three privacy examples, negative control, new target noun and pedestrian identity. {destination}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", default=str(ROOT/"task1/outputs"))
    main(p.parse_args().runs_root)
