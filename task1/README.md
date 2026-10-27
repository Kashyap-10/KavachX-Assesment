# Task 1: English rule to a privacy-masked event

This is a completed CPU demonstration on **documented substitute footage**. The original loading-bay clip was not provided. No synthetic scene, slowed video, hardcoded detection, trained custom detector or API key is used.

## What was actually run

| Check | Observed result |
|---|---|
| Person carrying a bag, walkway, more than 10 seconds | One candidate event at video t=34.3s, accumulated qualifying dwell 10.1s |
| Five-second excerpt of the same video and same ten-second rule | Zero events |
| Change the noun to car, parking zone, more than 3 seconds | One event at t=3.1s, using identical pretrained model bytes |
| Independent pedestrian identity check | Alert track 244 matches MOT ground-truth ID 100 in all 143 sampled observations; mean box IoU 0.8557 |
| Privacy artifacts | Three masked examples plus the event's exact embedded JPEG |
| Automated checks | 17 Task 1 regression tests; artifact/source/timing/identity audit passes |

The bag event confidence is 0.110155, below the configured 0.65 verification threshold, so **needs_verification is true**. It is a candidate for a secondary verifier, not an automatically verified intrusion or compliance decision. The confidence is the minimum target/item detector score across qualifying observations, not a calibrated probability.

The independent annotations establish continuity for the alerting pedestrian only. They do not label bags, carrying, zone occupancy or all tracker identities. Visual review of source crops at 24, 26, 28, 30, 32 and 34 seconds shows the same person with a backpack throughout. No precision/recall or global tracking accuracy is claimed from these examples.

## Run from a clean environment

Tested on Python 3.12.14, Apple M4 CPU, PyTorch 2.8.0 and Ultralytics 8.3.235. A separate virtual environment installed only task1/requirements.txt and passed the Task 1 tests, the saved-output audit, and a real car-detection smoke run. Task 1 needs no GPU or training.

From the extracted package/repository root:

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r task1/requirements.txt
    python task1/prepare_demo.py

The setup verifies or downloads the original research clip, its independent tracking annotations and YOLO-World weights using recorded SHA-256 hashes. It never overwrites a differing local asset. Initial class registration also downloads the approximately 338 MiB OpenAI CLIP encoder through the pinned Ultralytics CLIP fork; internet access is needed for the first run. Later runs use its cache. Direct dependencies are pinned; transitive versions from the verified environment are in task1/outputs/environment.txt.

Run the actual bag rule:

    python task1/run_task1.py --model models/yolov8s-worldv2.pt --output-dir task1/outputs/reproduce_bag

An output directory must be empty to prevent stale or mixed results. Omitting --output-dir creates a timestamped run directory. Existing packaged measurements remain untouched.

Explicit equivalent command:

    python task1/run_task1.py \
      --rule "Alert me if a person carrying a bag enters the walkway and stays more than 10 seconds." \
      --video clips/mot17-04.mp4 \
      --zones task1/demo_zones.json \
      --model models/yolov8s-worldv2.pt \
      --stride 3 --imgsz 960 \
      --output-dir task1/outputs/reproduce_bag

The source is 30fps and every third frame is inferred, giving ten samples per video second. Dwell uses source frame index / FPS, not processing speed. This file implementation assumes constant-rate footage; it is not a live-stream or variable-frame-rate timing implementation.

A real negative control:

    python task1/run_task1.py --model models/yolov8s-worldv2.pt --max-frames 150 --output-dir task1/outputs/reproduce_short

A real new-noun control, with no retraining:

    python task1/run_task1.py \
      --rule "Alert me if a car enters the parking and stays more than 3 seconds." \
      --model models/yolov8s-worldv2.pt --max-frames 150 \
      --output-dir task1/outputs/reproduce_car

The two five-second controls are explicitly marked incomplete-video runs. Their lack of three quarter-point review frames does not affect the complete main demo, which has all three.

## Supported English and detection

The bounded parser accepts:

    Alert me if a <target> [carrying/wearing/holding a <item>] enters the <zone> and stays more than/at least <number> seconds/minutes.

It also supports the documented aliases visible in rule_parser.py: notify me if/alert if/if; moves into/is inside/is in; dwells/remains; over/greater than; secs/s/mins/m. Object nouns and multiword names are extracted from the input, not chosen from a closed list. Worker/human/pedestrian normalize to person; holding normalizes to carrying.

Unsupported text, missing durations, negation, compound conditions, invalid polygons, unknown zones and missing videos raise errors. There is no LLM call, API-key requirement, default-person fallback, zone substitution or synthetic-video fallback.

YOLO-World registers the extracted target and requested item vocabulary. A requested generic bag includes bag, backpack, handbag and suitcase. A toolbox accepts only toolbox. The same bag cannot qualify two actors, duplicate synonym detections are suppressed, and wearing items must lie near the upper body. Spatial association is an approximation: proximity does not prove physical ownership or carrying.

## Tracking, dwell and event semantics

Ultralytics ByteTrack tracks all target detections, including people without a currently observed bag. A missing item does not immediately replace a person's identity. New tracks require confidence 0.25; scores down to 0.1 can support tracking/association and are exposed for verification.

The bottom-center of the detected target box is tested against the polygon, including its boundary. Total dwell accumulates across visits while an identity survives. Only adjacent samples that both satisfy the zone and attribute conditions contribute time. Missing detections, occlusions, outside-zone samples and missing attributes add no time. This is a conservative lower bound, rather than a claim about hidden intervals. Objects already inside when monitoring starts are counted from the first qualifying observation.

More than is strictly greater than; at least includes equality. A track emits once for this rule over the run. After a long occlusion or identity switch a physical person can receive a new track ID; no cross-ID duplicate guarantee or re-identification is claimed.

ISO timestamps are based on either --video-start with a timezone-aware capture start, or a documented run-start UTC anchor. The original clip's recording time is unknown. The latter timestamps must not be presented as authentic historical capture times.

Events validate against event_schema.json, including types, UUID, confidence bounds and ordered timestamps. Invalid events stop the run. sample_event.json is an exact copy of the first event, not a second violation. The evidence JPEG file contains exactly the bytes embedded in that event.

## Privacy and artifacts

The entire evidence image is Gaussian-blurred to cover background people missed by detectors. Frontal/profile face detections and generous head regions for every detected person receive additional coarse mean masks. Nonqualifying bystanders are included. Bounding boxes and zone lines are drawn only after masking. This sacrifices visual detail for conservative privacy; it is not a formal de-identification guarantee.

The packaged three images and both event images were visually reviewed. Source video is retained privately for reproducibility. Do not publish raw footage or original assessment assets.

Packaged verified output folders:

- task1/outputs/mot17_bag_demo/: sample_event.json, one event JSON, its exact evidence JPEG, three masked_frame images, trace.jsonl and run_manifest.json.
- task1/outputs/car_noun_smoke/: real target-noun smoke test.
- task1/outputs/short_clip_negative/: real negative time-threshold control.
- task1/outputs/task1_verification.json: independent audit and 143 ground-truth matches.

Run the checks:

    python -m unittest discover -s tests -p test_task1.py -v
    python task1/verify_demo.py

verify_demo.py audits the three named packaged runs. To audit a newly regenerated suite with that script, run the scenarios into the same names under a fresh alternate directory and pass --runs-root. UUIDs and UTC anchors vary across runs; event policy, times and source/model hashes are what matter.

## Provenance and remaining scope

Source: https://motchallenge.net/data/MOT17/ . The exact unmodified sequence preview is 960x540, 30fps and 35 seconds. Ground-truth coordinates are scaled from 1920x1080. See task1/demo_source/provenance.json and ATTRIBUTION.md for exact URLs, hashes and the publisher's CC BY-NC-SA 3.0 terms. Evidence derivatives retain those dataset terms. This is a private non-commercial research demonstration, not a commercial-use license.

The walkway and parking polygons match this substitute. The scene is not a loading bay. Original zones.json remains separate and is not silently reused or rescaled.

AI assistance was used to implement and review the parser, detector association, tracking/dwell, privacy, event validation, setup/audit scripts and tests, and to source/review demo assets. The candidate must inspect, understand, explain and modify the code; this package does not establish that they already have.

No services, repositories, controllers or new architecture layers were added. Existing Task 1 modules were repaired. Setup and artifact-audit scripts, a demo zone/source manifest, tests and pinned Task 1 dependencies were added. ByteTrack uses the existing Ultralytics dependency plus lap; CLIP is pinned to a specific upstream commit.

This package completes the Task 1 substitute demonstration. It does not complete Task 3, Task 2 motion gating, PPE-data cleanup or retraining.
