# Task 3: measured secondary verifier

Working CLIP implementation and a reproducible **substitute demonstration**. The original 30 KavachX snapshots and their labels were not supplied. This does not complete validation against those missing assets.

## Measured result

| Pipeline on 30 test images | TP | FP | FN | TN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| All edge candidates forwarded | 15 | 15 | 0 | 0 | 50.0% | 100.0% |
| CLIP score >= 0.3187942803 | 11 | 6 | 4 | 9 | 64.7% | 73.3% |

Nine of fifteen false alarms are filtered (60%), but four human-positive candidates are held. **The recall target fails on the test set. Do not automatically dismiss alerts using this model.** The useful result here is an honest baseline and an auditable evaluation, not a production-ready intrusion detector. Held candidates should remain available in a review queue; this command-line prototype writes decisions, not a dashboard or notification service.

## Run

Use Python 3.12 (tested: 3.12.14), Git and a CPU. From the repository/package root:

```bash
python3 -m venv .venv-task3
source .venv-task3/bin/activate
python -m pip install -r task3/requirements.txt
python task3/prepare_demo.py
python -m unittest discover -s tests -p test_task3.py -v
python task3/verifier.py --output-dir task3/outputs/my_run
python task3/evaluate.py --scores task3/outputs/my_run/verifier_scores.csv --output-dir task3/outputs/my_run
python task3/verify_demo.py --run-dir task3/outputs/my_run --rescore
```

Use a new empty scoring output directory. `prepare_demo.py` restores missing public images/annotations from the fixed manifest, verifies SHA-256 hashes, re-derives labels from source annotations and creates `alerts/task3_demo/labels.csv`. It rejects a changed existing asset. Missing input, corrupt image, duplicate bytes, source-group leakage or model download failure aborts; there is no mock-image or alternative-classifier fallback. An interrupted scoring run leaves its configuration for diagnosis: choose a fresh output directory to retry. The local Python environment must trust HTTPS; do not disable certificate verification.

First model use downloads approximately 338 MiB from OpenAI to `models/clip/ViT-B-32.pt`; no API key, paid service, GPU or training is required. Weights are excluded from the compact ZIP. Their SHA-256 is `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`. Thereafter scoring runs offline. Data and original annotations are included in the ZIP. Model loading and text encoding are excluded from the saved per-image latency. The measured warm CPU average was 18.79 ms, p95 19.48 ms, using four Torch threads on this development machine; these are not edge-hardware measurements.

## Source and labels

The inputs are original real nighttime thermal images from [BIRDSAI / LILA](https://lila.science/datasets/conservationdrones/), credited to Elizabeth Bondi and collaborators: *BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos*, WACV 2020. See the [author dataset page](https://sites.google.com/view/elizabethbondi/dataset), [original annotation documentation](https://github.com/exb7900/BIRDSAI), [fixed manifest](demo_source/manifest.json) and [CDLA-Permissive-1.0 terms](demo_source/LICENSE.txt).

- 20 calibration images: 10 human-positive, 10 animal-only; official TrainReal partition, 11 recording groups.
- 30 test images: 15 human-positive, 15 animal-only; official TestReal partition, 16 recording groups.
- Any original class-1 box means positive. Negative means annotated animals with no human box. This is a **human-presence proxy**, not evidence of unauthorized intrusion, identity or intent.
- Selection used seed 42, recording-group round robin over annotated-frame quartiles, before model scoring. The manifest fixes every source URL, source annotation and hash. No synthetic BIRDSAI frames, label-coded generated inputs, ground-truth-guided crops or score-selected samples were used.
- Calibration and test recording prefixes are disjoint; all 50 image hashes are unique. The positive test images come from only five recordings (three frames each), so these are correlated examples, not 30 independent sites. Broader site/individual independence is unknown.
- Aerial thermal data differs from fixed CCTV. Rain, shadows, insects and authorized humans are not represented by a validated negative set. Absence of a human annotation is not independently exhaustive manual ground truth.

Original JPGs and annotation CSVs are unchanged. `labels.csv` and the manifest are derived additions. `annotation_review.jpg` is a derived review sheet with source boxes/labels; it was never passed to CLIP. The selection implementation is retained in `demo_source/select_subset.py` for transparency; it expects the original full annotation inventory in the documented scratch layout. Routine restoration uses `prepare_demo.py` and requires no full dataset archive.

## Scoring and threshold

OpenAI CLIP ViT-B/32, loaded through the pinned Ultralytics CLIP fork, encodes pixels and two predeclared prompt ensembles from `prompts.json`. Each text embedding is normalized; each class mean is normalized again. Image cosine similarities are divided by temperature 0.05 and a two-class softmax supplies the human score. **It is not a calibrated probability.** Images are converted to RGB and padded with black to preserve the whole frame before official 224-pixel preprocessing. The filename, label and source group are not supplied to the image encoder.

We first measured the 20 calibration images with this configuration, then kept the model, prompts, preprocessing and selection policy fixed for the test evaluation. Full-frame CLIP was retained as the simplest baseline; no test-driven prompt or crop optimization was performed. The low resolution can erase small human detail. Proposed changes now require fresh validation data because these test outcomes have been seen.

The threshold minimizes calibration false positives subject to empirical recall >=95%; ties retain more positives, then choose the largest threshold. With only ten calibration positives, the floor requires retaining all ten. The selected threshold retained 10/10 and rejected 4/10 animal-only calibration images. This small result does not demonstrate 95% population recall. The separate test then exposed four misses. `threshold.json` fingerprints the score table; `scoring_config.json` records weight, prompt, source-code and label versions. The full calibration sweep is saved. No test labels are used to choose the threshold, and the threshold has not been changed to improve the reported test result.

At threshold zero all candidates would be forwarded: recall stays 100% within this supplied candidate set, but no false-alarm benefit remains. Neither baseline establishes upstream edge-detector recall; the edge itself was not run on these 30 candidates. Balanced subset precision also does not transfer to a site's actual event prevalence.

## Artifacts and remaining work

`outputs/real_demo/` contains per-image scores/timing, calibration sweep, fixed threshold, 30 test decisions, JSON/Markdown metrics, failure review, source-box contact sheet and verification evidence. Legacy files directly under `task3/outputs/` belong to the old unverified implementation and are excluded from the Task 3 ZIP. Only `real_demo` results are accepted here.

See [NOTES.md](NOTES.md) for monthly cost, feedback design, AI disclosure and the next-week plan. Before actual deployment/submission acceptance, obtain the assessment's original snapshots or explicitly agree the substitute with the reviewer; collect representative CCTV calibration and unseen evaluation data across cameras/days/weather; verify authorization labels; and re-evaluate recall before enabling suppression. No original-asset accuracy claim is made.
