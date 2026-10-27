# Research footage and annotation attribution

The demo uses the **MOT17-04-DPM raw sequence preview**, introduced in MOT16 and distributed by MOTChallenge. The official sequence page does not name an individual photographer.

- Publisher and sequence: https://motchallenge.net/data/MOT17/
- Unmodified source video: https://motchallenge.net/sequenceVideos/MOT17-04-DPM-raw.mp4
- Independent pedestrian annotations: https://motchallenge.net/data/MOT17Labels.zip
- Publisher-declared dataset license: **Creative Commons Attribution-NonCommercial-ShareAlike 3.0**: https://creativecommons.org/licenses/by-nc-sa/3.0/

This package is a private non-commercial research/evaluation demonstration. Preserve attribution and the same dataset-license terms for footage derivatives. This is not a grant of commercial deployment rights. MOTChallenge notes that additional original-source terms can apply to third-party sequences.

The source MP4 bytes are unchanged. It is the official 960x540, 30fps, 1,050-frame preview, while the annotation coordinates refer to 1920x1080 source frames; the audit scales coordinates by 0.5. No slowing, looping, frame duplication or artificial dwell extension is used.

Evidence images are derivatives: the whole frame is Gaussian-blurred, detected face/head regions receive stronger mean masks, and zone/object/track annotations are added. These footage derivatives retain the dataset's CC BY-NC-SA 3.0 terms.

The image-space walkway and parking polygons were defined for this substitute. It is a pedestrian street, not KavachX's unavailable loading bay. Person IDs and boxes come from the original annotations; bag/carrying relationships are not annotated by MOTChallenge and are not validated by that ground truth.

Model software/weights: Ultralytics YOLO-World, https://docs.ultralytics.com/models/yolo-world/ and https://www.ultralytics.com/license . Text encoder: the pinned Ultralytics CLIP fork, https://github.com/ultralytics/CLIP . These components retain their respective upstream licenses.

Exact download URLs, byte hashes and transformation notes are in provenance.json and task1/prepare_demo.py.
