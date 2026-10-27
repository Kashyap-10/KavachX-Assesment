# Visual review of the five observed failures

These interpretations follow inspection of the actual annotated validation images. Error counts are in failure_cases.json. Confidence is fixed at 0.25 for this diagnostic, whereas validation recall uses Ultralytics' operating point. Causes below are hypotheses, not established internal model explanations.

1. **image1121.jpg:** A crowded water-sports scene contains overlapping, partly obscured helmets. Source boxes often cover faces as well as helmets. Small targets, occlusion and the annotation convention plausibly contribute to localization errors. This is outside factory CCTV; audit the labels before changing the model.
2. **image144.jpg:** A crowded close-up contains occluded heads and several bare heads labeled helmet. Annotation noise is visible alongside genuine detection misses. Re-annotate with a consistent helmet/head convention before interpreting every mismatch as a safety failure.
3. **image130.jpg:** Workers and helmets are obscured by dense vertical rods, and some targets are partly cropped. Occlusion is a plausible cause. Audit visible-region labels and add representative occluded training examples.
4. **image117.jpg:** A line of small helmeted workers includes overlapping boxes and targets cut off at the right edge. Small scale and truncation plausibly contribute. Check annotation alignment and higher-resolution inference before adding a new detection head.
5. **image1217.jpg:** All four no_helmet boxes in a non-construction group portrait are missed at confidence 0.25. This demonstrates poor bare-head generalization and source-domain mismatch. It does not establish performance on factory workers. Collect and review site-relevant negative examples.

The substitute data is suitable for exercising the pipeline, but this visual audit prevents treating its score as a clean factory-PPE benchmark. Keep the original validation labels fixed for the reported comparison; a corrected dataset needs a separately versioned evaluation.
