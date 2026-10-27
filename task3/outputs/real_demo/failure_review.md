# Held-out failure review

The calibration-only cutoff is 0.31879428029060364. Four of fifteen human-positive test images fall below it; six of fifteen animal-only images exceed it. Review `test_decisions.csv` for every decision and `annotation_review.jpg` for the 30 original frames with publisher boxes overlaid. Contact-sheet captions/boxes are review aids and were not model inputs. Human boxes are red; animal boxes are cyan. All labels are presence proxies from source annotations.

| Held human-positive frame (sequence / frame) | Score | Source human box size(s), pixels |
|---|---:|---|
| 0000000362_0000000000 / 518 | 0.301367 | 14 × 26 |
| 0000000355_0000000000 / 1586 | 0.298873 | 42 × 50 |
| 0000000371_0000000000 / 1358 | 0.299073 | Eight boxes, widths 14–24, heights 24–36 |
| 0000000355_0000000000 / 1900 | 0.271180 | 109 × 53 |

Visual review confirms very small/distant thermal subjects in much of the benchmark, with vegetation, changing contrast and varied body orientations. The last missed frame contains a wider, near-horizontal source human box, so small size alone cannot explain every miss. Full-frame downsampling can lose detail; thermal-domain mismatch and pose are plausible contributors, not experimentally isolated causes. Multiple animals also have bright upright silhouettes. The source-box sheet is evidence of annotation locations, not proof that every label is correct.

The initial calibration run retained 10/10 positives, but this failed to transfer to the official TestReal subset. Ten calibration positives and five positive test recordings cannot establish a production recall guarantee. Do not adjust the threshold or prompts on these known test failures and call the same images an untouched holdout. Keep these as regression cases, collect fresh camera/day-separated data, and validate a small-target or domain-specific improvement on new data. Until then, hold/review decisions must not become silent suppression.
