# Two-dataset fog validation implementation plan

## Goal and scope

Make the analysis answer two questions separately: does density conditioning preserve normal-use detection, and does it help on foggy images? Evaluate (1) the existing nuScenes benchmark and (2) labeled Foggy Cityscapes with matched clear/foggy images. Existing profiles and run directories remain readable.

The fog result must state training scope. A detector trained and tested on Foggy Cityscapes is the primary in-domain fog track. Testing a nuScenes-trained detector there is a secondary cross-domain diagnostic because domain shift can mimic a fog effect.

## Contracts and proposed changes

`fog_prepare.py` already provides scene-safe nuScenes partitions and projected 2D boxes; beta is unknown, so nuScenes allows only `rgb_control` and `predicted_density`. `fog_data.py` parses Foggy Cityscapes beta, builds boxes from `gtFine`, and can retain clear counterparts. Make those pairs and dataset provenance first-class records.

Add `evaluation_datasets`, `fog_detector_scope`, and `fog_density_bins` to `SETTINGS` and the run fingerprint. Changing them after a checkpoint exists fails with the existing mismatch error. Foggy Cityscapes allows `oracle_density`; nuScenes rejects it. Missing masks, malformed/out-of-range beta, duplicate pair IDs, and split leakage fail before training. Legacy reports load with null dataset/bin fields.

## Imbalance policy

Counts are unequal (about 204,894 nuScenes versus 10,425 Foggy Cityscapes), but these are separate tasks, not one pooled class balance. Do not downsample nuScenes or pool scores. Report each dataset and delta separately; an optional headline may weight datasets equally. Keep clear/foggy pairs intact, stratify Foggy and density training by beta bin, and report natural-distribution plus balanced-bin results. Bootstrap within dataset/bin so nuScenes volume cannot dominate the fog claim.

## Data, outputs, and flow

1. `notebooks/_support/fog_data.py` emits dataset, clear/foggy condition, pair/group ID, beta and bin metadata; `fog_prepare.py` keeps nuScenes provenance.
2. `notebooks/_support/fog_training.py` resolves track/model/arm/seed/profile, trains the density estimator, then trains and evaluates using the declared scope. Dataset-track directories and locks prevent checkpoint mixing.
3. `notebooks/_support/fog_research.py` runs the same COCO bbox metrics per dataset/model/arm. Foggy Cityscapes adds clear-vs-foggy paired deltas and beta-bin results; density quality and latency stay separate. `full` keeps the nuScenes held-out test; partials keep preliminary validation.
4. `fog_analytics.py`, `fog_notebook.py`, `scripts/create_colab_notebook.py`, and the consolidated notebook expose scopes and export dataset/condition/bin columns in `results.csv`, paired rows, seed summaries and side-by-side figures.

The manifest plus run `settings.json` is authoritative. Reports are reproducible; completed reports reuse safely, interrupted evaluation resumes per dataset, and writes are atomic.

## Validation and rollout

Extend the existing tests or add a focused suite covering:

- positive: both datasets, all three models, valid control/predicted arms, Foggy Cityscapes oracle, pairs and bins;
- negative/boundary: missing masks, malformed/edge beta, empty bins, duplicate pairs, no boxes/detections, unsupported nuScenes oracle, mixed scopes, stale manifests and legacy CSV;
- correctness: pycocotools parity, exact pairing, deterministic bins, per-class metrics, seed aggregation and finite latency;
- recovery/compatibility: interrupted density/detector/evaluation resume, idempotent completed runs, lock races, changed-setting rejection, partial writes and dependency failure;
- performance: bounded validation, no duplicate paired inference, and full-track time/memory.

Roll out with `setup`, a partial profile on both tracks, then `full`. Existing partials remain nuScenes-only until rerun; missing fog results cannot be backfilled. Record the selected scope in configuration and reports.
