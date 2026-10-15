# Fog notebook visible workflow implementation plan

## Goal and scope

Make top-level code show concrete datasets, model names, arms, and validation targets. Beginners should not need to decode generic calls such as `load_study_data(...)` and `train_detector_profile(...)`. Keep one self-contained notebook; add no helper modules or copies.

Keep the research question, profile budgets, six detector comparisons, matched clear/fog analysis, checkpoint resume, run identity, and Colab A100 optimizations unchanged. Change presentation and function boundaries, not the experiment.

## Data roles and visible flow

Name datasets by purpose:

- `nuscenes_data`: labeled camera images for detector training and its usual validation/test splits.
- `fog_density_training_data`: Foggy Cityscapes images and filename-derived beta labels used only to fit the frozen density predictor. These images are not detector negatives.
- `labeled_clear_fog_data`: matched Cityscapes clear/fog images with `gtFine` labels, used only for the separate fog hypothesis evaluation. Require labels for both conditions; density-only images are not detection validation data.

Visible cells read in this order:

1. Choose profile and data/output locations.
2. Load nuScenes detector data, Foggy Cityscapes beta-training data, and (for its analysis) labeled clear/fog evaluation data in separately named variables.
3. `fog_predictor = fit_fog_density_model(fog_density_training_data, ...)` always restores the checkpoint when present or initializes a fresh model; it resumes training and returns frozen weights, or stops immediately when complete.
4. Prepare nuScenes partitions once. Show six `train_and_validate_detector(...)` calls with named results and explicit model/arm arguments: YOLO11n, YOLO11m, and DETR, each RGB-only and with predicted density. Apply profile limits only when selecting detector-training rows.
5. Each named training call evaluates its model on the profile’s nuScenes validation/test split. A separate cell calls `load_labeled_clear_fog_data()` and passes those paired labels to saved-model fog evaluation.
6. Display the saved training, detection, density, fog-versus-clear, and latency analyses; export the run.

Share the repeated training mechanics below the visible calls; do not add six trivial wrappers. Name progress/results by model and arm.

## Existing code and proposed changes

- `notebooks/fog_density_study.ipynb`: separate dataset calls; replace generic profile-wide training with six explicit model/arm calls; show validation targets and keep matched clear/fog analysis separate. Group implementation in collapsed sections by workflow stage; keep boundary checks step-level.
- `scripts/check_notebook_readability.py`: assert call order, dataset/result names, all models and arms, validation targets, collapsed helpers, and no generated support files.
- `tests/`: check model/arm selection, profile budget, checkpoint namespace, data/split separation, and evaluation of saved models without retraining.
- `README.md`: link this plan and describe the explicit flow.

No schema migration or persistence layer is needed. Preserve manifest formats, profile versions, checkpoint paths/payloads, result columns, resume behavior, and the single Drive notebook.

## Risks and safeguards

Explicit calls can drift apart; keep mechanics shared and assert the six calls are present once. Keep density-training images separate from labeled detection-evaluation pairs. Do not present nuScenes scores as fog-specific evidence or claim detection validation without `gtFine` labels. Preserve scene splits, seeds, arm parity, checkpoint recovery, and batch/data-loader optimization.

## Test and rollout

Run unit tests and the notebook code, readability, and bounded end-to-end checks. Use ZIP mode only when full extraction is intended. Compare profile fingerprints, checkpoints, arms, splits, and report schema before/after. Confirm names in the notebook before syncing it to Drive.
