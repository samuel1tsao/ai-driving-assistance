# Fog notebook visible workflow implementation plan

## Goal and scope

Make top-level code show concrete datasets, model names, arms, and validation targets. Train detector comparisons on labeled fog and clear examples as well as nuScenes, so the predicted-density arm can learn from foggy object-detection cases. Keep one self-contained notebook; add no helper modules or copies.

Keep the six detector comparisons, matched clear/fog analysis, checkpoint recovery, run identity, and Colab A100 optimizations. This changes the training data, so use fresh profile versions and preserve previous checkpoints.

## Data roles and visible flow

Name datasets by purpose:

- `nuscenes_data`: labeled camera images for detector training and its usual validation/test splits.
- `fog_density_training_data`: Foggy Cityscapes images and filename-derived beta labels used to fit the frozen density predictor.
- `labeled_cityscapes_data`: the same Foggy Cityscapes images joined to `gtFine` object boxes, plus matched clear images and boxes. These rows train and validate the detector; they use the beta predictor's scene split so its training groups stay out of detector validation and test.
- Both RGB and predicted-density detector arms train on the same nuScenes and labeled Cityscapes examples. The RGB arm gets no density signal; the density arm gets the predictor's estimate. This isolates the effect of supplying the density value while allowing both models to learn object detection under fog.

Visible cells read in this order:

1. Choose profile and data/output locations.
2. Load nuScenes detector data and Foggy Cityscapes beta-training data, then attach `gtFine` boxes and matched clear examples from the configured `cityscapes-labeled` folder.
3. `fog_predictor = fit_fog_density_model(fog_density_training_data, ...)` always restores the checkpoint when present or initializes a fresh model; it resumes training and returns frozen weights, or stops immediately when complete.
4. Prepare combined nuScenes and labeled Cityscapes partitions once. Show six `train_and_validate_detector(...)` calls with named results and explicit model/arm arguments: YOLO11n, YOLO11m, and DETR, each RGB-only and with predicted density. Partial caps are shared evenly across the two datasets and applied only to training rows.
5. Each named call reports nuScenes and held-out Cityscapes results. The analysis cell calls `load_labeled_clear_fog_data()`, which selects the held-out split from the same cached labeled manifest used by training.
6. Display the saved training, detection, density, fog-versus-clear, and latency analyses; export the run.

Share the repeated training mechanics below the visible calls; do not add six trivial wrappers. Name progress/results by model and arm.

## Existing code and proposed changes

- `notebooks/fog_density_study.ipynb`: load labeled Cityscapes for detector training, align its scene split with the beta predictor, combine it with nuScenes, and retain six explicit model/arm calls plus held-out matched clear/fog analysis from the shared labeled manifest. The configured labeled root must contain `gtFine/` and matched `leftImg8bit/` files.
- `scripts/check_notebook_readability.py`: assert call order, dataset/result names, all models and arms, validation targets, collapsed helpers, and no generated support files.
- `tests/`: check model/arm selection, profile budget, checkpoint namespace, data/split separation, and evaluation of saved models without retraining.
- `README.md`: link this plan and describe the explicit flow.

The labeled detector manifest adds a detector-specific split alongside the predictor split. Preserve checkpoint payload formats and result columns; use new profile versions because the input manifest changes. Keep the single Drive notebook.

## Risks and safeguards

Explicit calls can drift apart; keep mechanics shared and assert the six calls are present once. Require real `gtFine` boxes and matched clear images before training on fog. Use scene-level splits shared with beta training so no beta-training scene appears in detector validation/test. Report results separately by dataset and condition. Preserve seeds, arm parity, checkpoint recovery, and batch/data-loader optimization.

## Test and rollout

Run unit tests and the notebook code, readability, and bounded end-to-end checks. Use ZIP mode only when full extraction is intended. Compare profile fingerprints, checkpoints, arms, splits, and report schema before/after. Confirm names in the notebook before syncing it to Drive.
